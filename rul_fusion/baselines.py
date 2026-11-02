"""Unified-protocol adapters for external baseline models.

The fusion model in `rul_fusion.model` borrows mechanisms only and never
imports the reference repositories.  Baseline comparison is a different
matter: to keep the comparison faithful to the published architecture, this
module loads the *unmodified* FC-STGNN model code from its repository and
wraps it behind the call contract used by `rul_fusion.runner`:

    prediction = model(x, w)
    prediction, regularization = model(x, w, return_aux=True)

Everything else -- engine-grouped split, train-only scaler, validation-based
checkpoint selection, single frozen test evaluation, metrics -- comes from
the shared runner, so the baseline is trained under exactly the same protocol
as the fusion model.  The original FC-STGNN training loop (window-level
validation folds and best-epoch selection by test RMSE) is deliberately not
used.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

# Per-dataset architecture hyperparameters, copied verbatim from
# FC_STGNN/main_RUL.py::args_config (the entries the model constructor needs).
FC_STGNN_HPARAMS = {
    "FD002": dict(patch_size=5, conv_out=7, lstmhidden_dim=8, lstmout_dim=12,
                  conv_kernel=2, hidden_dim=8, conv_time_CNN=10,
                  num_windows=14, moving_window=[2, 2], stride=[1, 2],
                  decay=0.7, pool_choice="mean"),
    "FD004": dict(patch_size=5, conv_out=7, lstmhidden_dim=8, lstmout_dim=6,
                  conv_kernel=2, hidden_dim=8, conv_time_CNN=10,
                  num_windows=14, moving_window=[2, 2], stride=[1, 2],
                  decay=0.7, pool_choice="mean"),
}

DEFAULT_FC_STGNN_ROOT = Path("/home/v2/FCSTGNN/FC_STGNN")


def load_fc_stgnn_class(repo_root: Path = DEFAULT_FC_STGNN_ROOT):
    """Import FC_STGNN_RUL from the unmodified reference repository."""
    repo_root = Path(repo_root)
    model_path = repo_root / "Model.py"
    if not model_path.exists():
        raise FileNotFoundError(
            f"FC-STGNN model not found at {model_path}; pass --fc_stgnn_root")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "the upstream FC-STGNN code calls .cuda() at construction time "
            "(Model_Base.py PositionalEncoding), so a CUDA device is required "
            "to build this baseline")
    # Model.py does `from Model_Base import *`, so its directory must be on
    # sys.path during execution; it is removed again right after.
    sys.path.insert(0, str(repo_root))
    try:
        spec = importlib.util.spec_from_file_location(
            "rul_fusion_baselines_fc_stgnn", model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(repo_root))
    return module.FC_STGNN_RUL


def to_fc_stgnn_layout(x: torch.Tensor, time_len: int) -> torch.Tensor:
    """[B, history, sensors] -> [B, time_len, sensors, patch].

    Reproduces CMPDataIter_graph.data_sampling followed by the transpose in
    Train.data_preprocess_transpose: the history axis is cut into time_len
    consecutive patches, then the patch axis moves behind the sensor axis.
    """
    if x.ndim != 3:
        raise ValueError("x must have shape [batch, time, sensors]")
    batch, history, sensors = x.shape
    if history % time_len:
        raise ValueError(
            f"history {history} is not divisible by time_len {time_len}; "
            "FC-STGNN needs history == patch_size * conv_time_CNN")
    patch = history // time_len
    return x.reshape(batch, time_len, patch, sensors).transpose(2, 3)


class FCSTGNNWrapper(nn.Module):
    """FC-STGNN baseline behind the runner's model(x, w) contract.

    The reference model never consumed its prepared operating-condition
    inputs (main_RUL.py calls net(data) only), so `condition` is accepted
    for interface compatibility and ignored, keeping the comparison honest
    to the published architecture.
    """

    def __init__(self, sensor_dim: int, history: int, dataset: str,
                 repo_root: Path = DEFAULT_FC_STGNN_ROOT,
                 hparams: Optional[dict] = None):
        super().__init__()
        dataset = dataset.upper()
        if dataset not in FC_STGNN_HPARAMS:
            raise ValueError(
                f"FC-STGNN hyperparameters are only registered for "
                f"{sorted(FC_STGNN_HPARAMS)} (tuned in the reference "
                "main_RUL.py), not for " + dataset)
        hp = dict(FC_STGNN_HPARAMS[dataset])
        if hparams:
            hp.update(hparams)
        self.hparams = hp
        self.sensor_dim = sensor_dim
        self.history = history
        self.time_len = hp["conv_time_CNN"]
        if history % self.time_len:
            raise ValueError(
                f"history {history} must equal patch_size * conv_time_CNN "
                f"(= multiple of {self.time_len}) for FC-STGNN")
        model_cls = load_fc_stgnn_class(Path(repo_root))
        # Argument order mirrors main_RUL.py:
        # FC_STGNN_RUL(patch_size, conv_out, lstmhidden_dim, lstmout_dim,
        #   conv_kernel, hidden_dim, conv_time_CNN, num_sensor, num_windows,
        #   moving_window, stride, decay, pool_choice, n_class)
        self.net = model_cls(
            hp["patch_size"], hp["conv_out"], hp["lstmhidden_dim"],
            hp["lstmout_dim"], hp["conv_kernel"], hp["hidden_dim"],
            hp["conv_time_CNN"], sensor_dim, hp["num_windows"],
            hp["moving_window"], hp["stride"], hp["decay"],
            hp["pool_choice"], 1,
        )

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None,
                return_aux: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        features = to_fc_stgnn_layout(x, self.time_len)
        prediction = self.net(features).squeeze(-1)
        if not return_aux:
            return prediction
        return prediction, prediction.new_zeros(())
