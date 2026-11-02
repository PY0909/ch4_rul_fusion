import sys
import csv
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from rul_fusion.data import DataBundle, FeatureScaler, WindowDataset
from rul_fusion.probabilistic import (EvidentialNIGRUL, HeteroscedasticRUL,
                                      gaussian_nll, nig_evidential_loss)
from rul_fusion.probabilistic_runner import (_edl_sensor_positions,
                                            fit_probabilistic)


def _samples(count, offset=0):
    rng = np.random.default_rng(19 + offset)
    rows = []
    for i in range(count):
        x = rng.normal(size=(6, 3)).astype(np.float32)
        w = rng.normal(size=(6, 2)).astype(np.float32)
        y = float(15.0 + x[-1].mean())
        rows.append((x, w, y, offset + i, i))
    return rows


def _bundle():
    scaler = FeatureScaler().fit(np.random.default_rng(1).normal(size=(8, 5)))
    return DataBundle(
        train=WindowDataset(_samples(12, 0)), val=WindowDataset(_samples(6, 20)),
        test=WindowDataset(_samples(4, 40)), scaler=scaler,
        metadata={"dataset": "FD002", "split_seed": 0, "train_units": list(range(12)),
                  "val_units": list(range(20, 26)), "calibration_units": list(range(30, 36)),
                  "test_units": list(range(40, 44))}, sensor_dim=3, condition_dim=2,
        val_endpoint=WindowDataset(_samples(4, 20)),
    )


def test_hnn_and_nig_outputs_are_valid_and_losses_are_finite():
    x = torch.randn(5, 6, 3)
    w = torch.randn(5, 6, 2)
    y = torch.rand(5) * 30.0
    hnn = HeteroscedasticRUL(3, 2, window_size=6)
    hnn_params = hnn(x, w)
    assert torch.all(hnn_params[:, 0] >= 0)
    assert torch.all(hnn_params[:, 1] > 0)
    assert torch.isfinite(gaussian_nll(hnn_params, y))
    edl = EvidentialNIGRUL(3, 2)
    nig_params = edl(x, w)
    assert torch.all(nig_params[:, 1] > 0)
    assert torch.all(nig_params[:, 2] > 1)
    assert torch.all(nig_params[:, 3] > 0)
    loss = nig_evidential_loss(nig_params, y)
    assert torch.isfinite(loss)
    loss.backward()


def test_paper_edl_is_five_layer_sensor_only_endpoint_mlp():
    model = EvidentialNIGRUL(4, 2, dropout=0.3, hidden_dim=16,
                             sensor_indices=(0, 2))
    linear_layers = [module for module in model.modules()
                     if isinstance(module, torch.nn.Linear)]
    assert len(linear_layers) == 5
    assert not any(isinstance(module, torch.nn.Conv1d) for module in model.modules())
    assert _edl_sensor_positions(
        {"sensor_indices": [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]},
        14) == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11)
    x = torch.randn(3, 6, 4)
    condition = torch.randn(3, 6, 2)
    model.eval()
    with torch.no_grad():
        output = model(x, condition)
        changed = x.clone()
        changed[:, :-1] += 100.0
        condition += 100.0
        output_changed = model(changed, condition)
    assert output.shape == (3, 4)
    assert torch.allclose(output, output_changed)


def test_probabilistic_runner_writes_metrics_and_recomputable_predictions(tmp_path):
    result = fit_probabilistic(_bundle(), "hnn", tmp_path / "hnn", epochs=2,
                               batch_size=4, patience=2,
                               dropout=0.0, device="cpu", seed=0)
    assert np.isfinite(result["NLL"])
    assert np.isfinite(result["CRPS"])
    assert 0.0 <= result["PICP"] <= 1.0
    for name in ("best.pt", "metrics.json", "history.csv", "predictions.csv",
                 "config.json", "manifest.json"):
        assert (tmp_path / "hnn" / name).exists()

    edl_result = fit_probabilistic(_bundle(), "edl_nig", tmp_path / "edl", epochs=1,
                                   batch_size=4, patience=1,
                                   device="cpu", seed=1)
    assert np.isfinite(edl_result["NLL"])
    assert np.isfinite(edl_result["CRPS"])
    with (tmp_path / "edl" / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        assert {"nig_gamma", "epistemic_variance", "aleatoric_variance"}.issubset(
            next(csv.reader(handle)))
    manifest = json.loads((tmp_path / "edl" / "manifest.json").read_text(encoding="utf-8"))
    assert "code_dirty" in manifest and "source_sha256" in manifest
    config = json.loads((tmp_path / "edl" / "config.json").read_text(encoding="utf-8"))
    assert edl_result["selection_criterion"] == "val_rmse"
    assert config["selection_criterion"] == "val_rmse"
    assert config["dropout"] == 0.3 and config["edl_hidden_dim"] == 128
    assert config["edl_sensor_ids_original"] is None  # synthetic bundle has no CMAPSS IDs
    try:
        fit_probabilistic(_bundle(), "hnn", tmp_path / "hnn", epochs=1,
                          batch_size=4, patience=1, device="cpu")
    except FileExistsError:
        pass
    else:
        raise AssertionError("runner must not overwrite a previous run")


def test_ensemble_trains_independent_members(tmp_path):
    result = fit_probabilistic(_bundle(), "ensemble", tmp_path / "ensemble",
                               epochs=1, batch_size=4, patience=1,
                               dropout=0.0, ensemble_members=2, device="cpu", seed=3)
    assert result["ensemble_members"] == 2
    assert (tmp_path / "ensemble" / "member_0.pt").exists()
    assert (tmp_path / "ensemble" / "member_1.pt").exists()
