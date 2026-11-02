import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from rul_fusion.baselines import (DEFAULT_FC_STGNN_ROOT, FCSTGNNWrapper,
                                  load_fc_stgnn_class, to_fc_stgnn_layout)

requires_repo = pytest.mark.skipif(
    not (Path(DEFAULT_FC_STGNN_ROOT) / "Model.py").exists(),
    reason="FC-STGNN reference repository not available")
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="upstream FC-STGNN calls .cuda() at construction time")


def test_layout_matches_reference_pipeline():
    # Reference: CMPDataIter_graph.data_sampling slices the history axis into
    # consecutive patches, then main_RUL transposes patch and sensor axes.
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(size=(3, 50, 14)).astype("f4"))
    patches = np.stack([x.numpy()[:, i * 5:(i + 1) * 5, :] for i in range(10)], axis=1)
    reference = torch.tensor(patches).transpose(2, 3)
    assert torch.equal(to_fc_stgnn_layout(x, time_len=10), reference)


def test_layout_rejects_indivisible_history():
    with pytest.raises(ValueError):
        to_fc_stgnn_layout(torch.randn(2, 47, 14), time_len=10)


def test_wrapper_rejects_untuned_dataset():
    with pytest.raises(ValueError):
        FCSTGNNWrapper(sensor_dim=14, history=50, dataset="DS03",
                       repo_root=DEFAULT_FC_STGNN_ROOT)


@requires_repo
@requires_cuda
def test_wrapper_forward_contract():
    model = FCSTGNNWrapper(sensor_dim=14, history=50, dataset="FD002").cuda()
    x = torch.randn(4, 50, 14).cuda()
    w = torch.randn(4, 50, 3).cuda()
    prediction = model(x, w)
    assert prediction.shape == (4,)
    assert torch.isfinite(prediction).all()
    prediction, regularization = model(x, w, return_aux=True)
    assert prediction.shape == (4,)
    assert regularization.ndim == 0 and float(regularization) == 0.0


@requires_repo
@requires_cuda
def test_wrapper_train_step_updates_batchnorm():
    model = FCSTGNNWrapper(sensor_dim=14, history=50, dataset="FD004").cuda()
    model.train()
    before = [b.running_mean.clone() for b in model.modules()
              if isinstance(b, torch.nn.BatchNorm1d)]
    x = torch.randn(8, 50, 14).cuda()
    y = torch.rand(8, device="cuda") * 100
    loss = torch.nn.functional.mse_loss(model(x, None), y)
    loss.backward()
    assert torch.isfinite(loss)
    after = [b.running_mean for b in model.modules()
             if isinstance(b, torch.nn.BatchNorm1d)]
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))


@requires_repo
@requires_cuda
def test_wrapper_runs_inside_shared_runner(tmp_path):
    from rul_fusion.data import DataBundle, FeatureScaler, WindowDataset
    from rul_fusion.runner import fit_and_evaluate

    rng = np.random.default_rng(3)

    def samples(n, offset):
        return [(rng.normal(size=(50, 14)).astype("f4"),
                 rng.normal(size=(50, 3)).astype("f4"),
                 float(offset + i), 100 + i, i) for i in range(n)]

    bundle = DataBundle(WindowDataset(samples(8, 1)), WindowDataset(samples(4, 2)),
                        WindowDataset(samples(2, 3)),
                        FeatureScaler(np.zeros(17), np.ones(17)),
                        {"dataset": "SYNTH", "train_units": list(range(8)),
                         "val_units": list(range(4)), "test_units": [100, 101]}, 14, 3)
    model = FCSTGNNWrapper(sensor_dim=14, history=50, dataset="FD002")
    result = fit_and_evaluate(bundle, model, tmp_path, epochs=1, batch_size=4,
                              lr=1e-3, patience=2, device="cuda", seed=0,
                              train_drop_last=True, config={"model": "fc_stgnn"})
    assert (tmp_path / "metrics.json").exists()
    assert set(["MAE", "RMSE", "NASA_Score"]).issubset(result)
    config = (tmp_path / "config.json").read_text()
    assert '"train_drop_last": true' in config


def test_cli_parser_accepts_fc_stgnn_choice():
    from rul_fusion.cli import build_parser
    args = build_parser().parse_args([
        "--dataset", "FD002", "--data_root", "/tmp", "--model", "fc_stgnn"])
    assert args.model == "fc_stgnn"
