import sys
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))

from rul_fusion.data import (DataBundle, FeatureScaler, GroupedStandardScaler,
                             WindowDataset)
from rul_fusion.mc_dropout import predict_mc_dropout
from rul_fusion.probabilistic import (LSTMProbRUL, MSAUQRUL,
                                      PAPER_METHOD_DEFAULTS,
                                      TCNTransformerMCDropoutRUL,
                                      BayesianCooperativeTransformerRUL,
                                      bcpt_channel_loss, gaussian_nll)
from rul_fusion.probabilistic_runner import (_bcpt_channel_weights,
                                             _bcpt_draw_components, _bcpt_fuse,
                                             crps_from_draws, fit_probabilistic,
                                             gaussian_fit_outputs,
                                             kde_quantiles)
import rul_fusion.probabilistic_runner as probabilistic_runner


def _samples(count, offset=0):
    rng = np.random.default_rng(23 + offset)
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


def test_lstm_prob_outputs_and_loss():
    x = torch.randn(5, 6, 3)
    w = torch.randn(5, 6, 2)
    y = torch.rand(5) * 30.0
    model = LSTMProbRUL(3, 2)
    params = model(x, w)
    assert params.shape == (5, 2)
    assert torch.all(params[:, 1] > 0)
    loss = gaussian_nll(params, y)
    assert torch.isfinite(loss)
    loss.backward()


def test_tcn_transformer_point_head_and_mc_draws():
    x = torch.randn(4, 6, 3)
    model = TCNTransformerMCDropoutRUL(3, 2, window_size=6, dropout=0.2)
    assert model(x).shape == (4, 1)
    loader = DataLoader(_bundle().test, batch_size=4)
    draws = predict_mc_dropout(model, loader, torch.device("cpu"), samples=5)["draws"]
    assert draws.shape[0] == 5 and draws.shape[1] == 4
    assert draws.std(axis=0).min() > 0.0  # dropout active -> draws differ


def test_msauq_smoothing_and_fd_parameter_count():
    x = torch.randn(2, 6, 3)
    model = MSAUQRUL(3, window_size=6)
    assert model(x).shape == (2, 1)
    smoothed = model.smoothing(x)
    assert not torch.allclose(smoothed, x)
    fd_model = MSAUQRUL(14, window_size=50)
    count = sum(p.numel() for p in fd_model.parameters())
    # Paper total is 80,225 with window 30; widths are sized to land near it.
    assert 55_000 < count < 105_000, count


def test_kde_quantiles_crps_and_gaussian_fit_outputs():
    rng = np.random.default_rng(7)
    draws = rng.normal(20.0, 5.0, size=(100, 50))
    target = rng.normal(20.0, 5.0, size=50)
    bounds = kde_quantiles(draws)
    assert bounds.shape == (2, 50)
    mean, scale, lower, upper, nll, crps = gaussian_fit_outputs(target, draws, bounds)
    assert np.all(lower < mean) and np.all(mean < upper)
    assert np.isfinite(nll).all() and np.isfinite(crps).all()
    assert np.isfinite(crps_from_draws(target, draws)).all()


def test_bcpt_channel_outputs_loss_components_and_fusion():
    torch.manual_seed(3)
    x = torch.randn(4, 6, 3)
    y = torch.rand(4) * 30.0
    model = BayesianCooperativeTransformerRUL(3, window_size=6)
    out = model.forward_channels(x, sample_tau=True)
    assert out["mean"].shape == (4, 3) and out["log_var"].shape == (4, 3)
    weights = torch.full((3,), 1.0)
    loss = bcpt_channel_loss(out, y, weights)
    assert torch.isfinite(loss)
    loss.backward()
    loader = DataLoader(_bundle().test, batch_size=4)
    components = _bcpt_draw_components(model, loader, torch.device("cpu"), m=3)
    for key in ("mean", "epistemic", "attention_mi", "aleatoric"):
        assert components[key].shape == (4, 3)
        assert np.isfinite(components[key]).all()
    assert (components["epistemic"] >= -1e-8).all()
    fused = _bcpt_fuse(components, components)
    for array in fused:
        assert np.isfinite(array).all()
    mean, total_u = fused[0], fused[1]
    assert (total_u > 0).all()
    assert (mean - 1.2816 * np.sqrt(total_u) < mean + 1.2816 * np.sqrt(total_u)).all()
    weights_low = _bcpt_channel_weights(model, loader, torch.device("cpu"), m=3,
                                        high=False)
    assert weights_low.shape == (3,) and np.isfinite(weights_low).all()
    assert np.all(weights_low <= 1.0 + 1e-6)


def test_grouped_standard_scaler_groups_and_fallback():
    rng = np.random.default_rng(11)
    values = np.vstack([rng.normal(0.0, 1.0, (40, 2)), rng.normal(50.0, 3.0, (40, 2))])
    keys = np.array([0.0] * 40 + [1.0] * 40)
    scaler = GroupedStandardScaler().fit(values, keys)
    scaled = scaler.transform(values, keys)
    assert abs(scaled.mean()) < 1e-5
    assert 0.9 < scaled.std() < 1.1
    unseen = scaler.transform(np.array([[1.0, 2.0]]), np.array([7.0]))
    assert np.isfinite(unseen).all() and scaler.unseen == [7.0]
    state = json.dumps(scaler.state_dict())
    assert "groups" in state and "condition_scaler" in state


def test_paper_defaults_match_registration():
    assert PAPER_METHOD_DEFAULTS["lstm_prob"][:6] == (
        14, 43, 5.613e-4, 0.0, 0.03, "minmax")
    assert PAPER_METHOD_DEFAULTS["tcn_transformer_mc"] == (
        50, 128, 1e-3, 0.0, 0.2, "minmax", "val_rmse", 30)
    assert PAPER_METHOD_DEFAULTS["msauq"] == (
        50, 128, 1e-3, 0.0, 0.1, "zscore_grouped", "val_rmse", 100)
    assert PAPER_METHOD_DEFAULTS["bcpt"] == (
        200, 256, 5e-3, 0.0, 0.3, "minmax", "val_fused_rmse", 30)


def test_runner_smoke_for_four_new_methods(tmp_path):
    for kind in ("lstm_prob", "tcn_transformer_mc", "msauq", "bcpt"):
        result = fit_probabilistic(_bundle(), kind, tmp_path / kind, epochs=1,
                                   batch_size=4, patience=1, device="cpu", seed=0,
                                   mc_samples=3)
        assert np.isfinite(result["NLL"]) and np.isfinite(result["CRPS"])
        assert 0.0 <= result["PICP"] <= 1.0
        assert result["mc_samples"] == (3 if kind != "lstm_prob" else 0)
        for name in ("best.pt", "metrics.json", "history.csv", "predictions.csv",
                     "config.json", "manifest.json"):
            assert (tmp_path / kind / name).exists(), (kind, name)
    assert (tmp_path / "tcn_transformer_mc" / "mc_draws.npy").exists()
    assert (tmp_path / "msauq" / "mc_draws.npy").exists()
    assert (tmp_path / "bcpt" / "bcpt_test_arrays.npz").exists()
    state = torch.load(tmp_path / "bcpt" / "best.pt", map_location="cpu",
                       weights_only=False)
    assert {"lu_state", "hu_state"}.issubset(state)
    metrics = json.loads((tmp_path / "msauq" / "metrics.json").read_text())
    assert metrics["selection_criterion"] == "val_rmse"
    assert json.loads((tmp_path / "bcpt" / "config.json").read_text())["bcpt_gamma"] == 0.2


def test_defer_test_skips_evaluation_for_new_methods(tmp_path):
    for kind in ("msauq", "bcpt"):
        result = fit_probabilistic(_bundle(), kind, tmp_path / f"defer_{kind}",
                                   epochs=1, batch_size=4, patience=1, device="cpu",
                                   seed=1, mc_samples=3, defer_test=True)
        assert result["test_evaluated"] is False
        assert not (tmp_path / f"defer_{kind}" / "predictions.csv").exists()


def test_fixed_epoch_probabilistic_training_needs_no_validation(tmp_path):
    bundle = _bundle()
    bundle.val_endpoint = None
    result = fit_probabilistic(
        bundle, "lstm_prob", tmp_path / "lstm_fixed", epochs=2,
        batch_size=4, patience=30, fixed_epochs=True, device="cpu",
        seed=0, defer_test=True)
    assert result["best_epochs"] == [2]
    assert result["selection_criterion"] == "fixed_epoch"
    assert result["test_evaluated"] is False
    state = torch.load(tmp_path / "lstm_fixed/best.pt", map_location="cpu",
                       weights_only=False)
    assert state["best_epoch"] == 2


def test_fixed_epoch_bcpt_uses_training_units_for_channel_weights(tmp_path):
    bundle = _bundle()
    bundle.val_endpoint = None
    result = fit_probabilistic(
        bundle, "bcpt", tmp_path / "bcpt_fixed", epochs=1,
        batch_size=4, patience=30, fixed_epochs=True, mc_samples=3,
        device="cpu", seed=0, defer_test=True)
    assert result["best_epochs"] == [1]
    assert result["selection_criterion"] == "fixed_epoch"
    state = torch.load(tmp_path / "bcpt_fixed/best.pt", map_location="cpu",
                       weights_only=False)
    assert {"lu_state", "hu_state"}.issubset(state)


def test_bcpt_validation_uses_post_update_models(tmp_path, monkeypatch):
    original = probabilistic_runner._bcpt_draw_components
    calls = []

    def tracked(model, loader, device, draws):
        calls.append(id(model))
        return original(model, loader, device, draws)

    monkeypatch.setattr(probabilistic_runner, "_bcpt_draw_components", tracked)
    fit_probabilistic(_bundle(), "bcpt", tmp_path / "bcpt_post_update",
                      epochs=1, batch_size=4, patience=1, device="cpu",
                      seed=0, mc_samples=3, defer_test=True)
    assert len(calls) == 4  # two pre-update training weights, two post-update val draws
