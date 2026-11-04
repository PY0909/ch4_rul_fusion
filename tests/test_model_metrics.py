import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from rul_fusion.metrics import mae, nasa_score, rmse
from rul_fusion.model import MultiScalePhaseGraphRUL


def test_model_outputs_nonnegative_scalar_and_gradients():
    model = MultiScalePhaseGraphRUL(
        sensor_dim=14, condition_dim=3, d_model=32, num_scales=3,
        num_heads=4, num_routers=2, graph_layers=1,
    )
    x = torch.randn(4, 50, 14)
    w = torch.randn(4, 50, 3)
    y = model(x, w)
    assert y.shape == (4,)
    assert torch.isfinite(y).all()
    assert (y >= 0).all()
    y.mean().backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_model_supports_variable_length_and_sensor_count():
    model = MultiScalePhaseGraphRUL(
        sensor_dim=28, condition_dim=4, d_model=32, num_scales=3,
        num_heads=4, num_routers=2,
    )
    y = model(torch.randn(1, 37, 28), torch.randn(1, 37, 4))
    assert y.shape == (1,)


def test_model_patch_cnn_encoder_and_bn_input_norm():
    for encoder, input_norm in [("patch_cnn", "none"), ("linear", "bn"), ("patch_cnn", "bn")]:
        model = MultiScalePhaseGraphRUL(
            sensor_dim=14, condition_dim=3, d_model=32, num_scales=3,
            num_heads=4, num_routers=2, encoder=encoder, input_norm=input_norm,
        )
        y = model(torch.randn(4, 50, 14), torch.randn(4, 50, 3), return_aux=True)
        pred, reg = y
        assert pred.shape == (4,)
        assert torch.isfinite(pred).all() and (pred >= 0).all()
        assert float(reg) == 0.0
        pred.mean().backward()
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_metrics_known_values():
    target = np.array([1.0, 2.0, 4.0])
    pred = np.array([2.0, 2.0, 1.0])
    assert mae(target, pred) == np.mean([1.0, 0.0, 3.0])
    assert np.isclose(rmse(target, pred), np.sqrt((1 + 0 + 9) / 3))
    assert nasa_score(np.array([1.0]), np.array([2.0])) > nasa_score(
        np.array([1.0]), np.array([0.0])
    )


def test_quantile_crps_matches_closed_form_normal():
    """CRPS built from an exact quantile function must equal the normal CRPS."""
    from scipy.special import ndtr
    from scipy.stats import norm

    from rul_fusion.metrics import quantile_metrics

    rng = np.random.default_rng(0)
    y = rng.normal(5.0, 3.0, size=40)
    mu, sigma = 6.0, 2.5

    def normal_crps(target):
        z = (target - mu) / sigma
        phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
        return sigma * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))

    # A dense quantile grid reproduces the analytic score to grid accuracy.
    taus = np.linspace(1e-6, 1.0 - 1e-6, 20001)
    values = mu + sigma * norm.ppf(taus)
    losses = np.maximum(taus[:, None] * (y[None, :] - values[:, None]),
                        (taus[:, None] - 1.0) * (y[None, :] - values[:, None]))
    approx = 2.0 * np.mean(np.trapezoid(losses, taus, axis=0))
    assert np.allclose(approx, np.mean(normal_crps(y)), atol=1e-5)

    # The three-level head is consistent: a wider spread cannot score better
    # than the true distribution on its own samples.
    q10, q50, q90 = (mu + sigma * norm.ppf(t) for t in (0.1, 0.5, 0.9))
    tight = quantile_metrics(y, np.full_like(y, q10), np.full_like(y, q50),
                             np.full_like(y, q90))
    wide = quantile_metrics(y, np.full_like(y, q10 - 10.0),
                            np.full_like(y, q50), np.full_like(y, q90 + 10.0))
    assert np.isfinite(tight["CRPS_3pt"]) and tight["CRPS_3pt"] > 0
    assert wide["CRPS_3pt"] > tight["CRPS_3pt"]


def test_pinball_loss_matches_definition():
    from rul_fusion.metrics import pinball_loss

    y = np.array([0.0, 10.0])
    assert np.isclose(pinball_loss(y, np.array([0.0, 0.0]), 0.5),
                      np.mean([0.0, 5.0]))
    # Under-prediction is penalised more at the upper quantile.
    assert pinball_loss(y, np.array([0.0, 0.0]), 0.9) > pinball_loss(
        y, np.array([0.0, 0.0]), 0.1)
