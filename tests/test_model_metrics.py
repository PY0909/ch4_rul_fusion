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
