import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from rul_fusion.model import (
    EndpointAlignedAverageDownsample,
    MomentConstrainedDownsample,
    MultiScalePhaseGraphRUL,
    SensorGraphBlock,
    compute_trend_stats,
)


def test_endpoint_average_keeps_latest_value_and_ceil_length():
    pool = EndpointAlignedAverageDownsample(factor=2)
    x = torch.arange(5.0).reshape(1, 1, 5, 1)
    y, coords = pool(x, torch.arange(5.0))
    assert y.shape == (1, 1, 3, 1)
    assert coords.shape == (3,)
    assert torch.isclose(y[0, 0, -1, 0], torch.tensor(3.5))
    assert torch.isclose(coords[-1], torch.tensor(3.5))


def test_moment_downsample_preserves_affine_signal_and_endpoint():
    pool = MomentConstrainedDownsample(d_model=2, factor=2, window_size=5)
    pool.gamma.data.fill_(1.0)
    coords = torch.arange(9.0)
    signal = torch.stack([2.0 * coords + 1.0, -coords + 3.0], dim=-1)
    x = signal.reshape(1, 1, 9, 2)
    y, out_coords = pool(x, coords)
    expected = torch.stack([2.0 * out_coords + 1.0, -out_coords + 3.0], dim=-1)
    assert y.shape[2] == 5
    assert torch.allclose(y[0, 0], expected, atol=1e-5)
    perturbed = x.clone()
    perturbed[:, :, -1] += 10.0
    y_perturbed, _ = pool(perturbed, coords)
    assert not torch.allclose(y_perturbed[:, :, -1], y[:, :, -1])


def test_trend_stats_returns_reliable_affine_slope():
    coords = torch.arange(6.0)
    slope = torch.tensor([2.0, -1.0])
    x = (coords[:, None] * slope[None, :] + torch.tensor([1.0, 4.0]))
    sequence = x.reshape(1, 1, 6, 2)
    trend, reliability = compute_trend_stats(sequence, coords)
    assert trend.shape == (1, 1, 2)
    assert reliability.shape == (1, 1)
    assert torch.allclose(trend[0, 0], slope * (coords[-1] - coords[0]))
    assert reliability[0, 0] > 0.99


def test_variants_a1_to_a7_build_and_forward():
    x = torch.randn(2, 16, 4)
    condition = torch.randn(2, 16, 2)
    for variant in ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]:
        model = MultiScalePhaseGraphRUL(
            sensor_dim=4, condition_dim=2, d_model=16, num_scales=3,
            num_heads=4, num_routers=2, variant=variant,
        )
        pred, reg = model(x, condition, return_aux=True)
        assert pred.shape == (2,)
        assert torch.isfinite(pred).all()
        assert torch.isfinite(reg)


def test_variant_configuration_is_explicit():
    model = MultiScalePhaseGraphRUL(
        sensor_dim=4, condition_dim=2, d_model=16, num_scales=3,
        num_heads=4, num_routers=2, variant="A7",
    )
    assert model.readout == "single"
    assert model.downsample_mode == "moment"
    assert model.graph_mode == "trend"
    assert model.use_trend_features


def test_endpoint_scale_lengths_follow_history_length():
    model = MultiScalePhaseGraphRUL(
        sensor_dim=4, condition_dim=2, d_model=8, num_scales=3,
        num_heads=2, num_routers=2, variant="A7",
    )
    x = torch.randn(1, 4, 50, 8)
    coords = torch.arange(50.0)
    x1, c1 = model.downsamplers[0](x, coords)
    x2, c2 = model.downsamplers[1](x1, c1)
    assert [x.shape[2], x1.shape[2], x2.shape[2]] == [50, 25, 13]
    x = torch.randn(1, 4, 16, 8)
    coords = torch.arange(16.0)
    x1, c1 = model.downsamplers[0](x, coords)
    x2, _ = model.downsamplers[1](x1, c1)
    assert [x.shape[2], x1.shape[2], x2.shape[2]] == [16, 8, 4]


def test_trend_graph_uses_trend_input_when_dynamic_term_is_enabled():
    graph = SensorGraphBlock(sensor_dim=4, d_model=8, graph_mode="trend", dropout=0.0)
    graph.eval()
    graph.dynamic_scale.data.fill_(1.0)
    x = torch.randn(2, 4, 8)
    reliability = torch.ones(2, 4)
    out_a = graph(x, torch.zeros_like(x), reliability)
    out_b = graph(x, torch.randn_like(x), reliability)
    assert not torch.allclose(out_a, out_b)
