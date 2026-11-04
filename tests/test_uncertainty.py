import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from rul_fusion.metrics import prediction_interval_metrics
from rul_fusion.model import MultiScalePhaseGraphRUL
from rul_fusion.uncertainty import (CQRCalibrator, ConditionStageQualityGrouper,
                                    NonCrossingQuantileHead, interval_metrics,
                                    normalize_quantile_features, pinball_loss,
                                    ScaleAdaptiveCQRCalibrator,
                                    weighted_conformal_threshold)


def test_non_crossing_head_and_pinball_gradient():
    torch.manual_seed(0)
    head = NonCrossingQuantileHead(8, dropout=0.0)
    prediction = head(torch.randn(16, 8))
    assert prediction.shape == (16, 3)
    assert torch.all(prediction[:, 0] <= prediction[:, 1])
    assert torch.all(prediction[:, 1] <= prediction[:, 2])
    assert torch.all(prediction[:, 0] >= 0.0)
    loss = pinball_loss(prediction, torch.rand(16))
    loss.backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_a6_encode_features_matches_point_forward():
    torch.manual_seed(0)
    model = MultiScalePhaseGraphRUL(
        sensor_dim=4, condition_dim=2, d_model=8, num_scales=3,
        num_heads=2, num_routers=2, encoder="patch_cnn", variant="A6",
    ).eval()
    x = torch.randn(3, 8, 4)
    w = torch.randn(3, 8, 2)
    with torch.no_grad():
        point = model(x, w)
        features, regularization = model.encode_features(x, w)
        point_again = model.head(features).squeeze(-1)
    assert torch.allclose(point, point_again, atol=1e-6)
    assert regularization.ndim == 0


def test_cqr_global_and_small_group_fallback():
    lower = np.zeros(5)
    upper = np.ones(5)
    target = np.array([0.2, 0.8, 0.1, 0.9, 0.5])
    calibrator = CQRCalibrator(alpha=0.2, min_group_size=3, shrink_tau=10.0)
    calibrator.fit(target, lower, upper, groups=np.array(["a", "a", "b", "b", "b"]))
    assert calibrator.threshold_for("a") == calibrator.global_threshold
    assert np.isfinite(calibrator.threshold_for("b"))
    calibrated_lower, calibrated_upper = calibrator.transform(
        lower, upper, groups=np.array(["a", "a", "b", "b", "unknown"]))
    assert np.all(calibrated_lower <= calibrated_upper)
    assert np.isfinite(calibrated_lower).all()


def test_scale_adaptive_cqr_scores_and_width_scaled_transform():
    lower = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    upper = np.array([2.0, 4.0, 1.0, 8.0, 2.0])
    target = np.array([3.0, 6.0, 0.5, 12.0, 1.0])
    eps = 1e-6
    expected_scores = np.array([0.5, 0.5, 0.0, 0.5, 0.0])
    scores = ScaleAdaptiveCQRCalibrator.scores(target, lower, upper, eps=eps)
    assert np.allclose(scores, expected_scores, atol=1e-6)

    calibrator = ScaleAdaptiveCQRCalibrator(alpha=0.2, eps=eps).fit(
        target, lower, upper)
    test_lower = np.array([10.0, 10.0])
    test_upper = np.array([12.0, 18.0])
    out_lower, out_upper = calibrator.transform(test_lower, test_upper)
    correction = calibrator.global_threshold * (test_upper - test_lower + eps)
    assert np.allclose(out_lower, test_lower - correction)
    assert np.allclose(out_upper, test_upper + correction)
    assert out_upper[1] - test_upper[1] > out_upper[0] - test_upper[0]


def test_scale_adaptive_cqr_group_fallback_and_validation():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    groups = np.array(["a", "a", "b", "b", "b"])
    calibrator = ScaleAdaptiveCQRCalibrator(
        alpha=0.2, min_group_size=3, shrink_tau=10.0).fit_scores(scores, groups)
    assert calibrator.threshold_for("a") == calibrator.global_threshold
    assert np.isfinite(calibrator.threshold_for("b"))
    with np.testing.assert_raises(ValueError):
        ScaleAdaptiveCQRCalibrator.scores([1.0], [2.0], [1.0])
    with np.testing.assert_raises(ValueError):
        ScaleAdaptiveCQRCalibrator(eps=0.0)


def test_cqr_block_aggregation_uses_one_conservative_score_per_unit():
    from rul_fusion.quantile_runner import _aggregate_calibration_blocks

    y = np.array([0.0, 1.0, 0.0, 2.0])
    q = np.array([[0.0, 0.5, 0.5], [0.0, 0.5, 0.6],
                  [0.0, 0.5, 0.5], [0.0, 0.5, 1.5]])
    units = np.array([1, 1, 2, 2])
    ids = np.array([10, 11, 20, 21])
    conditions = np.zeros((4, 2))
    groups = np.array(["a", "b", "a", "b"])
    reduced = _aggregate_calibration_blocks(y, q, units, ids, conditions, groups)
    assert reduced[2].tolist() == [1, 2]
    assert reduced[1][:, 2].tolist() == [0.6, 1.5]
    assert reduced[5].tolist() == ["b", "b"]


def test_mc_dropout_restores_eval_mode():
    from torch.utils.data import DataLoader
    from rul_fusion.mc_dropout import predict_mc_dropout

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, x, w):
            return self.dropout(x).sum(dim=-1)

    class LoaderDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return (torch.ones(3, 2), torch.zeros(3, 1), torch.tensor(1.0), i, i)

    model = Tiny()
    predict_mc_dropout(model, DataLoader(LoaderDataset(), batch_size=2), "cpu", samples=2)
    assert not model.training and not model.dropout.training

    from rul_fusion.mc_dropout import _enable_dropout
    attention = torch.nn.MultiheadAttention(4, 1, dropout=0.2, batch_first=True)
    attention.eval()
    attention.apply(_enable_dropout)
    assert attention.training


def test_interval_metrics_and_alias():
    y = np.array([0.0, 1.0, 2.0])
    lower = np.array([-0.5, 0.5, 1.5])
    upper = np.array([0.5, 1.5, 2.5])
    expected = interval_metrics(y, lower, upper, alpha=0.2)
    assert expected["PICP"] == 1.0
    assert expected["MPIW"] == 1.0
    assert prediction_interval_metrics(y, lower, upper, alpha=0.2) == expected


def test_weighted_threshold_and_parameterized_normalization():
    scores = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 4.0])
    assert weighted_conformal_threshold(scores, weights, 0.2) == 3.0
    assert np.allclose(normalize_quantile_features([[10.0, 20.0]], 10.0), [[1.0, 2.0]])

    grouper = ConditionStageQualityGrouper(n_clusters=2, random_state=0)
    conditions = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]])
    labels = np.array([10.0, 12.0, 80.0, 90.0])
    grouper.fit(conditions, labels)
    groups = grouper.groups(conditions, np.array([11.0, 13.0, 81.0, 89.0]))
    assert groups.shape == (4,)
    assert len(set(groups.tolist())) >= 2
