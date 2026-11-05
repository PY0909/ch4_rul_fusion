from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F


class NonCrossingQuantileHead(nn.Module):
    """Return q(0.1), q(0.5), q(0.9) in guaranteed sorted order."""

    def __init__(self, d_model: int, dropout: float = 0.1,
                 activation: str = "legacy"):
        super().__init__()
        modules = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
        if activation == "legacy":
            nonlinearity = nn.GELU()
        elif activation in modules:
            nonlinearity = modules[activation]()
        else:
            raise ValueError(f"activation must be 'legacy' or one of {sorted(modules)}")
        self.base = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nonlinearity, nn.Dropout(dropout)
        )
        self.lower = nn.Linear(2 * d_model, 1)
        self.middle_gap = nn.Linear(2 * d_model, 1)
        self.upper_gap = nn.Linear(2 * d_model, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, d_model]")
        hidden = self.base(features)
        lower = F.softplus(self.lower(hidden).squeeze(-1))
        middle = lower + F.softplus(self.middle_gap(hidden).squeeze(-1))
        upper = middle + F.softplus(self.upper_gap(hidden).squeeze(-1))
        return torch.stack((lower, middle, upper), dim=-1)


class A6QuantileRUL(nn.Module):
    """Quantile wrapper around an A6 single-readout backbone."""

    def __init__(self, backbone: nn.Module, dropout: float = 0.1,
                 activation: str = "legacy"):
        super().__init__()
        self.backbone = backbone
        self.backbone.head = nn.Identity()
        self.head = NonCrossingQuantileHead(backbone.d_model, dropout,
                                            activation=activation)

    def forward(self, x: torch.Tensor, condition: torch.Tensor,
                return_aux: bool = False):
        features, regularization = self.backbone.encode_features(x, condition)
        quantiles = self.head(features)
        return (quantiles, regularization) if return_aux else quantiles


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor,
                 taus: Sequence[float] = (0.1, 0.5, 0.9)) -> torch.Tensor:
    if prediction.ndim != 2 or prediction.shape[-1] != len(taus):
        raise ValueError("prediction must have shape [batch, number of quantiles]")
    target = target.reshape(-1, 1).to(dtype=prediction.dtype)
    if target.shape[0] != prediction.shape[0]:
        raise ValueError("target and prediction batch sizes must match")
    tau = prediction.new_tensor(tuple(taus)).reshape(1, -1)
    error = target - prediction
    return torch.maximum(tau * error, (tau - 1.0) * error).mean()


def normalize_quantile_features(quantiles, normalization: float | Sequence[float] = 1.0):
    """Normalize observable quantile features for density-ratio fitting."""
    values = np.asarray(quantiles, dtype=np.float64)
    scale = np.asarray(normalization, dtype=np.float64)
    if np.any(scale <= 0):
        raise ValueError("normalization must be positive")
    return values / scale


def weighted_conformal_threshold(scores, weights, alpha: float) -> float:
    """Conservative weighted conformal quantile using the max-weight atom."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(scores) == 0 or len(scores) != len(weights):
        raise ValueError("scores and weights must be non-empty and aligned")
    if not (np.isfinite(scores).all() and np.isfinite(weights).all() and
            np.all(scores >= 0) and np.all(weights > 0) and 0 < alpha < 1):
        raise ValueError("invalid scores, weights, or alpha")
    order = np.argsort(scores)
    target = (1.0 - alpha) * (weights.sum() + weights.max())
    idx = int(np.searchsorted(np.cumsum(weights[order]), target, side="left"))
    return float(scores[order[min(idx, len(order) - 1)]])


def finite_sample_quantile(scores, coverage: float) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty finite array")
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage must be in [0, 1]")
    index = int(np.ceil((values.size + 1) * coverage)) - 1
    index = min(max(index, 0), values.size - 1)
    return float(np.partition(values, index)[index])


class CQRCalibrator:
    """Global CQR with optional group thresholds and global shrinkage fallback."""

    def __init__(self, alpha: float = 0.2, min_group_size: int = 20,
                 shrink_tau: float = 20.0):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if min_group_size < 1 or shrink_tau < 0.0:
            raise ValueError("invalid group calibration parameters")
        self.alpha = float(alpha)
        self.min_group_size = int(min_group_size)
        self.shrink_tau = float(shrink_tau)
        self.global_threshold: Optional[float] = None
        self.group_thresholds: dict[str, float] = {}
        self.group_counts: dict[str, int] = {}

    @staticmethod
    def _scores(y, lower, upper) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        lower = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(upper, dtype=np.float64).reshape(-1)
        if not (y.shape == lower.shape == upper.shape):
            raise ValueError("calibration arrays must have equal shape")
        if not (np.isfinite(y).all() and np.isfinite(lower).all()
                and np.isfinite(upper).all()):
            raise ValueError("calibration arrays must be finite")
        if np.any(lower > upper):
            raise ValueError("lower interval must not exceed upper interval")
        # CQR's nonconformity score is non-negative even when the target is
        # already inside the base interval.  Keeping the explicit zero also
        # makes the persisted block scores interpretable.
        return np.maximum(0.0, np.maximum(lower - y, y - upper))

    def fit(self, y, lower, upper, groups=None):
        scores = self._scores(y, lower, upper)
        return self.fit_scores(scores, groups=groups)

    def fit_scores(self, scores, groups=None):
        """Fit thresholds from one already-aggregated score per unit."""
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.size == 0 or not np.isfinite(scores).all() or np.any(scores < 0.0):
            raise ValueError("scores must be a non-empty finite non-negative array")
        self.global_threshold = max(0.0, finite_sample_quantile(scores, 1.0 - self.alpha))
        self.group_thresholds = {}
        self.group_counts = {}
        if groups is None:
            return self
        groups = np.asarray(groups).reshape(-1)
        if len(groups) != len(scores):
            raise ValueError("groups and calibration arrays must have equal length")
        for key in sorted({str(value) for value in groups}):
            mask = np.asarray([str(value) == key for value in groups])
            count = int(mask.sum())
            self.group_counts[key] = count
            if count < self.min_group_size:
                continue
            local = max(0.0, finite_sample_quantile(scores[mask], 1.0 - self.alpha))
            weight = count / (count + self.shrink_tau)
            self.group_thresholds[key] = float(
                weight * local + (1.0 - weight) * self.global_threshold
            )
        return self

    def threshold_for(self, group) -> float:
        if self.global_threshold is None:
            raise RuntimeError("fit calibrator before requesting thresholds")
        return float(self.group_thresholds.get(str(group), self.global_threshold))

    def transform(self, lower, upper, groups=None):
        if self.global_threshold is None:
            raise RuntimeError("fit calibrator before transforming intervals")
        lower = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(upper, dtype=np.float64).reshape(-1)
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have equal shape")
        if groups is None:
            thresholds = np.full(lower.shape, self.global_threshold)
        else:
            groups = np.asarray(groups).reshape(-1)
            if len(groups) != len(lower):
                raise ValueError("groups and intervals must have equal length")
            thresholds = np.asarray([self.threshold_for(value) for value in groups])
        return lower - thresholds, upper + thresholds

    def state_dict(self) -> dict:
        if self.global_threshold is None:
            raise RuntimeError("fit calibrator before serialization")
        return {
            "alpha": self.alpha,
            "min_group_size": self.min_group_size,
            "shrink_tau": self.shrink_tau,
            "global_threshold": self.global_threshold,
            "group_thresholds": self.group_thresholds,
            "group_counts": self.group_counts,
        }

    def save(self, path):
        path.write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")


class ScaleAdaptiveCQRCalibrator(CQRCalibrator):
    """CQR whose conformal correction scales with each base interval width."""

    def __init__(self, alpha: float = 0.2, min_group_size: int = 20,
                 shrink_tau: float = 20.0, eps: float = 1e-6):
        super().__init__(alpha=alpha, min_group_size=min_group_size,
                         shrink_tau=shrink_tau)
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)

    @staticmethod
    def _scale(lower, upper):
        lower = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(upper, dtype=np.float64).reshape(-1)
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have equal shape")
        if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
            raise ValueError("interval bounds must be finite")
        if np.any(lower > upper):
            raise ValueError("lower interval must not exceed upper interval")
        return upper - lower

    @classmethod
    def scores(cls, y, lower, upper, eps: float = 1e-6):
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        lower = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(upper, dtype=np.float64).reshape(-1)
        scale = cls._scale(lower, upper)
        if not (len(y) == len(lower) == len(upper)):
            raise ValueError("targets and interval bounds must have equal length")
        if not np.isfinite(y).all():
            raise ValueError("targets must be finite")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        return np.maximum(0.0, np.maximum(lower - y, y - upper) / (scale + eps))

    def fit_scores(self, scores, groups=None):
        """Fit thresholds from one already-aggregated score per exchangeable unit."""
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.size == 0 or not np.isfinite(scores).all() or np.any(scores < 0.0):
            raise ValueError("scores must be a non-empty finite non-negative array")
        self.global_threshold = finite_sample_quantile(scores, 1.0 - self.alpha)
        self.group_thresholds = {}
        self.group_counts = {}
        if groups is None:
            return self
        groups = np.asarray(groups).reshape(-1)
        if len(groups) != len(scores):
            raise ValueError("groups and scores must have equal length")
        for key in sorted({str(value) for value in groups}):
            mask = np.asarray([str(value) == key for value in groups])
            count = int(mask.sum())
            self.group_counts[key] = count
            if count < self.min_group_size:
                continue
            local = finite_sample_quantile(scores[mask], 1.0 - self.alpha)
            weight = count / (count + self.shrink_tau)
            self.group_thresholds[key] = float(
                weight * local + (1.0 - weight) * self.global_threshold)
        return self

    def fit(self, y, lower, upper, groups=None):
        return self.fit_scores(self.scores(y, lower, upper, self.eps), groups=groups)

    def transform(self, lower, upper, groups=None):
        if self.global_threshold is None:
            raise RuntimeError("fit calibrator before transforming intervals")
        lower = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(upper, dtype=np.float64).reshape(-1)
        scale = self._scale(lower, upper)
        if groups is None:
            thresholds = np.full(lower.shape, self.global_threshold)
        else:
            groups = np.asarray(groups).reshape(-1)
            if len(groups) != len(lower):
                raise ValueError("groups and intervals must have equal length")
            thresholds = np.asarray([self.threshold_for(value) for value in groups])
        expansion = thresholds * (scale + self.eps)
        return lower - expansion, upper + expansion

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update({"method": "scale_adaptive_cqr", "eps": self.eps})
        return state


@dataclass
class ConditionStageQualityGrouper:
    n_clusters: int = 4
    random_state: int = 0
    stage_quantiles: tuple[float, float] = (0.33, 0.66)
    quality_bins: tuple[float, ...] = (0.0, 0.01, 0.05, 1.0)
    centers: Optional[np.ndarray] = None
    stage_thresholds: Optional[np.ndarray] = None

    def fit(self, condition_means, train_labels):
        condition_means = np.asarray(condition_means, dtype=np.float64)
        train_labels = np.asarray(train_labels, dtype=np.float64).reshape(-1)
        if condition_means.ndim != 2 or len(condition_means) != len(train_labels):
            raise ValueError("condition means and labels must align")
        unique = np.unique(condition_means, axis=0)
        n_clusters = min(self.n_clusters, len(unique), len(condition_means))
        if n_clusters < 1:
            raise ValueError("at least one training condition is required")
        estimator = KMeans(n_clusters=n_clusters, random_state=self.random_state,
                           n_init=10).fit(condition_means)
        self.centers = estimator.cluster_centers_.astype(np.float64)
        self.stage_thresholds = np.quantile(train_labels, self.stage_quantiles)
        return self

    def groups(self, condition_means, q50, missing_rate=None):
        if self.centers is None or self.stage_thresholds is None:
            raise RuntimeError("fit grouper before generating groups")
        condition_means = np.asarray(condition_means, dtype=np.float64)
        q50 = np.asarray(q50, dtype=np.float64).reshape(-1)
        if condition_means.ndim != 2 or len(condition_means) != len(q50):
            raise ValueError("condition means and q50 must align")
        distance = ((condition_means[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        condition_id = distance.argmin(axis=1)
        stage_id = np.digitize(q50, self.stage_thresholds, right=False)
        if missing_rate is None:
            missing_rate = np.zeros(len(q50), dtype=np.float64)
        missing_rate = np.asarray(missing_rate, dtype=np.float64).reshape(-1)
        if len(missing_rate) != len(q50):
            raise ValueError("missing_rate and q50 must align")
        quality_id = np.digitize(missing_rate, self.quality_bins[1:-1], right=False)
        return np.asarray([f"c{c}|s{s}|q{q}" for c, s, q
                           in zip(condition_id, stage_id, quality_id)], dtype=object)

    def state_dict(self):
        if self.centers is None or self.stage_thresholds is None:
            raise RuntimeError("fit grouper before serialization")
        return {"n_clusters": self.n_clusters, "random_state": self.random_state,
                "stage_quantiles": list(self.stage_quantiles),
                "quality_bins": list(self.quality_bins),
                "centers": self.centers.tolist(),
                "stage_thresholds": self.stage_thresholds.tolist()}


def interval_metrics(y, lower, upper, alpha: float = 0.2) -> dict:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    if not (y.shape == lower.shape == upper.shape):
        raise ValueError("interval arrays must have equal shape")
    if np.any(lower > upper):
        raise ValueError("lower interval must not exceed upper interval")
    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    score = width + (2.0 / alpha) * (lower - y) * (y < lower)
    score += (2.0 / alpha) * (y - upper) * (y > upper)
    return {"PICP": float(covered.mean()), "MPIW": float(width.mean()),
            "IntervalScore": float(score.mean()),
            "coverage_gap": float(covered.mean() - (1.0 - alpha))}
