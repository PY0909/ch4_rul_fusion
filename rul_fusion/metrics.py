from __future__ import annotations

import numpy as np

from .uncertainty import interval_metrics


def _arrays(target, prediction):
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    p = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if y.shape != p.shape:
        raise ValueError("target and prediction must have equal shape")
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("target and prediction must be finite")
    return y, p


def mae(target, prediction) -> float:
    y, p = _arrays(target, prediction)
    return float(np.mean(np.abs(p - y)))


def rmse(target, prediction) -> float:
    y, p = _arrays(target, prediction)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def nasa_score(target, prediction) -> float:
    y, p = _arrays(target, prediction)
    error = p - y
    penalties = np.where(error >= 0.0, np.exp(error / 10.0) - 1.0,
                         np.exp(-error / 13.0) - 1.0)
    return float(np.sum(penalties))


def regression_metrics(target, prediction) -> dict:
    return {"MAE": mae(target, prediction), "RMSE": rmse(target, prediction),
            "NASA_Score": nasa_score(target, prediction)}


def prediction_interval_metrics(target, lower, upper, alpha: float = 0.2) -> dict:
    """Metrics for an interval at the same estimand as the point target."""
    return interval_metrics(target, lower, upper, alpha=alpha)
