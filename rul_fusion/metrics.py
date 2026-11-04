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


def pinball_loss(target, quantile, tau: float) -> float:
    """Quantile (pinball) loss for one predicted quantile level."""
    y, q = _arrays(target, quantile)
    return float(np.mean(np.maximum(tau * (y - q), (tau - 1.0) * (y - q))))


def quantile_metrics(target, q10, q50, q90) -> dict:
    """Pinball losses plus CRPS for a three-quantile head.

    CRPS uses the Gneiting-Raftery identity CRPS = 2 * integral_0^1 pinball_tau
    dtau, which makes it well defined for quantile/CQR heads that expose no
    density.  With only three quantile levels the quantile function is linearly
    interpolated between tau=0.1/0.5/0.9 and held flat outside that range, so the
    score is a consistent but coarse approximation of the full distribution.
    """
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    levels = np.asarray([0.1, 0.5, 0.9])
    nodes = np.stack([
        np.asarray(q10, dtype=np.float64).reshape(-1),
        np.asarray(q50, dtype=np.float64).reshape(-1),
        np.asarray(q90, dtype=np.float64).reshape(-1),
    ], axis=1)
    if not np.isfinite(nodes).all():
        raise ValueError("quantiles must be finite")
    taus = np.linspace(1e-6, 1.0 - 1e-6, 20001)
    interp = np.stack([np.interp(taus, levels, row) for row in nodes])
    losses = np.maximum(taus[None, :] * (y[:, None] - interp),
                        (taus[None, :] - 1.0) * (y[:, None] - interp))
    return {
        "pinball_0.1": pinball_loss(y, nodes[:, 0], 0.1),
        "pinball_0.5": pinball_loss(y, nodes[:, 1], 0.5),
        "pinball_0.9": pinball_loss(y, nodes[:, 2], 0.9),
        "CRPS_3pt": float(2.0 * np.mean(np.trapezoid(losses, taus, axis=1))),
    }
