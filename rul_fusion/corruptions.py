from __future__ import annotations

from typing import Tuple

import numpy as np


def _array(x):
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("sensor windows must have shape [batch, time, sensors]")
    return values.copy()


def inject_missing(x, rate: float, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    if not 0.0 <= rate <= 1.0:
        raise ValueError("missing rate must be in [0, 1]")
    values = _array(x)
    mask = rng.random(values.shape) < float(rate)
    values[mask] = 0.0
    return values, {"missing_rate": float(mask.mean()), "n_missing": int(mask.sum())}


def inject_noise(x, std: float, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    if std < 0.0:
        raise ValueError("noise std must be non-negative")
    values = _array(x)
    values += rng.normal(0.0, float(std), size=values.shape).astype(np.float32)
    return values, {"noise_std": float(std)}


def inject_bias(x, magnitude: float, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    values = _array(x)
    n_sensors = values.shape[-1]
    count = max(1, n_sensors // 4)
    sensors = np.sort(rng.choice(n_sensors, size=count, replace=False))
    values[:, :, sensors] += float(magnitude)
    return values, {"bias": float(magnitude), "sensors": sensors.tolist()}


def inject_drift(x, slope: float, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    del rng
    values = _array(x)
    timeline = np.linspace(0.0, 1.0, values.shape[1], dtype=np.float32)
    values += float(slope) * timeline[None, :, None]
    return values, {"drift_end": float(slope)}
