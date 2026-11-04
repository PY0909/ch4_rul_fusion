from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FeatureScaler:
    """Train-only standardizer for concatenated sensor and condition features."""

    def __init__(self, mean: Optional[np.ndarray] = None, scale: Optional[np.ndarray] = None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float32)

    def fit(self, values: np.ndarray) -> "FeatureScaler":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("scaler expects a two-dimensional array")
        self.mean = np.nanmean(values, axis=0).astype(np.float32)
        scale = np.nanstd(values, axis=0).astype(np.float32)
        self.scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("fit scaler before transform")
        return ((np.asarray(values, dtype=np.float32) - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("fit scaler before inverse_transform")
        return (np.asarray(values, dtype=np.float32) * self.scale + self.mean).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


class MinMaxScaler:
    """Train-only per-column min/max scaler."""

    def __init__(self):
        self.min = None
        self.range = None

    def fit(self, values: np.ndarray) -> "MinMaxScaler":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("scaler expects a two-dimensional array")
        self.min = values.min(axis=0).astype(np.float32)
        span = values.max(axis=0).astype(np.float32) - self.min
        self.range = np.where(span < 1e-6, 1.0, span).astype(np.float32)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.min is None or self.range is None:
            raise RuntimeError("fit scaler before transform")
        return ((np.asarray(values, dtype=np.float32) - self.min) / self.range).astype(np.float32)

    def state_dict(self) -> dict:
        return {"min": self.min.tolist(), "range": self.range.tolist()}


class GroupedMinMaxScaler:
    """Min/max scaling per operating-condition group keyed by rounded setting.

    Sensor statistics are fitted on train rows only; rows whose group never
    appears during fitting fall back to the global per-column min/max instead
    of being dropped. The optional ``condition_scaler`` is fitted globally and
    stored so the whole protocol serializes through one ``state_dict``.
    """

    def __init__(self):
        self.groups: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        self.fallback = MinMaxScaler()
        self.condition_scaler: Optional[MinMaxScaler] = None
        self.unseen: List[float] = []

    def fit(self, values: np.ndarray, keys: np.ndarray) -> "GroupedMinMaxScaler":
        values = np.asarray(values, dtype=np.float64)
        self.fallback.fit(values)
        self.groups = {}
        for key in np.unique(np.asarray(keys)):
            scaler = MinMaxScaler().fit(values[np.asarray(keys) == key])
            self.groups[float(key)] = (scaler.min, scaler.range)
        return self

    def transform(self, values: np.ndarray, keys: np.ndarray) -> np.ndarray:
        if not self.groups:
            raise RuntimeError("fit scaler before transform")
        values = np.asarray(values, dtype=np.float32)
        keys = np.asarray(keys)
        scaled = np.empty_like(values)
        self.unseen = []
        for key in np.unique(keys):
            mask = keys == key
            entry = self.groups.get(float(key))
            if entry is None:
                self.unseen.append(float(key))
                lo, span = self.fallback.min, self.fallback.range
            else:
                lo, span = entry
            scaled[mask] = ((values[mask] - lo) / span).astype(np.float32)
        return scaled

    def state_dict(self) -> dict:
        return {
            "fallback": self.fallback.state_dict(),
            "groups": {str(k): [v[0].tolist(), v[1].tolist()] for k, v in self.groups.items()},
            "condition_scaler": self.condition_scaler.state_dict() if self.condition_scaler else None,
        }


class GroupedStandardScaler:
    """Z-score scaling per operating-condition group (B6 paper preprocessing).

    Mirrors :class:`GroupedMinMaxScaler`: mean/std fitted on train rows per
    rounded setting group, global per-column z-score as fallback for unseen
    groups, and one ``state_dict`` for serialization.
    """

    def __init__(self):
        self.groups: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        self.fallback = FeatureScaler()
        self.condition_scaler: Optional[MinMaxScaler] = None
        self.unseen: List[float] = []

    def fit(self, values: np.ndarray, keys: np.ndarray) -> "GroupedStandardScaler":
        values = np.asarray(values, dtype=np.float64)
        self.fallback.fit(values)
        self.groups = {}
        for key in np.unique(np.asarray(keys)):
            rows = values[np.asarray(keys) == key]
            mean = np.nanmean(rows, axis=0).astype(np.float32)
            scale = np.nanstd(rows, axis=0).astype(np.float32)
            scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
            self.groups[float(key)] = (mean, scale)
        return self

    def transform(self, values: np.ndarray, keys: np.ndarray) -> np.ndarray:
        if not self.groups:
            raise RuntimeError("fit scaler before transform")
        values = np.asarray(values, dtype=np.float32)
        keys = np.asarray(keys)
        scaled = np.empty_like(values)
        self.unseen = []
        for key in np.unique(keys):
            mask = keys == key
            entry = self.groups.get(float(key))
            if entry is None:
                self.unseen.append(float(key))
                mean, scale = self.fallback.mean, self.fallback.scale
            else:
                mean, scale = entry
            scaled[mask] = ((values[mask] - mean) / scale).astype(np.float32)
        return scaled

    def state_dict(self) -> dict:
        return {
            "fallback": {"mean": self.fallback.mean.tolist(),
                         "scale": self.fallback.scale.tolist()},
            "groups": {str(k): [v[0].tolist(), v[1].tolist()]
                       for k, v in self.groups.items()},
            "condition_scaler": self.condition_scaler.state_dict() if self.condition_scaler else None,
        }


class WindowDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[np.ndarray, np.ndarray, float, int, int]]):
        self.x = np.asarray([s[0] for s in samples], dtype=np.float32)
        self.w = np.asarray([s[1] for s in samples], dtype=np.float32)
        self.y = np.asarray([s[2] for s in samples], dtype=np.float32)
        self.units = np.asarray([s[3] for s in samples], dtype=np.int64)
        self.sample_ids = np.asarray([s[4] for s in samples], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.x[index]), torch.from_numpy(self.w[index]),
            torch.tensor(self.y[index], dtype=torch.float32),
            int(self.units[index]), int(self.sample_ids[index]),
        )


@dataclass
class DataBundle:
    train: WindowDataset
    val: WindowDataset
    test: WindowDataset
    scaler: FeatureScaler
    metadata: dict
    sensor_dim: int
    condition_dim: int
    val_endpoint: Optional[WindowDataset] = None
    test_endpoint: Optional[WindowDataset] = None
    calibration_endpoint: Optional[WindowDataset] = None


def _pad_window(x: np.ndarray, w: np.ndarray, history: int, end: int):
    start = max(0, end - history + 1)
    xw, ww = x[start:end + 1], w[start:end + 1]
    if len(xw) < history:
        pad = history - len(xw)
        xw = np.concatenate([np.repeat(xw[:1], pad, axis=0), xw], axis=0)
        ww = np.concatenate([np.repeat(ww[:1], pad, axis=0), ww], axis=0)
    return xw[-history:], ww[-history:]


def _endpoint_samples(records: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                      raw_labels: Dict[int, np.ndarray], units: Iterable[int],
                      history: int, rng: np.random.Generator,
                      cut_lo: float = 0.5, cut_hi: float = 1.0,
                      fracs: Optional[Sequence[float]] = None):
    """Test-protocol windows per unit, labeled with raw remaining RUL.

    With ``fracs`` (fixed-cuts protocol) each unit contributes one window per
    listed life fraction, so validation units carry equal weight and the cut
    rule is locked before any comparison. Otherwise a single cut is drawn
    uniformly from [cut_lo, cut_hi] (legacy endpoint protocol).
    """
    samples, cuts = [], []
    for unit in sorted(units):
        x, w, _ = records[int(unit)]
        y_raw = raw_labels[int(unit)]
        n = len(x)
        if n == 0:
            continue
        unit_fracs = (list(fracs) if fracs is not None
                      else [float(rng.uniform(cut_lo, cut_hi))])
        for frac in unit_fracs:
            cut = min(max(int(round(frac * (n - 1))), 0), n - 1)
            xw, ww = _pad_window(x, w, history, cut)
            samples.append((xw, ww, float(y_raw[cut]), int(unit), int(cut)))
            cuts.append({"unit": int(unit), "frac": round(float(frac), 4),
                         "cut": int(cut), "length": int(n),
                         "label_raw": float(y_raw[cut])})
    return samples, cuts


def _make_windows(records: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                  units: Iterable[int], history: int, final_only: bool = False):
    samples = []
    for unit in sorted(units):
        x, w, y = records[int(unit)]
        if len(x) == 0 or len(y) == 0:
            continue
        ends = [len(x) - 1] if final_only else range(min(len(x), len(y)))
        for sample_id, end in enumerate(ends):
            xw, ww = _pad_window(x, w, history, end)
            target = y[-1] if final_only else y[end]
            samples.append((xw, ww, float(target), int(unit), int(sample_id)))
    return samples


def _read_cmapss(path: Path):
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    values = frame.to_numpy(dtype=np.float64)
    engine = values[:, 0].astype(np.int64)
    cycle = values[:, 1].astype(np.int64)
    condition = values[:, 2:5].astype(np.float32)
    all_sensors = values[:, 5:26].astype(np.float32)
    keep = [i for i in range(21) if i not in {0, 4, 5, 9, 15, 17, 18}]
    sensor = all_sensors[:, keep]
    return engine, cycle, sensor, condition


def build_cmapss_datasets(data_root, dataset: str, history: int = 50,
                          rul_cap: int = 0, split_seed: int = 0,
                          val_ratio: float = 0.2,
                          calibration_ratio: Optional[float] = None,
                          scaler_mode: str = "global",
                          condition_features: str = "settings",
                          validation_protocol: str = "legacy_endpoint") -> DataBundle:
    dataset = dataset.upper()
    if dataset not in {"FD002", "FD004"}:
        raise ValueError("C-MAPSS dataset must be FD002 or FD004")
    if scaler_mode not in {"global", "grouped", "minmax", "zscore_grouped"}:
        raise ValueError("scaler_mode must be 'global', 'grouped', 'minmax', "
                         "or 'zscore_grouped'")
    if condition_features not in {"settings", "settings_cycle"}:
        raise ValueError("condition_features must be 'settings' or 'settings_cycle'")
    if validation_protocol not in {"legacy_endpoint", "fixed_cuts"}:
        raise ValueError("validation_protocol must be 'legacy_endpoint' or 'fixed_cuts'")
    if calibration_ratio is not None and not 0.0 < calibration_ratio < 1.0:
        raise ValueError("calibration_ratio must be between 0 and 1")
    root = Path(data_root) / "CMAPSSData"
    train_path = root / f"train_{dataset}.txt"
    test_path = root / f"test_{dataset}.txt"
    rul_path = root / f"RUL_{dataset}.txt"
    for path in (train_path, test_path, rul_path):
        if not path.exists():
            raise FileNotFoundError(path)

    engine, cycle, sensor, condition = _read_cmapss(train_path)
    if condition_features == "settings_cycle":
        condition = np.concatenate(
            [condition, cycle[:, None].astype(np.float32)], axis=1)
    records: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    raw_labels: Dict[int, np.ndarray] = {}
    for unit in np.unique(engine):
        idx = np.where(engine == unit)[0]
        idx = idx[np.argsort(cycle[idx])]
        x, w = sensor[idx], condition[idx]
        raw_y = (cycle[idx].max() - cycle[idx]).astype(np.float32)
        y = np.minimum(raw_y, rul_cap) if rul_cap > 0 else raw_y
        records[int(unit)] = (x, w, y)
        raw_labels[int(unit)] = raw_y

    units = sorted(records)
    rng = np.random.default_rng(split_seed)
    shuffled = np.asarray(units, dtype=np.int64)
    rng.shuffle(shuffled)
    n_holdout = max(1, int(round(len(units) * val_ratio))) if len(units) > 1 else 0
    holdout_units = shuffled[:n_holdout].tolist()
    train_units = sorted(shuffled[n_holdout:].tolist())
    if calibration_ratio is None:
        val_units = sorted(holdout_units)
        calibration_units = []
    else:
        if len(holdout_units) < 2:
            raise ValueError("calibration_ratio requires at least 2 holdout engines")
        n_calibration = (max(1, int(round(len(holdout_units) * calibration_ratio)))
                         if len(holdout_units) > 1 else 0)
        n_calibration = min(len(holdout_units) - 1, n_calibration)
        val_units = sorted(holdout_units[:-n_calibration])
        calibration_units = sorted(holdout_units[-n_calibration:])
    if not train_units:
        raise ValueError("engine split leaves no training engines")

    sensor_dim = sensor.shape[1]
    scaled_records: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    group_scaler: Optional[Union[GroupedMinMaxScaler, GroupedStandardScaler]] = None
    cond_scaler: Optional[MinMaxScaler] = None
    if scaler_mode in {"grouped", "zscore_grouped"}:
        group_scaler = (GroupedMinMaxScaler() if scaler_mode == "grouped"
                        else GroupedStandardScaler()).fit(
            np.concatenate([records[u][0] for u in train_units], axis=0),
            np.concatenate([np.round(records[u][1][:, 0], 1) for u in train_units]))
        cond_scaler = MinMaxScaler().fit(
            np.concatenate([records[u][1] for u in train_units], axis=0))
        group_scaler.condition_scaler = cond_scaler
        scaler = group_scaler
        for u, (x, w, y) in records.items():
            scaled_records[u] = (group_scaler.transform(x, np.round(w[:, 0], 1)),
                                 cond_scaler.transform(w), y)
    elif scaler_mode == "global":
        train_rows = np.concatenate([np.concatenate([records[u][0], records[u][1]], axis=1)
                                     for u in train_units], axis=0)
        scaler = FeatureScaler().fit(train_rows)
        for u, (x, w, y) in records.items():
            z = scaler.transform(np.concatenate([x, w], axis=1))
            scaled_records[u] = (z[:, :sensor_dim], z[:, sensor_dim:], y)
    else:
        train_rows = np.concatenate([np.concatenate([records[u][0], records[u][1]], axis=1)
                                     for u in train_units], axis=0)
        scaler = MinMaxScaler().fit(train_rows)
        for u, (x, w, y) in records.items():
            z = scaler.transform(np.concatenate([x, w], axis=1))
            scaled_records[u] = (z[:, :sensor_dim], z[:, sensor_dim:], y)

    test_engine, test_cycle, test_sensor, test_condition = _read_cmapss(test_path)
    if condition_features == "settings_cycle":
        test_condition = np.concatenate(
            [test_condition, test_cycle[:, None].astype(np.float32)], axis=1)
    test_rul = np.loadtxt(rul_path, dtype=np.float32).reshape(-1)
    test_records = {}
    unseen_before_test = list(group_scaler.unseen) if group_scaler else []
    for pos, unit in enumerate(sorted(np.unique(test_engine))):
        idx = np.where(test_engine == unit)[0]
        idx = idx[np.argsort(test_cycle[idx])]
        x, w = test_sensor[idx], test_condition[idx]
        if scaler_mode in {"grouped", "zscore_grouped"}:
            test_records[int(unit)] = (group_scaler.transform(x, np.round(w[:, 0], 1)),
                                       cond_scaler.transform(w),
                                       np.asarray([test_rul[pos]], dtype=np.float32))
        elif scaler_mode == "global":
            z = scaler.transform(np.concatenate([x, w], axis=1))
            test_records[int(unit)] = (z[:, :sensor_dim], z[:, sensor_dim:],
                                       np.asarray([test_rul[pos]], dtype=np.float32))
        else:
            z = scaler.transform(np.concatenate([x, w], axis=1))
            test_records[int(unit)] = (z[:, :sensor_dim], z[:, sensor_dim:],
                                       np.asarray([test_rul[pos]], dtype=np.float32))
    unseen_test_groups = ([g for g in group_scaler.unseen if g not in unseen_before_test]
                          if group_scaler else [])

    endpoint_fracs = [0.25, 0.5, 0.75, 0.9] if validation_protocol == "fixed_cuts" else None
    endpoint_rng = np.random.default_rng(1_000_003 * split_seed + 11)
    endpoint_samples, endpoint_cuts = _endpoint_samples(
        scaled_records, raw_labels, val_units, history, endpoint_rng, fracs=endpoint_fracs)
    calibration_endpoint_samples, calibration_endpoint_cuts = _endpoint_samples(
        scaled_records, raw_labels, calibration_units, history, endpoint_rng, fracs=endpoint_fracs)

    metadata = {
        "dataset": dataset, "train_units": train_units, "val_units": val_units,
        "calibration_units": calibration_units,
        "test_units": sorted(test_records), "sensor_indices": [i + 1 for i in range(21)
        if i not in {0, 4, 5, 9, 15, 17, 18}], "history": history,
        "rul_cap": rul_cap, "split_seed": split_seed,
        "scaler_mode": scaler_mode, "unseen_test_groups": unseen_test_groups,
        "condition_features": condition_features,
        "validation_protocol": validation_protocol,
        "calibration_ratio": calibration_ratio,
        "val_endpoint": endpoint_cuts,
        "calibration_endpoint": calibration_endpoint_cuts,
        "val_endpoint_protocol": ("fixed cuts at life fractions {0.25, 0.5, 0.75, 0.9} per "
                                  "validation engine, endpoint windows, raw remaining RUL "
                                  "labels (never capped)"
                                  if validation_protocol == "fixed_cuts" else
                                  "per validation engine: trajectory truncated at a uniform "
                                  "[0.5, 1.0] life fraction, endpoint window, raw remaining "
                                  "RUL label (never capped)"),
    }
    return DataBundle(
        WindowDataset(_make_windows(scaled_records, train_units, history)),
        WindowDataset(_make_windows(scaled_records, val_units, history)),
        WindowDataset(_make_windows(test_records, sorted(test_records), history, final_only=True)),
        scaler, metadata, sensor_dim, condition.shape[1],
        WindowDataset(endpoint_samples),
        None,
        WindowDataset(calibration_endpoint_samples) if calibration_endpoint_samples else None,
    )


def _decode_names(dataset):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in dataset[:]]


def _iter_nc_cycles(path: Path, split: str, include_virtual: bool = False,
                    chunk_size: int = 200_000):
    """Yield unit-cycle aggregates without loading the full HDF5 arrays."""
    with h5py.File(path, "r") as f:
        a_ds, w_ds, xs_ds, y_ds = (f[f"{k}_{split}"] for k in ("A", "W", "X_s", "Y"))
        xv_ds = f.get(f"X_v_{split}") if include_virtual else None
        n = a_ds.shape[0]
        current_key = None
        a_buf, w_buf, xs_buf, xv_buf, y_buf = [], [], [], [], []

        def flush():
            if current_key is None:
                return None
            ys = np.asarray(y_buf).reshape(-1)
            if len(ys) == 0:
                return None
            if not np.all(ys == ys[0]):
                raise ValueError(f"Y is not constant within unit-cycle {current_key}")
            x = np.asarray(xs_buf, dtype=np.float32).mean(axis=0)
            if xv_ds is not None:
                x = np.concatenate([x, np.asarray(xv_buf, dtype=np.float32).mean(axis=0)])
            w = np.asarray(w_buf, dtype=np.float32).mean(axis=0)
            return int(current_key[0]), int(current_key[1]), x, w, float(ys[0])

        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            a = np.asarray(a_ds[start:end])
            w = np.asarray(w_ds[start:end], dtype=np.float32)
            xs = np.asarray(xs_ds[start:end], dtype=np.float32)
            ys = np.asarray(y_ds[start:end]).reshape(-1)
            xv = np.asarray(xv_ds[start:end], dtype=np.float32) if xv_ds is not None else None
            for i in range(end - start):
                key = (int(a[i, 0]), int(a[i, 1]))
                if current_key is not None and key != current_key:
                    item = flush()
                    if item is not None:
                        yield item
                    a_buf, w_buf, xs_buf, xv_buf, y_buf = [], [], [], [], []
                current_key = key
                w_buf.append(w[i])
                xs_buf.append(xs[i])
                y_buf.append(ys[i])
                if xv is not None:
                    xv_buf.append(xv[i])
        item = flush()
        if item is not None:
            yield item


def _records_from_nc(path: Path, split: str, include_virtual: bool):
    records: Dict[int, List[Tuple[int, np.ndarray, np.ndarray, float]]] = {}
    for unit, cycle, x, w, y in _iter_nc_cycles(path, split, include_virtual):
        records.setdefault(unit, []).append((cycle, x, w, y))
    result = {}
    for unit, values in records.items():
        values.sort(key=lambda z: z[0])
        result[unit] = (
            np.stack([v[1] for v in values]).astype(np.float32),
            np.stack([v[2] for v in values]).astype(np.float32),
            np.asarray([v[3] for v in values], dtype=np.float32),
        )
    return result


def build_ncmapss_datasets(data_root, dataset: str, history: int = 16,
                           split_seed: int = 0, val_ratio: float = 0.2,
                           calibration_ratio: Optional[float] = None,
                           include_virtual: bool = False,
                           validation_protocol: str = "legacy_endpoint",
                           dev_path: Optional[Path] = None,
                           test_path: Optional[Path] = None,
                           registered_split: bool = False,
                           min_calibration_units: int = 1) -> DataBundle:
    dataset = dataset.upper()
    if dataset not in {"DS03", "DS08A"}:
        raise ValueError("N-CMAPSS dataset must be DS03 or DS08a")
    if validation_protocol not in {"legacy_endpoint", "fixed_cuts"}:
        raise ValueError("validation_protocol must be 'legacy_endpoint' or 'fixed_cuts'")
    if calibration_ratio is not None and not 0.0 < calibration_ratio < 1.0:
        raise ValueError("calibration_ratio must be between 0 and 1")
    if min_calibration_units < 0:
        raise ValueError("min_calibration_units must be non-negative")
    if registered_split:
        if calibration_ratio is None:
            calibration_ratio = 0.34
        # Registered N-CMAPSS protocol: ratios are applied to the dev-unit
        # pool, yielding 3 validation and 2 calibration units for the
        # nine-unit release split (val_ratio=.55, calibration_ratio=.34).
        val_ratio = 0.55 if val_ratio == 0.2 else val_ratio
        min_calibration_units = max(2, min_calibration_units)
    canonical = "DS08a" if dataset == "DS08A" else "DS03"
    data_dir = Path(data_root) / "data_set"
    if dev_path is None:
        dev_path = data_dir / ("N-CMAPSS_DS03-012.h5" if canonical == "DS03"
                               else "N-CMAPSS_DS08a-009.h5")
    if test_path is None:
        test_path = dev_path
    if not Path(dev_path).exists() or not Path(test_path).exists():
        raise FileNotFoundError("N-CMAPSS HDF5 file not found")

    dev = _records_from_nc(Path(dev_path), "dev", include_virtual)
    test = _records_from_nc(Path(test_path), "test", include_virtual)
    dev_units = sorted(dev)
    rng = np.random.default_rng(split_seed)
    shuffled = np.asarray(dev_units, dtype=np.int64)
    rng.shuffle(shuffled)
    n_holdout = max(1, int(round(len(dev_units) * val_ratio))) if len(dev_units) > 1 else 0
    if registered_split and len(dev_units) >= 5:
        # Registered external check reserves exactly five units when possible:
        # three validation and two calibration units.
        n_holdout = min(len(dev_units) - 1, 5)
    holdout_units = shuffled[:n_holdout].tolist()
    train_units = sorted(shuffled[n_holdout:].tolist())
    if calibration_ratio is None:
        val_units = sorted(holdout_units)
        calibration_units = []
    else:
        if len(holdout_units) < 2:
            raise ValueError("calibration_ratio requires at least 2 holdout units")
        n_calibration = (max(1, int(round(len(holdout_units) * calibration_ratio)))
                         if len(holdout_units) > 1 else 0)
        n_calibration = min(len(holdout_units) - 1, n_calibration)
        if n_calibration < min_calibration_units:
            raise ValueError(
                f"calibration split has {n_calibration} units; "
                f"requires at least {min_calibration_units}")
        val_units = sorted(holdout_units[:-n_calibration])
        calibration_units = sorted(holdout_units[-n_calibration:])
    if not train_units:
        raise ValueError("unit split leaves no training units")

    train_rows = np.concatenate([np.concatenate([dev[u][0], dev[u][1]], axis=1)
                                 for u in train_units], axis=0)
    scaler = FeatureScaler().fit(train_rows)
    sensor_dim = dev[dev_units[0]][0].shape[1]
    scaled_dev = {}
    for u, (x, w, y) in dev.items():
        z = scaler.transform(np.concatenate([x, w], axis=1))
        scaled_dev[u] = (z[:, :sensor_dim], z[:, sensor_dim:], y)
    scaled_test = {}
    for u, (x, w, y) in test.items():
        z = scaler.transform(np.concatenate([x, w], axis=1))
        scaled_test[u] = (z[:, :sensor_dim], z[:, sensor_dim:], y)

    endpoint_fracs = [0.25, 0.5, 0.75, 0.9] if validation_protocol == "fixed_cuts" else None
    endpoint_rng = np.random.default_rng(1_000_003 * split_seed + 11)
    endpoint_samples, endpoint_cuts = _endpoint_samples(
        scaled_dev, {u: scaled_dev[u][2] for u in dev_units}, val_units, history,
        endpoint_rng, fracs=endpoint_fracs)
    calibration_endpoint_samples, calibration_endpoint_cuts = _endpoint_samples(
        scaled_dev, {u: scaled_dev[u][2] for u in dev_units}, calibration_units, history,
        endpoint_rng, fracs=endpoint_fracs)

    metadata = {
        "dataset": canonical, "dev_units": dev_units, "train_units": train_units,
        "val_units": val_units, "calibration_units": calibration_units,
        "test_units": sorted(test), "history": history,
        "include_virtual": include_virtual, "uses_T": False,
        "aggregation": "unit-cycle mean; T is degradation metadata and is excluded",
        "split_seed": split_seed, "scaler_mode": "global",
        "calibration_ratio": calibration_ratio,
        "registered_split": registered_split,
        "min_calibration_units": min_calibration_units,
        "validation_protocol": validation_protocol,
        "val_endpoint": endpoint_cuts,
        "calibration_endpoint": calibration_endpoint_cuts,
        "val_endpoint_protocol": ("fixed cuts at life fractions {0.25, 0.5, 0.75, 0.9} per "
                                  "validation unit, endpoint windows, raw remaining flight "
                                  "RUL labels"
                                  if validation_protocol == "fixed_cuts" else
                                  "per validation unit: flight history truncated at a uniform "
                                  "[0.5, 1.0] life fraction, endpoint window, raw remaining "
                                  "flight RUL label"),
        "test_endpoint_protocol": ("test trajectories in the public release run to failure "
                                   "(final Y=0), so the frozen evaluation uses the same fixed "
                                   "cuts {0.25, 0.5, 0.75, 0.9} per test unit; labels are the "
                                   "file's Y at each cut"
                                   if validation_protocol == "fixed_cuts" else None),
    }
    test_endpoint_samples, test_endpoint_cuts = [], []
    if validation_protocol == "fixed_cuts":
        test_rng = np.random.default_rng(7_000_031 * split_seed + 13)
        test_endpoint_samples, test_endpoint_cuts = _endpoint_samples(
            scaled_test, {u: scaled_test[u][2] for u in scaled_test}, sorted(scaled_test),
            history, test_rng, fracs=endpoint_fracs)
    metadata["test_endpoint"] = test_endpoint_cuts
    return DataBundle(
        WindowDataset(_make_windows(scaled_dev, train_units, history)),
        WindowDataset(_make_windows(scaled_dev, val_units, history)),
        WindowDataset(_make_windows(scaled_test, sorted(scaled_test), history, final_only=True)),
        scaler, metadata, sensor_dim, 4,
        WindowDataset(endpoint_samples),
        WindowDataset(test_endpoint_samples) if test_endpoint_samples else None,
        WindowDataset(calibration_endpoint_samples) if calibration_endpoint_samples else None,
    )


def save_scaler(scaler: FeatureScaler, path: Path):
    path.write_text(json.dumps(scaler.state_dict(), indent=2), encoding="utf-8")
