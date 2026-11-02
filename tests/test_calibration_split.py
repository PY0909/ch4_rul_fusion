import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from rul_fusion.data import build_cmapss_datasets


def _write_cmapss(root: Path):
    data = root / "CMAPSSData"
    data.mkdir()
    rows = []
    for engine in range(1, 11):
        for cycle in range(1, 7):
            rows.append([engine, cycle, 0.1, 0.2, 0.3] +
                        [float(engine + cycle + i) for i in range(21)])
    np.savetxt(data / "train_FD002.txt", np.asarray(rows), fmt="%.6f")
    test_rows = []
    for engine in [1, 2]:
        for cycle in range(1, 4):
            test_rows.append([engine, cycle, 0.1, 0.2, 0.3] +
                             [float(engine + cycle + i) for i in range(21)])
    np.savetxt(data / "test_FD002.txt", np.asarray(test_rows), fmt="%.6f")
    np.savetxt(data / "RUL_FD002.txt", np.asarray([[2], [4]]), fmt="%d")


def test_calibration_units_are_disjoint_and_endpoint_labels_are_raw(tmp_path):
    _write_cmapss(tmp_path)
    bundle = build_cmapss_datasets(
        tmp_path, "FD002", history=3, rul_cap=2, split_seed=7, val_ratio=0.2,
        calibration_ratio=0.5, scaler_mode="grouped", validation_protocol="fixed_cuts")
    train = set(bundle.metadata["train_units"])
    val = set(bundle.metadata["val_units"])
    calibration = set(bundle.metadata["calibration_units"])
    assert not (train & val or train & calibration or val & calibration)
    assert len(bundle.calibration_endpoint) == 4 * len(calibration)
    assert bundle.metadata["calibration_endpoint"]
    for sample in bundle.calibration_endpoint:
        assert sample[2] == bundle.metadata["calibration_endpoint"][0]["label_raw"] or \
            sample[2] in [entry["label_raw"] for entry in bundle.metadata["calibration_endpoint"]]


def test_high_calibration_ratio_keeps_validation_engine(tmp_path):
    _write_cmapss(tmp_path)
    bundle = build_cmapss_datasets(
        tmp_path, "FD002", history=3, split_seed=7, val_ratio=0.2,
        calibration_ratio=0.99, scaler_mode="grouped", validation_protocol="fixed_cuts")
    assert len(bundle.metadata["val_units"]) >= 1
    assert len(bundle.metadata["calibration_units"]) >= 1
