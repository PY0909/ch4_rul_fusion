import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from rul_fusion.data import (_endpoint_samples, _make_windows, GroupedMinMaxScaler,
                             MinMaxScaler, build_cmapss_datasets, build_ncmapss_datasets)


def _write_cmapss(root: Path):
    data = root / "CMAPSSData"
    data.mkdir()
    rows = []
    for engine in [1, 2, 3, 4, 5]:
        for cycle in range(1, 6):
            rows.append([engine, cycle, 0.1, 0.2, 0.3] + [float(engine + cycle + i) for i in range(21)])
    np.savetxt(data / "train_FD002.txt", np.asarray(rows), fmt="%.6f")
    test_rows = []
    for engine in [1, 2]:
        for cycle in range(1, 4):
            test_rows.append([engine, cycle, 0.1, 0.2, 0.3] + [float(engine + cycle + i) for i in range(21)])
    np.savetxt(data / "test_FD002.txt", np.asarray(test_rows), fmt="%.6f")
    np.savetxt(data / "RUL_FD002.txt", np.asarray([[2], [4]]), fmt="%d")


def test_cmapss_engine_split_and_final_test_window(tmp_path):
    _write_cmapss(tmp_path)
    bundle = build_cmapss_datasets(tmp_path, "FD002", history=3, split_seed=7, val_ratio=0.2)
    assert set(bundle.metadata["train_units"]).isdisjoint(bundle.metadata["val_units"])
    assert len(bundle.test) == 2
    x, w, y, unit, sample = bundle.test[0]
    assert x.shape == (3, 14)
    assert w.shape == (3, 3)
    assert float(y) in {2.0, 4.0}
    assert unit in {1, 2}


def test_final_only_uses_last_observation_but_single_official_rul_label():
    x = np.arange(5, dtype=np.float32).reshape(5, 1)
    w = (10 + np.arange(5, dtype=np.float32)).reshape(5, 1)
    records = {7: (x, w, np.asarray([12.0], dtype=np.float32))}
    samples = _make_windows(records, [7], history=3, final_only=True)
    window_x, window_w, target, unit, sample_id = samples[0]
    np.testing.assert_array_equal(window_x[:, 0], [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(window_w[:, 0], [12.0, 13.0, 14.0])
    assert target == 12.0
    assert unit == 7


def test_minmax_scaler_handles_constant_columns():
    values = np.array([[1.0, 5.0], [3.0, 5.0]], dtype=np.float64)
    scaler = MinMaxScaler().fit(values)
    scaled = scaler.transform(values)
    np.testing.assert_allclose(scaled[:, 0], [0.0, 1.0])
    np.testing.assert_allclose(scaled[:, 1], [0.0, 0.0])


def test_grouped_minmax_uses_group_stats_and_global_fallback():
    values = np.array([[0.0, 100.0], [10.0, 200.0], [1000.0, -100.0], [2000.0, 100.0]],
                      dtype=np.float64)
    keys = np.array([1.0, 1.0, 2.0, 2.0])
    scaler = GroupedMinMaxScaler().fit(values, keys)
    scaled = scaler.transform(values, keys)
    np.testing.assert_allclose(scaled[:, 0], [0.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(scaled[:, 1], [0.0, 1.0, 0.0, 1.0])
    # unseen group falls back to global min/max instead of being dropped
    unseen_rows = np.array([[3000.0, 0.0]], dtype=np.float64)
    scaled_unseen = scaler.transform(unseen_rows, np.array([9.0]))
    assert scaler.unseen == [9.0]
    assert scaled_unseen.shape == (1, 2)
    assert np.isfinite(scaled_unseen).all()


def test_endpoint_samples_use_cut_window_and_raw_label():
    n = 6
    x = np.arange(n, dtype=np.float32).reshape(n, 1)
    w = np.zeros((n, 1), dtype=np.float32)
    raw = np.arange(n, 0, -1, dtype=np.float32)          # remaining RUL before any cap
    capped = np.minimum(raw, 2.0).astype(np.float32)
    records = {1: (x, w, capped)}
    rng = np.random.default_rng(5)
    samples, cuts = _endpoint_samples(records, {1: raw}, [1], history=3, rng=rng)
    cut = cuts[0]["cut"]
    assert cut == int(round(cuts[0]["frac"] * (n - 1)))
    window_x, _, target, unit, sample_id = samples[0]
    np.testing.assert_allclose(window_x[:, 0], x[max(0, cut - 2):cut + 1, 0])
    assert target == raw[cut]          # endpoint label is the raw remaining RUL
    assert target != capped[cut] or raw[cut] <= 2.0
    assert unit == 1 and sample_id == cut


def test_endpoint_samples_fixed_cuts_lock_four_windows_per_unit():
    n = 8
    x = np.arange(n, dtype=np.float32).reshape(n, 1)
    w = np.zeros((n, 1), dtype=np.float32)
    raw = (n - np.arange(n)).astype(np.float32)
    records = {3: (x, w, np.minimum(raw, 125.0).astype(np.float32))}
    samples, cuts = _endpoint_samples({3: records[3]}, {3: raw}, [3], history=4,
                                      rng=np.random.default_rng(0),
                                      fracs=[0.25, 0.5, 0.75, 0.9])
    assert [c["frac"] for c in cuts] == [0.25, 0.5, 0.75, 0.9]
    assert [c["cut"] for c in cuts] == [2, 4, 5, 6]
    assert [s[4] for s in samples] == [c["cut"] for c in cuts]
    for s, c in zip(samples, cuts):
        assert s[2] == raw[c["cut"]]       # raw labels regardless of the cap
        assert s[3] == 3


def test_cmapss_bundle_has_one_endpoint_window_per_val_engine(tmp_path):
    _write_cmapss(tmp_path)
    bundle = build_cmapss_datasets(tmp_path, "FD002", history=3, split_seed=7,
                                   val_ratio=0.2, rul_cap=2, scaler_mode="grouped")
    val_units = bundle.metadata["val_units"]
    assert bundle.val_endpoint is not None
    assert len(bundle.val_endpoint) == len(val_units)
    cuts = bundle.metadata["val_endpoint"]
    assert [s[3] for s in bundle.val_endpoint] == sorted(val_units)
    for i, entry in enumerate(cuts):
        _, _, y, unit, sid = bundle.val_endpoint[i]
        assert unit == entry["unit"]
        assert y == entry["label_raw"]         # label recorded at the cut
        assert sid == entry["cut"]
        assert entry["label_raw"] <= entry["length"] - entry["cut"] + 1
    assert bundle.metadata["scaler_mode"] == "grouped"


def test_cmapss_fixed_cuts_protocol_locks_four_cuts_per_val_engine(tmp_path):
    _write_cmapss(tmp_path)
    bundle = build_cmapss_datasets(tmp_path, "FD002", history=3, split_seed=7,
                                   val_ratio=0.2, scaler_mode="grouped",
                                   validation_protocol="fixed_cuts")
    val_units = bundle.metadata["val_units"]
    assert bundle.metadata["validation_protocol"] == "fixed_cuts"
    assert len(bundle.val_endpoint) == 4 * len(val_units)
    cuts = bundle.metadata["val_endpoint"]
    for unit in val_units:
        fracs = sorted(e["frac"] for e in cuts if e["unit"] == unit)
        assert fracs == [0.25, 0.5, 0.75, 0.9]
    # deterministic: rebuilding yields identical endpoint windows
    again = build_cmapss_datasets(tmp_path, "FD002", history=3, split_seed=7,
                                  val_ratio=0.2, scaler_mode="grouped",
                                  validation_protocol="fixed_cuts")
    for a, b in zip(bundle.val_endpoint, again.val_endpoint):
        assert (a[3], a[4], float(a[2])) == (b[3], b[4], float(b[2]))


def _write_ncmapss(path: Path, inconsistent=False):
    n = 12
    A = []
    W = []
    Xs = []
    Xv = []
    Y = []
    for unit in [1, 2, 3]:
        for cycle in [1, 2]:
            for row in range(2):
                A.append([unit, cycle, 1, 1])
                W.append([100 + row, 0.1, 0.5, 20])
                Xs.append([unit + cycle + row + j for j in range(14)])
                Xv.append([unit + cycle + row + j for j in range(14)])
                Y.append([10 - cycle if not (inconsistent and unit == 1 and cycle == 1 and row == 1) else 99])
    with h5py.File(path, "w") as f:
        f.create_dataset("A_dev", data=np.asarray(A, dtype=np.float64))
        f.create_dataset("W_dev", data=np.asarray(W, dtype=np.float64))
        f.create_dataset("X_s_dev", data=np.asarray(Xs, dtype=np.float64))
        f.create_dataset("X_v_dev", data=np.asarray(Xv, dtype=np.float64))
        f.create_dataset("Y_dev", data=np.asarray(Y, dtype=np.int64))
        # one test unit with a full run-to-failure trajectory (final Y=0), as in the release
        for cycle in [1, 2]:
            A.append([10, cycle, 1, 1])
            W.append([100 + cycle, 0.1, 0.5, 20])
            Xs.append([10 + cycle + j for j in range(14)])
            Xv.append([10 + cycle + j for j in range(14)])
            Y.append([7 - cycle])
        test_data = {"A_test": np.asarray(A, dtype=np.float64)[-2:],
                     "W_test": np.asarray(W, dtype=np.float64)[-2:],
                     "X_s_test": np.asarray(Xs, dtype=np.float64)[-2:],
                     "X_v_test": np.asarray(Xv, dtype=np.float64)[-2:],
                     "Y_test": np.asarray(Y, dtype=np.int64)[-2:]}
        for key, data in test_data.items():
            f.create_dataset(key, data=data)


def test_cmapss_settings_cycle_appends_cycle_column(tmp_path):
    _write_cmapss(tmp_path)
    bundle = build_cmapss_datasets(tmp_path, "FD002", history=3, split_seed=7,
                                   scaler_mode="grouped",
                                   condition_features="settings_cycle")
    assert bundle.condition_dim == 4
    x, w, y, unit, sid = bundle.train[0]
    assert w.shape == (3, 4)
    assert np.isfinite(w.numpy()).all()
    assert bundle.metadata["condition_features"] == "settings_cycle"
    # group keys are still the raw setting1: identical split as settings-only build
    plain = build_cmapss_datasets(tmp_path, "FD002", history=3, split_seed=7,
                                  scaler_mode="grouped")
    assert bundle.metadata["train_units"] == plain.metadata["train_units"]
    assert bundle.metadata["val_units"] == plain.metadata["val_units"]


def test_ncmapss_groups_unit_cycles_and_uses_cycle_labels(tmp_path):
    path = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss(path)
    bundle = build_ncmapss_datasets(tmp_path, "DS03", history=2, dev_path=path,
                                    test_path=path, split_seed=0)
    assert bundle.sensor_dim == 14
    assert bundle.condition_dim == 4
    assert len(bundle.metadata["dev_units"]) == 3
    assert len(bundle.train) > 0 and len(bundle.val) > 0
    x, w, y, unit, sample = bundle.train[0]
    assert x.shape == (2, 14)
    assert w.shape == (2, 4)
    assert np.isfinite(x.numpy()).all()
    assert bundle.test_endpoint is None          # legacy protocol: no mid-life test cuts


def test_ncmapss_fixed_cuts_builds_val_and_test_endpoints(tmp_path):
    path = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss(path)
    bundle = build_ncmapss_datasets(tmp_path, "DS03", history=2, dev_path=path,
                                    test_path=path, split_seed=0,
                                    validation_protocol="fixed_cuts")
    n_val, n_test = len(bundle.metadata["val_units"]), len(bundle.metadata["test_units"])
    assert len(bundle.val_endpoint) == 4 * n_val
    assert bundle.test_endpoint is not None
    assert len(bundle.test_endpoint) == 4 * n_test
    for s in bundle.test_endpoint:               # mid-life cuts never land on the Y=0 tail
        assert float(s[2]) > 0


def test_ncmapss_rejects_nonconstant_cycle_label(tmp_path):
    path = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss(path, inconsistent=True)
    with pytest.raises(ValueError, match="not constant"):
        build_ncmapss_datasets(tmp_path, "DS03", history=2, dev_path=path,
                               test_path=path, split_seed=0)
