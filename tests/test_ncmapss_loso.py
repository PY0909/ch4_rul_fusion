import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from test_data import _write_ncmapss_nine_units  # noqa: E402
from rul_fusion.data import build_ncmapss_datasets  # noqa: E402
from rul_fusion.model import MultiScalePhaseGraphRUL  # noqa: E402
from rul_fusion.uncertainty import (A6QuantileRUL,  # noqa: E402
                                    NonCrossingQuantileHead)
from scripts.run_ncmapss_loso_search import (main as search_main,  # noqa: E402
                                             parse_configs, pick_budget)
from scripts.run_ncmapss_loso_final import (  # noqa: E402
    _frozen_selection, main as final_main)


def _quantile_model(activation, seed=3):
    torch.manual_seed(seed)
    backbone = MultiScalePhaseGraphRUL(sensor_dim=14, condition_dim=4,
                                       d_model=32, num_heads=2, num_routers=2,
                                       dropout=0.1, encoder="patch_cnn",
                                       variant="A6", condition_mode="full",
                                       activation=activation)
    return A6QuantileRUL(backbone, dropout=0.1, activation=activation)


def test_legacy_activation_is_bit_identical_to_default():
    x = torch.randn(2, 8, 14)
    w = torch.randn(2, 8, 4)
    default = _quantile_model("legacy")
    torch.manual_seed(3)
    backbone = MultiScalePhaseGraphRUL(sensor_dim=14, condition_dim=4,
                                       d_model=32, num_heads=2, num_routers=2,
                                       dropout=0.1, encoder="patch_cnn",
                                       variant="A6", condition_mode="full")
    plain = A6QuantileRUL(backbone, dropout=0.1)
    for key in plain.state_dict():
        assert torch.equal(plain.state_dict()[key], default.state_dict()[key])
    plain.eval(); default.eval()
    with torch.no_grad():
        assert torch.equal(plain(x, w), default(x, w))


def test_activation_variants_are_finite_and_sorted():
    x = torch.randn(2, 8, 14)
    w = torch.randn(2, 8, 4)
    outputs = {}
    for activation in ("legacy", "relu", "gelu", "silu"):
        model = _quantile_model(activation)
        model.eval()
        with torch.no_grad():
            q = model(x, w)
        assert torch.isfinite(q).all()
        assert torch.all(q[:, 0] <= q[:, 1]) and torch.all(q[:, 1] <= q[:, 2])
        outputs[activation] = q
    assert not torch.allclose(outputs["legacy"], outputs["relu"])
    assert not torch.allclose(outputs["gelu"], outputs["silu"])
    head = NonCrossingQuantileHead(16, activation="silu")
    q = head(torch.randn(3, 16))
    assert q.shape == (3, 3) and torch.isfinite(q).all()


def test_unit_override_loso_fold_partition(tmp_path):
    path = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss_nine_units(path)
    bundle = build_ncmapss_datasets(
        tmp_path, "DS03", history=2, dev_path=path, test_path=path,
        split_seed=0, validation_protocol="fixed_cuts", registered_split=True,
        unit_override={"train": [1, 2, 3, 4, 5, 6, 7, 8], "val": [9],
                       "calibration": []})
    meta = bundle.metadata
    assert meta["train_units"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert meta["val_units"] == [9] and meta["calibration_units"] == []
    assert len(bundle.val_endpoint) == 4                 # held engine's cuts
    assert not bundle.calibration_endpoint               # empty -> None or 0
    assert meta["unit_override"]["val"] == [9]
    train_units_seen = sorted({int(u) for u in bundle.train.units})
    assert train_units_seen == [1, 2, 3, 4, 5, 6, 7, 8]


def test_dev_only_bundle_does_not_open_test_file(tmp_path):
    path = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss_nine_units(path)
    bundle = build_ncmapss_datasets(
        tmp_path, "DS03", history=2, dev_path=path,
        test_path=tmp_path / "missing_test.h5", load_test=False,
        validation_protocol="fixed_cuts", registered_split=True)
    assert bundle.metadata["test_units"] == []
    assert bundle.metadata["test_endpoint"] == []
    assert len(bundle.test) == 0
    assert bundle.test_endpoint is None
    assert len(bundle.val_endpoint) == 3 * 4


def test_unit_override_final_arm_shape(tmp_path):
    path = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss_nine_units(path)
    g9 = [round(0.1 * step, 2) for step in range(1, 10)]
    bundle = build_ncmapss_datasets(
        tmp_path, "DS03", history=2, dev_path=path, test_path=path,
        split_seed=0, validation_protocol="fixed_cuts", registered_split=True,
        calibration_fracs=g9,
        unit_override={"train": [3, 4, 5, 6, 7, 8, 9], "val": [],
                       "calibration": [1, 2]})
    assert bundle.metadata["train_units"] == [3, 4, 5, 6, 7, 8, 9]
    assert not bundle.val_endpoint                       # empty -> None or 0
    assert len(bundle.calibration_endpoint) == 2 * 9     # G9 on two engines
    assert bundle.metadata["calibration_fracs"] == g9


def test_unit_override_rejects_overlap_and_unknown(tmp_path):
    path = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss_nine_units(path)
    common = dict(history=2, dev_path=path, test_path=path, split_seed=0,
                  validation_protocol="fixed_cuts", registered_split=True)
    for override in ({"train": [1, 2], "val": [2], "calibration": []},
                     {"train": [99], "val": [], "calibration": []},
                     {"train": [], "val": [1], "calibration": []},
                     {"train": [1], "val": [2]}):
        try:
            build_ncmapss_datasets(tmp_path, "DS03", unit_override=override,
                                   **common)
        except ValueError:
            continue
        raise AssertionError(f"override should have been rejected: {override}")


def test_pick_budget_and_parse_configs():
    curve = [9.0, 8.0, 7.0, 6.0, 5.0, 5.0, 6.0, 7.0]
    budget = pick_budget(curve)
    assert 4 <= budget <= 6
    configs = parse_configs("gelu:0.001:0.2:0.001, silu:0.0003:0.1:0.001")
    assert configs[0] == {"activation": "gelu", "lr": 0.001, "dropout": 0.2,
                          "weight_decay": 0.001}
    assert configs[1]["activation"] == "silu"
    for bad in ("gelu:0.001", "gelu:0.001:0.1:0.001,"):
        try:
            parse_configs(bad)
        except ValueError:
            continue
        raise AssertionError(f"should reject {bad!r}")


def test_loso_search_smoke_on_synthetic(tmp_path):
    path = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss_nine_units(path)
    out = tmp_path / "loso_out"
    search_main(["--data_root", str(tmp_path), "--datasets", "DS03",
                 "--dev_path", str(path), "--test_path",
                 str(tmp_path / "missing_test.h5"),
                 "--grid", "custom", "--configs", "legacy:0.001:0.1:0.001",
                 "--max_epochs", "2", "--folds", "3",
                 "--output_dir", str(out), "--device", "cpu"])
    curves = list(csv.DictReader(open(out / "curves_DS03_custom.csv")))
    assert len(curves) == 3 * 2                     # 3 folds x 2 epochs
    selection = list(csv.DictReader(open(out / "selection_custom.csv")))
    assert len(selection) == 1 and selection[0]["config"].startswith("legacy")
    assert np.isfinite(float(selection[0]["pinball_mean"]))
    budgets = list(csv.DictReader(open(out / "criterion_budgets_custom.csv")))
    assert {row["criterion"] for row in budgets} == {
        "pinball_mean", "q50_rmse", "raw_interval_score"}
    assert all("raw_MPIW" in row for row in budgets)
    aggregate = list(csv.DictReader(open(out / "aggregate_custom.csv")))
    assert all("raw_MPIW" in row for row in aggregate)


def test_loso_search_excludes_registered_calibration_units(tmp_path, monkeypatch):
    path = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss_nine_units(path)
    out = tmp_path / "pilot"
    args = ["--data_root", str(tmp_path), "--datasets", "DS03",
            "--dev_path", str(path), "--test_path",
            str(tmp_path / "missing_test.h5"), "--grid", "pilot",
            "--max_epochs", "1", "--output_dir", str(out),
            "--device", "cpu"]
    search_main(args)
    curves = list(csv.DictReader(open(out / "curves_DS03_pilot.csv")))
    assert len(curves) == 7 * 2
    assert {int(row["held_unit"]) for row in curves} == {1, 2, 3, 5, 6, 8, 9}
    assert {row["config"] for row in curves} == {
        "legacy_lr0.001_do0.1_wd0.001", "legacy_lr0.0003_do0.1_wd0.001"}
    assert len(list((out / "folds" / "DS03").rglob("held_*.csv"))) == 14
    search = importlib.import_module("scripts.run_ncmapss_loso_search")

    def refuse_retraining(*_args, **_kwargs):
        raise AssertionError("completed folds must be reused")

    monkeypatch.setattr(search, "run_fold_config", refuse_retraining)
    search_main(args)


def test_loso_search_rejects_changed_calibration_ids(tmp_path, monkeypatch):
    path = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss_nine_units(path)
    search = importlib.import_module("scripts.run_ncmapss_loso_search")
    real_build = search.build_ncmapss_datasets

    def changed_split(*args, **kwargs):
        bundle = real_build(*args, **kwargs)
        if kwargs.get("unit_override") is None:
            bundle.metadata["calibration_units"] = [1, 2]
        return bundle

    monkeypatch.setattr(search, "build_ncmapss_datasets", changed_split)
    with pytest.raises(RuntimeError, match="registered calibration units"):
        search_main(["--data_root", str(tmp_path), "--datasets", "DS03",
                     "--dev_path", str(path), "--grid", "pilot",
                     "--max_epochs", "1", "--folds", "1",
                     "--output_dir", str(tmp_path / "out"), "--device", "cpu"])


def test_loso_final_requires_frozen_selection_and_explicit_test(tmp_path, capsys):
    args = ["--dataset", "DS03", "--data_root", str(tmp_path)]
    with pytest.raises(SystemExit):
        final_main(args)
    assert "--selection_file" in capsys.readouterr().err

    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "status": "frozen", "datasets": {"DS03": {
            "config": {"activation": "legacy", "lr": 0.001,
                       "dropout": 0.1, "weight_decay": 0.001},
            "budget_epoch": 26, "criterion": "pinball_mean",
            "n_folds": 7, "boundary_censored": False,
            "search_units": [1, 2, 3, 5, 6, 8, 9],
            "calibration_units": [4, 7],
        }},
    }))
    with pytest.raises(SystemExit):
        final_main(args + ["--selection_file", str(selection)])
    assert "--eval_test" in capsys.readouterr().err

    record = json.loads(selection.read_text())
    record["datasets"]["DS03"].pop("n_folds")
    selection.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="complete seven-fold"):
        _frozen_selection(selection, "DS03")


def test_loso_final_synthetic_frozen_run(tmp_path, monkeypatch):
    final = importlib.import_module("scripts.run_ncmapss_loso_final")
    data_dir = tmp_path / "data_set"
    data_dir.mkdir()
    _write_ncmapss_nine_units(data_dir / "N-CMAPSS_DS03-012.h5")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "status": "frozen", "datasets": {"DS03": {
            "config": {"activation": "legacy", "lr": 0.001,
                       "dropout": 0.1, "weight_decay": 0.001},
            "budget_epoch": 1, "criterion": "pinball_mean",
            "n_folds": 7, "boundary_censored": False,
            "search_units": [1, 2, 3, 5, 6, 8, 9],
            "calibration_units": [4, 7],
        }},
    }))
    monkeypatch.setattr(final, "ROOT", tmp_path)
    monkeypatch.setattr(final, "_registered_calibration_units",
                        lambda _dataset: [4, 7])
    final.main(["--dataset", "DS03", "--data_root", str(tmp_path),
                "--selection_file", str(selection), "--eval_test",
                "--output_dir", str(tmp_path / "final"), "--device", "cpu"])
    result = json.loads((tmp_path / "final/DS03/seed_0/summary.json").read_text())
    assert result["train_units"] == [1, 2, 3, 5, 6, 8, 9]
    assert result["calibration_units"] == [4, 7]
    assert len(result["rows"]) == 3
    assert all(row["n_cal_windows"] == 18 for row in result["rows"])
    touch = (tmp_path / "results_v4/chapter4_main_logs/test_touches.log").read_text()
    assert "test access started" in touch
    assert "official test evaluated once" in touch
