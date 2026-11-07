import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from test_data import _write_ncmapss_nine_units  # noqa: E402
from scripts.run_ncmapss_loso_budget_compare import (  # noqa: E402
    _load_cache, main, should_stop)


def test_stopping_waits_for_both_budget_rules():
    pinball = [5, 4, 3, 4, 5]
    interval = [9, 8, 7, 6, 5]
    assert not should_stop(pinball, interval, patience=2, window=1)
    interval.extend([6, 7])
    pinball.extend([6, 7])
    assert should_stop(pinball, interval, patience=2, window=1)


def test_stopping_uses_centered_five_epoch_window():
    descending = [9, 8, 7, 6, 5]
    assert not should_stop(descending, descending, patience=1)
    values = descending + [6, 7, 8, 9]
    assert should_stop(values, values, patience=1)


def test_incomplete_fold_cache_cannot_be_reused(tmp_path):
    path = tmp_path / "held_1.json"
    identity = {"patience": 30}
    path.write_text(json.dumps({
        "identity": identity,
        "rows": [{"epoch": epoch, "pinball_mean": 10.0 - epoch,
                  "raw_interval_score": 100.0 - epoch}
                 for epoch in range(1, 6)],
    }))
    with pytest.raises(RuntimeError, match="invalid completed-fold cache"):
        _load_cache(path, identity, max_epochs=120)


def test_budget_comparison_smoke_excludes_test_and_registered_calibration(tmp_path):
    path = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss_nine_units(path)
    out = tmp_path / "comparison"
    args = ["--data_root", str(tmp_path), "--datasets", "DS03",
            "--dev_path", str(path), "--test_path",
            str(tmp_path / "missing_test.h5"), "--max_epochs", "5",
            "--patience", "2", "--folds", "1", "--output_dir", str(out),
            "--device", "cpu"]
    main(args)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "partial_smoke"
    assert manifest["test_loaded"] is False
    assert manifest["search_units"] == [1, 2, 3, 5, 6, 8, 9]
    assert manifest["calibration_units_excluded"] == [4, 7]
    budgets = list(csv.DictReader((out / "fold_budgets.csv").open()))
    assert len(budgets) == 4  # two learning rates x two budget rules
    assert {row["criterion"] for row in budgets} == {
        "pinball_mean", "raw_interval_score"}
    assert {int(row["held_unit"]) for row in budgets} == {1}
    assert all(1 <= int(row["budget_epoch"]) <= int(row["stop_epoch"]) <= 5
               for row in budgets)
    comparison = list(csv.DictReader((out / "comparison.csv").open()))
    assert len(comparison) == 4
    assert all(int(row["n_folds"]) == 1 for row in comparison)


def test_full_comparison_rejects_unregistered_epoch_or_patience(tmp_path):
    with pytest.raises(SystemExit):
        main(["--data_root", str(tmp_path), "--max_epochs", "5",
              "--device", "cpu"])
    with pytest.raises(SystemExit):
        main(["--data_root", str(tmp_path), "--patience", "2",
              "--device", "cpu"])
