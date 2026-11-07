import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.merge_ncmapss_loso_baselines import merge
from scripts.run_ncmapss_loso_baselines import METHODS, SEARCH_UNITS

FIELDS = ("best_epoch", "dataset", "held_unit", "method", "selection_criterion", "val_rmse")


def _write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _original(tmp_path):
    root = tmp_path / "original"
    rows = []
    for dataset in ("DS03", "DS08a"):
        for method in METHODS:
            for index, held in enumerate(SEARCH_UNITS):
                best = 5 + index
                rows.append({"best_epoch": best, "dataset": dataset, "held_unit": held,
                             "method": method, "selection_criterion": "val_rmse",
                             "val_rmse": 10.0 + index})
                stop = 120 if (method == "hnn" and held == 1) else best + 30
                _write(root / "folds" / dataset / method / f"held_{held}" / "history.csv",
                       [{"epoch": e} for e in range(1, stop + 1)], ("epoch",))
    _write(root / "development.csv", rows, FIELDS)
    (root / "manifest.json").write_text(json.dumps({
        "status": "complete", "test_loaded": False, "max_epochs": 120, "patience": 30}))
    return root


def _corrected(tmp_path, d_model=32, held_units=SEARCH_UNITS):
    root = tmp_path / "corrected"
    rows = [{"best_epoch": 40 + index, "dataset": dataset, "held_unit": held,
             "method": "a6_point", "selection_criterion": "val_endpoint_rmse",
             "val_rmse": 7.0, "stop_epoch": 70 + index, "stop_reason": "patience",
             "boundary_censored": False}
            for dataset in ("DS03", "DS08a") for index, held in enumerate(held_units)]
    _write(root / "development.csv", rows, sorted(rows[0]))
    (root / "manifest.json").write_text(json.dumps({
        "status": "complete_methods_subset", "methods": ["a6_point"],
        "test_loaded": False, "max_epochs": 120, "patience": 30,
        "a6_point_dims": {"d_model": d_model, "num_heads": 2, "num_routers": 2}}))
    return root


def test_merge_takes_a6_point_from_rerun_and_derives_stop_reasons(tmp_path):
    summary = merge(_original(tmp_path), _corrected(tmp_path), tmp_path / "merged")
    assert len(summary) == 2 * len(METHODS)
    a6 = [row for row in summary if row["method"] == "a6_point"]
    assert all(row["source"] == "a6_point_p0_lock_rerun" for row in a6)
    assert all(row["candidate_fixed_epochs"] == 43 for row in a6)   # low median of 40..46
    hnn = next(r for r in summary if r["method"] == "hnn" and r["dataset"] == "DS03")
    assert hnn["n_stopped_by_cap"] == 1 and hnn["source"] == "original_120_30_run"
    rows = list(csv.DictReader((tmp_path / "merged" / "development.csv").open()))
    assert len(rows) == 2 * len(METHODS) * len(SEARCH_UNITS)
    manifest = json.loads((tmp_path / "merged" / "manifest.json").read_text())
    assert manifest["status"] == "complete_merged" and manifest["test_loaded"] is False


def test_merge_rejects_wrong_lock_and_missing_folds(tmp_path):
    original = _original(tmp_path)
    with pytest.raises(ValueError):
        merge(original, _corrected(tmp_path, d_model=64), tmp_path / "bad1")
    with pytest.raises(ValueError):
        merge(original, _corrected(tmp_path, held_units=SEARCH_UNITS[:6]),
              tmp_path / "bad2")
