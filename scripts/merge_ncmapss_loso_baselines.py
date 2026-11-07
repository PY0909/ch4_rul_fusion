"""Merge the corrected A6-Point LOSO rerun with the unchanged seven baselines.

docs/ncmapss-loso-baseline-correction-amendment-2026-11-07.md: only A6-Point
was rerun (P0 lock d32/2/2, weight decay 1e-3) under the common 120/30
budget; the other seven baselines keep their original development results,
whose configuration and budget did not change. Stop epochs and reasons for
the original folds are derived from their saved per-fold histories. Nothing
here loads data, calibration labels or test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ncmapss_loso_baselines import (BOUNDARY_MARGIN,  # noqa: E402
                                                DEV_MAX_EPOCHS, DEV_PATIENCE,
                                                METHODS, SEARCH_UNITS)

CORRECTED_METHODS = ("a6_point",)


def _rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_manifest(directory, expected_methods=None):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    if manifest.get("test_loaded") is not False:
        raise ValueError(f"{directory} must not have loaded test")
    if (manifest.get("max_epochs"), manifest.get("patience")) != (
            DEV_MAX_EPOCHS, DEV_PATIENCE):
        raise ValueError(f"{directory} was not run at {DEV_MAX_EPOCHS}/{DEV_PATIENCE}")
    if expected_methods is not None:
        if manifest.get("status") not in {"complete", "complete_methods_subset"}:
            raise ValueError(f"{directory} is not a complete seven-fold run")
        if sorted(manifest.get("methods", METHODS)) != sorted(expected_methods):
            raise ValueError(f"{directory} does not hold exactly {expected_methods}")
    elif manifest.get("status") != "complete":
        raise ValueError(f"{directory} is not a complete eight-baseline run")
    return manifest


def _stop_from_history(directory, dataset, method, held):
    history = Path(directory) / "folds" / dataset / method / f"held_{held}" / "history.csv"
    stop_epoch = len(_rows(history))
    return stop_epoch, ("cap" if stop_epoch >= DEV_MAX_EPOCHS else "patience")


def merge(original, corrected, output):
    original, corrected, output = Path(original), Path(corrected), Path(output)
    _check_manifest(original)
    corrected_manifest = _check_manifest(corrected, CORRECTED_METHODS)
    if corrected_manifest.get("a6_point_dims", {}).get("d_model") != 32:
        raise ValueError("corrected run does not use the P0 A6-Point lock")

    merged = []
    for row in _rows(original / "development.csv"):
        if row["method"] in CORRECTED_METHODS:
            continue
        stop_epoch, reason = _stop_from_history(original, row["dataset"],
                                                row["method"], row["held_unit"])
        merged.append({**row, "stop_epoch": stop_epoch, "stop_reason": reason,
                       "boundary_censored": (
                           int(row["best_epoch"]) > DEV_MAX_EPOCHS - BOUNDARY_MARGIN),
                       "source": "original_120_30_run"})
    for row in _rows(corrected / "development.csv"):
        if row["method"] not in CORRECTED_METHODS:
            raise ValueError("corrected run contains an unexpected method")
        merged.append({**row, "source": "a6_point_p0_lock_rerun"})

    summary = []
    for dataset in ("DS03", "DS08a"):
        for method in METHODS:
            group = [r for r in merged if r["dataset"] == dataset and r["method"] == method]
            held = sorted(int(r["held_unit"]) for r in group)
            if held != sorted(SEARCH_UNITS):
                raise ValueError(f"{dataset}/{method} lacks the seven registered folds")
            summary.append({
                "dataset": dataset, "method": method, "n_folds": len(group),
                "candidate_fixed_epochs": int(statistics.median_low(
                    [int(r["best_epoch"]) for r in group])),
                "min_best_epoch": min(int(r["best_epoch"]) for r in group),
                "max_best_epoch": max(int(r["best_epoch"]) for r in group),
                "mean_held_rmse": float(np.mean([float(r["val_rmse"]) for r in group])),
                "n_boundary_censored": int(sum(
                    str(r["boundary_censored"]) == "True" for r in group)),
                "n_stopped_by_cap": int(sum(r["stop_reason"] == "cap" for r in group)),
                "selection_criterion": group[0]["selection_criterion"],
                "selection_status": "development_candidate",
                "source": group[0]["source"]})

    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("development.csv", merged), ("development_summary.csv", summary)):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)
    (output / "manifest.json").write_text(json.dumps({
        "phase": "development", "status": "complete_merged", "test_loaded": False,
        "amendment": "docs/ncmapss-loso-baseline-correction-amendment-2026-11-07.md",
        "max_epochs": DEV_MAX_EPOCHS, "patience": DEV_PATIENCE,
        "boundary_margin": BOUNDARY_MARGIN, "search_units": SEARCH_UNITS,
        "corrected_methods": list(CORRECTED_METHODS),
        "sources": {
            "original": {"path": str(original),
                         "development_sha256": _sha256(original / "development.csv")},
            "corrected": {"path": str(corrected),
                          "development_sha256": _sha256(corrected / "development.csv")}},
    }, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default=str(ROOT / "results_v4/ncmapss_loso_baselines"))
    parser.add_argument("--corrected", default=str(ROOT / "results_v4/ncmapss_loso_baselines_v2"))
    parser.add_argument("--output", default=str(ROOT / "results_v4/ncmapss_loso_baselines_corrected"))
    args = parser.parse_args(argv)
    for row in merge(args.original, args.corrected, args.output):
        print(f"{row['dataset']:6s} {row['method']:20s} budget={row['candidate_fixed_epochs']:>3d} "
              f"range=[{row['min_best_epoch']},{row['max_best_epoch']}] "
              f"heldRMSE={row['mean_held_rmse']:.3f} cap_stops={row['n_stopped_by_cap']} "
              f"censored={row['n_boundary_censored']} ({row['source']})")


if __name__ == "__main__":
    main()
