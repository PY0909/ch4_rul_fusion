"""Independent pre-test verification of the N-CMAPSS final-arm training stage.

Checks every A6-Quantile and baseline training output against the frozen
record (results_v4/ncmapss_loso_frozen/selection.json): frozen config and
epochs, seven search engines with no validation set, calibration engines
[4, 7], checkpoint and calibration-file SHA-256, selection-file hash, and
``test_loaded: false``. Reads only local artifacts; never touches data or test.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "results_v4/ncmapss_loso_frozen/selection.json"
A6_ROOT = ROOT / "results_v4/ncmapss_loso_final"
BASELINE_ROOT = ROOT / "results_v4/ncmapss_loso_final_baselines/final"
SEARCH_UNITS = [1, 2, 3, 5, 6, 8, 9]
CALIBRATION_UNITS = [4, 7]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    selection = json.loads(SELECTION.read_text())
    selection_sha = _sha(SELECTION)
    rows = []
    for dataset, entry in selection["datasets"].items():
        run = A6_ROOT / dataset / "seed_0"
        m = json.loads((run / "train_manifest.json").read_text())
        checks = {
            "test_not_loaded": m["test_loaded"] is False,
            "frozen_config": m["config"] == entry["config"],
            "frozen_epochs": m["epochs"] == entry["budget_epoch"] == m["epochs_trained"],
            "search_engines": m["train_units"] == SEARCH_UNITS,
            "calibration_engines": m["calibration_units"] == CALIBRATION_UNITS,
            "g9_windows": m["n_calibration_windows"] == 18,
            "checkpoint_hash": m["checkpoint_sha256"] == _sha(run / "best.pt"),
            "calibration_hash": m["calibration_sha256"] == _sha(run / "calibration_predictions.npz"),
            "selection_hash": m["selection_sha256"] == selection_sha,
        }
        # information only: an existing evaluation directory means this run was
        # already evaluated; it is not a training-integrity failure
        rows_extra = {"already_evaluated": (run / "evaluation").exists()}
        rows.append({"dataset": dataset, "model": "A6-Quantile (main)",
                     "frozen_epochs": entry["budget_epoch"],
                     "epochs_trained": m["epochs_trained"],
                     "checkpoint_sha256": m["checkpoint_sha256"][:12],
                     "all_checks_pass": all(checks.values()),
                     **rows_extra,
                     "failed": ";".join(k for k, ok in checks.items() if not ok)})
        for method, spec in entry["methods"].items():
            run = BASELINE_ROOT / dataset / method / "train"
            m = json.loads((run / "manifest.json").read_text())
            checks = {
                "phase": m["phase"] == "final_train",
                "test_not_loaded": m["test_loaded"] is False,
                "frozen_epochs": (m["identity"]["epochs"] == spec["epochs"] ==
                                  m["epochs_trained"]),
                "search_engines": m["train_units"] == SEARCH_UNITS,
                "no_validation": m["val_units"] == [],
                "calibration_engines_unused": m["calibration_units"] == CALIBRATION_UNITS,
                "checkpoint_hash": m["checkpoint_sha256"] == _sha(run / "best.pt"),
                "selection_hash": m["identity"]["selection_sha256"] == selection_sha,
            }
            rows.append({"dataset": dataset, "model": method,
                         "frozen_epochs": spec["epochs"],
                         "epochs_trained": m["epochs_trained"],
                         "checkpoint_sha256": m["checkpoint_sha256"][:12],
                         "all_checks_pass": all(checks.values()),
                         "already_evaluated": (run.parent / "eval").exists(),
                         "failed": ";".join(k for k, ok in checks.items() if not ok)})
    return rows


def main():
    rows = verify()
    out = ROOT / "results_v4/ncmapss_loso_final_verification.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['dataset']:6s} {row['model']:20s} {row['epochs_trained']:>3}/"
              f"{row['frozen_epochs']:<3} ckpt {row['checkpoint_sha256']} "
              f"{'PASS' if row['all_checks_pass'] else 'FAIL ' + row['failed']}")
    passed = sum(row["all_checks_pass"] for row in rows)
    print(f"{passed}/{len(rows)} training runs verified -> {out}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
