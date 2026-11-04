"""Recount the chapter-4 panel under two consistent label conventions.

The pipeline trains on RUL labels capped at 125 but the official test labels
are read uncapped from RUL_FD00x.txt, so 22%/27% of test engines sit above the
cap where the model was never supervised. This script reports every method
under both conventions so the discrepancy is visible rather than silently
favouring any single row:

  raw     official test RUL as distributed (current panel numbers)
  capped  test RUL truncated at the training cap, matching what was learned

It only reads persisted prediction CSVs; no checkpoint or GPU is required.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rul_fusion.metrics import regression_metrics  # noqa: E402

CAP = 125.0

# (panel label, path template, point column)
METHODS = [
    ("A6-Point", "results_v4/chapter4_main/A6-point/{ds}/seed_0/predictions.csv",
     "predicted_rul"),
    ("A6-Quantile (q50)",
     "results_v4/chapter4_main/quantile_cqr/global/{ds}/seed_0/predictions.csv", "q50"),
    ("HNN", "results_v4/chapter4_main/{ds}/hnn/seed_0/predictions.csv",
     "predicted_mean"),
    ("EDL-NIG", "results_v4/chapter4_main/{ds}/edl_nig/seed_0/predictions.csv",
     "predicted_mean"),
]


def _read(path, column):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y = np.asarray([float(r["target_rul"]) for r in rows], dtype=np.float64)
    p = np.asarray([float(r[column]) for r in rows], dtype=np.float64)
    return y, p


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(
        ROOT / "results_v4/chapter4_main_logs/panel_recount.json"))
    args = parser.parse_args(argv)

    table = {}
    for dataset in ("FD002", "FD004"):
        rows = []
        for label, template, column in METHODS:
            path = ROOT / template.format(ds=dataset)
            if not path.exists():
                print(f"[{dataset}] missing {label}: {path}", flush=True)
                continue
            y, p = _read(path, column)
            above = int((y > CAP).sum())
            entry = {"method": label, "n_test_engines": int(len(y)),
                     "n_above_cap": above,
                     "frac_above_cap": float(above / len(y))}
            for convention, target in (("raw", y), ("capped", np.minimum(y, CAP))):
                m = regression_metrics(target, p)
                entry[f"{convention}_MAE"] = m["MAE"]
                entry[f"{convention}_RMSE"] = m["RMSE"]
                entry[f"{convention}_NASA"] = m["NASA_Score"]
            rows.append(entry)
        rows.sort(key=lambda r: r["capped_RMSE"])
        table[dataset] = rows
        print(f"\n== {dataset}  (n={rows[0]['n_test_engines']}, "
              f"{rows[0]['n_above_cap']} engines above cap=", flush=True)
        for r in rows:
            print("  {:<18s} raw MAE/RMSE/NASA {:6.2f}/{:6.2f}/{:8.1f}   "
                  "capped {:5.2f}/{:5.2f}/{:7.1f}".format(
                      r["method"], r["raw_MAE"], r["raw_RMSE"], r["raw_NASA"],
                      r["capped_MAE"], r["capped_RMSE"], r["capped_NASA"]),
                  flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "cap": CAP,
        "note": ("raw = official test labels; capped = labels truncated at the "
                 "training cap. Both are reported so the protocol mismatch is "
                 "visible for every method, not just the proposed one."),
        "datasets": table,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
