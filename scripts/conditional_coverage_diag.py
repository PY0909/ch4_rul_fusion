"""P3 conditional-coverage diagnostics (CPU post-processing, frozen artifacts).

Reproduces the frozen window-weighted CQR intervals deterministically
(same helpers as run_weighted_cqr.py: frozen chapter4_main seed-0
checkpoint, ratio-0.5 grouped split, density-ratio weights) and reports
per-group coverage on the official test windows, grouped by

1. fitted condition/stage cluster (ConditionStageQualityGrouper, refit
   deterministically on train predictions), and
2. RUL horizon bins of the true label (<=30 / 30-60 / 60-125 / >125) —
   the stratification the asymmetric NASA cost cares about.

Per the registered matrix this is DIAGNOSTIC ONLY: small group sizes are
reported explicitly and nothing here feeds back into thresholds,
groupings or any hyperparameter. The extra test read is logged to
test_touches.log.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rul_fusion.data import build_cmapss_datasets  # noqa: E402
from rul_fusion.quantile_runner import _fit_grouper, _predict  # noqa: E402
from scripts.run_weighted_cqr import (  # noqa: E402
    ALPHA, _density_ratio, _features, _loader, _load_model,
    _weighted_threshold, _window_scores,
)

HORIZON_BINS = [(0.0, 30.0), (30.0, 60.0), (60.0, 125.0), (125.0, float("inf"))]


def _group_table(name, labels, y, lower, upper):
    rows = []
    for g in sorted(set(labels.tolist())):
        m = labels == g
        cover = ((y >= lower) & (y <= upper))[m].mean()
        rows.append({"group_axis": name, "group": str(g), "n": int(m.sum()),
                     "PICP": float(cover),
                     "mean_width": float((upper - lower)[m].mean())})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint_root", default=str(
        ROOT / "results_v4/chapter4_main/quantile_cqr/global"))
    parser.add_argument("--output_root", default=str(
        ROOT / "results_v4/conditional_coverage_diag"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for dataset in ("FD002", "FD004"):
        bundle = build_cmapss_datasets(
            args.data_root, dataset, history=50, rul_cap=125, split_seed=0,
            calibration_ratio=0.5, scaler_mode="grouped",
            validation_protocol="fixed_cuts")
        checkpoint = Path(args.checkpoint_root) / dataset / "seed_0" / "best.pt"
        model = _load_model(bundle, checkpoint, device)

        cal_y, cal_q, cal_units, _, cal_cond = _predict(
            model, _loader(bundle.calibration_endpoint), device)
        cal_scores = _window_scores(cal_y, cal_q)
        cal_feats = _features(cal_q, cal_cond)

        tgt_y, tgt_q, tgt_units, _, tgt_cond = _predict(
            model, _loader(bundle.test), device)
        tgt_feats = _features(tgt_q, tgt_cond)
        weights = _density_ratio(cal_feats, tgt_feats)
        threshold = _weighted_threshold(cal_scores, weights, ALPHA)
        lower = tgt_q[:, 0] - threshold
        upper = tgt_q[:, 2] + threshold

        grouper = _fit_grouper(bundle, model, _loader(bundle.train), device)
        stage = grouper.groups(tgt_cond, tgt_q[:, 1])
        horizon = np.full(len(tgt_y), -1, dtype=int)
        for i, (lo, hi) in enumerate(HORIZON_BINS):
            horizon[(tgt_y >= lo) & (tgt_y < hi)] = i

        rows = _group_table("stage_cluster", np.asarray(stage), tgt_y, lower, upper)
        rows += _group_table("rul_horizon", horizon, tgt_y, lower, upper)
        for r in rows:
            r["dataset"] = dataset
            r["threshold"] = float(threshold)
        all_rows.extend(rows)
        print(f"[{dataset}] thr={threshold:.2f} overall PICP="
              f"{((tgt_y >= lower) & (tgt_y <= upper)).mean():.3f}")
        for r in rows:
            print(f"  {r['group_axis']:14s} g={r['group']:>4s} n={r['n']:4d} "
                  f"PICP={r['PICP']:.3f} width={r['mean_width']:6.1f}")

    fields = sorted({k for r in all_rows for k in r})
    with (out / "groups.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    (out / "summary.json").write_text(json.dumps(
        {"diagnostic_only": True, "generated": datetime.now().isoformat(
            timespec="seconds"), "rows": all_rows}, indent=2, default=float),
        encoding="utf-8")
    touch = ROOT / "results_v4/chapter4_main_logs/test_touches.log"
    with touch.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] "
                f"conditional_coverage_diag: official test read for per-group "
                f"coverage DIAGNOSTIC on frozen window-weighted intervals "
                f"(no parameter or threshold changes)\n")
    print("wrote", out / "groups.csv")


if __name__ == "__main__":
    raise SystemExit(main())
