"""One-shot official-test evaluation of the frozen checkpoint-B candidates.

Method frozen at checkpoint A (2026-11-02): late-endpoint aggregation +
global threshold + raw CQR scores. Candidates: FD002/FD004 x ratio
{0.6, 0.7}, each evaluated with its OWN matched retrained checkpoint and
its own calibration engines (fixes the stage-1 'exploratory_split_overlap'
caveat — those rows mixed checkpoints whose validation engines overlapped
the calibration split). The official test split is read only for final
metric computation.

DISCLOSURE (pre-registered boundary): official test engines were already
used by the stage-1 candidate matrix (results_v4/calibration_stage1_test),
so all test numbers in this project are exploratory/descriptive. The
0.6-vs-0.7 ratio decision is made after this comparison at the user's
explicit direction — one disclosed degree of test-informed selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rul_fusion.data import build_cmapss_datasets  # noqa: E402
from rul_fusion.metrics import regression_metrics  # noqa: E402
from rul_fusion.uncertainty import interval_metrics  # noqa: E402
from scripts.evaluate_calibration_stage1_test import _load_model, _loader  # noqa: E402
from scripts.run_calibration_stage1 import (  # noqa: E402
    _aggregate, _fit_thresholds, _transform,
)


def evaluate(bundle, model, device, mode="late"):
    grouper_unused = None  # global thresholds only (checkpoint-A freeze)
    cal = _predict_wrapped(model, _loader(bundle.calibration_endpoint), device)
    cal_y, cal_q, cal_units, cal_ids, cal_conditions = cal
    scores, reduced_groups, _ = _aggregate(
        cal_y, cal_q, cal_units, cal_ids, None, mode)
    threshold, group_thresholds, group_counts = _fit_thresholds(
        scores, reduced_groups, 20, 20.0)
    test = _predict_wrapped(model, _loader(bundle.test), device)
    y, q, _, _, conditions = test
    lower, upper = _transform(q, None, threshold, group_thresholds)
    point = regression_metrics(y, q[:, 1])
    interval = interval_metrics(y, lower, upper, alpha=0.2)
    return {**point, **interval, "calibration_threshold": float(threshold),
            "n_test_engines": int(len(y))}


def _predict_wrapped(model, loader, device):
    from rul_fusion.quantile_runner import _predict
    return _predict(model, loader, device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint_root", default=str(
        ROOT / "results_v4/calibration_ratio_earlystop_v1"))
    parser.add_argument("--output_root", default=str(
        ROOT / "results_v4/frozen_candidates_test"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in ("FD002", "FD004"):
        for ratio in (0.6, 0.7):
            bundle = build_cmapss_datasets(
                args.data_root, dataset, history=50, rul_cap=125, split_seed=0,
                calibration_ratio=ratio, scaler_mode="grouped",
                validation_protocol="fixed_cuts")
            ckpt = (Path(args.checkpoint_root) / f"ratio_{ratio}" / "global"
                    / dataset / "seed_0" / "best.pt")
            model = _load_model(bundle, ckpt, device)
            result = evaluate(bundle, model, device, mode="late")
            row = {"dataset": dataset, "ratio": ratio, "aggregation": "late",
                   "calibration_mode": "global", "score_mode": "raw",
                   "checkpoint": str(ckpt), "checkpoint_split_match": True,
                   "official_test_evaluated": True,
                   "disclosure": "test previously seen by stage-1 matrix; "
                                 "ratio 0.6-vs-0.7 chosen after this "
                                 "comparison (user-directed, disclosed)",
                   **result}
            rows.append(row)
            print(dataset, ratio, f"PICP={result['PICP']:.3f} "
                  f"MPIW={result['MPIW']:.2f} IS={result['IntervalScore']:.2f} "
                  f"RMSE={result['RMSE']:.2f}", flush=True)
    fields = sorted({k for r in rows for k in r})
    with (out / "frozen_candidates_test.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(
        {"official_test_evaluated": True, "method": "late/global/raw (checkpoint A)",
         "rows": rows}, indent=2), encoding="utf-8")
    print("wrote", out / "frozen_candidates_test.csv")


if __name__ == "__main__":
    raise SystemExit(main())
