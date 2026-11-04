"""One-shot official-test evaluation of the P1 raw-label (cap=0) quantile arm.

Mirrors the main arm's interval variants for a symmetric comparison table:
global block-max CQR and window-weighted CQR, both fitted only on the arm's
own calibration engines, evaluated once on the official test windows using
the frozen cap=0 checkpoint. Test labels are read only for metrics; the
touch is logged. rul_cap=0 bundle (training-label parity with the arm).
"""
from __future__ import annotations

import argparse
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
from rul_fusion.metrics import regression_metrics  # noqa: E402
from rul_fusion.uncertainty import interval_metrics  # noqa: E402
from scripts.run_weighted_cqr import (  # noqa: E402
    ALPHA, _block_max, _density_ratio, _features, _loader, _load_model,
    _plain_threshold, _weighted_threshold, _window_scores,
)
from rul_fusion.quantile_runner import _predict  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--arm_root", default=str(ROOT / "results_v4/fd_cap0_arm"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    out_path = Path(args.output or (Path(args.arm_root) / "quantile_test_eval.json"))

    rows = []
    for dataset in ("FD002", "FD004"):
        bundle = build_cmapss_datasets(
            args.data_root, dataset, history=50, rul_cap=0, split_seed=0,
            calibration_ratio=0.5, scaler_mode="grouped",
            validation_protocol="fixed_cuts")
        ckpt = Path(args.arm_root) / "quantile_cqr" / "global" / dataset / "seed_0" / "best.pt"
        model = _load_model(bundle, ckpt, device)

        cal_y, cal_q, cal_units, _, cal_cond = _predict(
            model, _loader(bundle.calibration_endpoint), device)
        cal_scores = _window_scores(cal_y, cal_q)
        cal_feats = _features(cal_q, cal_cond)
        block_scores, _ = _block_max(cal_scores, cal_units)

        tgt_y, tgt_q, tgt_units, _, tgt_cond = _predict(
            model, _loader(bundle.test), device)
        tgt_feats = _features(tgt_q, tgt_cond)
        weights = _density_ratio(cal_feats, tgt_feats)

        point = regression_metrics(tgt_y, tgt_q[:, 1])
        row = {"dataset": dataset, "endpoint": "official test", "rul_cap": 0,
               "checkpoint": str(ckpt), "official_test_evaluated": True,
               **{f"q50_{k}": v for k, v in point.items()}}
        for name, scores, w in (
                ("global_block_max", block_scores, None),
                ("window_weighted", cal_scores, weights)):
            thr = (_weighted_threshold(scores, w, ALPHA) if w is not None
                   else _plain_threshold(scores, ALPHA))
            lower, upper = tgt_q[:, 0] - thr, tgt_q[:, 2] + thr
            m = interval_metrics(tgt_y, lower, upper, alpha=ALPHA)
            row[f"{name}_threshold"] = float(thr)
            for k, v in m.items():
                row[f"{name}_{k}"] = v
        rows.append(row)
        print(dataset, "q50 RMSE=%.3f | block-max PICP=%.3f IS=%.2f | "
              "window-weighted PICP=%.3f IS=%.2f" % (
                  point["RMSE"], row["global_block_max_PICP"],
                  row["global_block_max_IntervalScore"],
                  row["window_weighted_PICP"],
                  row["window_weighted_IntervalScore"]), flush=True)

    out_path.write_text(json.dumps(
        {"official_test_evaluated": True, "generated": datetime.now().isoformat(
            timespec="seconds"), "rows": rows}, indent=2, default=float),
        encoding="utf-8")
    touch = ROOT / "results_v4/chapter4_main_logs/test_touches.log"
    with touch.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] "
                f"eval_cap0_arm_test: P1 raw-cap arm one-shot test evaluation "
                f"(q50 + block-max + window-weighted variants, frozen "
                f"checkpoints, no selection)\n")
    print("wrote", out_path)


if __name__ == "__main__":
    raise SystemExit(main())
