"""Ensemble CQR: average the A6-Quantile members' quantiles, then calibrate.

Members are the seed-0..4 global quantile checkpoints trained on the identical
engine split (split_seed=0).  Quantile averaging happens before scoring, then
the same four threshold variants as run_weighted_cqr.py are fitted against the
calibration cut windows and evaluated once on the official test endpoint.
Official test is touched once per invocation; the touch is logged.
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
from rul_fusion.metrics import regression_metrics  # noqa: E402
from rul_fusion.quantile_runner import _predict  # noqa: E402
from scripts.run_weighted_cqr import (  # noqa: E402
    ALPHA, _apply, _block_max, _density_ratio, _features, _load_model,
    _plain_threshold, _rank_key, _weighted_threshold, _window_scores,
)
from torch.utils.data import DataLoader  # noqa: E402


def _loader(dataset):
    return DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)


def _average(predictions):
    ys = [p[0] for p in predictions]
    for y in ys[1:]:
        assert np.allclose(y, ys[0])
    q = np.mean([p[1] for p in predictions], axis=0)
    return (ys[0], q, predictions[0][2], predictions[0][3], predictions[0][4])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint_root", default=str(
        ROOT / "results_v4/chapter4_main/quantile_cqr/global"))
    parser.add_argument("--output_root", default=str(ROOT / "results_v4/ensemble_cqr"))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for dataset in ("FD002", "FD004"):
        bundle = build_cmapss_datasets(
            args.data_root, dataset, history=50, rul_cap=125, split_seed=0,
            calibration_ratio=0.5, scaler_mode="grouped",
            validation_protocol="fixed_cuts")
        checkpoints = [Path(args.checkpoint_root) / dataset / f"seed_{s}" / "best.pt"
                       for s in seeds]
        missing = [str(p) for p in checkpoints if not p.exists()]
        if missing:
            print(f"[{dataset}] SKIP missing checkpoints: {missing}", flush=True)
            continue

        def predict(dataset):
            preds = [_predict(_load_model(bundle, ckpt, device),
                              _loader(dataset), device) for ckpt in checkpoints]
            return _average(preds)

        cal_y, cal_q, cal_units, cal_ids, cal_cond = predict(bundle.calibration_endpoint)
        cal_scores_w = _window_scores(cal_y, cal_q)
        cal_feats = _features(cal_q, cal_cond)
        block_scores, _ = _block_max(cal_scores_w, cal_units)

        tgt_y, tgt_q, tgt_units, _, tgt_cond = predict(bundle.test)
        tgt_feats = _features(tgt_q, tgt_cond)
        weights_w = _density_ratio(cal_feats, tgt_feats)
        weights_b = np.asarray([
            weights_w[cal_units == unit].mean() for unit in np.unique(cal_units)])

        variants = {
            "ens_window_weighted": (cal_scores_w, weights_w, _weighted_threshold),
            "ens_window_plain": (cal_scores_w, np.ones(len(cal_scores_w)),
                                 _plain_threshold),
            "ens_block_max_plain": (block_scores, np.ones(len(block_scores)),
                                    _plain_threshold),
            "ens_block_max_weighted": (block_scores, weights_b, _weighted_threshold),
        }
        rows = []
        for name, (scores, weights, fitter) in variants.items():
            threshold = (fitter(scores, weights, ALPHA)
                         if fitter is _weighted_threshold
                         else fitter(scores, ALPHA))
            metrics = _apply(tgt_y, tgt_q, threshold)
            point = regression_metrics(tgt_y, tgt_q[:, 1])
            rows.append({"dataset": dataset, "variant": name, "seeds": args.seeds,
                         "n_members": len(seeds),
                         "n_cal_points": int(len(scores)),
                         "n_test_engines": int(len(tgt_y)),
                         "point_RMSE": point["RMSE"], "point_MAE": point["MAE"],
                         **metrics})
        rows.sort(key=_rank_key)
        all_rows.extend(rows)
        print(f"[{dataset}] ensemble members={len(seeds)}", flush=True)
        for row in rows:
            print("  {:<22s} thr={:7.2f} PICP={:.3f} MPIW={:7.2f} IS={:7.2f} "
                  "RMSE={:.2f}".format(
                      row["variant"], row["threshold"], row["PICP"],
                      row["MPIW"], row["IntervalScore"], row["point_RMSE"]),
                  flush=True)
        print(f"  selected: {rows[0]['variant']}", flush=True)

    if not all_rows:
        print("no datasets evaluated", flush=True)
        return
    with (out_root / "matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    generated = datetime.now().isoformat(timespec="seconds")
    (out_root / "summary.json").write_text(json.dumps(
        {"generated": generated, "official_test_evaluated": True,
         "device": str(device), "rows": all_rows}, indent=2, default=float),
        encoding="utf-8")
    touch_log = ROOT / "results_v4/chapter4_main_logs/test_touches.log"
    with touch_log.open("a", encoding="utf-8") as f:
        f.write(f"[{generated}] ensemble_cqr: official test evaluated for 4 "
                f"threshold variants x FD002/FD004 (seeds {args.seeds})\n")


if __name__ == "__main__":
    main()
