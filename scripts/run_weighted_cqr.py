"""Density-ratio weighted CQR against the endpoint estimand shift.

Calibration windows sit at fixed life-fraction cuts while the official test
endpoint is each engine's final window.  Tibshirani et al. (2019) weighted
conformal corrects exactly this covariate shift: each calibration score is
weighted by the density ratio between the target window distribution and the
calibration window distribution, estimated with logistic regression on
observable covariates only (predicted q50, interval width, condition mean; no
labels, no unit lengths).

Endpoint protocol:
  --endpoint val   thresholds are fitted against val engines' final windows
                   (constructed from bundle.val) and coverage is measured on
                   the same val final windows.  Official test is NOT read.
  --endpoint test  weights are fitted against unlabeled test-window
                   covariates and every variant is evaluated once on the
                   official test endpoint (disclosure logged).

Variants share the frozen seed-0 global quantile checkpoint; only thresholds
differ, so no retraining is involved.
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rul_fusion.data import build_cmapss_datasets  # noqa: E402
from rul_fusion.metrics import interval_metrics, quantile_metrics, rmse  # noqa: E402
from rul_fusion.model import MultiScalePhaseGraphRUL  # noqa: E402
from rul_fusion.quantile_runner import _predict  # noqa: E402
from rul_fusion.uncertainty import (A6QuantileRUL, CQRCalibrator,
                                    normalize_quantile_features,
                                    weighted_conformal_threshold)  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ALPHA = 0.2


def _load_model(bundle, checkpoint, device):
    backbone = MultiScalePhaseGraphRUL(
        sensor_dim=bundle.sensor_dim, condition_dim=bundle.condition_dim,
        d_model=64, num_heads=4, num_routers=4, dropout=0.1,
        encoder="patch_cnn", variant="A6", condition_mode="full")
    model = A6QuantileRUL(backbone, dropout=0.1).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def _loader(dataset):
    return DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)


def _latest_mask(units, sample_ids):
    mask = np.zeros(len(units), dtype=bool)
    for unit in np.unique(units):
        indices = np.flatnonzero(units == unit)
        mask[indices[np.argmax(sample_ids[indices])]] = True
    return mask


def _features(q, cond_means, normalization=1.0, feature_spec=("q50", "width", "cond")):
    """Observable window covariates; no labels, no unit lengths.

    ``feature_spec`` selects columns: q50 -> normalized median, width ->
    normalized q90-q10, cond -> condition mean. Default is the frozen
    production triple (plan F5).
    """
    normalized = normalize_quantile_features(q, normalization)
    columns = {"q50": normalized[:, 1],
               "width": normalized[:, 2] - normalized[:, 0],
               "cond": np.asarray(cond_means, dtype=np.float64)}
    if not feature_spec or any(name not in columns for name in feature_spec):
        raise ValueError(f"unknown feature spec: {feature_spec}")
    return np.column_stack([columns[name] for name in feature_spec])


def _density_ratio(cal_feats, target_feats, clip=20.0, lr_c=1.0):
    """w(x) = p_target(x) / p_cal(x) via the logistic odds identity."""
    x = np.vstack([cal_feats, target_feats])
    y = np.concatenate([np.zeros(len(cal_feats)), np.ones(len(target_feats))])
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(lr_c), max_iter=2000, class_weight="balanced"),
    )
    clf.fit(x, y)
    p = clf.predict_proba(cal_feats)[:, 1]
    prior = len(cal_feats) / max(len(target_feats), 1)
    ratio = prior * p / np.clip(1.0 - p, 1e-6, None)
    return np.clip(ratio, 1.0 / clip, clip)


def _window_scores(y, q):
    return CQRCalibrator._scores(y, q[:, 0], q[:, 2])


def _block_max(scores, units):
    out, keep = [], []
    for unit in np.unique(units):
        idx = np.flatnonzero(units == unit)
        local = idx[np.argmax(scores[idx])]
        keep.append(local)
        out.append(float(scores[local]))
    return np.asarray(out), np.asarray(keep, dtype=np.int64)


def _weighted_threshold(scores, weights, alpha):
    """Weighted conformal quantile with the conservative sup-weight atom."""
    return weighted_conformal_threshold(scores, weights, alpha)


def _plain_threshold(scores, alpha):
    order = np.argsort(scores)
    s = scores[order]
    idx = int(np.ceil((1.0 - alpha) * (len(s) + 1))) - 1
    return float(s[max(0, min(idx, len(s) - 1))])


def _apply(y, q, threshold, alpha=ALPHA):
    lower, upper = q[:, 0] - threshold, q[:, 2] + threshold
    out = interval_metrics(y, lower, upper, alpha=alpha)
    out["q50_RMSE"] = float(rmse(y, q[:, 1]))
    out["threshold"] = float(threshold)
    out.update(quantile_metrics(y, q[:, 0], q[:, 1], q[:, 2]))
    return out


def _rank_key(metrics, alpha=ALPHA):
    coverage = metrics["PICP"]
    eligible = 0.75 <= coverage <= 0.85
    return (0 if eligible else 1,
            metrics["IntervalScore"] + 100.0 * abs(coverage - (1.0 - alpha)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint_root", default=str(
        ROOT / "results_v4/chapter4_main/quantile_cqr/global"))
    parser.add_argument("--output_root", default=str(ROOT / "results_v4/weighted_cqr"))
    parser.add_argument("--endpoint", choices=("val", "test"), required=True)
    parser.add_argument("--ratio_features", default="q50,width,cond",
                        help="comma list of q50,width,cond (default: frozen F5)")
    parser.add_argument("--clip", type=float, default=20.0,
                        help="density-ratio clip bound (lower bound is 1/clip)")
    parser.add_argument("--lr_c", type=float, default=1.0,
                        help="logistic regression inverse regularization C")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    feature_spec = tuple(part.strip() for part in args.ratio_features.split(",")
                         if part.strip())
    alpha = float(args.alpha)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    out_root = Path(args.output_root) / args.endpoint
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for dataset in ("FD002", "FD004"):
        bundle = build_cmapss_datasets(
            args.data_root, dataset, history=50, rul_cap=125, split_seed=0,
            calibration_ratio=0.5, scaler_mode="grouped",
            validation_protocol="fixed_cuts")
        checkpoint = Path(args.checkpoint_root) / dataset / "seed_0" / "best.pt"
        model = _load_model(bundle, checkpoint, device)

        cal_y, cal_q, cal_units, cal_ids, cal_cond = _predict(
            model, _loader(bundle.calibration_endpoint), device)
        cal_scores_w = _window_scores(cal_y, cal_q)
        cal_feats = _features(cal_q, cal_cond, feature_spec=feature_spec)
        block_scores, block_idx = _block_max(cal_scores_w, cal_units)

        if args.endpoint == "val":
            val_y, val_q, val_units, val_ids, val_cond = _predict(
                model, _loader(bundle.val), device)
            latest = _latest_mask(val_units, val_ids)
            tgt_y, tgt_q = val_y[latest], val_q[latest]
            tgt_units = val_units[latest]
            tgt_cond = val_cond[latest]
            tgt_name = "val final windows"
        else:
            tgt_y, tgt_q, tgt_units, tgt_ids, tgt_cond = _predict(
                model, _loader(bundle.test), device)
            tgt_name = "official test"

        tgt_feats = _features(tgt_q, tgt_cond, feature_spec=feature_spec)
        weights_w = _density_ratio(cal_feats, tgt_feats, clip=args.clip,
                                   lr_c=args.lr_c)
        weights_b = np.asarray([
            weights_w[cal_units == unit].mean() for unit in np.unique(cal_units)])

        variants = {
            "block_max_plain": (block_scores, np.ones(len(block_scores)),
                                _plain_threshold),
            "window_plain": (cal_scores_w, np.ones(len(cal_scores_w)),
                             _plain_threshold),
            "window_weighted": (cal_scores_w, weights_w, _weighted_threshold),
            "block_max_weighted": (block_scores, weights_b, _weighted_threshold),
        }
        rows = []
        for name, (scores, weights, fitter) in variants.items():
            threshold = (fitter(scores, weights, alpha)
                         if fitter is _weighted_threshold
                         else fitter(scores, alpha))
            metrics = _apply(tgt_y, tgt_q, threshold, alpha=alpha)
            rows.append({"dataset": dataset, "variant": name,
                         "endpoint": args.endpoint, "target": tgt_name,
                         "n_cal_points": int(len(scores)),
                         "n_target_engines": int(len(np.unique(tgt_units))),
                         "weight_min": float(weights.min()),
                         "weight_max": float(weights.max()),
                         **metrics})
        rows.sort(key=lambda row: _rank_key(row, alpha))
        best = rows[0]
        all_rows.extend(rows)
        print(f"[{dataset}] target={tgt_name} n={len(tgt_y)}", flush=True)
        for row in rows:
            print("  {:<18s} thr={:7.2f} PICP={:.3f} MPIW={:7.2f} IS={:7.2f}".format(
                row["variant"], row["threshold"], row["PICP"],
                row["MPIW"], row["IntervalScore"]), flush=True)
        print(f"  selected: {best['variant']}", flush=True)

    with (out_root / "matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "endpoint": args.endpoint,
        "device": str(device),
        "official_test_evaluated": args.endpoint == "test",
        "ratio_features": list(feature_spec),
        "clip": args.clip,
        "lr_c": args.lr_c,
        "alpha": alpha,
        "ratio_fit": ("logistic regression on observable covariates; test run "
                      "fits weights against unlabeled test-window features"),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "rows": all_rows,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    if args.endpoint == "test":
        touch_log = ROOT / "results_v4/chapter4_main_logs/test_touches.log"
        with touch_log.open("a", encoding="utf-8") as f:
            f.write(f"[{summary['generated']}] weighted_cqr: official test "
                    f"evaluated for 4 threshold variants x FD002/FD004 "
                    f"(features-only ratio fit, no labels)\n")


if __name__ == "__main__":
    main()
