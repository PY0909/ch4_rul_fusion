"""Validation-only comparison of raw and scale-adaptive CQR on FD002/FD004."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rul_fusion.data import build_cmapss_datasets  # noqa: E402
from rul_fusion.metrics import interval_metrics, rmse  # noqa: E402
from rul_fusion.model import MultiScalePhaseGraphRUL  # noqa: E402
from rul_fusion.quantile_runner import _fit_grouper, _predict  # noqa: E402
from rul_fusion.uncertainty import (  # noqa: E402
    A6QuantileRUL,
    CQRCalibrator,
    ScaleAdaptiveCQRCalibrator,
)
from scripts.run_calibration_stage1 import (  # noqa: E402
    WEIGHTS, _aggregate, _fit_thresholds, _latest_mask, _transform,
)


def _model(bundle, checkpoint, device):
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


def _normalized_unit_scores(y, q, units, sample_ids, groups, aggregation, eps):
    scores = ScaleAdaptiveCQRCalibrator.scores(y, q[:, 0], q[:, 2], eps=eps)
    out_scores, out_groups = [], []
    for unit in np.unique(units):
        inds = np.flatnonzero(units == unit)
        inds = inds[np.argsort(sample_ids[inds])]
        if aggregation == "block_max":
            score_index = inds[np.argmax(scores[inds])]
            value = scores[score_index]
        elif aggregation == "late":
            score_index = inds[-1]
            value = scores[score_index]
        elif aggregation == "weighted_late":
            if len(inds) != len(WEIGHTS):
                raise ValueError("weighted_late requires exactly four fixed cuts per engine")
            score_index = inds[-1]
            value = float(np.dot(WEIGHTS, scores[inds]))
        else:
            raise ValueError(f"unknown aggregation {aggregation}")
        out_scores.append(float(value))
        out_groups.append(groups[score_index] if groups is not None else "global")
    return np.asarray(out_scores), np.asarray(out_groups, dtype=object)


def _evaluate(bundle, model, aggregation, score_mode, calibration_mode,
              min_group_size, shrink_tau, eps, device):
    train_loader = _loader(bundle.train)
    cal = _predict(model, _loader(bundle.calibration_endpoint), device)
    val = _predict(model, _loader(bundle.val_endpoint), device)
    grouper = _fit_grouper(bundle, model, train_loader, device)
    cal_y, cal_q, cal_units, cal_ids, cal_conditions = cal
    val_y, val_q, val_units, val_ids, val_conditions = val
    hierarchical = calibration_mode == "hierarchical"
    cal_groups = grouper.groups(cal_conditions, cal_q[:, 1]) if hierarchical else None
    val_groups = grouper.groups(val_conditions, val_q[:, 1]) if hierarchical else None

    if score_mode == "raw":
        scores, score_groups, _ = _aggregate(
            cal_y, cal_q, cal_units, cal_ids, cal_groups, aggregation)
        calibrator = CQRCalibrator(alpha=0.2, min_group_size=min_group_size,
                                   shrink_tau=shrink_tau)
        calibrator.fit_scores(scores, groups=score_groups)
    elif score_mode == "scale_adaptive":
        scores, score_groups = _normalized_unit_scores(
            cal_y, cal_q, cal_units, cal_ids, cal_groups, aggregation, eps)
        calibrator = ScaleAdaptiveCQRCalibrator(
            alpha=0.2, min_group_size=min_group_size,
            shrink_tau=shrink_tau, eps=eps)
        calibrator.fit_scores(scores, groups=score_groups)
    else:
        raise ValueError(score_mode)

    if score_mode == "scale_adaptive":
        lower, upper = calibrator.transform(val_q[:, 0], val_q[:, 2], val_groups)
    else:
        lower, upper = _transform(val_q, val_groups, calibrator.global_threshold,
                                  calibrator.group_thresholds)
    latest = _latest_mask(val_units, val_ids)
    metrics = interval_metrics(val_y[latest], lower[latest], upper[latest], alpha=0.2)
    q50_rmse = float(rmse(val_y[latest], val_q[latest, 1]))
    active = {k: v for k, v in calibrator.group_thresholds.items()
              if k != "global" and not np.isclose(v, calibrator.global_threshold,
                                                   rtol=0.0, atol=1e-12)}
    return {
        **metrics, "q50_RMSE": q50_rmse,
        "n_calibration_engines": int(len(np.unique(cal_units))),
        "n_validation_engines": int(len(np.unique(val_units))),
        "calibration_threshold": float(calibrator.global_threshold),
        "group_counts": calibrator.group_counts,
        "group_thresholds": calibrator.group_thresholds,
        "n_active_groups": len(active), "active_group_thresholds": active,
        "calibration_scores": scores.tolist(),
        "calibration_score_semantics": "per_engine_aggregated_nonconformity",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint_root", default=str(
        ROOT / "results_v4/chapter4_main/quantile_cqr/global"))
    parser.add_argument("--output_root", default=str(
        ROOT / "results_v4/scale_adaptive_cqr_validation"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--calibration_ratio", type=float, default=0.5,
                        help="bundle split ratio AND the retrained-checkpoint "
                             "split it must match (0.5 = frozen stage-1 default; "
                             "0.6/0.7 = registered extension per "
                             "scale_adaptive_cqr_stage2_experiment_matrix.md)")
    args = parser.parse_args(argv)
    if args.eps <= 0:
        parser.error("--eps must be positive")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dataset in ("FD002", "FD004"):
        bundle = build_cmapss_datasets(
            args.data_root, dataset, history=50, rul_cap=125, split_seed=0,
            calibration_ratio=args.calibration_ratio, scaler_mode="grouped",
            validation_protocol="fixed_cuts")
        checkpoint = (Path(args.checkpoint_root) / dataset / "seed_0" / "best.pt")
        model = _model(bundle, checkpoint, device)
        rows, results = [], {}
        candidates = [
            ("global", "block_max", "raw", 20, 20.0),
            ("global", "late", "raw", 20, 20.0),
            ("global", "weighted_late", "raw", 20, 20.0),
            ("global", "block_max", "scale_adaptive", 20, 20.0),
            ("global", "late", "scale_adaptive", 20, 20.0),
            ("global", "weighted_late", "scale_adaptive", 20, 20.0),
            ("hierarchical", "block_max", "raw", 8, 10.0),
            ("hierarchical", "block_max", "scale_adaptive", 8, 10.0),
            ("hierarchical", "late", "scale_adaptive", 8, 10.0),
            ("hierarchical", "block_max", "scale_adaptive", 10, 20.0),
            ("hierarchical", "late", "scale_adaptive", 10, 20.0),
            ("hierarchical", "weighted_late", "scale_adaptive", 10, 20.0),
        ]
        for calibration_mode, aggregation, score_mode, min_group_size, shrink_tau in candidates:
            result = _evaluate(bundle, model, aggregation, score_mode,
                               calibration_mode, min_group_size, shrink_tau,
                               args.eps, device)
            key = (f"{calibration_mode}__{aggregation}__{score_mode}__"
                   f"min_{min_group_size}__tau_{shrink_tau:g}")
            results[key] = result
            rows.append({"dataset": dataset, "key": key,
                         "checkpoint_split_ratio": args.calibration_ratio,
                         "calibration_ratio": args.calibration_ratio,
                         "calibration_mode": calibration_mode,
                         "aggregation": aggregation,
                         "score_mode": score_mode,
                         "min_group_size": min_group_size,
                         "shrink_tau": shrink_tau,
                         "eps": args.eps,
                         "checkpoint_split_match": True,
                         "test_evaluated": False,
                         "val_latest_PICP": result["PICP"],
                         "val_latest_MPIW": result["MPIW"],
                         "val_latest_IntervalScore": result["IntervalScore"],
                         "val_latest_coverage_gap": result["coverage_gap"],
                         "val_latest_q50_RMSE": result["q50_RMSE"],
                         "n_active_groups": result["n_active_groups"],
                         "threshold": result["calibration_threshold"]})
            print(dataset, key, f"PICP={result['PICP']:.3f}",
                  f"MPIW={result['MPIW']:.2f}", flush=True)
        ds_out = output_root / dataset
        ds_out.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0])
        with (ds_out / "matrix.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        summary = {"dataset": dataset, "checkpoint": str(checkpoint),
                   "checkpoint_split_ratio": args.calibration_ratio, "test_evaluated": False,
                   "eps": args.eps, "results": results}
        (ds_out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
    (output_root / "summary.json").write_text(json.dumps(
        {"device": str(device), "test_evaluated": False, "datasets": summaries}, indent=2),
        encoding="utf-8")
if __name__ == "__main__":
    main()
