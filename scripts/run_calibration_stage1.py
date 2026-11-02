"""Stage-1 HCQ-CQR calibration sweep on frozen quantile checkpoints.

This script deliberately evaluates validation and calibration engines only.  It
does not run the official test endpoint, so the selected calibration protocol
can be frozen before the final test evaluation.
"""

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
    finite_sample_quantile,
)


WEIGHTS = np.asarray([0.05, 0.15, 0.30, 0.50], dtype=np.float64)
ALPHA = 0.2


def _model(bundle, checkpoint, device):
    backbone = MultiScalePhaseGraphRUL(
        sensor_dim=bundle.sensor_dim,
        condition_dim=bundle.condition_dim,
        d_model=64,
        num_heads=4,
        num_routers=4,
        dropout=0.1,
        encoder="patch_cnn",
        variant="A6",
        condition_mode="full",
    )
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


def _weighted_unit_scores(y, q, units, sample_ids, groups):
    """Aggregate four fixed cuts to one late-weighted score per engine."""
    scores = CQRCalibrator._scores(y, q[:, 0], q[:, 2])
    selected = []
    aggregate_scores = []
    aggregate_groups = []
    for unit in np.unique(units):
        indices = np.flatnonzero(units == unit)
        order = indices[np.argsort(sample_ids[indices])]
        if len(order) != len(WEIGHTS):
            raise ValueError("weighted-late calibration requires four fixed cuts per unit")
        aggregate_scores.append(float(np.dot(WEIGHTS, scores[order])))
        selected.append(order[-1])
        aggregate_groups.append(groups[order[-1]] if groups is not None else "global")
    return (np.asarray(aggregate_scores, dtype=np.float64),
            np.asarray(aggregate_groups, dtype=object),
            np.asarray(selected, dtype=np.int64))


def _aggregate(y, q, units, sample_ids, groups, mode):
    scores = CQRCalibrator._scores(y, q[:, 0], q[:, 2])
    selected = []
    aggregate_scores = []
    aggregate_groups = []
    for unit in np.unique(units):
        indices = np.flatnonzero(units == unit)
        if mode == "block_max":
            local = indices[np.argmax(scores[indices])]
            selected.append(local)
            aggregate_scores.append(float(scores[local]))
            aggregate_groups.append(groups[local] if groups is not None else "global")
        elif mode == "late":
            local = indices[np.argmax(sample_ids[indices])]
            selected.append(local)
            aggregate_scores.append(float(scores[local]))
            aggregate_groups.append(groups[local] if groups is not None else "global")
        elif mode == "weighted_late":
            weighted, weighted_groups, weighted_selected = _weighted_unit_scores(
                y, q, units, sample_ids, groups)
            return weighted, weighted_groups, weighted_selected
        else:
            raise ValueError(f"unknown aggregation mode: {mode}")
    return (np.asarray(aggregate_scores, dtype=np.float64),
            np.asarray(aggregate_groups, dtype=object),
            np.asarray(selected, dtype=np.int64))


def _fit_thresholds(scores, groups, min_group_size, shrink_tau):
    threshold = finite_sample_quantile(scores, 1.0 - ALPHA)
    group_counts = {}
    group_thresholds = {}
    for key in sorted(set(str(value) for value in groups)):
        mask = np.asarray([str(value) == key for value in groups])
        count = int(mask.sum())
        group_counts[key] = count
        if count < min_group_size:
            continue
        local = finite_sample_quantile(scores[mask], 1.0 - ALPHA)
        weight = count / (count + shrink_tau)
        group_thresholds[key] = float(weight * local + (1.0 - weight) * threshold)
    return float(threshold), group_thresholds, group_counts


def _transform(q, groups, threshold, group_thresholds):
    thresholds = np.asarray([
        group_thresholds.get(str(group), threshold) if groups is not None else threshold
        for group in (groups if groups is not None else ["global"] * len(q))
    ])
    return q[:, 0] - thresholds, q[:, 2] + thresholds


def _metrics(y, q, units, sample_ids, groups, threshold, group_thresholds):
    lower, upper = _transform(q, groups, threshold, group_thresholds)
    interval = interval_metrics(y, lower, upper, alpha=ALPHA)
    return {
        **interval,
        "q50_RMSE": float(rmse(y, q[:, 1])),
        "n_samples": int(len(y)),
        "n_engines": int(len(np.unique(units))),
        "latest_only": bool(len(np.unique(sample_ids)) > 1),
    }


def _evaluate(bundle, model, mode, calibration_mode, min_group_size, shrink_tau, device):
    train_loader = _loader(bundle.train)
    val_loader = _loader(bundle.val_endpoint)
    cal_loader = _loader(bundle.calibration_endpoint)
    # The grouper is fitted only on training windows and labels.
    grouper = _fit_grouper(bundle, model, train_loader, device)
    cal_y, cal_q, cal_units, cal_ids, cal_conditions = _predict(model, cal_loader, device)
    val_y, val_q, val_units, val_ids, val_conditions = _predict(model, val_loader, device)
    cal_groups = (grouper.groups(cal_conditions, cal_q[:, 1])
                  if calibration_mode == "hierarchical" else None)
    val_groups = (grouper.groups(val_conditions, val_q[:, 1])
                  if calibration_mode == "hierarchical" else None)
    cal_scores, cal_agg_groups, cal_selected = _aggregate(
        cal_y, cal_q, cal_units, cal_ids, cal_groups, mode)
    threshold, group_thresholds, group_counts = _fit_thresholds(
        cal_scores, cal_agg_groups, min_group_size, shrink_tau)
    active_group_thresholds = {
        key: value for key, value in group_thresholds.items()
        if key != "global" and not np.isclose(value, threshold, rtol=0.0, atol=1e-12)
    }
    latest = _latest_mask(val_units, val_ids)
    all_metrics = _metrics(val_y, val_q, val_units, val_ids, val_groups,
                           threshold, group_thresholds)
    latest_metrics = _metrics(val_y[latest], val_q[latest], val_units[latest],
                              val_ids[latest], val_groups[latest] if val_groups is not None else None,
                              threshold, group_thresholds)
    return {
        "calibration_threshold": threshold,
        "group_thresholds": group_thresholds,
        "group_counts": group_counts,
        "n_group_thresholds": sum(key != "global" for key in group_thresholds),
        "n_active_groups": len(active_group_thresholds),
        "active_group_thresholds": active_group_thresholds,
        "calibration_scores": cal_scores.tolist(),
        "calibration_selected_sample_ids": cal_ids[cal_selected].tolist(),
        "all_cuts": all_metrics,
        "latest_cut": latest_metrics,
    }


def _rank_key(result):
    latest = result["latest_cut"]
    coverage = latest["PICP"]
    interval_score = latest["IntervalScore"]
    eligible = 0.75 <= coverage <= 0.85
    return (0 if eligible else 1, interval_score + 100.0 * abs(coverage - (1.0 - ALPHA)))


def _run_dataset(dataset, data_root, checkpoint, output_root, device):
    dataset_root = Path(output_root) / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    rows = []
    all_results = {}
    for ratio in (0.5, 0.6, 0.7):
        bundle = build_cmapss_datasets(
            data_root, dataset, history=50, rul_cap=125, split_seed=0,
            calibration_ratio=ratio, scaler_mode="grouped",
            validation_protocol="fixed_cuts")
        model = _model(bundle, checkpoint, device)
        for mode in ("block_max", "late", "weighted_late"):
            result = _evaluate(bundle, model, mode, "global", 20, 20.0, device)
            key = f"global__{mode}__ratio_{ratio:g}"
            all_results[key] = result
            latest = result["latest_cut"]
            rows.append({"stage": "protocol", "dataset": dataset, "key": key,
                         "calibration_mode": "global", "aggregation": mode,
                         "ratio": ratio, "min_group_size": 20, "shrink_tau": 20.0,
                         "checkpoint_split_match": ratio == 0.5,
                         "n_active_groups": 0, "threshold": result["calibration_threshold"],
                         "val_latest_PICP": latest["PICP"],
                         "val_latest_MPIW": latest["MPIW"],
                         "val_latest_IntervalScore": latest["IntervalScore"],
                         "val_latest_coverage_gap": latest["coverage_gap"],
                         "val_latest_q50_RMSE": latest["q50_RMSE"]})

    protocol_rows = [row for row in rows if row["stage"] == "protocol"]
    # This checkpoint was selected using the ratio=0.5 validation engines.
    # Ratios 0.6/0.7 reuse some of those engines for calibration, so retain
    # them as exploratory diagnostics but never use them to select the protocol.
    eligible_protocol_rows = [row for row in protocol_rows if row["ratio"] == 0.5]
    best_protocol = min(eligible_protocol_rows,
                        key=lambda row: _rank_key(all_results[row["key"]]))
    best_ratio = float(best_protocol["ratio"])
    best_mode = str(best_protocol["aggregation"])
    bundle = build_cmapss_datasets(
        data_root, dataset, history=50, rul_cap=125, split_seed=0,
        calibration_ratio=best_ratio, scaler_mode="grouped",
        validation_protocol="fixed_cuts")
    model = _model(bundle, checkpoint, device)
    for min_group_size in (8, 10, 15):
        for shrink_tau in (10.0, 20.0):
            result = _evaluate(bundle, model, best_mode, "hierarchical",
                               min_group_size, shrink_tau, device)
            key = (f"hierarchical__{best_mode}__ratio_{best_ratio:g}__"
                   f"min_{min_group_size}__tau_{shrink_tau:g}")
            all_results[key] = result
            latest = result["latest_cut"]
            rows.append({"stage": "hierarchy", "dataset": dataset, "key": key,
                         "calibration_mode": "hierarchical", "aggregation": best_mode,
                         "ratio": best_ratio, "min_group_size": min_group_size,
                         "shrink_tau": shrink_tau,
                         "checkpoint_split_match": best_ratio == 0.5,
                         "n_active_groups": result["n_active_groups"],
                         "threshold": result["calibration_threshold"],
                         "val_latest_PICP": latest["PICP"],
                         "val_latest_MPIW": latest["MPIW"],
                         "val_latest_IntervalScore": latest["IntervalScore"],
                         "val_latest_coverage_gap": latest["coverage_gap"],
                         "val_latest_q50_RMSE": latest["q50_RMSE"]})
    selected_hierarchy = min(
        [row for row in rows if row["stage"] == "hierarchy"],
        key=lambda row: _rank_key(all_results[row["key"]]),
    )
    summary = {
        "dataset": dataset,
        "checkpoint": str(checkpoint),
        "checkpoint_source": "existing_global_cqr_seed_0_epoch_selected_checkpoint",
        "checkpoint_split_ratio": 0.5,
        "ratio_0_6_0_7_status": "exploratory_only_checkpoint_selection_overlap",
        "test_evaluated": False,
        "protocol_candidates": 9,
        "hierarchy_candidates": 6,
        "best_protocol": best_protocol,
        "best_hierarchy": selected_hierarchy,
        "results": all_results,
    }
    (dataset_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (dataset_root / "matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint_root", default=str(
        ROOT / "results_v4/chapter4_main/quantile_cqr/global"))
    parser.add_argument("--output_root", default=str(ROOT / "results_v4/calibration_stage1"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    root = Path(args.checkpoint_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dataset in ("FD002", "FD004"):
        checkpoint = root / dataset / "seed_0" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        summaries.append(_run_dataset(dataset, args.data_root, checkpoint,
                                      output_root, device))
    (output_root / "summary.json").write_text(
        json.dumps({"device": str(device), "test_evaluated": False,
                    "datasets": summaries}, indent=2), encoding="utf-8")
    for summary in summaries:
        print(summary["dataset"], "best protocol:", summary["best_protocol"]["key"],
              "best hierarchy:", summary["best_hierarchy"]["key"], flush=True)


if __name__ == "__main__":
    main()
