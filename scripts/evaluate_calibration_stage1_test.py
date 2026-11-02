"""Evaluate the frozen stage-1 calibration matrix once on official test engines.

Calibration thresholds are fitted only from each candidate's calibration
engines. The official test split is read only for final metric computation.
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
from rul_fusion.metrics import regression_metrics  # noqa: E402
from rul_fusion.model import MultiScalePhaseGraphRUL  # noqa: E402
from rul_fusion.quantile_runner import _fit_grouper, _predict  # noqa: E402
from rul_fusion.uncertainty import A6QuantileRUL, CQRCalibrator, interval_metrics  # noqa: E402
from scripts.run_calibration_stage1 import (  # noqa: E402
    _aggregate, _fit_thresholds, _transform,
)


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


def _test_metrics(y, q, groups, threshold, group_thresholds):
    lower, upper = _transform(q, groups, threshold, group_thresholds)
    point = regression_metrics(y, q[:, 1])
    interval = interval_metrics(y, lower, upper, alpha=0.2)
    return {**point, **interval, "n_test_engines": int(len(y))}


def _evaluate_one(bundle, model, test, test_groups, mode, hierarchical,
                  min_group_size, shrink_tau, device):
    grouper = _fit_grouper(bundle, model, _loader(bundle.train), device)
    cal = _predict(model, _loader(bundle.calibration_endpoint), device)
    cal_y, cal_q, cal_units, cal_ids, cal_conditions = cal
    cal_groups = grouper.groups(cal_conditions, cal_q[:, 1]) if hierarchical else None
    scores, reduced_groups, _ = _aggregate(
        cal_y, cal_q, cal_units, cal_ids, cal_groups, mode)
    threshold, group_thresholds, group_counts = _fit_thresholds(
        scores, reduced_groups, min_group_size, shrink_tau)
    y, q, _, _, conditions = test
    groups = grouper.groups(conditions, q[:, 1]) if hierarchical else None
    metrics = _test_metrics(y, q, groups, threshold, group_thresholds)
    active = {key: value for key, value in group_thresholds.items()
              if key != "global" and not np.isclose(value, threshold, rtol=0.0, atol=1e-12)}
    return {**metrics, "calibration_threshold": threshold,
            "group_thresholds": group_thresholds, "group_counts": group_counts,
            "n_active_groups": len(active), "active_group_thresholds": active}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint_root", default=str(
        ROOT / "results_v4/chapter4_main/quantile_cqr/global"))
    parser.add_argument("--stage1_root", default=str(ROOT / "results_v4/calibration_stage1"))
    parser.add_argument("--output_root", default=str(ROOT / "results_v4/calibration_stage1_test"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dataset in ("FD002", "FD004"):
        ds_out = output_root / dataset
        ds_out.mkdir(parents=True, exist_ok=True)
        checkpoint = Path(args.checkpoint_root) / dataset / "seed_0" / "best.pt"
        stage1 = json.loads((Path(args.stage1_root) / dataset / "summary.json").read_text())
        selected_mode = stage1["best_protocol"]["aggregation"]
        rows = []
        # The train split and its fitted scaler are constant across ratios.
        for ratio in (0.5, 0.6, 0.7):
            bundle = build_cmapss_datasets(
                args.data_root, dataset, history=50, rul_cap=125, split_seed=0,
                calibration_ratio=ratio, scaler_mode="grouped",
                validation_protocol="fixed_cuts")
            model = _load_model(bundle, checkpoint, device)
            test = _predict(model, _loader(bundle.test), device)
            for mode in ("block_max", "late", "weighted_late"):
                result = _evaluate_one(bundle, model, test, None, mode, False,
                                       20, 20.0, device)
                rows.append({"stage": "protocol", "dataset": dataset,
                             "aggregation": mode, "ratio": ratio,
                             "checkpoint_split_match": ratio == 0.5,
                             "status": "matched" if ratio == 0.5 else "exploratory_split_overlap",
                             **result})
            if ratio == 0.5:
                for min_group_size in (8, 10, 15):
                    for shrink_tau in (10.0, 20.0):
                        result = _evaluate_one(bundle, model, test, None, selected_mode,
                                               True, min_group_size, shrink_tau, device)
                        rows.append({"stage": "hierarchy", "dataset": dataset,
                                     "aggregation": selected_mode, "ratio": ratio,
                                     "checkpoint_split_match": True, "status": "matched",
                                     "min_group_size": min_group_size,
                                     "shrink_tau": shrink_tau, **result})
        fields = sorted({key for row in rows for key in row})
        with (ds_out / "test_matrix.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        summary = {"dataset": dataset, "checkpoint": str(checkpoint),
                   "official_test_evaluated": True,
                   "warning": "ratio 0.6/0.7 overlap checkpoint-selection validation engines; exploratory only",
                   "best_validation_protocol_mode": selected_mode,
                   "results": rows}
        (ds_out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
        print(dataset, "test candidates:", len(rows), flush=True)
    (output_root / "summary.json").write_text(json.dumps(
        {"device": str(device), "official_test_evaluated": True,
         "datasets": summaries}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
