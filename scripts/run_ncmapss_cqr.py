"""N-CMAPSS DS03/DS08a A6-Quantile + global/window-weighted CQR external check.

Pipeline (registered external check, M5.1 of docs/chapter4-experiment-matrix.md):
1. Train the locked A6 quantile model (d32 / 2 heads / 2 routers, history 16,
   batch 32, Adam lr 1e-3 wd 1e-3, 300 epochs) on the registered N-CMAPSS dev
   split: split_seed=0, val_ratio=0.55, calibration_ratio=0.34 -> 3 validation
   units + 2 calibration units (4 train units on the 9-unit dev pool), guarded
   by min_calibration_units=2. Checkpoint selection uses validation endpoint
   q50 RMSE only; the official test set is never read during training.
2. Fit a global CQR threshold on the calibration units (block-max aggregation,
   same conservative per-unit reduction as the FD pipeline).
3. Fit the window-weighted conformal threshold with the density-ratio weights
   of run_weighted_cqr.py, computed on observable window covariates only and
   with a parameterized normalization (the FD cap of 125 never applies here;
   the DS default is the raw flight-RUL scale, i.e. 1.0).

`--defer_test` (default) stops after calibration-only artifacts and writes no
test touch. Without it, the frozen checkpoint/calibrators are evaluated once on
the official test units and the touch is appended to
results_v4/chapter4_main_logs/test_touches.log. NASA score is a C-MAPSS
convention and is never reported for N-CMAPSS.
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

from rul_fusion.data import build_ncmapss_datasets  # noqa: E402
from rul_fusion.model import MultiScalePhaseGraphRUL  # noqa: E402
from rul_fusion.quantile_runner import (  # noqa: E402
    _aggregate_calibration_blocks, _predict, fit_quantile_and_calibrate)
from rul_fusion.uncertainty import (  # noqa: E402
    A6QuantileRUL, CQRCalibrator, interval_metrics, pinball_loss)
from scripts.run_weighted_cqr import (  # noqa: E402
    _density_ratio, _features, _loader, _rank_key)

ALPHA = 0.2
# Registered external-check split (9 dev units -> 4 train / 3 val / 2 cal).
REGISTERED_VAL_RATIO = 0.55
REGISTERED_CALIBRATION_RATIO = 0.34
MIN_CALIBRATION_UNITS = 2


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("DS03", "DS08a"))
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default=str(ROOT / "results_v4/ncmapss_cqr"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--defer_test", action="store_true",
                        help="stop after calibration-only artifacts; official test is not read")
    parser.add_argument("--history", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_routers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--normalization", type=float, default=1.0,
                        help="feature scale for the density-ratio covariates "
                             "(flight RUL units; the FD cap of 125 is never assumed)")
    parser.add_argument("--device", default="auto")
    return parser


def _touch_log_path() -> Path:
    return ROOT / "results_v4/chapter4_main_logs/test_touches.log"


def _log_test_touch(message: str) -> None:
    path = _touch_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def _load_quantile_model(bundle, checkpoint: Path, device, dims) -> A6QuantileRUL:
    backbone = MultiScalePhaseGraphRUL(
        sensor_dim=bundle.sensor_dim, condition_dim=bundle.condition_dim,
        d_model=dims["d_model"], num_heads=dims["num_heads"],
        num_routers=dims["num_routers"], dropout=dims["dropout"],
        encoder="patch_cnn", variant="A6", condition_mode="full")
    model = A6QuantileRUL(backbone, dropout=dims["dropout"]).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def _threshold_variants(cal_y, cal_q, cal_units, cal_ids, cal_conditions,
                        target_feats, alpha, normalization=1.0):
    """Return the global and window-weighted CQR thresholds."""
    cal_y_block, cal_q_block, cal_units_block, cal_ids_block, _, _, block_scores = \
        _aggregate_calibration_blocks(cal_y, cal_q, cal_units, cal_ids, cal_conditions)
    calibrator = CQRCalibrator(alpha=alpha).fit_scores(block_scores)

    window_scores = CQRCalibrator._scores(cal_y, cal_q[:, 0], cal_q[:, 2])
    cal_feats = _features(cal_q, cal_conditions, normalization=normalization)
    weights = _density_ratio(cal_feats, target_feats)
    weighted = {}
    for name, scores, score_weights in (
            ("window_weighted", window_scores, weights),
            ("block_max_weighted", block_scores, np.asarray([
                weights[cal_units == unit].mean() for unit in np.unique(cal_units)]))):
        from rul_fusion.uncertainty import weighted_conformal_threshold
        weighted[name] = weighted_conformal_threshold(scores, score_weights, alpha)
    return calibrator, weighted, cal_feats


def _apply(y, q, threshold, alpha):
    lower, upper = q[:, 0] - threshold, q[:, 2] + threshold
    return interval_metrics(y, lower, upper, alpha=alpha)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if abs(args.alpha - ALPHA) > 1e-12:
        raise SystemExit("alpha must stay 0.2 for the frozen q0.1/q0.5/q0.9 head")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)

    bundle = build_ncmapss_datasets(
        args.data_root, args.dataset, history=args.history,
        split_seed=args.split_seed, val_ratio=REGISTERED_VAL_RATIO,
        calibration_ratio=REGISTERED_CALIBRATION_RATIO,
        validation_protocol="fixed_cuts", registered_split=True,
        min_calibration_units=MIN_CALIBRATION_UNITS)
    dims = {"d_model": args.d_model, "num_heads": args.num_heads,
            "num_routers": args.num_routers, "dropout": args.dropout}
    backbone = MultiScalePhaseGraphRUL(
        sensor_dim=bundle.sensor_dim, condition_dim=bundle.condition_dim,
        d_model=args.d_model, num_heads=args.num_heads,
        num_routers=args.num_routers, dropout=args.dropout,
        encoder="patch_cnn", variant="A6", condition_mode="full")
    model = A6QuantileRUL(backbone, dropout=args.dropout)
    run_dir = Path(args.output_dir) / args.dataset / f"seed_{args.seed}"
    config = {**vars(args), "registered_split": True,
              "resolved_val_ratio": REGISTERED_VAL_RATIO,
              "resolved_calibration_ratio": REGISTERED_CALIBRATION_RATIO,
              "resolved_min_calibration_units": MIN_CALIBRATION_UNITS,
              "resolved_architecture": "locked A6 d32/2-head/2-router",
              "runner": "scripts/run_ncmapss_cqr.py",
              "density_ratio_normalization": args.normalization}
    train_result = fit_quantile_and_calibrate(
        bundle, model, run_dir, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, patience=args.patience,
        alpha=args.alpha, calibration_mode="global", seed=args.seed,
        device=args.device, config=config, evaluate_test=False)
    # fit_quantile_and_calibrate already wrote the global CQR calibrator and
    # calibration-only artifacts with test_evaluated=false. Reload the frozen
    # checkpoint for the weighted variants so nothing downstream can reselect.
    frozen = _load_quantile_model(bundle, run_dir / "best.pt", device, dims)

    cal_y, cal_q, cal_units, cal_ids, cal_conditions = _predict(
        frozen, _loader(bundle.calibration_endpoint), device)
    val_y, val_q, val_units, val_ids, val_conditions = _predict(
        frozen, _loader(bundle.val_endpoint), device)
    val_feats = _features(val_q, val_conditions, normalization=args.normalization)

    calibrator, weighted_thresholds, cal_feats = _threshold_variants(
        cal_y, cal_q, cal_units, cal_ids, cal_conditions, val_feats, args.alpha,
        normalization=args.normalization)
    threshold = calibrator.global_threshold
    rows = [{"dataset": args.dataset, "variant": "global_block_max_cqr",
             "endpoint": "val", "threshold": float(threshold),
             "n_cal_units": int(len(np.unique(cal_units))),
             **_apply(val_y, val_q, threshold, args.alpha)}]
    for name, weighted_threshold in weighted_thresholds.items():
        rows.append({"dataset": args.dataset, "variant": name,
                     "endpoint": "val", "threshold": float(weighted_threshold),
                     "n_cal_units": int(len(np.unique(cal_units))),
                     **_apply(val_y, val_q, weighted_threshold, args.alpha)})
    rows.sort(key=_rank_key)
    validation_path = run_dir / "cqr_variants_validation.csv"
    with validation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "cqr_variants_calibrators.json").write_text(json.dumps({
        "alpha": args.alpha,
        "global_threshold": calibrator.global_threshold,
        "weighted_thresholds": weighted_thresholds,
        "density_ratio_normalization": args.normalization,
        "density_ratio_features": ["q50/norm", "width/norm", "condition_mean"],
        "calibration_units": bundle.metadata["calibration_units"],
        "val_units": bundle.metadata["val_units"],
        "fitted_against": "validation endpoint covariates (no labels used for weights)",
    }, indent=2), encoding="utf-8")
    for row in rows:
        print("[{}] {:<22s} thr={:8.3f} PICP={:.3f} MPIW={:9.3f} IS={:9.3f}".format(
            args.dataset, row["variant"], row["threshold"], row["PICP"],
            row["MPIW"], row["IntervalScore"]), flush=True)
    print(f"selected: {rows[0]['variant']} (validation endpoint)", flush=True)

    if args.defer_test:
        return

    test_dataset = bundle.test_endpoint if bundle.test_endpoint is not None else bundle.test
    test_y, test_q, test_units, test_ids, test_conditions = _predict(
        frozen, _loader(test_dataset), device)
    test_feats = _features(test_q, test_conditions, normalization=args.normalization)
    calibrator, weighted_thresholds, _ = _threshold_variants(
        cal_y, cal_q, cal_units, cal_ids, cal_conditions, test_feats, args.alpha,
        normalization=args.normalization)
    test_rows = [{"dataset": args.dataset, "variant": "global_block_max_cqr",
                  "endpoint": "test", "threshold": float(calibrator.global_threshold),
                  "n_cal_units": int(len(np.unique(cal_units))),
                  **_apply(test_y, test_q, calibrator.global_threshold, args.alpha)}]
    for name, weighted_threshold in weighted_thresholds.items():
        test_rows.append({"dataset": args.dataset, "variant": name,
                          "endpoint": "test", "threshold": float(weighted_threshold),
                          "n_cal_units": int(len(np.unique(cal_units))),
                          **_apply(test_y, test_q, weighted_threshold, args.alpha)})
    test_rows.sort(key=_rank_key)
    with (run_dir / "cqr_variants_test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(test_rows)
    point = {"MAE": float(np.mean(np.abs(test_q[:, 1] - test_y))),
             "RMSE": float(np.sqrt(np.mean((test_q[:, 1] - test_y) ** 2)))}
    summary = {"dataset": args.dataset, "seed": args.seed, "endpoint": "test",
               "official_test_evaluated": True,
               "test_protocol": bundle.metadata.get("test_endpoint_protocol"),
               "n_test_units": int(len(np.unique(test_units))),
               "q50_point": point, "rows": test_rows,
               "generated": datetime.now().isoformat(timespec="seconds")}
    (run_dir / "cqr_variants_test_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    for row in test_rows:
        print("[test] {:<22s} thr={:8.3f} PICP={:.3f} MPIW={:9.3f} IS={:9.3f}".format(
            row["variant"], row["threshold"], row["PICP"], row["MPIW"],
            row["IntervalScore"]), flush=True)
    _log_test_touch(
        f"[{summary['generated']}] ncmapss_cqr: official test evaluated once for "
        f"{args.dataset} seed {args.seed} (global + window-weighted CQR variants, "
        f"registered split, frozen checkpoint)")


if __name__ == "__main__":
    main()
