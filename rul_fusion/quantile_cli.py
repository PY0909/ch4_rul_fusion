from __future__ import annotations

import argparse
from pathlib import Path

from .data import build_cmapss_datasets, build_ncmapss_datasets
from .model import MultiScalePhaseGraphRUL
from .quantile_runner import fit_quantile_and_calibrate
from .runner import set_seed
from .uncertainty import A6QuantileRUL


def build_parser():
    parser = argparse.ArgumentParser(description="A6 quantile and conformal RUL prediction")
    parser.add_argument("--dataset", required=True, choices=["FD002", "FD004", "DS03", "DS08a"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default="results_v4/uncertainty")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--calibration_ratio", type=float, default=0.5)
    parser.add_argument("--registered_split", action="store_true",
                        help="use the registered N-CMAPSS validation/calibration split")
    parser.add_argument("--min_calibration_units", type=int, default=1)
    parser.add_argument("--defer_test", action="store_true",
                        help="train/select/calibrate without evaluating official test engines")
    parser.add_argument("--history", type=int, default=None)
    parser.add_argument("--rul_cap", type=int, default=None)
    parser.add_argument("--scaler", choices=["global", "grouped"], default=None)
    parser.add_argument("--condition_mode", choices=["full", "mean", "none"], default="full")
    parser.add_argument("--include_virtual_sensors", action="store_true")
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--num_routers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None,
                        help="early-stopping patience on validation q50 RMSE; defaults to max epochs")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--calibration_mode", choices=["global", "hierarchical"],
                        default="global")
    parser.add_argument("--min_group_size", type=int, default=20)
    parser.add_argument("--shrink_tau", type=float, default=20.0)
    parser.add_argument("--corruption", choices=["missing", "noise", "bias", "drift"],
                        default=None)
    parser.add_argument("--corruption_severity", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    is_fd = args.dataset.startswith("FD")
    history = args.history or (50 if is_fd else 16)
    rul_cap = (125 if is_fd else 0) if args.rul_cap is None else args.rul_cap
    scaler = ("grouped" if is_fd else "global") if args.scaler is None else args.scaler
    d_model = (64 if is_fd else 32) if args.d_model is None else args.d_model
    num_heads = (4 if is_fd else 2) if args.num_heads is None else args.num_heads
    num_routers = (4 if is_fd else 2) if args.num_routers is None else args.num_routers
    epochs = (41 if is_fd else 300) if args.epochs is None else args.epochs
    batch_size = (100 if is_fd else 32) if args.batch_size is None else args.batch_size
    weight_decay = (0.0 if is_fd else 1e-3) if args.weight_decay is None else args.weight_decay
    patience = epochs if args.patience is None else args.patience
    set_seed(args.seed)

    if is_fd:
        bundle = build_cmapss_datasets(
            args.data_root, args.dataset, history=history, rul_cap=rul_cap,
            split_seed=args.split_seed, calibration_ratio=args.calibration_ratio, scaler_mode=scaler,
            validation_protocol="fixed_cuts")
    else:
        bundle = build_ncmapss_datasets(
            args.data_root, args.dataset, history=history, split_seed=args.split_seed,
            calibration_ratio=args.calibration_ratio, include_virtual=args.include_virtual_sensors,
            registered_split=args.registered_split,
            min_calibration_units=args.min_calibration_units,
            validation_protocol="fixed_cuts")

    backbone = MultiScalePhaseGraphRUL(
        sensor_dim=bundle.sensor_dim, condition_dim=bundle.condition_dim,
        d_model=d_model, num_heads=num_heads, num_routers=num_routers,
        dropout=args.dropout, encoder="patch_cnn", variant="A6",
        condition_mode=args.condition_mode)
    model = A6QuantileRUL(backbone, dropout=args.dropout)
    run_dir = Path(args.output_dir) / args.calibration_mode / args.dataset / f"seed_{args.seed}"
    config = {**vars(args), "resolved_history": history, "resolved_rul_cap": rul_cap,
              "resolved_scaler": scaler, "resolved_d_model": d_model,
              "resolved_num_heads": num_heads, "resolved_num_routers": num_routers,
              "resolved_epochs": epochs, "resolved_batch_size": batch_size,
              "resolved_weight_decay": weight_decay, "resolved_patience": patience,
              "resolved_early_stopping_metric": "val_q50_rmse"}
    fit_quantile_and_calibrate(
        bundle, model, run_dir, epochs=epochs, batch_size=batch_size, lr=args.lr,
        weight_decay=weight_decay, patience=patience, alpha=args.alpha,
        calibration_mode=args.calibration_mode, min_group_size=args.min_group_size,
        shrink_tau=args.shrink_tau, seed=args.seed, device=args.device,
        num_workers=args.num_workers, corruption=args.corruption,
        corruption_severity=args.corruption_severity, config=config,
        evaluate_test=not args.defer_test)


if __name__ == "__main__":
    main()
