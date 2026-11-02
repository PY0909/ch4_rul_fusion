from __future__ import annotations

import argparse
from pathlib import Path

from .data import build_cmapss_datasets
from .probabilistic_runner import fit_probabilistic


def build_parser():
    parser = argparse.ArgumentParser(description="Probabilistic RUL baselines for C-MAPSS")
    parser.add_argument("--dataset", required=True, choices=["FD002", "FD004"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--method", required=True, choices=["hnn", "edl_nig", "ensemble"])
    parser.add_argument("--output_dir", default="results_v4/chapter4_main")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--history", type=int, default=50)
    parser.add_argument("--rul_cap", type=int, default=125)
    parser.add_argument("--calibration_ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--ensemble_members", type=int, default=5)
    parser.add_argument("--evidence_weight", type=float, default=1e-2)
    parser.add_argument("--edl_hidden_dim", type=int, default=128,
                        help="reconstruction choice; paper does not report hidden widths")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    edl = args.method == "edl_nig"
    epochs = args.epochs if args.epochs is not None else (50 if edl else 41)
    batch_size = args.batch_size if args.batch_size is not None else (32 if edl else 100)
    lr = args.lr if args.lr is not None else 1e-3
    weight_decay = args.weight_decay if args.weight_decay is not None else (0.0 if edl else 1e-3)
    patience = args.patience if args.patience is not None else epochs
    dropout = args.dropout if args.dropout is not None else (0.3 if edl else 0.0)
    bundle = build_cmapss_datasets(
        args.data_root, args.dataset, history=args.history, rul_cap=args.rul_cap,
        split_seed=args.split_seed, calibration_ratio=args.calibration_ratio,
        scaler_mode="minmax" if edl else "grouped",
        validation_protocol="fixed_cuts")
    output = Path(args.output_dir) / args.dataset / args.method / f"seed_{args.seed}"
    fit_probabilistic(
        bundle, args.method, output, seed=args.seed, epochs=epochs,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        patience=patience, dropout=dropout,
        ensemble_members=args.ensemble_members, evidence_weight=args.evidence_weight,
        edl_hidden_dim=args.edl_hidden_dim,
        device=args.device, num_workers=args.num_workers,
        config={**vars(args), "epochs": epochs, "batch_size": batch_size,
                "lr": lr, "weight_decay": weight_decay, "patience": patience,
                "dropout": dropout, "scaler_mode": "minmax" if edl else "grouped"})


if __name__ == "__main__":
    main()
