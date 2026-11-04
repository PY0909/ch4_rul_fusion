"""Registered N-CMAPSS DS03/DS08a probabilistic baseline runner."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rul_fusion.data import build_ncmapss_datasets  # noqa: E402
from rul_fusion.probabilistic_runner import fit_probabilistic  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("DS03", "DS08a"))
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--method", required=True, choices=("hnn", "edl_nig", "ensemble"))
    parser.add_argument("--output_dir", default=str(ROOT / "results_v4/ncmapss_probabilistic"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--history", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--evidence_weight", type=float, default=1e-2)
    parser.add_argument("--ensemble_members", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--defer_test", action="store_true", default=True,
                        help="train/validate only; this is the default")
    parser.add_argument("--eval_test", action="store_true",
                        help="evaluate the frozen checkpoint on official test endpoints")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    edl = args.method == "edl_nig"
    dropout = args.dropout if args.dropout is not None else (0.3 if edl else 0.0)
    weight_decay = args.weight_decay if args.weight_decay is not None else (0.0 if edl else 1e-3)
    bundle = build_ncmapss_datasets(
        args.data_root, args.dataset, history=args.history, split_seed=args.split_seed,
        val_ratio=0.55, calibration_ratio=0.34, validation_protocol="fixed_cuts",
        registered_split=True, min_calibration_units=2)
    out = Path(args.output_dir) / args.dataset / args.method / f"seed_{args.seed}"
    explicit_test = bool(args.eval_test)
    fit_probabilistic(
        bundle, args.method, out, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, weight_decay=weight_decay,
        patience=args.patience, dropout=dropout, evidence_weight=args.evidence_weight,
        ensemble_members=args.ensemble_members, device=args.device,
        test_dataset=bundle.test_endpoint, defer_test=not explicit_test,
        config={**vars(args), "registered_split": True, "val_ratio": 0.55,
                "calibration_ratio": 0.34, "min_calibration_units": 2,
                "resolved_dropout": dropout, "resolved_weight_decay": weight_decay,
                "edl_features": "all physical sensors" if edl else None,
                "NASA_Score": "not applicable for N-CMAPSS"})
    if explicit_test:
        touch = ROOT / "results_v4/chapter4_main_logs/test_touches.log"
        touch.parent.mkdir(parents=True, exist_ok=True)
        with touch.open("a", encoding="utf-8") as handle:
            from datetime import datetime
            handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] ncmapss_probabilistic: official test evaluated for {args.dataset} {args.method} seed {args.seed}\n")


if __name__ == "__main__":
    main()
