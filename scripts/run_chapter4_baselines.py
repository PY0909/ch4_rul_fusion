"""Print or execute FD002/FD004 main-model baselines, excluding HCQ-CQR."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_DATA_ROOT = "/home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset"


def commands(dataset, seed, data_root, output_root):
    root = Path(__file__).parents[1]
    common = ["--dataset", dataset, "--data_root", data_root, "--history", "50",
              "--rul_cap", "125", "--split_seed", "0", "--seed", str(seed)]
    base = [sys.executable, "-m", "rul_fusion.cli", *common,
            "--validation_protocol", "fixed_cuts",
            "--calibration_ratio", "0.5", "--model", "fusion", "--scaler", "grouped",
            "--variant", "A6", "--encoder", "patch_cnn", "--d_model", "64",
            "--num_heads", "4", "--num_routers", "4", "--epochs", "41",
            "--batch_size", "100", "--lr", "0.001", "--weight_decay", "0",
            "--patience", "41", "--optimizer", "adam", "--no_scheduler",
            "--no_grad_clip", "--output_dir", str(Path(output_root) / "A6-point")]
    quantile = [sys.executable, "-m", "rul_fusion.quantile_cli", *common,
                "--output_dir", str(Path(output_root) / "quantile_cqr"),
                "--calibration_mode", "global", "--alpha", "0.2", "--epochs", "41",
                "--batch_size", "100", "--device", "auto"]
    prob = []
    for method in ("hnn", "edl_nig", "ensemble"):
        args = [sys.executable, "-m", "rul_fusion.probabilistic_cli", *common,
                "--method", method, "--output_dir", str(output_root), "--device", "auto"]
        if method == "edl_nig":
            args.extend(["--epochs", "50", "--batch_size", "32", "--lr", "0.001",
                         "--weight_decay", "0", "--patience", "50", "--dropout", "0.3"])
        else:
            args.extend(["--epochs", "41", "--batch_size", "100", "--lr", "0.001",
                         "--weight_decay", "0", "--patience", "41"])
        prob.append((args, root))
    return [(base, root), (quantile, root), *prob]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default="results_v4/chapter4_main")
    parser.add_argument("--dataset", choices=["FD002", "FD004"], action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    datasets = args.dataset or ["FD002", "FD004"]
    seeds = args.seed or [0, 1, 2]
    for dataset in datasets:
        for seed in seeds:
            for argv, root in commands(dataset, seed, args.data_root, args.output_dir):
                print(" ".join(shlex.quote(item) for item in argv), flush=True)
                if args.run:
                    subprocess.run(argv, cwd=root, check=True)


if __name__ == "__main__":
    main()
