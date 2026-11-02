"""Launch the registered fourth-chapter quantile matrix.

The script only constructs commands by default. Pass --run to execute them
serially, keeping each run's output under results_v4/uncertainty.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DATA_ROOT = "/home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset"


def command(dataset, method, seed, output_dir, data_root):
    root = Path(__file__).parents[1]
    args = [sys.executable, "-m", "rul_fusion.quantile_cli",
            "--dataset", dataset, "--data_root", data_root,
            "--output_dir", str(output_dir), "--seed", str(seed),
            "--calibration_mode", method]
    return args, root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--output_dir", default="results_v4/uncertainty")
    parser.add_argument("--dataset", choices=["FD002", "FD004", "DS03", "DS08a"],
                        action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--corruption", choices=["missing", "noise", "bias", "drift"])
    parser.add_argument("--corruption_severity", type=float)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.corruption and args.corruption_severity is None:
        parser.error("--corruption_severity is required with --corruption")
    datasets = args.dataset or ["FD002", "FD004"]
    seeds = args.seed or [0, 1, 2]
    methods = ["global", "hierarchical"]
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                argv, root = command(dataset, method, seed, args.output_dir, args.data_root)
                if args.corruption:
                    argv.extend(["--corruption", args.corruption,
                                 "--corruption_severity", str(args.corruption_severity)])
                print(" ".join(shlex.quote(value) for value in argv), flush=True)
                if args.run:
                    subprocess.run(argv, cwd=root, check=True)


if __name__ == "__main__":
    main()
