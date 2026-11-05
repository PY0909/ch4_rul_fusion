"""Plan or run dev-only A6-Quantile tuning on the registered N-CMAPSS split.

The frozen A6 architecture, alpha, density-ratio rule and test endpoint are
unchanged. Each trial changes one training hyperparameter from P0 and passes
--defer_test, so this entry point never evaluates official test labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results_v4/ncmapss_a6_dev_tuning"
BASELINE_ROOT = ROOT / "results_v4/ncmapss_comparison/cqr"
DATASETS = ("DS03", "DS08a")
BASE = {"lr": "0.001", "weight_decay": "0.001", "dropout": "0.1"}
TRIALS = (
    ("lr3e4", {**BASE, "lr": "0.0003"}),
    ("wd3e3", {**BASE, "weight_decay": "0.003"}),
    ("dropout02", {**BASE, "dropout": "0.2"}),
)


def commands(data_root, python=sys.executable, output_root=OUTPUT_ROOT):
    output_root = Path(output_root)
    trials = []
    for dataset in DATASETS:
        for name, settings in TRIALS:
            run_dir = output_root / name / dataset / "seed_0"
            command = [
                python, str(ROOT / "scripts/run_ncmapss_cqr.py"),
                "--dataset", dataset, "--data_root", str(data_root),
                "--output_dir", str(output_root / name), "--seed", "0",
                "--history", "16", "--d_model", "32",
                "--num_heads", "2", "--num_routers", "2",
                "--epochs", "300", "--patience", "300",
                "--alpha", "0.2", "--device", "cuda", "--defer_test",
                "--lr", settings["lr"],
                "--weight_decay", settings["weight_decay"],
                "--dropout", settings["dropout"],
            ]
            trials.append((dataset, name, command, run_dir))
    return trials


def summary_rows(output_root=OUTPUT_ROOT, baseline_root=BASELINE_ROOT):
    output_root = Path(output_root)
    baseline_root = Path(baseline_root)
    rows = []
    for dataset in DATASETS:
        for name, settings in (("p0", BASE), *TRIALS):
            run_dir = ((baseline_root if name == "p0" else output_root / name)
                       / dataset / "seed_0")
            path = run_dir / "calibration_only.json"
            row = {"dataset": dataset, "config": name, "seed": 0,
                   **settings, "status": "pending", "path": str(path)}
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                if result.get("test_evaluated") is not False:
                    raise ValueError(f"expected dev-only result: {path}")
                row.update({"status": "complete",
                            "best_epoch": result["best_epoch"],
                            "val_q50_RMSE": result["val_q50_RMSE"],
                            "calibration_threshold": result["calibration_threshold"]})
            rows.append(row)
    return rows


def write_summary(output_root=OUTPUT_ROOT, baseline_root=BASELINE_ROOT):
    output_root = Path(output_root)
    rows = summary_rows(output_root, baseline_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "dev_summary.csv"
    fields = ("dataset", "config", "seed", "lr", "weight_decay", "dropout",
              "status", "best_epoch", "val_q50_RMSE",
              "calibration_threshold", "path")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def cuda_available():
    import torch
    return bool(torch.cuda.is_available())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root")
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true", help="train all pending trials")
    action.add_argument("--summarize", action="store_true", help="write dev-only summary")
    args = parser.parse_args(argv)
    if args.summarize:
        print(write_summary(args.output_root))
        return
    if not args.data_root:
        parser.error("--data_root is required for listing or running trials")
    if args.run and not cuda_available():
        raise SystemExit("CUDA is unavailable; switch the server to GPU mode before --run")

    trials = commands(args.data_root, args.python, args.output_root)
    for dataset, name, command, run_dir in trials:
        print(f"[{dataset} {name}] {shlex.join(command)}", flush=True)
        if not args.run:
            continue
        if (run_dir / "calibration_only.json").exists():
            print(f"already complete: {run_dir}", flush=True)
            continue
        if run_dir.exists():
            raise SystemExit(f"partial trial requires inspection: {run_dir}")
        subprocess.run(command, cwd=ROOT, check=True)
    if args.run:
        print(write_summary(args.output_root))


if __name__ == "__main__":
    main()
