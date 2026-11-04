"""Print or run the stage-2 external-baseline matrix (B3-B6 on FD002/FD004).

Methods B1 (hnn) and B2 (edl_nig) already exist under
results_v4/chapter4_main; this matrix trains the four new external baselines
with their registered paper defaults (see references/probabilistic_baselines.md)
and evaluates the official test once per run. Runs are fault-isolated: one
failed training is recorded in failures.json and the matrix continues, and
completed runs (metrics.json present) are skipped so the matrix is resumable.
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "results_v4/external_baselines"
METHODS = ("lstm_prob", "tcn_transformer_mc", "msauq", "bcpt")


def commands(data_root, python=sys.executable, datasets=("FD002", "FD004")):
    result = []
    for dataset in datasets:
        for method in METHODS:
            result.append([
                python, "-m", "rul_fusion.probabilistic_cli",
                "--dataset", dataset, "--method", method,
                "--data_root", str(data_root),
                "--output_dir", str(OUTPUT),
                "--seed", "0", "--split_seed", "0",
                "--calibration_ratio", "0.5"])
    return result


def write_summary(path=OUTPUT / "summary.csv"):
    rows = []
    for metrics_path in sorted(OUTPUT.glob("*/**/metrics.json")):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({"path": str(metrics_path.relative_to(ROOT)), **data})
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--run", action="store_true",
                        help="execute the matrix; default only prints commands")
    parser.add_argument("--datasets", nargs="+", default=["FD002", "FD004"])
    parser.add_argument("--summary_only", action="store_true",
                        help="only (re)write summary.csv from existing metrics")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.summary_only:
        summary, rows = write_summary()
        print(f"summary: {summary} ({len(rows)} rows)")
        return
    matrix = commands(args.data_root, args.python, tuple(args.datasets))
    for command in matrix:
        print(" ".join(command))
    if not args.run:
        return
    failures = []
    for command in matrix:
        dataset = command[command.index("--dataset") + 1]
        method = command[command.index("--method") + 1]
        run_dir = OUTPUT / dataset / method / "seed_0"
        if (run_dir / "metrics.json").exists():
            print(f"skip completed: {run_dir}", flush=True)
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            # Interrupted run without a final metrics.json: clear partial
            # artifacts so the fresh rerun does not hit the no-overwrite guard.
            print(f"clearing interrupted run: {run_dir}", flush=True)
            shutil.rmtree(run_dir)
        try:
            subprocess.run(command + ["--device", args.device], cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            failures.append({"dataset": dataset, "method": method,
                             "returncode": int(error.returncode),
                             "at": datetime.now().isoformat(timespec="seconds")})
            print(f"FAILED (matrix continues): {dataset}/{method}", flush=True)
    (OUTPUT / "failures.json").write_text(
        json.dumps({"failures": failures,
                    "generated": datetime.now().isoformat(timespec="seconds")},
                   indent=2), encoding="utf-8")
    summary, rows = write_summary()
    print(f"summary: {summary} ({len(rows)} rows, {len(failures)} failures)")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
