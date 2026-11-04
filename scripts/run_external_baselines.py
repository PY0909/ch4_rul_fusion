"""Print or run the stage-2 external-baseline matrix (B3-B6 on FD002/FD004).

Methods B1 (hnn) and B2 (edl_nig) already exist under
results_v4/chapter4_main; this matrix trains the four new external baselines
with their registered paper defaults (see references/probabilistic_baselines.md)
and evaluates the official test once per run.
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "results_v4/external_baselines"
METHODS = ("lstm_prob", "tcn_transformer_mc", "msauq", "bcpt")


def commands(data_root, python=sys.executable):
    result = []
    for dataset in ("FD002", "FD004"):
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
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    matrix = commands(args.data_root, args.python)
    for command in matrix:
        print(" ".join(command))
    if not args.run:
        return
    for command in matrix:
        subprocess.run(command + ["--device", args.device], cwd=ROOT, check=True)
    summary, rows = write_summary()
    print(f"summary: {summary} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
