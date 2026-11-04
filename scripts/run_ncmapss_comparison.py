"""Print or run the registered N-CMAPSS baseline comparison matrix."""
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
COMPARISON = ROOT / "results_v4/ncmapss_comparison"


def commands(data_root, python=sys.executable):
    scripts = ROOT / "scripts"
    common = ["--data_root", str(data_root), "--registered_split"]
    result = []
    for ds in ("DS03", "DS08a"):
        for method in ("A6", "fc_stgnn"):
            command = [python, str(scripts / "run_ncmapss.py"), "--dataset", ds,
                       "--model", method, "--output_dir", str(COMPARISON / "point" / method),
                       "--seed", "0", *common]
            if method == "A6":
                command.extend(["--d_model", "32", "--num_heads", "2", "--num_routers", "2",
                                "--dropout", "0.1", "--weight_decay", "1e-3"])
            result.append(command)
        result.append([python, str(scripts / "run_ncmapss_cqr.py"), "--dataset", ds,
                       "--seed", "0", "--output_dir", str(COMPARISON / "cqr"),
                       "--data_root", str(data_root)])
        for method in ("hnn", "edl_nig"):
            result.append([python, str(scripts / "run_ncmapss_probabilistic.py"),
                           "--dataset", ds, "--method", method, "--seed", "0",
                           "--output_dir", str(COMPARISON / "probabilistic"),
                           "--eval_test", "--data_root", str(data_root)])
    return result


def _summary_rows(root=COMPARISON):
    rows = []
    for path in root.glob("**/metrics.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"path": str(path.relative_to(ROOT)), "source": "metrics.json", **data})
        except (OSError, json.JSONDecodeError):
            continue
    for path in root.glob("**/cqr_variants_test_summary.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            base = {"path": str(path.relative_to(ROOT)), "source": "cqr_variants_test_summary.json",
                    "dataset": data.get("dataset"), "seed": data.get("seed"),
                    "official_test_evaluated": data.get("official_test_evaluated")}
            point = data.get("q50_point", {})
            for variant in data.get("rows", []):
                rows.append({**base, "method": variant.get("variant"),
                             "endpoint": variant.get("endpoint"), **point, **variant})
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def write_summary(path=COMPARISON / "summary.csv"):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _summary_rows()
    fields = sorted({key for row in rows for key in row}) or ["path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--run", action="store_true", help="execute commands; default only prints")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    matrix = commands(args.data_root, args.python)
    for command in matrix:
        print(" ".join(command))
    if args.run:
        for command in matrix:
            subprocess.run(command, cwd=ROOT, check=True)
    summary, _ = write_summary()
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
