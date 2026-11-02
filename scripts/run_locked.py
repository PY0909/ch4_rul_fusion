"""Launch the locked A6 configuration for a dataset.

Reads locked_configs.json (the single source of truth for the frozen
per-dataset configurations) and invokes the matching runner. `--seed` and
`--output_dir` are the only overrides; changing anything else means changing
the lock, which requires editing locked_configs.json and its provenance.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
DEFAULT_DATA_ROOT = "/home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset"
DEFAULT_OUTPUT_ROOT = str(ROOT / "results_locked")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["FD002", "FD004", "DS03", "DS08a"])
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    lock = json.loads((ROOT / "locked_configs.json").read_text())
    entry = lock["datasets"][args.dataset]
    output_dir = args.output_dir or str(Path(DEFAULT_OUTPUT_ROOT) / args.dataset)

    if entry["runner"] == "rul_fusion.cli":
        cmd = [sys.executable, "-m", "rul_fusion.cli",
               "--dataset", args.dataset, "--data_root", args.data_root,
               "--seed", str(args.seed), "--output_dir", output_dir] + entry["args"]
        env_pythonpath = True
    elif entry["runner"] == "scripts.run_ncmapss":
        cmd = [sys.executable, str(ROOT / "scripts" / "run_ncmapss.py"),
               "--dataset", args.dataset, "--data_root", args.data_root,
               "--seed", str(args.seed), "--output_dir", output_dir] + entry["args"]
        env_pythonpath = True
    else:
        raise SystemExit(f"unknown runner: {entry['runner']}")

    if args.dry_run:
        print(" ".join(cmd))
        print(f"(run with PYTHONPATH={ROOT})")
        return
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
