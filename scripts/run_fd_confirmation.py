"""FD001/FD003 external confirmation arm (preregistered).

Plan: docs/fd001-fd003-confirmation-prereg-2026-11-08.md. The FD001/FD003
official test sets have never been evaluated by this project; everything here
is organized around spending that one-shot value once.

``--stage train``  trains A6-Point (locked FD config), A6-Quantile (locked FD
                    config, calibration-only artifacts) and the six external
                    baselines (paper-native defaults) with
                    ``load_test=False``; nothing test-related is read.
``--stage verify`` checks every training artifact against the preregistered
                    lock values and hashes; no data is loaded.
``--stage evaluate --eval_test`` re-verifies, then loads test exactly once per
                    run, computes the preregistered row set (point metrics,
                    four CQR variants incl. Window-Weighted, baseline
                    probabilistic metrics) and writes the disclosure entry.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DATASETS = ("FD001", "FD003")
BASELINES = ("hnn", "edl_nig", "lstm_prob", "tcn_transformer_mc", "msauq", "bcpt")
OUTPUT = ROOT / "results_v4/fd_confirmation"
PREREG = "docs/fd001-fd003-confirmation-prereg-2026-11-08.md"

# Locked A6-Point command (locked_configs.json, locked 2026-10-29) + seed/split.
A6_POINT_ARGS = ["--history", "50", "--rul_cap", "125", "--scaler", "grouped",
                 "--encoder", "patch_cnn", "--variant", "A6",
                 "--validation_protocol", "fixed_cuts", "--optimizer", "adam",
                 "--weight_decay", "0", "--batch_size", "100", "--epochs", "41",
                 "--patience", "41", "--no_scheduler", "--no_grad_clip",
                 "--d_model", "64", "--num_heads", "4", "--num_routers", "4",
                 "--calibration_ratio", "0.5"]
# Locked A6-Quantile resolution (chapter4_main frozen config.json): every value
# explicit so nothing depends on CLI default drift.
A6_QUANTILE_ARGS = ["--history", "50", "--rul_cap", "125", "--scaler", "grouped",
                    "--d_model", "64", "--num_heads", "4", "--num_routers", "4",
                    "--epochs", "41", "--batch_size", "100", "--lr", "1e-3",
                    "--weight_decay", "0.0", "--patience", "41",
                    "--calibration_ratio", "0.5", "--calibration_mode", "global"]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run(cmd, log):
    print(" ".join(str(part) for part in cmd), flush=True)
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(f"\n=== {' '.join(str(p) for p in cmd)}\n")
        handle.flush()
        subprocess.run([str(p) for p in cmd], cwd=ROOT, check=True,
                       stdout=handle, stderr=subprocess.STDOUT)


def _python(args):
    return [sys.executable, *args]


def _dataset_args(dataset):
    return ["--dataset", dataset, "--split_seed", "0", "--seed", "0"]


def stage_train(args):
    log = OUTPUT / "train.log"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        _run(_python(["-m", "rul_fusion.cli", *_dataset_args(dataset),
                      *A6_POINT_ARGS, "--defer_test",
                      "--data_root", args.data_root,
                      "--output_dir", OUTPUT / "a6_point"]),
             log)
        _run(_python(["-m", "rul_fusion.quantile_cli", *_dataset_args(dataset),
                      *A6_QUANTILE_ARGS, "--defer_test",
                      "--data_root", args.data_root,
                      "--output_dir", OUTPUT / "a6_quantile"]),
             log)
        for method in BASELINES:
            _run(_python(["-m", "rul_fusion.probabilistic_cli",
                          *(_dataset_args(dataset) + ["--method", method]),
                          "--defer_test",
                          "--data_root", args.data_root,
                          "--output_dir", OUTPUT / "baselines"]),
                 log)
    print("training stage complete; test never loaded", flush=True)


def _expected_paths(dataset):
    return {
        "a6_point": (OUTPUT / "a6_point" / dataset / "seed_0",
                     ("best.pt", "history.csv", "config.json",
                      "calibration_only.json")),
        "a6_quantile": (OUTPUT / "a6_quantile" / "global" / dataset / "seed_0",
                        ("best.pt", "history.csv", "config.json",
                         "calibration_only.json", "calibrator.json",
                         "calibration_scores.csv", "grouping.json")),
        **{method: (OUTPUT / "baselines" / dataset / method / "seed_0",
                    ("best.pt", "history.csv", "config.json"))
           for method in BASELINES},
    }


def stage_verify(args):
    rows = []
    for dataset in DATASETS:
        for model, (run, required) in _expected_paths(dataset).items():
            checks = {"required_files": all((run / name).is_file()
                                            for name in required)}
            if checks["required_files"]:
                config = json.loads((run / "config.json").read_text(encoding="utf-8"))
                metadata = config.get("data", {})
                calib_only = (run / "calibration_only.json")
                checks.update({
                    "test_not_evaluated": (
                        (json.loads(calib_only.read_text())["test_evaluated"] is False)
                        if calib_only.is_file() else
                        not (run / "metrics.json").is_file()),
                    "no_predictions": not (run / "predictions.csv").is_file(),
                    "locked_seed": config.get("seed") == 0,
                    "locked_split": metadata.get("split_seed") == 0,
                    "locked_history": metadata.get("history") == 50,
                    "locked_cap": metadata.get("rul_cap") == 125,
                    "fixed_cuts": metadata.get("validation_protocol") == "fixed_cuts",
                    "test_not_loaded": metadata.get("test_loaded", True) is False,
                })
            rows.append({"dataset": dataset, "model": model, "run": str(run),
                         "all_checks_pass": all(checks.values()),
                         "failed": ";".join(k for k, ok in checks.items() if not ok)})
    out = OUTPUT / "verification.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['dataset']} {row['model']:20s} "
              f"{'PASS' if row['all_checks_pass'] else 'FAIL ' + row['failed']}")
    passed = sum(row["all_checks_pass"] for row in rows)
    print(f"{passed}/{len(rows)} verified -> {out}")
    return 0 if passed == len(rows) else 1


def stage_evaluate(args):
    if not args.eval_test:
        raise SystemExit("--stage evaluate requires --eval_test")
    if stage_verify(args):
        raise SystemExit("verification failed; refusing test access")
    touch = OUTPUT / "test_touches_fd_confirmation.log"
    stamp = datetime.now().isoformat(timespec="seconds")
    with touch.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] fd_confirmation: FIRST-TOUCH test evaluation "
                     f"started for {', '.join(DATASETS)} (prereg {PREREG}); "
                     f"rows fixed before evaluation\n")
    log = OUTPUT / "evaluate.log"
    for dataset in DATASETS:
        _run(_python(["-m", "rul_fusion.cli", *_dataset_args(dataset),
                      *A6_POINT_ARGS, "--data_root", args.data_root,
                      "--output_dir", OUTPUT / "a6_point_eval"]),
             log)
        _run(_python(["-m", "rul_fusion.quantile_cli", *_dataset_args(dataset),
                      *A6_QUANTILE_ARGS, "--data_root", args.data_root,
                      "--output_dir", OUTPUT / "a6_quantile_eval"]),
             log)
        for method in BASELINES:
            _run(_python(["-m", "rul_fusion.probabilistic_cli",
                          *(_dataset_args(dataset) + ["--method", method]),
                          "--data_root", args.data_root,
                          "--output_dir", OUTPUT / "baselines_eval"]),
                 log)
    with touch.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] "
                     f"fd_confirmation: test evaluation complete for "
                     f"{', '.join(DATASETS)}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("train", "verify", "evaluate"))
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--eval_test", action="store_true")
    args = parser.parse_args(argv)
    {"train": stage_train, "verify": stage_verify,
     "evaluate": stage_evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
