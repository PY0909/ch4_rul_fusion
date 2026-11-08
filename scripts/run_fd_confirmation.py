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
LOCKED_HISTORY = 50
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
                          "--history", str(LOCKED_HISTORY), "--rul_cap", "125",
                          "--calibration_ratio", "0.5", "--defer_test",
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
                    "locked_history": metadata.get("history") == LOCKED_HISTORY,
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
    """Evaluate the ALREADY TRAINED checkpoints on the untouched test, once."""
    if not args.eval_test:
        raise SystemExit("--stage evaluate requires --eval_test")
    if stage_verify(args):
        raise SystemExit("verification failed; refusing test access")
    import numpy as np
    import torch
    from rul_fusion.data import build_cmapss_datasets
    from rul_fusion.metrics import regression_metrics
    from rul_fusion.model import MultiScalePhaseGraphRUL
    from rul_fusion.probabilistic_runner import fit_probabilistic
    from rul_fusion.quantile_runner import _predict
    from rul_fusion.uncertainty import A6QuantileRUL
    from run_weighted_cqr import _apply, _features, _loader

    touch = OUTPUT / "test_touches_fd_confirmation.log"
    with touch.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] "
                     f"fd_confirmation: FIRST-TOUCH test evaluation started for "
                     f"{', '.join(DATASETS)} (prereg {PREREG}); rows fixed "
                     f"before evaluation\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for dataset in DATASETS:
        print(f"=== evaluate {dataset} ({device}) ===", flush=True)
        point_cfg = json.loads((OUTPUT / "a6_point" / dataset / "seed_0" /
                                "config.json").read_text(encoding="utf-8"))
        bundle = build_cmapss_datasets(
            args.data_root, dataset, history=int(point_cfg["data"]["history"]),
            rul_cap=125, split_seed=0, calibration_ratio=0.5, scaler_mode="grouped",
            validation_protocol="fixed_cuts", load_test=True)

        # --- A6-Point: load frozen checkpoint, predict test once ---
        run = OUTPUT / "a6_point" / dataset / "seed_0"
        state = torch.load(run / "best.pt", map_location=device, weights_only=False)
        model = MultiScalePhaseGraphRUL(
            sensor_dim=bundle.sensor_dim, condition_dim=bundle.condition_dim,
            d_model=int(point_cfg["d_model"]), num_heads=int(point_cfg["num_heads"]),
            num_routers=int(point_cfg["num_routers"]), dropout=0.1,
            encoder="patch_cnn", variant="A6", condition_mode="full").to(device)
        model.load_state_dict(state["model_state"])
        model.eval()
        ys, ps, us, ids = [], [], [], []
        with torch.no_grad():
            for x, w, y, unit, sid in _loader(bundle.test):
                pred = model(x.to(device), w.to(device))
                ys.append(y.numpy()); ps.append(pred.reshape(-1).cpu().numpy())
                us.append(np.asarray(unit)); ids.append(np.asarray(sid))
        y = np.concatenate(ys); pred = np.concatenate(ps)
        point = regression_metrics(y, pred)
        rows.append({"dataset": dataset, "model": "A6-Point",
                     **{k: point[k] for k in ("MAE", "RMSE", "NASA_Score")}})

        # --- A6-Quantile + four CQR variants on the frozen checkpoint ---
        run = OUTPUT / "a6_quantile" / "global" / dataset / "seed_0"
        state = torch.load(run / "best.pt", map_location=device, weights_only=False)
        q_cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        backbone = MultiScalePhaseGraphRUL(
            sensor_dim=bundle.sensor_dim, condition_dim=bundle.condition_dim,
            d_model=int(q_cfg["resolved_d_model"]),
            num_heads=int(q_cfg["resolved_num_heads"]),
            num_routers=int(q_cfg["resolved_num_routers"]), dropout=0.1,
            encoder="patch_cnn", variant="A6", condition_mode="full")
        qmodel = A6QuantileRUL(backbone, dropout=0.1).to(device)
        qmodel.load_state_dict(state["model_state"])
        qmodel.eval()
        cal_y, cal_q, cal_units, cal_ids, cal_cond = _predict(
            qmodel, _loader(bundle.calibration_endpoint), device)
        test_y, test_q, test_units, _ids, test_cond = _predict(
            qmodel, _loader(bundle.test), device)
        q50 = {"MAE": float(np.mean(np.abs(test_q[:, 1] - test_y))),
               "RMSE": float(np.sqrt(np.mean((test_q[:, 1] - test_y) ** 2)))}
        rows.append({"dataset": dataset, "model": "A6-Quantile q50",
                     **q50, "NASA_Score": None})
        test_feats = _features(test_q, test_cond, normalization=1.0)
        from run_weighted_cqr import (_block_max, _density_ratio,
                                      _plain_threshold, _window_scores,
                                      _weighted_threshold)
        scores_w = _window_scores(cal_y, cal_q)
        block_scores, _ = _block_max(scores_w, cal_units)
        weights_w = _density_ratio(_features(cal_q, cal_cond, normalization=1.0),
                                   test_feats, clip=20.0, lr_c=1.0)
        weights_b = np.asarray([weights_w[cal_units == unit].mean()
                                for unit in np.unique(cal_units)])
        for name, threshold in (
                ("window_plain", _plain_threshold(scores_w, 0.2)),
                ("block_max_plain", _plain_threshold(block_scores, 0.2)),
                ("window_weighted",
                 _weighted_threshold(scores_w, weights_w, 0.2)),
                ("block_max_weighted",
                 _weighted_threshold(block_scores, weights_b, 0.2))):
            metrics = _apply(test_y, test_q, threshold, alpha=0.2)
            rows.append({"dataset": dataset, "model": f"A6-Quantile+CQR/{name}",
                         "threshold": metrics.pop("threshold"),
                         **{k: metrics[k] for k in
                            ("PICP", "MPIW", "IntervalScore", "q50_RMSE")}})

        # --- baselines: evaluate the trained checkpoints on test ---
        for method in BASELINES:
            run = OUTPUT / "baselines" / dataset / method / "seed_0"
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            out = OUTPUT / "baselines_eval" / dataset / method / "seed_0"
            if out.exists() and (out / "metrics.json").is_file():
                print(f"[{dataset}/{method}] already evaluated; skipping",
                      flush=True)
                result = json.loads((out / "metrics.json").read_text())
            else:
                from rul_fusion.probabilistic import PAPER_METHOD_DEFAULTS
                _, _, lr_default, wd_default, do_default, _sc, _crit, mc_default = \
                    PAPER_METHOD_DEFAULTS[method]
                out.parent.mkdir(parents=True, exist_ok=True)
                result = fit_probabilistic(
                    bundle, method, out, seed=0, epochs=config.get("epochs", 41),
                    batch_size=config.get("batch_size"),
                    lr=lr_default, weight_decay=wd_default,
                    patience=config.get("patience"),
                    dropout=do_default, mc_samples=mc_default,
                    device=str(device), defer_test=False,
                    evaluate_checkpoint=run / "best.pt",
                    config={"prereg": PREREG, "first_touch": True})
            rows.append({"dataset": dataset, "model": method,
                         **{k: result.get(k) for k in
                            ("MAE", "RMSE", "NASA_Score", "NLL", "CRPS",
                             "PICP", "MPIW", "IntervalScore")}})

    out = OUTPUT / "confirmation_rows.csv"
    fields = sorted({k for row in rows for k in row})
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with touch.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] "
                     f"fd_confirmation: FIRST-TOUCH test evaluation complete; "
                     f"{len(rows)} preregistered rows -> {out}\n")
    for row in rows:
        rmse = row.get("RMSE"); picp = row.get("PICP"); is_ = row.get("IntervalScore")
        print(f"{row['dataset']} {row['model']:32s} "
              f"RMSE={rmse if rmse is None else round(rmse, 2)} "
              f"PICP={picp if picp is None else round(picp, 3)} "
              f"IS={is_ if is_ is None else round(is_, 2)}", flush=True)


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
