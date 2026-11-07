"""N-CMAPSS LOSO-tuned A6-Quantile final arm, in two separated stages.

Frozen record: results_v4/ncmapss_loso_frozen/selection.json
(docs/ncmapss-loso-selection-freeze-2026-11-07.md).

``--stage train`` never loads the test split. It trains ONE model per dataset
on the seven search engines for exactly the frozen epoch budget (no validation,
no early stopping), predicts the G9 calibration endpoints of the registered
calibration engines, and writes the checkpoint, calibration predictions,
history and a manifest with SHA-256 hashes and ``test_loaded: false``.

``--stage evaluate --eval_test`` re-verifies the selection hash, frozen
config/epochs and the checkpoint and calibration hashes, recomputes the
calibration predictions from the checkpoint and checks they match, and only
then predicts the six test engines' fixed-cut endpoints once. Window-Weighted
CQR weights use unlabeled test covariates as in production. Each test access
is logged in test_touches.log; results are a descriptive recheck.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from rul_fusion.data import build_ncmapss_datasets  # noqa: E402
from rul_fusion.runner import set_seed  # noqa: E402
from rul_fusion.uncertainty import pinball_loss  # noqa: E402
from rul_fusion.quantile_runner import _predict  # noqa: E402
from run_ncmapss_cqr import (REGISTERED_CALIBRATION_RATIO,  # noqa: E402
                             MIN_CALIBRATION_UNITS, REGISTERED_VAL_RATIO,
                             _apply, _threshold_variants)
from run_ncmapss_loso_search import (BATCH, CALIBRATION_UNITS, HISTORY,  # noqa: E402
                                     SEARCH_UNITS, _build_model)
from run_weighted_cqr import _features, _loader  # noqa: E402

ALPHA = 0.2
G9 = [round(0.1 * step, 2) for step in range(1, 10)]
FROZEN_CRITERION = "raw_interval_score"
# Training runs on CUDA and evaluation may run on CPU; the same checkpoint then
# reproduces its calibration quantiles to about 2e-3 cycles. A tolerance of
# 0.01 cycles separates that float noise from a changed model or data path.
REPRODUCTION_TOLERANCE = 0.01


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _registered_calibration_units(dataset):
    path = (ROOT / "results_v4/ncmapss_comparison/cqr" / dataset
            / "seed_0" / "cqr_variants_calibrators.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [int(u) for u in data["calibration_units"]]


def _frozen_selection(path, dataset):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("status") != "frozen":
        raise ValueError("selection status must be frozen")
    entry = data.get("datasets", {}).get(dataset)
    if not isinstance(entry, dict):
        raise ValueError(f"selection is missing {dataset}")
    if (entry.get("criterion") != FROZEN_CRITERION or
            sorted(entry.get("search_units", [])) != SEARCH_UNITS or
            sorted(entry.get("calibration_units", [])) != CALIBRATION_UNITS):
        raise ValueError("selection criterion or registered units do not match")
    if entry.get("n_folds") != 7 or entry.get("boundary_censored") is not False:
        raise ValueError("selection requires a complete seven-fold, uncensored search")
    config = entry.get("config", {})
    if (set(config) != {"activation", "lr", "dropout", "weight_decay"} or
            config["activation"] not in {"legacy", "gelu", "silu"} or
            config["lr"] not in {1e-3, 3e-4} or
            config["dropout"] not in {0.1, 0.2} or
            config["weight_decay"] != 1e-3):
        raise ValueError("selection config is outside the registered grid")
    epochs = entry.get("budget_epoch")
    if not isinstance(epochs, int) or not 1 <= epochs <= 120:
        raise ValueError("selection budget_epoch must be in [1, 120]")
    return config, epochs


def _bundle(args, train_units, calibration_units, load_test):
    return build_ncmapss_datasets(
        args.data_root, args.dataset, history=HISTORY, split_seed=0,
        val_ratio=REGISTERED_VAL_RATIO,
        calibration_ratio=REGISTERED_CALIBRATION_RATIO,
        validation_protocol="fixed_cuts", registered_split=True,
        min_calibration_units=MIN_CALIBRATION_UNITS,
        calibration_fracs=G9, dev_path=args.dev_path, test_path=args.test_path,
        unit_override={"train": train_units, "val": [],
                       "calibration": calibration_units}, load_test=load_test)


def _partition(args):
    calibration_units = (CALIBRATION_UNITS if args.dev_path
                         else _registered_calibration_units(args.dataset))
    if sorted(calibration_units) != CALIBRATION_UNITS:
        raise RuntimeError("registered calibration units changed")
    base = build_ncmapss_datasets(
        args.data_root, args.dataset, history=HISTORY, split_seed=0,
        val_ratio=REGISTERED_VAL_RATIO,
        calibration_ratio=REGISTERED_CALIBRATION_RATIO,
        validation_protocol="fixed_cuts", registered_split=True,
        min_calibration_units=MIN_CALIBRATION_UNITS, dev_path=args.dev_path,
        test_path=args.test_path, load_test=False)
    dev_units = [int(u) for u in base.metadata["dev_units"]]
    train_units = [u for u in dev_units if u not in calibration_units]
    if (sorted(train_units) != SEARCH_UNITS or
            sorted(train_units + calibration_units) != sorted(dev_units)):
        raise RuntimeError("final-arm partition must cover all dev units")
    return train_units, calibration_units


def _stage_train(args, out_dir, config, epochs, selection_path):
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite final-arm output: {out_dir}")
    train_units, calibration_units = _partition(args)
    bundle = _bundle(args, train_units, calibration_units, load_test=False)
    if getattr(bundle, "test_endpoint", None) is not None:
        raise RuntimeError("training stage must not load the test split")
    device = _device(args)
    set_seed(args.seed)
    model = _build_model(bundle, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    loader = DataLoader(bundle.train, batch_size=BATCH, shuffle=True)
    history_rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, w, y, _unit, _sid in loader:
            x, w, y = x.to(device), w.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            q, regularization = model(x, w, return_aux=True)
            loss = pinball_loss(q, y) + regularization
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
        history_rows.append({"epoch": epoch, "train_loss": total / max(count, 1)})
        print(f"epoch:{epoch}/{epochs} train_loss:{total / max(count, 1):.5f}",
              flush=True)
    cal_y, cal_q, cal_units, cal_ids, cal_cond = _predict(
        model, _loader(bundle.calibration_endpoint), device)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config,
                "epochs": epochs}, out_dir / "best.pt")
    np.savez(out_dir / "calibration_predictions.npz", y=cal_y, q=cal_q,
             units=cal_units, ids=cal_ids, cond=cal_cond)
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(history_rows)
    manifest = {
        "stage": "train", "dataset": args.dataset, "seed": args.seed,
        "selection_file": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "config": config, "epochs": epochs, "epochs_trained": len(history_rows),
        "train_units": train_units, "calibration_units": calibration_units,
        "calibration_fracs": G9, "n_calibration_windows": int(len(cal_y)),
        "checkpoint_sha256": _sha256(out_dir / "best.pt"),
        "calibration_sha256": _sha256(out_dir / "calibration_predictions.npz"),
        "runner_sha256": _sha256(Path(__file__)),
        "test_loaded": False,
        "generated": datetime.now().isoformat(timespec="seconds")}
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                 encoding="utf-8")
    print(f"[{args.dataset}] trained {len(history_rows)} epochs; "
          f"checkpoint {manifest['checkpoint_sha256'][:12]}; test not loaded",
          flush=True)


def _verify_training(out_dir, config, epochs, selection_path):
    manifest_path = out_dir / "train_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("no completed training stage to evaluate")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "stage": manifest.get("stage") == "train",
        "test_not_loaded_in_training": manifest.get("test_loaded") is False,
        "selection_unchanged": manifest.get("selection_sha256") == _sha256(selection_path),
        "config_matches_frozen": manifest.get("config") == config,
        "epochs_match_frozen": (manifest.get("epochs") == epochs and
                                manifest.get("epochs_trained") == epochs),
        "checkpoint_hash": manifest.get("checkpoint_sha256") == _sha256(out_dir / "best.pt"),
        "calibration_hash": (manifest.get("calibration_sha256") ==
                             _sha256(out_dir / "calibration_predictions.npz")),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"training verification failed: {failed}")
    return manifest, checks


def _per_engine(test_y, test_q, test_units, threshold):
    lower, upper = test_q[:, 0] - threshold, test_q[:, 2] + threshold
    rows = []
    for unit in np.unique(test_units):
        mask = test_units == unit
        err = test_q[mask, 1] - test_y[mask]
        rows.append({"unit": int(unit), "n_endpoints": int(mask.sum()),
                     "q50_rmse": float(np.sqrt(np.mean(err ** 2))),
                     "q50_mae": float(np.mean(np.abs(err))),
                     "picp": float(np.mean((test_y[mask] >= lower[mask]) &
                                           (test_y[mask] <= upper[mask]))),
                     "mpiw": float(np.mean(upper[mask] - lower[mask]))})
    return rows


def _stage_evaluate(args, out_dir, config, epochs, selection_path):
    manifest, checks = _verify_training(out_dir, config, epochs, selection_path)
    eval_dir = out_dir / "evaluation"
    if eval_dir.exists() and any(eval_dir.iterdir()):
        raise FileExistsError("test evaluation already recorded for this run")
    device = _device(args)
    train_units, calibration_units = manifest["train_units"], manifest["calibration_units"]

    # Reproduce the saved calibration predictions before the test split is
    # loaded at all; a failure here aborts without any test access.
    check_bundle = _bundle(args, train_units, calibration_units, load_test=False)
    model = _build_model(check_bundle, config, device)
    state = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    saved = np.load(out_dir / "calibration_predictions.npz")
    cal_y, cal_q, cal_units, cal_ids, cal_cond = _predict(
        model, _loader(check_bundle.calibration_endpoint), device)
    max_abs_diff = float(np.max(np.abs(cal_q - saved["q"])))
    if not (np.array_equal(cal_units, saved["units"]) and
            np.array_equal(cal_ids, saved["ids"]) and
            np.array_equal(cal_y, saved["y"]) and
            max_abs_diff <= REPRODUCTION_TOLERANCE):
        raise RuntimeError("checkpoint no longer reproduces its calibration predictions "
                           f"(max abs diff {max_abs_diff:.4g} cycles)")
    checks["calibration_reproduced"] = True

    touch = Path(args.touch_log)
    touch.parent.mkdir(parents=True, exist_ok=True)
    with touch.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] "
                     f"ncmapss_loso_final A6-Quantile: test access started for "
                     f"{args.dataset} seed {args.seed} after verified training "
                     f"stage (checkpoint {manifest['checkpoint_sha256'][:12]}, "
                     f"calibration reproduced within {max_abs_diff:.2g} cycles)\n")
    bundle = _bundle(args, train_units, calibration_units, load_test=True)
    test_y, test_q, test_units, test_ids, test_cond = _predict(
        model, _loader(bundle.test_endpoint), device)
    test_feats = _features(test_q, test_cond, normalization=1.0)
    calibrator, weighted, _ = _threshold_variants(
        cal_y, cal_q, cal_units, cal_ids, cal_cond, test_feats, ALPHA,
        normalization=1.0)
    thresholds = {"global_block_max_cqr": float(calibrator.global_threshold),
                  **{name: float(value) for name, value in weighted.items()}}
    rows = [{"dataset": args.dataset, "arm": "loso_tuned_G9", "variant": name,
             "threshold": threshold, "n_cal_windows": int(len(cal_y)),
             **_apply(test_y, test_q, threshold, ALPHA)}
            for name, threshold in thresholds.items()]
    references = _references(args)

    eval_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows + references for key in row})
    with (eval_dir / "arm_comparison.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows + references)
    per_engine = _per_engine(test_y, test_q, test_units, thresholds["window_weighted"])
    with (eval_dir / "per_engine.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=sorted(per_engine[0]))
        writer.writeheader()
        writer.writerows(per_engine)
    point = {"MAE": float(np.mean(np.abs(test_q[:, 1] - test_y))),
             "RMSE": float(np.sqrt(np.mean((test_q[:, 1] - test_y) ** 2)))}
    (eval_dir / "summary.json").write_text(json.dumps({
        "freeze_record": "docs/ncmapss-loso-selection-freeze-2026-11-07.md",
        "result_role": "descriptive_recheck",
        "verification": checks, "config": config, "epochs": epochs,
        "evaluation_device": str(device),
        "calibration_reproduction_max_abs_diff": max_abs_diff,
        "reproduction_tolerance_cycles": REPRODUCTION_TOLERANCE,
        "train_units": train_units, "calibration_units": calibration_units,
        "calibration_fracs": G9, "q50_point": point, "rows": rows,
        "references": references, "per_engine_variant": "window_weighted",
        "test_loaded": True,
        "generated": datetime.now().isoformat(timespec="seconds")},
        indent=2, default=float), encoding="utf-8")
    with touch.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] "
                     f"ncmapss_loso_final A6-Quantile: official test evaluated once "
                     f"for {args.dataset} (descriptive recheck; P0/Arm B reported "
                     f"side by side)\n")
    print(f"[{args.dataset}] q50 point: {point}", flush=True)
    for row in rows + references:
        print(f"{row['arm']:<18s} {row['variant']:<22s} thr={row['threshold']:8.3f}"
              f" PICP={row['PICP']:.3f} MPIW={row['MPIW']:8.3f}"
              f" IS={row['IntervalScore']:8.3f}", flush=True)


def _references(args):
    references = []
    p0 = (ROOT / "results_v4/ncmapss_comparison/cqr" / args.dataset
          / f"seed_{args.seed}" / "cqr_variants_test_summary.json")
    if p0.exists():
        for row in json.loads(p0.read_text(encoding="utf-8"))["rows"]:
            references.append({"dataset": args.dataset, "arm": "A_P0_4cuts",
                               **{k: row[k] for k in ("variant", "threshold",
                                                      "PICP", "MPIW",
                                                      "IntervalScore")}})
    arm_b = (ROOT / "results_v4/ncmapss_cqr_expanded" / args.dataset
             / f"seed_{args.seed}" / "arm_comparison.csv")
    if arm_b.exists():
        with arm_b.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["arm"] == "B_G9" and row["variant"] == "window_weighted":
                    references.append({
                        "dataset": args.dataset, "arm": "B_G9_frozen_ckpt",
                        "variant": "window_weighted",
                        "threshold": float(row["threshold"]),
                        "PICP": float(row["PICP"]), "MPIW": float(row["MPIW"]),
                        "IntervalScore": float(row["IntervalScore"])})
                    break
    return references


def _device(args):
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                        if args.device == "auto" else args.device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("train", "evaluate"))
    parser.add_argument("--dataset", required=True, choices=("DS03", "DS08a"))
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dev_path")
    parser.add_argument("--test_path")
    parser.add_argument("--selection_file", default=str(
        ROOT / "results_v4/ncmapss_loso_frozen/selection.json"))
    parser.add_argument("--eval_test", action="store_true",
                        help="explicitly authorize this registered test evaluation")
    parser.add_argument("--output_dir", default=str(
        ROOT / "results_v4/ncmapss_loso_final"))
    parser.add_argument("--touch_log", default=str(
        ROOT / "results_v4/chapter4_main_logs/test_touches.log"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.seed != 0:
        parser.error("the registered final arm uses seed 0")
    if args.stage == "evaluate" and not args.eval_test:
        parser.error("the evaluation stage requires explicit --eval_test")
    if args.stage == "train" and args.eval_test:
        parser.error("the training stage never accesses test")
    selection_path = Path(args.selection_file)
    config, epochs = _frozen_selection(selection_path, args.dataset)
    out_dir = Path(args.output_dir) / args.dataset / f"seed_{args.seed}"
    if args.stage == "train":
        _stage_train(args, out_dir, config, epochs, selection_path)
    else:
        _stage_evaluate(args, out_dir, config, epochs, selection_path)


if __name__ == "__main__":
    main()
