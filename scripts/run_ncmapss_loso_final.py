"""N-CMAPSS LOSO-tuned final arm (preregistered).

Plan: docs/ncmapss-loso-tuning-prereg-2026-11-05.md. Trains ONE model per
dataset on the 7 dev engines that are not the two registered calibration
engines (unit_override train=7, val=[], calibration=registered 2), with the
frozen epoch budget and config chosen by the LOSO search — no early stopping,
no validation use. Calibrates Window-Weighted CQR on the G9 expanded cuts of
the registered calibration engines and evaluates the official 6-engine test
endpoints once (third touch of these endpoints; disclosed in
test_touches.log). P0 and Arm B rows are embedded side by side; nothing is
selected on test.
"""
from __future__ import annotations

import argparse
import csv
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
from rul_fusion.model import MultiScalePhaseGraphRUL  # noqa: E402
from rul_fusion.runner import set_seed  # noqa: E402
from rul_fusion.uncertainty import A6QuantileRUL, pinball_loss  # noqa: E402
from rul_fusion.quantile_runner import _predict  # noqa: E402
from run_ncmapss_cqr import (REGISTERED_CALIBRATION_RATIO,  # noqa: E402
                             MIN_CALIBRATION_UNITS, REGISTERED_VAL_RATIO,
                             _apply, _threshold_variants)
from run_ncmapss_loso_search import (BATCH, HISTORY, _build_model,  # noqa: E402
                                     _endpoint_metrics)
from run_weighted_cqr import _features, _loader  # noqa: E402

ALPHA = 0.2
G9 = [round(0.1 * step, 2) for step in range(1, 10)]


def _registered_calibration_units(dataset):
    path = (ROOT / "results_v4/ncmapss_comparison/cqr" / dataset
            / "seed_0" / "calibrator.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [int(u) for u in data["calibration_units"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("DS03", "DS08a"))
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--activation", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, required=True,
                        help="frozen LOSO budget; no early stopping")
    parser.add_argument("--output_dir", default=str(
        ROOT / "results_v4/ncmapss_loso_final"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    config = {"activation": args.activation, "lr": args.lr,
              "dropout": args.dropout, "weight_decay": args.weight_decay}

    calibration_units = _registered_calibration_units(args.dataset)
    base = build_ncmapss_datasets(
        args.data_root, args.dataset, history=HISTORY, split_seed=0,
        val_ratio=REGISTERED_VAL_RATIO,
        calibration_ratio=REGISTERED_CALIBRATION_RATIO,
        validation_protocol="fixed_cuts", registered_split=True,
        min_calibration_units=MIN_CALIBRATION_UNITS)
    dev_units = [int(u) for u in base.metadata["dev_units"]]
    train_units = [u for u in dev_units if u not in calibration_units]
    if sorted(train_units + calibration_units) != sorted(dev_units):
        raise RuntimeError("final-arm partition must cover all dev units")

    bundle = build_ncmapss_datasets(
        args.data_root, args.dataset, history=HISTORY, split_seed=0,
        val_ratio=REGISTERED_VAL_RATIO,
        calibration_ratio=REGISTERED_CALIBRATION_RATIO,
        validation_protocol="fixed_cuts", registered_split=True,
        min_calibration_units=MIN_CALIBRATION_UNITS,
        calibration_fracs=G9,
        unit_override={"train": train_units, "val": [],
                       "calibration": calibration_units})

    set_seed(args.seed)
    model = _build_model(bundle, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    loader = DataLoader(bundle.train, batch_size=BATCH, shuffle=True)
    history_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, w, y, _unit, _sid in loader:
            x, w, y = x.to(device), w.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            q, regularization = model(x, w, return_aux=True)
            loss = pinball_loss(q, y) + regularization
            loss.backward()
            optimizer.step()
        history_rows.append({"epoch": epoch, "train_loss": float(loss)})
        print(f"epoch:{epoch}/{args.epochs} train_loss:{float(loss):.5f}",
              flush=True)

    cal_y, cal_q, cal_units, cal_ids, cal_cond = _predict(
        model, _loader(bundle.calibration_endpoint), device)
    test_dataset = (bundle.test_endpoint if bundle.test_endpoint is not None
                    else bundle.test)
    test_y, test_q, test_units, test_ids, test_cond = _predict(
        model, _loader(test_dataset), device)
    test_feats = _features(test_q, test_cond, normalization=1.0)
    calibrator, weighted, _ = _threshold_variants(
        cal_y, cal_q, cal_units, cal_ids, cal_cond, test_feats, ALPHA,
        normalization=1.0)

    rows = [{"dataset": args.dataset, "arm": "loso_tuned_G9",
             "variant": "global_block_max_cqr",
             "threshold": float(calibrator.global_threshold),
             "n_cal_windows": int(len(cal_y)),
             **_apply(test_y, test_q, calibrator.global_threshold, ALPHA)}]
    for name, threshold in weighted.items():
        rows.append({"dataset": args.dataset, "arm": "loso_tuned_G9",
                     "variant": name, "threshold": float(threshold),
                     "n_cal_windows": int(len(cal_y)),
                     **_apply(test_y, test_q, threshold, ALPHA)})

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
                        "PICP": float(row["PICP"]),
                        "MPIW": float(row["MPIW"]),
                        "IntervalScore": float(row["IntervalScore"])})
                    break

    out_dir = Path(args.output_dir) / args.dataset / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config,
                "epochs": args.epochs}, out_dir / "best.pt")
    fields = sorted({key for row in rows + references for key in row})
    with (out_dir / "arm_comparison.csv").open("w", newline="",
                                               encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows + references)
    point = {"MAE": float(np.mean(np.abs(test_q[:, 1] - test_y))),
             "RMSE": float(np.sqrt(np.mean((test_q[:, 1] - test_y) ** 2)))}
    (out_dir / "summary.json").write_text(json.dumps({
        "preregistration": "docs/ncmapss-loso-tuning-prereg-2026-11-05.md",
        "config": config, "epochs": args.epochs,
        "train_units": train_units, "calibration_units": calibration_units,
        "calibration_fracs": G9, "q50_point": point, "rows": rows,
        "references": references,
        "generated": datetime.now().isoformat(timespec="seconds")},
        indent=2, default=float), encoding="utf-8")
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(history_rows)
    touch = ROOT / "results_v4/chapter4_main_logs/test_touches.log"
    with touch.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"ncmapss_loso_final (third touch of these endpoints, "
            f"pre-registered): official test evaluated once for {args.dataset} "
            f"seed {args.seed}; config frozen from LOSO search before this "
            f"run; P0/Arm B/LOSO-tuned reported side by side\n")
    print(f"q50 point: {point}", flush=True)
    for row in rows + references:
        print(f"{row['arm']:<18s} {row['variant']:<22s} thr={row['threshold']:8.3f}"
              f" PICP={row['PICP']:.3f} MPIW={row['MPIW']:8.3f}"
              f" IS={row['IntervalScore']:8.3f}", flush=True)


if __name__ == "__main__":
    main()
