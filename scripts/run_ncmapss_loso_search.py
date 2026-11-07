"""N-CMAPSS LOSO search over non-calibration development units.

Plan: docs/ncmapss-loso-preparation-amendment-2026-11-06.md. Seven folds
exclude the two registered calibration units; each fold trains six and records
per-epoch metrics on the held-out engine's fixed-cut endpoints. Primary
selection criterion: mean pinball loss over q10/q50/q90 on the held-out
endpoints; secondary: q50 RMSE, raw (no-conformal) Interval Score at nominal
80%. The official test set is never read. The pilot also compares budgets
chosen from all three metrics on the same training trajectories.
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
from rul_fusion.model import MultiScalePhaseGraphRUL  # noqa: E402
from rul_fusion.runner import set_seed  # noqa: E402
from rul_fusion.uncertainty import (A6QuantileRUL, interval_metrics,  # noqa: E402
                                    pinball_loss)
from run_ncmapss_cqr import (REGISTERED_CALIBRATION_RATIO,  # noqa: E402
                             MIN_CALIBRATION_UNITS, REGISTERED_VAL_RATIO)

HISTORY = 16
DIMS = dict(d_model=32, num_heads=2, num_routers=2, dropout=0.1)
MAX_EPOCHS = 300
BATCH = 32
SEARCH_UNITS = [1, 2, 3, 5, 6, 8, 9]
CALIBRATION_UNITS = [4, 7]
PILOT_GRID = [
    {"activation": "legacy", "lr": lr, "dropout": 0.1,
     "weight_decay": 1e-3} for lr in (1e-3, 3e-4)
]
S1_GRID = [
    {"activation": "legacy", "lr": 1e-3, "dropout": 0.1, "weight_decay": 1e-3},
    {"activation": "legacy", "lr": 3e-4, "dropout": 0.1, "weight_decay": 1e-3},
    {"activation": "gelu", "lr": 1e-3, "dropout": 0.1, "weight_decay": 1e-3},
    {"activation": "gelu", "lr": 3e-4, "dropout": 0.1, "weight_decay": 1e-3},
    {"activation": "silu", "lr": 1e-3, "dropout": 0.1, "weight_decay": 1e-3},
    {"activation": "silu", "lr": 3e-4, "dropout": 0.1, "weight_decay": 1e-3},
]


def parse_configs(text):
    """Parse "activation:lr:dropout:weight_decay,..." into config dicts."""
    configs = []
    for chunk in text.split(","):
        parts = [p.strip() for p in chunk.split(":") if p.strip()]
        if len(parts) != 4:
            raise ValueError(f"config needs activation:lr:dropout:wd, got {chunk!r}")
        configs.append({"activation": parts[0], "lr": float(parts[1]),
                        "dropout": float(parts[2]), "weight_decay": float(parts[3])})
    if not configs:
        raise ValueError("empty config list")
    return configs


def config_id(config):
    return (f"{config['activation']}_lr{config['lr']:g}"
            f"_do{config['dropout']:g}_wd{config['weight_decay']:g}")


def _fold_cache_path(output, dataset, config, held, train_units, dev_path,
                     max_epochs):
    source_paths = [Path(__file__), ROOT / "rul_fusion/data.py",
                    ROOT / "rul_fusion/model.py",
                    ROOT / "rul_fusion/uncertainty.py"]
    path = Path(dev_path).resolve()
    stat = path.stat()
    identity = json.dumps({
        "dataset": dataset, "config": config, "held": held,
        "train_units": train_units, "max_epochs": max_epochs,
        "dev_path": str(path), "dev_size": stat.st_size,
        "dev_mtime_ns": stat.st_mtime_ns,
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(
            p.read_bytes()).hexdigest() for p in source_paths},
    }, sort_keys=True)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return output / "folds" / dataset / f"{config_id(config)}_{digest}" / f"held_{held}.csv"


def _read_fold_cache(path, dataset, tag, held, max_epochs):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if (len(rows) != max_epochs or
            [int(row["epoch"]) for row in rows] != list(range(1, max_epochs + 1)) or
            any(row["dataset"] != dataset or row["config"] != tag or
                int(row["held_unit"]) != held for row in rows)):
        raise RuntimeError(f"invalid completed-fold cache: {path}")
    numeric_int = {"fold", "held_unit", "epoch"}
    numeric_float = {"lr", "dropout", "weight_decay", "pinball_mean",
                     "q50_rmse", "raw_interval_score", "raw_PICP", "raw_MPIW"}
    return [{key: (int(value) if key in numeric_int else
                   float(value) if key in numeric_float else value)
             for key, value in row.items()} for row in rows]


def _write_fold_cache(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def moving_average(values, window=5):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        return values.copy()
    return np.convolve(values, np.ones(window) / window, mode="valid")


def pick_budget(primary_curve, window=5):
    values = np.asarray(primary_curve, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("empty curve")
    if len(values) < window:
        return int(np.argmin(values)) + 1           # 1-based epoch
    smoothed = moving_average(values, window)
    budget = int(np.argmin(smoothed)) + (window + 1) // 2
    return min(max(budget, 1), len(values))


def _build_model(bundle, config, device):
    backbone = MultiScalePhaseGraphRUL(
        sensor_dim=bundle.sensor_dim, condition_dim=bundle.condition_dim,
        d_model=DIMS["d_model"], num_heads=DIMS["num_heads"],
        num_routers=DIMS["num_routers"], dropout=config["dropout"],
        encoder="patch_cnn", variant="A6", condition_mode="full",
        activation=config["activation"])
    model = A6QuantileRUL(backbone, dropout=config["dropout"],
                          activation=config["activation"]).to(device)
    return model


@torch.no_grad()
def _endpoint_metrics(model, loader, device):
    model.eval()
    ys, qs = [], []
    for x, w, y, _unit, _sid in loader:
        qs.append(model(x.to(device), w.to(device)).cpu().numpy())
        ys.append(y.numpy())
    y = np.concatenate(ys)
    q = np.concatenate(qs)
    pinball = float(pinball_loss(torch.as_tensor(q), torch.as_tensor(y)))
    q50_rmse = float(np.sqrt(np.mean((q[:, 1] - y) ** 2)))
    raw = interval_metrics(y, q[:, 0], q[:, 2], alpha=0.2)
    return {"pinball_mean": pinball, "q50_rmse": q50_rmse,
            "raw_interval_score": float(raw["IntervalScore"]),
            "raw_PICP": float(raw["PICP"]), "raw_MPIW": float(raw["MPIW"])}


def run_fold_config(bundle, config, device, max_epochs=MAX_EPOCHS):
    """Train one LOSO fold with one config; return per-epoch held-out rows."""
    set_seed(0)
    model = _build_model(bundle, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    loader = DataLoader(bundle.train, batch_size=BATCH, shuffle=True)
    held_loader = DataLoader(bundle.val_endpoint, batch_size=BATCH, shuffle=False)
    rows = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, w, y, _unit, _sid in loader:
            x, w, y = x.to(device), w.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            q, regularization = model(x, w, return_aux=True)
            loss = pinball_loss(q, y) + regularization
            loss.backward()
            optimizer.step()
        rows.append({"epoch": epoch, **_endpoint_metrics(model, held_loader, device)})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--datasets", nargs="+", default=["DS03", "DS08a"])
    parser.add_argument("--dev_path", default=None,
                        help="override dev HDF5 path (tests); default data_root/data_set")
    parser.add_argument("--test_path", default=None)
    parser.add_argument("--grid", choices=("pilot", "s1", "custom"),
                        default="pilot")
    parser.add_argument("--configs", help="custom S2 list, e.g. gelu:0.001:0.2:0.001,...")
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--output_dir", default=str(
        ROOT / "results_v4/ncmapss_loso_search"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--folds", type=int, default=0,
                        help="run only the first K folds (smoke/probe); 0 = all")
    args = parser.parse_args(argv)
    if args.grid == "custom" and not args.configs:
        parser.error("--grid custom requires --configs")
    if args.max_epochs < 1 or args.folds < 0:
        parser.error("max_epochs must be positive and folds must be nonnegative")
    configs = (PILOT_GRID if args.grid == "pilot" else
               S1_GRID if args.grid == "s1" else parse_configs(args.configs))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)

    def build(dataset, **kwargs):
        return build_ncmapss_datasets(
            args.data_root, dataset, history=HISTORY, split_seed=0,
            val_ratio=REGISTERED_VAL_RATIO,
            calibration_ratio=REGISTERED_CALIBRATION_RATIO,
            validation_protocol="fixed_cuts", registered_split=True,
            min_calibration_units=MIN_CALIBRATION_UNITS,
            dev_path=args.dev_path, test_path=args.test_path,
            load_test=False, **kwargs)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    curve_rows = []
    partitions = {}
    for dataset in args.datasets:
        base = build(dataset)
        dev_units = [int(u) for u in base.metadata["dev_units"]]
        calibration_units = [int(u) for u in base.metadata["calibration_units"]]
        search_units = [u for u in dev_units if u not in calibration_units]
        if (len(dev_units) != 9 or calibration_units != CALIBRATION_UNITS or
                search_units != SEARCH_UNITS):
            raise RuntimeError("registered calibration units or search pool changed")
        partitions[dataset] = {"search_units": search_units,
                               "calibration_units": calibration_units}
        fold_units = search_units[:args.folds] if args.folds else search_units
        for fold, held in enumerate(fold_units):
            train_units = [u for u in search_units if u != held]
            bundle = None
            for config in configs:
                tag = config_id(config)
                dev_path = (args.dev_path or Path(args.data_root) / "data_set" /
                            ("N-CMAPSS_DS03-012.h5" if dataset == "DS03" else
                             "N-CMAPSS_DS08a-009.h5"))
                cache = _fold_cache_path(output, dataset, config, held,
                                         train_units, dev_path, args.max_epochs)
                if cache.exists():
                    print(f"[{dataset}] reuse held={held} config={tag}", flush=True)
                    fold_rows = _read_fold_cache(cache, dataset, tag, held,
                                                 args.max_epochs)
                else:
                    print(f"[{dataset}] fold={fold} held={held} config={tag}",
                          flush=True)
                    if bundle is None:
                        bundle = build(dataset, unit_override={
                            "train": train_units, "val": [held],
                            "calibration": []})
                    fold_rows = [{"dataset": dataset, "fold": fold,
                                  "held_unit": held, "config": tag,
                                  **config, **row}
                                 for row in run_fold_config(
                                     bundle, config, device,
                                     max_epochs=args.max_epochs)]
                    _write_fold_cache(cache, fold_rows)
                curve_rows.extend(fold_rows)
        write_curves = output / f"curves_{dataset}_{args.grid}.csv"
        rows_here = [r for r in curve_rows if r["dataset"] == dataset]
        with write_curves.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows_here[0]))
            writer.writeheader()
            writer.writerows(rows_here)

    # Aggregate per (dataset, config, epoch) over folds; pick budgets; rank.
    aggregate = []
    for dataset in args.datasets:
        rows_here = [r for r in curve_rows if r["dataset"] == dataset]
        by_key = {}
        for row in rows_here:
            by_key.setdefault((row["config"], row["epoch"]), []).append(row)
        for (config, epoch), members in sorted(by_key.items()):
            aggregate.append({
                "dataset": dataset, "config": config, "epoch": epoch,
                "n_folds": len(members),
                "pinball_mean": float(np.mean([m["pinball_mean"] for m in members])),
                "q50_rmse": float(np.mean([m["q50_rmse"] for m in members])),
                "raw_interval_score": float(np.mean(
                    [m["raw_interval_score"] for m in members])),
                "raw_PICP": float(np.mean([m["raw_PICP"] for m in members])),
                "raw_MPIW": float(np.mean([m["raw_MPIW"] for m in members]))})
    with (output / f"aggregate_{args.grid}.csv").open("w", newline="",
                                                      encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    selection = []
    criterion_budgets = []
    for dataset in args.datasets:
        rows_here = [r for r in aggregate if r["dataset"] == dataset]
        for config in sorted({r["config"] for r in rows_here}):
            curve = [r for r in rows_here if r["config"] == config]
            curve.sort(key=lambda r: r["epoch"])
            for criterion in ("pinball_mean", "q50_rmse", "raw_interval_score"):
                criterion_budget = pick_budget([r[criterion] for r in curve])
                at_criterion = next(r for r in curve
                                    if r["epoch"] == criterion_budget)
                criterion_budgets.append({
                    "dataset": dataset, "config": config,
                    "criterion": criterion, "budget_epoch": criterion_budget,
                    "boundary_censored": criterion_budget > args.max_epochs - 10,
                    **{k: at_criterion[k] for k in
                       ("pinball_mean", "q50_rmse", "raw_interval_score",
                        "raw_PICP", "raw_MPIW")}})
            budget = next(r["budget_epoch"] for r in criterion_budgets
                          if r["dataset"] == dataset and r["config"] == config
                          and r["criterion"] == "pinball_mean")
            at_budget = next(r for r in curve if r["epoch"] == budget)
            selection.append({"dataset": dataset, "config": config,
                              "budget_epoch": budget,
                              **{k: at_budget[k] for k in
                                 ("pinball_mean", "q50_rmse",
                                  "raw_interval_score", "raw_PICP",
                                  "raw_MPIW")}})
    selection.sort(key=lambda r: (r["dataset"], r["pinball_mean"],
                                  r["q50_rmse"], r["raw_interval_score"]))
    with (output / f"selection_{args.grid}.csv").open("w", newline="",
                                                      encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(selection[0]))
        writer.writeheader()
        writer.writerows(selection)
    with (output / f"criterion_budgets_{args.grid}.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(criterion_budgets[0]))
        writer.writeheader()
        writer.writerows(criterion_budgets)
    (output / f"selection_{args.grid}.json").write_text(json.dumps({
        "preregistration": "docs/ncmapss-loso-preparation-amendment-2026-11-06.md",
        "grid": args.grid, "configs": configs, "max_epochs": args.max_epochs,
        "partitions": partitions,
        "folds_limited_to": args.folds or None,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "ranking": selection}, indent=2, default=float), encoding="utf-8")
    for row in selection:
        print(f"[{row['dataset']}] {row['config']:<36s} budget={row['budget_epoch']:>3d} "
              f"pinball={row['pinball_mean']:.4f} q50={row['q50_rmse']:.3f} "
              f"rawIS={row['raw_interval_score']:.2f} rawPICP={row['raw_PICP']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
