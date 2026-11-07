"""Development-only N-CMAPSS LOSO comparison of two epoch-budget rules.

See docs/ncmapss-loso-budget-comparison-amendment-2026-11-07.md. The
registered calibration engines and the test split are never used here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
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
from run_ncmapss_cqr import (MIN_CALIBRATION_UNITS,  # noqa: E402
                             REGISTERED_CALIBRATION_RATIO,
                             REGISTERED_VAL_RATIO)
from run_ncmapss_loso_search import (BATCH, CALIBRATION_UNITS, HISTORY,  # noqa: E402
                                     PILOT_GRID, SEARCH_UNITS, _build_model,
                                     _endpoint_metrics, config_id, pick_budget)

MAX_EPOCHS = 120
PATIENCE = 30
CRITERIA = ("pinball_mean", "raw_interval_score")
METRICS = ("pinball_mean", "q50_rmse", "raw_interval_score",
           "raw_PICP", "raw_MPIW")
PREREGISTRATION = (ROOT / "docs" /
                   "ncmapss-loso-budget-comparison-amendment-2026-11-07.md")
SOURCE_PATHS = (Path(__file__), ROOT / "scripts/run_ncmapss_loso_search.py",
                ROOT / "scripts/run_ncmapss_cqr.py",
                ROOT / "rul_fusion/data.py", ROOT / "rul_fusion/model.py",
                ROOT / "rul_fusion/runner.py", ROOT / "rul_fusion/uncertainty.py",
                PREREGISTRATION)


def should_stop(pinball, interval_score, patience=PATIENCE, window=5):
    """Stop only after both centered curves have been stale for patience."""
    if patience < 1 or window < 1 or window % 2 != 1:
        raise ValueError("patience and odd window must be positive")
    if len(pinball) != len(interval_score):
        raise ValueError("metric curves must be aligned")
    if len(pinball) < window:
        return False
    latest_center = len(pinball) - window // 2
    return all(latest_center - pick_budget(values, window) >= patience
               for values in (pinball, interval_score))


def _source_hashes():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in SOURCE_PATHS}


def _cache_path(output, identity):
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    return (output / "folds" / identity["dataset"] /
            f"{config_id(identity['config'])}_{digest}" /
            f"held_{identity['held_unit']}.json")


def _load_cache(path, identity, max_epochs):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    if (data.get("identity") != identity or not rows or
            len(rows) > max_epochs or
            [row["epoch"] for row in rows] != list(range(1, len(rows) + 1)) or
            (len(rows) < max_epochs and
             not should_stop([row["pinball_mean"] for row in rows],
                             [row["raw_interval_score"] for row in rows],
                             identity["patience"]))):
        raise RuntimeError(f"invalid completed-fold cache: {path}")
    return rows


def _save_cache(path, identity, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"identity": identity, "rows": rows}),
                         encoding="utf-8")
    temporary.replace(path)


def _train_fold(bundle, config, device, max_epochs, patience):
    set_seed(0)
    model = _build_model(bundle, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    train_loader = DataLoader(bundle.train, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(bundle.val_endpoint, batch_size=BATCH, shuffle=False)
    rows = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, condition, y, _unit, _sid in train_loader:
            x, condition, y = x.to(device), condition.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            q, regularization = model(x, condition, return_aux=True)
            loss = pinball_loss(q, y) + regularization
            loss.backward()
            optimizer.step()
        rows.append({"epoch": epoch, **_endpoint_metrics(model, val_loader, device)})
        if should_stop([row["pinball_mean"] for row in rows],
                       [row["raw_interval_score"] for row in rows], patience):
            break
    return rows


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _budget_rows(dataset, fold, held, config, rows, max_epochs):
    output = []
    stop_epoch = len(rows)
    for criterion in CRITERIA:
        budget = pick_budget([row[criterion] for row in rows])
        selected = rows[budget - 1]
        tail_sum = selected["raw_interval_score"] / 10.0
        output.append({
            "dataset": dataset, "fold": fold, "held_unit": held,
            "config": config_id(config), "criterion": criterion,
            "budget_epoch": budget, "stop_epoch": stop_epoch,
            "stop_reason": "cap" if stop_epoch == max_epochs else "joint_patience",
            "boundary_censored": (stop_epoch == max_epochs and
                                  budget > max_epochs - 10),
            "tail_pinball_sum": tail_sum,
            "q50_pinball": 3.0 * selected["pinball_mean"] - tail_sum,
            **{metric: selected[metric] for metric in METRICS},
        })
    return output


def _comparison_rows(fold_budgets):
    groups = {}
    for row in fold_budgets:
        groups.setdefault((row["dataset"], row["config"],
                           row["criterion"]), []).append(row)
    output = []
    for (dataset, config, criterion), members in sorted(groups.items()):
        output.append({
            "dataset": dataset, "config": config, "criterion": criterion,
            "n_folds": len(members),
            "median_budget_epoch": statistics.median(
                row["budget_epoch"] for row in members),
            "min_budget_epoch": min(row["budget_epoch"] for row in members),
            "max_budget_epoch": max(row["budget_epoch"] for row in members),
            "n_boundary_censored": sum(row["boundary_censored"] for row in members),
            **{metric: float(np.mean([row[metric] for row in members]))
               for metric in (*METRICS, "tail_pinball_sum", "q50_pinball")},
        })
    for dataset in sorted({row["dataset"] for row in output}):
        ranked = sorted((row for row in output if row["dataset"] == dataset),
                        key=lambda row: (row["raw_interval_score"],
                                         row["q50_rmse"]))
        for rank, row in enumerate(ranked, 1):
            row["raw_is_development_rank"] = rank
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--datasets", nargs="+", choices=("DS03", "DS08a"),
                        default=["DS03", "DS08a"])
    parser.add_argument("--dev_path", default=None, help="synthetic smoke only")
    parser.add_argument("--test_path", default=None, help="must not be loaded")
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--folds", type=int, default=0,
                        help="first K folds for synthetic smoke; 0 = all seven")
    parser.add_argument("--output_dir", default=str(
        ROOT / "results_v4/ncmapss_loso_budget_compare"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.max_epochs < 1 or args.patience < 1 or not 0 <= args.folds <= 7:
        parser.error("max_epochs and patience must be positive; folds must be 0..7")
    if args.folds == 0 and (args.max_epochs, args.patience) != (MAX_EPOCHS, PATIENCE):
        parser.error("complete comparison requires 120 epochs and patience 30")
    if args.folds == 0 and (args.datasets != ["DS03", "DS08a"] or
                             args.dev_path is not None or args.test_path is not None):
        parser.error("complete comparison requires both registered datasets")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hashes = _source_hashes()
    curves, fold_budgets, data_identities = [], [], {}
    for dataset in args.datasets:
        dev_path = Path(args.dev_path or Path(args.data_root) / "data_set" /
                        ("N-CMAPSS_DS03-012.h5" if dataset == "DS03" else
                         "N-CMAPSS_DS08a-009.h5")).resolve()
        stat = dev_path.stat()
        data_identities[dataset] = {"path": str(dev_path), "size": stat.st_size,
                                    "mtime_ns": stat.st_mtime_ns}
        common = dict(data_root=args.data_root, dataset=dataset, history=HISTORY,
                      split_seed=0, val_ratio=REGISTERED_VAL_RATIO,
                      calibration_ratio=REGISTERED_CALIBRATION_RATIO,
                      validation_protocol="fixed_cuts", registered_split=True,
                      min_calibration_units=MIN_CALIBRATION_UNITS,
                      dev_path=args.dev_path, test_path=args.test_path,
                      load_test=False)
        base = build_ncmapss_datasets(**common)
        dev_units = [int(u) for u in base.metadata["dev_units"]]
        cal_units = [int(u) for u in base.metadata["calibration_units"]]
        search_units = [u for u in dev_units if u not in cal_units]
        if (len(dev_units) != 9 or cal_units != CALIBRATION_UNITS or
                search_units != SEARCH_UNITS):
            raise RuntimeError("registered calibration units or search pool changed")
        held_units = search_units[:args.folds] if args.folds else search_units
        for fold, held in enumerate(held_units):
            train_units = [u for u in search_units if u != held]
            bundle = None
            for config in PILOT_GRID:
                identity = {"dataset": dataset, "held_unit": held,
                            "train_units": train_units, "config": config,
                            "max_epochs": args.max_epochs, "patience": args.patience,
                            "data": data_identities[dataset], "source_hashes": hashes}
                cache = _cache_path(output, identity)
                if cache.exists():
                    rows = _load_cache(cache, identity, args.max_epochs)
                    print(f"[{dataset}] reuse held={held} {config_id(config)}",
                          flush=True)
                else:
                    if bundle is None:
                        bundle = build_ncmapss_datasets(**common, unit_override={
                            "train": train_units, "val": [held],
                            "calibration": []})
                    print(f"[{dataset}] fold={fold} held={held} {config_id(config)}",
                          flush=True)
                    rows = _train_fold(bundle, config, device, args.max_epochs,
                                       args.patience)
                    _save_cache(cache, identity, rows)
                curves.extend({"dataset": dataset, "fold": fold,
                               "held_unit": held, "config": config_id(config),
                               **row} for row in rows)
                fold_budgets.extend(_budget_rows(dataset, fold, held, config,
                                                 rows, args.max_epochs))
    comparison = _comparison_rows(fold_budgets)
    _write_csv(output / "curves.csv", curves)
    _write_csv(output / "fold_budgets.csv", fold_budgets)
    _write_csv(output / "comparison.csv", comparison)
    complete = args.folds == 0 and len(args.datasets) == 2
    (output / "manifest.json").write_text(json.dumps({
        "preregistration": str(PREREGISTRATION.relative_to(ROOT)),
        "status": "complete_development_comparison" if complete else "partial_smoke",
        "test_loaded": False, "selection_frozen": False,
        "search_units": SEARCH_UNITS,
        "calibration_units_excluded": CALIBRATION_UNITS,
        "datasets": args.datasets, "configs": PILOT_GRID,
        "max_epochs": args.max_epochs, "patience": args.patience,
        "folds_limited_to": args.folds or None,
        "n_fold_config_runs": len(fold_budgets) // len(CRITERIA),
        "data": data_identities, "source_hashes": hashes,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    for row in sorted(comparison, key=lambda r: (r["dataset"],
                                                  r["raw_is_development_rank"])):
        print(f"[{row['dataset']}] rank={row['raw_is_development_rank']} "
              f"{row['config']} {row['criterion']} "
              f"budget_median={row['median_budget_epoch']} "
              f"rawIS={row['raw_interval_score']:.2f} "
              f"q50={row['q50_rmse']:.2f} PICP={row['raw_PICP']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
