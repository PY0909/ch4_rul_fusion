"""N-CMAPSS Arm B: expanded calibration cut grids on the frozen P0 checkpoint.

Pre-registration: docs/ncmapss-calibration-expansion-prereg-2026-11-05.md.
The registered split, frozen A6-Quantile checkpoint, test endpoint, alpha and
density-ratio configuration are unchanged; the only variable is the
calibration cut grid (4 cuts -> G9/G17). Windows are selected from
bundle.calibration by (unit, window-end cycle index) using the same cut
formula as the fixed-cuts protocol. Official test endpoints are evaluated
once per grid under the registered amendment and logged to test_touches.log.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rul_fusion.data import build_ncmapss_datasets  # noqa: E402
from rul_fusion.quantile_runner import _predict  # noqa: E402
from run_ncmapss_cqr import (REGISTERED_CALIBRATION_RATIO,  # noqa: E402
                             MIN_CALIBRATION_UNITS, REGISTERED_VAL_RATIO,
                             _apply, _load_quantile_model, _threshold_variants)
from run_weighted_cqr import _features, _loader  # noqa: E402

GRIDS = {
    "G9": [round(0.1 * step, 2) for step in range(1, 10)],
    "G17": [round(0.05 * step, 2) for step in range(2, 19)],
}
DIMS = {"d_model": 32, "num_heads": 2, "num_routers": 2, "dropout": 0.1}
ALPHA = 0.2


def _select_cuts(all_y, all_q, all_units, all_ids, all_cond, unit_lengths, grid):
    """Rows of the full calibration window set at the grid's cut indices."""
    wanted = {}
    for unit, length in unit_lengths.items():
        for frac in grid:
            cut = min(max(int(round(frac * (length - 1))), 0), length - 1)
            wanted[(int(unit), cut)] = True
    keep = np.array([(int(u), int(i)) in wanted
                     for u, i in zip(all_units, all_ids)], dtype=bool)
    if keep.sum() != len(wanted):
        missing = [key for key in wanted
                   if not any((int(u) == key[0] and int(i) == key[1])
                              for u, i in zip(all_units[keep], all_ids[keep]))]
        raise RuntimeError(f"cut windows missing from calibration set: {missing[:5]}")
    return (all_y[keep], all_q[keep], all_units[keep], all_ids[keep],
            all_cond[keep])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("DS03", "DS08a"))
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint_root", default=str(
        ROOT / "results_v4/ncmapss_comparison/cqr"))
    parser.add_argument("--output_dir", default=str(
        ROOT / "results_v4/ncmapss_cqr_expanded"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)

    bundle = build_ncmapss_datasets(
        args.data_root, args.dataset, history=16, split_seed=0,
        val_ratio=REGISTERED_VAL_RATIO,
        calibration_ratio=REGISTERED_CALIBRATION_RATIO,
        validation_protocol="fixed_cuts", registered_split=True,
        min_calibration_units=MIN_CALIBRATION_UNITS)
    checkpoint = Path(args.checkpoint_root) / args.dataset / f"seed_{args.seed}" / "best.pt"
    frozen = _load_quantile_model(bundle, checkpoint, device, DIMS)

    all_y, all_q, all_units, all_ids, all_cond = _predict(
        frozen, _loader(bundle.calibration), device)
    test_dataset = (bundle.test_endpoint if bundle.test_endpoint is not None
                    else bundle.test)
    test_y, test_q, test_units, test_ids, test_cond = _predict(
        frozen, _loader(test_dataset), device)
    test_feats = _features(test_q, test_cond, normalization=1.0)

    unit_lengths = {}
    for entry in bundle.metadata["calibration_endpoint"]:
        unit_lengths[int(entry["unit"])] = int(entry["length"])
    cal_units_list = sorted(unit_lengths)
    if sorted(set(int(u) for u in all_units)) != cal_units_list:
        raise RuntimeError("calibration unit set mismatch with registry")

    rows = []
    reference = (ROOT / "results_v4/ncmapss_comparison/cqr" / args.dataset
                 / f"seed_{args.seed}" / "cqr_variants_test_summary.json")
    if reference.exists():
        arm_a = json.loads(reference.read_text(encoding="utf-8"))
        for row in arm_a["rows"]:
            rows.append({"dataset": args.dataset, "arm": "A_4cuts", "grid": "4cuts",
                         "n_cal_windows": int(len(cal_units_list) * 4),
                         **{k: row[k] for k in ("variant", "threshold", "PICP",
                                                "MPIW", "IntervalScore")}})

    out_dir = Path(args.output_dir) / args.dataset / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for grid_name, grid in GRIDS.items():
        cal_y, cal_q, cal_units, cal_ids, cal_cond = _select_cuts(
            all_y, all_q, all_units, all_ids, all_cond, unit_lengths, grid)
        calibrator, weighted, _ = _threshold_variants(
            cal_y, cal_q, cal_units, cal_ids, cal_cond, test_feats, ALPHA,
            normalization=1.0)
        for name, threshold in (("global_block_max_cqr",
                                 calibrator.global_threshold), *weighted.items()):
            metrics = _apply(test_y, test_q, threshold, ALPHA)
            rows.append({"dataset": args.dataset, "arm": f"B_{grid_name}",
                         "grid": grid_name,
                         "n_cal_windows": int(len(cal_y)),
                         "variant": name, "threshold": float(threshold),
                         **metrics})
        print(f"[{args.dataset}] {grid_name}: n_cal={len(cal_y)} "
              f"thresholds global={calibrator.global_threshold:.3f} "
              f"weighted={weighted['window_weighted']:.3f}", flush=True)

    fields = sorted({key for row in rows for key in row})
    with (out_dir / "arm_comparison.csv").open("w", newline="",
                                               encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"dataset": args.dataset, "seed": args.seed,
               "preregistration": "docs/ncmapss-calibration-expansion-prereg-2026-11-05.md",
               "grids": {name: grid for name, grid in GRIDS.items()},
               "frozen_checkpoint": str(checkpoint),
               "test_units": sorted(set(int(u) for u in test_units)),
               "calibration_units": cal_units_list,
               "generated": datetime.now().isoformat(timespec="seconds"),
               "rows": rows}
    (out_dir / "arm_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    touch = ROOT / "results_v4/chapter4_main_logs/test_touches.log"
    with touch.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{summary['generated']}] ncmapss_cqr_expanded_cuts (Arm B, "
            f"pre-registered): official test endpoints evaluated once for "
            f"{args.dataset} seed {args.seed} under grids G9/G17 on the frozen "
            f"checkpoint; both grids and Arm A reported side by side, no "
            f"selection\n")
    for row in rows:
        print("{:<10s} {:<8s} {:<22s} n_cal={:>3} thr={:8.3f} PICP={:.3f} "
              "MPIW={:8.3f} IS={:8.3f}".format(
                  row["arm"], row["grid"], row["variant"],
                  row["n_cal_windows"], row["threshold"], row["PICP"],
                  row["MPIW"], row["IntervalScore"]), flush=True)


if __name__ == "__main__":
    main()
