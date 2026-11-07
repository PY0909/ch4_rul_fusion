"""Diagnose frozen N-CMAPSS A6-Quantile checkpoints on validation endpoints.

Uses the registered three validation units only. Calibration and test labels
are not used to score or select checkpoints. See the 2026-11-06 LOSO amendment.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from rul_fusion.data import build_ncmapss_datasets  # noqa: E402
from rul_fusion.quantile_runner import _predict  # noqa: E402
from run_ncmapss_cqr import (MIN_CALIBRATION_UNITS,  # noqa: E402
                             REGISTERED_CALIBRATION_RATIO,
                             REGISTERED_VAL_RATIO, _load_quantile_model)

TAUS = np.asarray([0.1, 0.5, 0.9])


def _checkpoints(dataset):
    baseline = ROOT / "results_v4/ncmapss_comparison/cqr" / dataset / "seed_0/best.pt"
    tuning = ROOT / "results_v4/ncmapss_a6_dev_tuning"
    return [("p0", baseline, 0.1)] + [
        (name, tuning / name / dataset / "seed_0/best.pt", dropout)
        for name, dropout in (("lr3e4", 0.1), ("wd3e3", 0.1),
                              ("dropout02", 0.2))]


def _endpoint_rows(name, y, q, units, sample_ids, cuts):
    lookup = {(int(cut["unit"]), int(cut["cut"])): cut["frac"]
              for cut in cuts}
    errors = y[:, None] - q
    pinball = np.maximum(TAUS * errors, (TAUS - 1.0) * errors)
    residual = q[:, 1] - y
    lower_miss = np.maximum(q[:, 0] - y, 0.0)
    upper_miss = np.maximum(y - q[:, 2], 0.0)
    width = q[:, 2] - q[:, 0]
    score = width + 10.0 * (lower_miss + upper_miss)
    rows = []
    for i, (unit, sample_id) in enumerate(zip(units, sample_ids)):
        key = (int(unit), int(sample_id))
        rows.append({
            "checkpoint": name, "unit_id": key[0], "sample_id": key[1],
            "life_fraction": float(lookup[key]), "target_rul": float(y[i]),
            "q10": float(q[i, 0]), "q50": float(q[i, 1]),
            "q90": float(q[i, 2]),
            "pinball_q10": float(pinball[i, 0]),
            "pinball_q50": float(pinball[i, 1]),
            "pinball_q90": float(pinball[i, 2]),
            "pinball_mean": float(pinball[i].mean()),
            "q50_error": float(residual[i]),
            "q50_squared_error": float(residual[i] ** 2),
            "raw_covered": int(lower_miss[i] == 0 and upper_miss[i] == 0),
            "raw_width": float(width[i]),
            "raw_interval_score": float(score[i]),
        })
    return rows


def _summary_rows(checkpoint, endpoint_rows):
    groups = [("overall", "all", endpoint_rows)]
    for key, field in (("unit", "unit_id"),
                       ("life_fraction", "life_fraction")):
        for value in sorted({row[field] for row in endpoint_rows}):
            groups.append((key, value,
                           [row for row in endpoint_rows if row[field] == value]))
    summaries = []
    for kind, group_id, rows in groups:
        residual = np.asarray([r["q50_error"] for r in rows])
        summaries.append({
            "checkpoint": checkpoint, "group_kind": kind, "group_id": group_id,
            "n_endpoints": len(rows),
            "n_units": len({r["unit_id"] for r in rows}),
            "q50_bias": float(residual.mean()),
            "q50_residual_variance": float(residual.var()),
            "q50_rmse": float(np.sqrt(np.mean(residual ** 2))),
            **{field: float(np.mean([r[field] for r in rows]))
               for field in ("pinball_q10", "pinball_q50", "pinball_q90",
                             "pinball_mean", "raw_covered", "raw_width",
                             "raw_interval_score")},
        })
    return summaries


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("DS03", "DS08a"), required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dev_path", default=None)
    parser.add_argument("--test_path", default=None)
    parser.add_argument("--checkpoint", action="append", default=None,
                        help="NAME=PATH; repeat for multiple frozen checkpoints")
    parser.add_argument("--output_dir", default=str(
        ROOT / "results_v4/ncmapss_frozen_diagnostics"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if args.checkpoint:
        checkpoints = []
        for entry in args.checkpoint:
            if "=" not in entry:
                parser.error("checkpoint must be NAME=PATH")
            name, path = entry.split("=", 1)
            if not name or not path:
                parser.error("checkpoint must be NAME=PATH")
            checkpoints.append((name, Path(path), 0.1))
    else:
        checkpoints = _checkpoints(args.dataset)
    if len({name for name, _, _ in checkpoints}) != len(checkpoints):
        parser.error("checkpoint names must be unique")
    for name, path, _ in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(f"{name} checkpoint missing: {path}")

    bundle = build_ncmapss_datasets(
        args.data_root, args.dataset, history=16, split_seed=0,
        val_ratio=REGISTERED_VAL_RATIO,
        calibration_ratio=REGISTERED_CALIBRATION_RATIO,
        validation_protocol="fixed_cuts", registered_split=True,
        min_calibration_units=MIN_CALIBRATION_UNITS,
        dev_path=args.dev_path, test_path=args.test_path, load_test=False)
    if bundle.metadata["val_units"] != [3, 5, 6]:
        raise RuntimeError("registered validation units changed")
    loader = DataLoader(bundle.val_endpoint, batch_size=32, shuffle=False)
    endpoints, summaries = [], []
    for name, path, dropout in checkpoints:
        dims = {"d_model": 32, "num_heads": 2, "num_routers": 2,
                "dropout": dropout}
        model = _load_quantile_model(bundle, path, device, dims)
        y, q, units, sample_ids, _ = _predict(model, loader, device)
        rows = _endpoint_rows(name, y, q, units, sample_ids,
                              bundle.metadata["val_endpoint"])
        endpoints.extend(rows)
        summaries.extend(_summary_rows(name, rows))

    output = Path(args.output_dir) / args.dataset
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "endpoints.csv", endpoints)
    _write_csv(output / "summary.csv", summaries)
    (output / "manifest.json").write_text(json.dumps({
        "dataset": args.dataset, "endpoint": "registered_validation_fixed_cuts",
        "val_units": bundle.metadata["val_units"],
        "calibration_units_excluded_from_scoring": bundle.metadata["calibration_units"],
        "test_loaded": False,
        "checkpoints": {name: str(path) for name, path, _ in checkpoints},
    }, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
