"""Cost-optimal point estimate for an asymmetric evaluation cost.

The NASA score penalises late predictions with exp(e/10) and early ones with
exp(-e/13), so the point estimate that minimises expected cost is not the
conditional median q50.  For a predictive distribution F and an asymmetric cost
C(e), the minimiser is the cost-weighted quantile

    q* = argmin_a  E_Y[C(Y - a)],

which a quantile head can evaluate with its own predicted quantiles: no labels
and no test data are required.  A scalar head (MSE-trained) can only return the
mean, which is why this adjustment is specific to distributional heads.

The shift is solved per sample on a fixed quantile grid and clipped to a
registered range so a badly calibrated tail cannot move a prediction far.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rul_fusion.metrics import nasa_score, regression_metrics  # noqa: E402

LEVELS = np.asarray([0.1, 0.5, 0.9])
GRID = np.linspace(1e-4, 1.0 - 1e-4, 401)


def _cost(err):
    return np.where(err >= 0.0, np.exp(err / 10.0) - 1.0,
                    np.exp(-err / 13.0) - 1.0)


def expected_cost(q, q10, q50, q90, levels=LEVELS, grid=GRID):
    """E[C(Y - q)] under the predictive distribution implied by the quantiles."""
    nodes = np.stack([q10, q50, q90], axis=1)
    quantiles = np.stack([np.interp(grid, levels, row) for row in nodes]).T
    return np.trapezoid(_cost(quantiles - np.asarray(q)[None, :]), grid, axis=0)


def cost_optimal_shift(q10, q50, q90, search=np.arange(-8.0, 8.01, 0.25)):
    """Per-sample shift minimising expected cost; median of the grid is neutral."""
    base = expected_cost(q50, q10, q50, q90)
    best_shift = np.zeros_like(np.asarray(q50, dtype=np.float64))
    best_cost = base.copy()
    for shift in search:
        current = expected_cost(q50 + shift, q10, q50, q90)
        better = current < best_cost
        best_cost = np.where(better, current, best_cost)
        best_shift = np.where(better, shift, best_shift)
    return best_shift


def apply_cost_optimal(q10, q50, q90, search=np.arange(-8.0, 8.01, 0.25)):
    shift = cost_optimal_shift(q10, q50, q90, search)
    return np.asarray(q50, dtype=np.float64) + shift, shift


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", action="append", required=True,
                        metavar="DATASET=PATH", help="e.g. FD002=path/to/predictions.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    import csv

    report = {}
    for spec in args.predictions:
        dataset, _, path = spec.partition("=")
        with Path(path).open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        y = np.asarray([float(r["target_rul"]) for r in rows])
        q10 = np.asarray([float(r["q10"]) for r in rows])
        q50 = np.asarray([float(r["q50"]) for r in rows])
        q90 = np.asarray([float(r["q90"]) for r in rows])
        adjusted, shift = apply_cost_optimal(q10, q50, q90)
        before = regression_metrics(y, q50)
        after = regression_metrics(y, adjusted)
        report[dataset] = {
            "baseline_q50": before, "cost_optimal": after,
            "shift_mean": float(shift.mean()),
            "shift_min": float(shift.min()), "shift_max": float(shift.max()),
        }
        print(f"{dataset}: NASA {before['NASA_Score']:.1f} -> "
              f"{after['NASA_Score']:.1f}  (MAE {before['MAE']:.2f} -> "
              f"{after['MAE']:.2f}, RMSE {before['RMSE']:.2f} -> "
              f"{after['RMSE']:.2f}, mean shift {shift.mean():+.2f})", flush=True)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
