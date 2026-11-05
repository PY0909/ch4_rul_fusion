"""Engine-level mechanism ablation for Window-Weighted CQR.

Implements the chapter plan of 2026-11-04 (§2–§7): calibration engines are
partitioned into engine-disjoint score-calibration / pseudo-target folds; the
pseudo-target endpoint is the pre-registered 0.9 life-fraction cut. Component
ablations (A1–A4), single-axis sensitivity (H1–H4) and mechanism analyses
(M1–M5) all run as CPU post-processing on the frozen seed-0 A6-Quantile
checkpoints. The official test set is never read; no result here may be used
to re-tune the production configuration (plan §2/§6 conclusion rules).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from rul_fusion.data import build_cmapss_datasets  # noqa: E402
from run_weighted_cqr import (_apply, _density_ratio, _features,  # noqa: E402
                              _load_model, _loader, _plain_threshold,
                              _weighted_threshold, _window_scores)
from rul_fusion.quantile_runner import _predict  # noqa: E402

K_FOLDS = 5
FOLD_SEED = 0
PSEUDO_FRAC = 0.9
BOOTSTRAP_DRAWS = 200
BOOTSTRAP_SEED = 20261105
RUL_BINS = ((0.0, 30.0), (30.0, 60.0), (60.0, 125.0), (125.0, float("inf")))
CONDITION_QUANTILES = (1.0 / 3.0, 2.0 / 3.0)
FEATURE_SETS = {
    "F1": ("q50",),
    "F2": ("width",),
    "F3": ("cond",),
    "F4": ("q50", "width"),
    "F5": ("q50", "width", "cond"),
}
BASE = dict(name="base", feature_spec=FEATURE_SETS["F5"], clip=20.0, lr_c=1.0,
            alpha=0.2, weighting="weighted", score="window")


def config_matrix():
    matrix = [dict(BASE),
              {**BASE, "weighting": "plain", "name": "a1_plain_weights"},
              {**BASE, "score": "block", "name": "a2_block_weighted"},
              {**BASE, "score": "block", "weighting": "plain",
               "name": "a2_block_plain"}]
    for tag in ("F1", "F2", "F3", "F4"):
        matrix.append({**BASE, "feature_spec": FEATURE_SETS[tag],
                       "name": f"a3_{tag}"})
    for clip in (5.0, 10.0, 50.0):
        matrix.append({**BASE, "clip": clip, "name": f"h1_clip{clip:g}"})
    for lr_c in (0.1, 10.0):
        matrix.append({**BASE, "lr_c": lr_c, "name": f"h2_c{lr_c:g}"})
    for alpha in (0.1, 0.3):
        matrix.append({**BASE, "alpha": alpha, "name": f"h3_alpha{alpha:g}"})
    return matrix


def fold_splits(engines, k=K_FOLDS, seed=FOLD_SEED):
    engines = np.asarray(sorted(int(e) for e in engines))
    shuffled = engines.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    return {fold: shuffled[index::k].tolist() for fold, index in
            enumerate(range(k))}


def _endpoint_arrays(bundle, checkpoint, device):
    model = _load_model(bundle, checkpoint, device)
    y, q, units, ids, cond = _predict(model, _loader(
        bundle.calibration_endpoint), device)
    cuts = bundle.metadata["calibration_endpoint"]
    pseudo_rows = {}
    for entry in cuts:
        if abs(entry["frac"] - PSEUDO_FRAC) > 1e-9:
            continue
        unit, cut = int(entry["unit"]), int(entry["cut"])
        rows = np.flatnonzero((units == unit) & (ids == cut))
        if len(rows) != 1:
            raise RuntimeError(f"pseudo-target row not unique: unit {unit} cut {cut}")
        pseudo_rows[unit] = int(rows[0])
    if len(pseudo_rows) != len(set(int(u) for u in units)):
        raise RuntimeError("missing pseudo-target rows for some calibration engines")
    return {"y": y, "q": q, "units": units, "ids": ids, "cond": cond,
            "pseudo_rows": pseudo_rows}


def _engine_weights(weights, units):
    return np.asarray([weights[units == unit].mean()
                       for unit in np.unique(units)])


def _weight_diagnostics(weights, units, clip):
    weights = np.asarray(weights, dtype=np.float64)
    engine_w = _engine_weights(weights, units)
    return {
        "weight_min": float(weights.min()),
        "weight_median": float(np.median(weights)),
        "weight_p90": float(np.quantile(weights, 0.9)),
        "weight_max": float(weights.max()),
        "clip_fraction": float(np.mean((weights <= 1.0 / clip + 1e-9)
                                       | (weights >= clip - 1e-9))),
        "max_normalized_weight": float(weights.max() / weights.sum()),
        "ess_windows": float(weights.sum() ** 2 / np.sum(weights ** 2)),
        "ess_engines": float(engine_w.sum() ** 2 / np.sum(engine_w ** 2)),
    }


def _sides(y, q, threshold):
    lower, upper = q[:, 0] - threshold, q[:, 2] + threshold
    miss_low = y < lower
    miss_high = y > upper
    return {
        "lower_miss_rate": float(miss_low.mean()),
        "upper_miss_rate": float(miss_high.mean()),
        "mean_lower_gap": float(np.mean(np.clip(lower - y, 0.0, None))),
        "mean_upper_gap": float(np.mean(np.clip(y - upper, 0.0, None))),
    }


def evaluate_config(arrays, fold_engines, cfg):
    pseudo_engines = fold_engines["pseudo"]
    score_engines = fold_engines["score"]
    unit_set = set(int(u) for u in arrays["units"])
    cal_mask = np.isin(arrays["units"], score_engines)
    tgt_idx = np.asarray([arrays["pseudo_rows"][int(u)] for u in pseudo_engines],
                         dtype=np.int64)

    cal_feats = _features(arrays["q"][cal_mask], arrays["cond"][cal_mask],
                          feature_spec=cfg["feature_spec"])
    tgt_feats = _features(arrays["q"][tgt_idx], arrays["cond"][tgt_idx],
                          feature_spec=cfg["feature_spec"])
    window_scores = _window_scores(arrays["y"], arrays["q"])[cal_mask]
    cal_units = arrays["units"][cal_mask]
    tgt_y = arrays["y"][tgt_idx]
    tgt_q = arrays["q"][tgt_idx]

    if cfg["weighting"] == "weighted":
        weights_w = _density_ratio(cal_feats, tgt_feats, clip=cfg["clip"],
                                   lr_c=cfg["lr_c"])
    else:
        weights_w = np.ones(len(window_scores))

    if cfg["score"] == "window":
        scores, weights = window_scores, weights_w
    else:
        from run_weighted_cqr import _block_max
        scores, keep = _block_max(window_scores, cal_units)
        if cfg["weighting"] == "weighted":
            weights = _engine_weights(weights_w, cal_units)
        else:
            weights = np.ones(len(scores))

    if cfg["weighting"] == "weighted":
        threshold = _weighted_threshold(scores, weights, cfg["alpha"])
    else:
        threshold = _plain_threshold(scores, cfg["alpha"])

    row = {"dataset": None, "fold": fold_engines["fold"],
           "config": cfg["name"],
           "n_score_engines": len(score_engines),
           "n_score_points": int(len(scores)),
           "n_pseudo_engines": len(pseudo_engines),
           "threshold": float(threshold)}
    row.update(_apply(tgt_y, tgt_q, threshold, alpha=cfg["alpha"]))
    row.update(_sides(tgt_y, tgt_q, threshold))
    if cfg["weighting"] == "weighted":
        row.update(_weight_diagnostics(weights_w, cal_units, cfg["clip"]))
    else:
        row.update({key: float("nan") for key in _weight_diagnostics(
            np.ones(1), np.asarray([0]), cfg["clip"])})
    audit = {"scores": scores, "weights": weights, "threshold": threshold,
             "cfg": cfg}
    return row, audit


def threshold_audit_rows(dataset, fold, audit):
    scores = np.asarray(audit["scores"], dtype=np.float64)
    weights = np.asarray(audit["weights"], dtype=np.float64)
    alpha = audit["cfg"]["alpha"]
    order = np.argsort(scores)
    cum = np.cumsum(weights[order])
    target_with = (1.0 - alpha) * (weights.sum() + weights.max())
    idx_with = int(np.searchsorted(cum, target_with, side="left"))
    target_without = (1.0 - alpha) * weights.sum()
    idx_without = int(np.searchsorted(cum, target_without, side="left"))
    rows = []
    for rank in range(len(scores)):
        rows.append({"dataset": dataset, "fold": fold,
                     "rank": rank, "score": float(scores[order][rank]),
                     "weight": float(weights[order][rank]),
                     "cum_weight": float(cum[rank]),
                     "target_with_atom": float(target_with),
                     "target_without_atom": float(target_without),
                     "threshold_with_atom": float(
                         scores[order][min(idx_with, len(order) - 1)]),
                     "threshold_without_atom": float(
                         scores[order][min(idx_without, len(order) - 1)])})
    return rows


def separability_rows(arrays, fold_engines):
    pseudo_engines = fold_engines["pseudo"]
    score_engines = fold_engines["score"]
    cal_mask = np.isin(arrays["units"], score_engines)
    tgt_idx = np.asarray([arrays["pseudo_rows"][int(u)] for u in pseudo_engines],
                         dtype=np.int64)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows = []
    for tag, spec in FEATURE_SETS.items():
        cal_feats = _features(arrays["q"][cal_mask], arrays["cond"][cal_mask],
                              feature_spec=spec)
        tgt_feats = _features(arrays["q"][tgt_idx], arrays["cond"][tgt_idx],
                              feature_spec=spec)
        x = np.vstack([cal_feats, tgt_feats])
        labels = np.concatenate([np.zeros(len(cal_feats)),
                                 np.ones(len(tgt_feats))])
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=2000,
                                               class_weight="balanced"))
        clf.fit(x, labels)
        probs = clf.predict_proba(x)[:, 1]
        coefficients = clf.named_steps["logisticregression"].coef_.reshape(-1)
        row = {"dataset": None, "fold": fold_engines["fold"], "feature_set": tag,
               "features": "+".join(spec), "auc_insample": float(
                   roc_auc_score(labels, probs)),
               "n_cal_windows": int(len(cal_feats)),
               "n_target_engines": int(len(tgt_idx))}
        for index, name in enumerate(spec):
            row[f"coef_std_{name}"] = float(coefficients[index])
        rows.append(row)
    return rows


def conditional_coverage_rows(dataset, arrays, splits, base_cfg):
    per_engine = []
    for fold, engines in splits.items():
        fold_engines = {"fold": fold, "pseudo": engines,
                        "score": [e for f, es in splits.items() if f != fold
                                  for e in es]}
        row, _ = evaluate_config(arrays, fold_engines, base_cfg)
        for unit in engines:
            idx = arrays["pseudo_rows"][int(unit)]
            per_engine.append({
                "unit": int(unit), "fold": fold, "threshold": row["threshold"],
                "y": float(arrays["y"][idx]),
                "cond": float(np.mean(arrays["cond"][idx])),
                "q10": float(arrays["q"][idx, 0]),
                "q90": float(arrays["q"][idx, 2]),
            })
    cond_values = np.asarray([entry["cond"] for entry in per_engine])
    cond_cuts = np.quantile(cond_values, CONDITION_QUANTILES)
    rows = []
    for entry in per_engine:
        lower = entry["q10"] - entry["threshold"]
        upper = entry["q90"] + entry["threshold"]
        entry["covered"] = bool(lower <= entry["y"] <= upper)
        entry["width"] = upper - lower
        entry["rul_bin"] = next(
            name for name, (lo, hi) in zip(("0-30", "30-60", "60-125", "125+"),
                                           RUL_BINS)
            if lo <= entry["y"] < hi)
        entry["cond_bin"] = ("low" if entry["cond"] <= cond_cuts[0]
                             else "mid" if entry["cond"] <= cond_cuts[1]
                             else "high")
    alpha = base_cfg["alpha"]
    for group_key in ("rul_bin", "cond_bin"):
        for group in sorted({entry[group_key] for entry in per_engine}):
            members = [e for e in per_engine if e[group_key] == group]
            covered = np.asarray([e["covered"] for e in members])
            widths = np.asarray([e["width"] for e in members])
            y = np.asarray([e["y"] for e in members])
            lower = np.asarray([e["q10"] - e["threshold"] for e in members])
            upper = np.asarray([e["q90"] + e["threshold"] for e in members])
            miss_low = y < lower
            miss_high = y > upper
            interval_score = widths + (2.0 / alpha) * (lower - y) * miss_low \
                + (2.0 / alpha) * (y - upper) * miss_high
            rows.append({"dataset": dataset, "grouping": group_key,
                         "group": group, "n_engines": len(members),
                         "PICP": float(covered.mean()),
                         "MPIW": float(widths.mean()),
                         "IntervalScore": float(interval_score.mean()),
                         "lower_miss_rate": float(miss_low.mean()),
                         "upper_miss_rate": float(miss_high.mean()),
                         "mean_y": float(y.mean())})
    return rows


def engine_sensitivity_rows(dataset, arrays, splits, base_cfg):
    rows = []
    rng_offset = 0
    for fold, engines in splits.items():
        score_engines = [e for f, es in splits.items() if f != fold
                         for e in es]
        fold_engines = {"fold": fold, "pseudo": engines, "score": score_engines}
        base_row, _ = evaluate_config(arrays, fold_engines, base_cfg)
        base_threshold = base_row["threshold"]
        # Leave-one-engine-out on the score-calibration engines.
        for engine in score_engines:
            reduced = [e for e in score_engines if e != engine]
            row, _ = evaluate_config(arrays, {"fold": fold, "pseudo": engines,
                                              "score": reduced}, base_cfg)
            rows.append({"dataset": dataset, "fold": fold,
                         "analysis": "loo", "removed_engine": int(engine),
                         "threshold": row["threshold"],
                         "threshold_delta": row["threshold"] - base_threshold,
                         "IntervalScore": row["IntervalScore"],
                         "PICP": row["PICP"]})
        # Cluster bootstrap over score-calibration engines (engine = cluster).
        rng = np.random.default_rng(BOOTSTRAP_SEED + rng_offset)
        rng_offset += 1
        boot_thresholds, boot_is = [], []
        pool = np.asarray(score_engines)
        for _ in range(BOOTSTRAP_DRAWS):
            sample = sorted({int(e) for e in rng.choice(pool, size=len(pool),
                                                        replace=True)})
            if len(sample) < 2:
                continue
            row, _ = evaluate_config(arrays, {"fold": fold, "pseudo": engines,
                                              "score": sample}, base_cfg)
            boot_thresholds.append(row["threshold"])
            boot_is.append(row["IntervalScore"])
        rows.append({"dataset": dataset, "fold": fold, "analysis": "bootstrap",
                     "removed_engine": None,
                     "threshold": float(np.mean(boot_thresholds)),
                     "threshold_delta": float(np.std(boot_thresholds)),
                     "IntervalScore": float(np.mean(boot_is)),
                     "PICP": float("nan"),
                     "bootstrap_threshold_p5": float(
                         np.quantile(boot_thresholds, 0.05)),
                     "bootstrap_threshold_p95": float(
                         np.quantile(boot_thresholds, 0.95))})
    return rows


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint_root", default=str(
        ROOT / "results_v4/chapter4_main/quantile_cqr/global"))
    parser.add_argument("--output_root", default=str(
        ROOT / "results_v4/weighted_cqr_ablation"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    matrix = config_matrix()
    fold_rows, weight_rows, audit_rows, sep_rows, cond_rows, sens_rows = \
        [], [], [], [], [], []
    fold_registry = {}
    for dataset in ("FD002", "FD004"):
        bundle = build_cmapss_datasets(
            args.data_root, dataset, history=50, rul_cap=125, split_seed=0,
            calibration_ratio=0.5, scaler_mode="grouped",
            validation_protocol="fixed_cuts")
        checkpoint = Path(args.checkpoint_root) / dataset / "seed_0" / "best.pt"
        arrays = _endpoint_arrays(bundle, checkpoint, device)
        engines = sorted(set(int(u) for u in arrays["units"]))
        splits = fold_splits(engines)
        fold_registry[dataset] = splits
        print(f"[{dataset}] calibration engines={len(engines)} "
              f"folds={K_FOLDS} pseudo_frac={PSEUDO_FRAC}", flush=True)
        for fold, pseudo in splits.items():
            fold_engines = {"fold": fold, "pseudo": pseudo,
                            "score": [e for f, es in splits.items() if f != fold
                                      for e in es]}
            for cfg in matrix:
                row, audit = evaluate_config(arrays, fold_engines, cfg)
                row["dataset"] = dataset
                fold_rows.append(row)
                if cfg["weighting"] == "weighted":
                    weight_rows.append({"dataset": dataset, "fold": fold,
                                        "config": cfg["name"], **{
                        k: v for k, v in row.items()
                        if k.startswith(("weight_", "clip_", "max_norm",
                                         "ess_"))}})
                if cfg["name"] == "base":
                    audit_rows.extend(threshold_audit_rows(dataset, fold, audit))
            sep_rows.extend(separability_rows(arrays, fold_engines))
        for row in sep_rows:
            if row["dataset"] is None:
                row["dataset"] = dataset
        cond_rows.extend(conditional_coverage_rows(dataset, arrays, splits,
                                                   BASE))
        sens_rows.extend(engine_sensitivity_rows(dataset, arrays, splits, BASE))
        print(f"[{dataset}] configs={len(matrix)} folds={K_FOLDS} done",
              flush=True)

    write_csv(output / "folds.csv", fold_rows)
    write_csv(output / "weights.csv", weight_rows)
    write_csv(output / "threshold_audit.csv", audit_rows)
    write_csv(output / "m1_separability.csv", sep_rows)
    write_csv(output / "conditional_coverage.csv", cond_rows)
    write_csv(output / "m5_engine_sensitivity.csv", sens_rows)

    repo_root = ROOT.parent
    try:
        commit = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                                check=True, capture_output=True,
                                text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    config_out = {
        "plan": "chapter4-followup-ablation-mechanism-plan-2026-11-04 §2-§7",
        "k_folds": K_FOLDS, "fold_seed": FOLD_SEED,
        "pseudo_target_frac": PSEUDO_FRAC,
        "rul_bins": [[lo, None if np.isinf(hi) else hi] for lo, hi in RUL_BINS],
        "condition_quantiles": list(CONDITION_QUANTILES),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "config_matrix": [{k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in cfg.items()} for cfg in matrix],
        "fold_registry": fold_registry,
        "notes": [
            "features use the production normalization path (scale-invariant "
            "under StandardScaler); score-cal windows are the four fixed-cut "
            "endpoint windows per engine; pseudo-target is the 0.9 cut.",
            "official test is never read; results are mechanism diagnostics "
            "and must not re-tune the production configuration.",
        ],
        "code_commit": commit,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    (output / "config.json").write_text(json.dumps(config_out, indent=2),
                                        encoding="utf-8")
    print(f"outputs -> {output}", flush=True)


if __name__ == "__main__":
    main()
