from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .corruptions import inject_bias, inject_drift, inject_missing, inject_noise
from .data import WindowDataset
from .metrics import regression_metrics, rmse
from .uncertainty import (CQRCalibrator, ConditionStageQualityGrouper,
                          interval_metrics, pinball_loss)
from .runner import set_seed


def _predict(model, loader, device):
    model.eval()
    ys, qs, units, sample_ids, condition_means = [], [], [], [], []
    with torch.no_grad():
        for x, w, y, unit, sample_id in loader:
            quantiles = model(x.to(device), w.to(device)).detach().cpu().numpy()
            ys.extend(y.numpy().tolist())
            qs.extend(quantiles.tolist())
            units.extend(np.asarray(unit).reshape(-1).tolist())
            sample_ids.extend(np.asarray(sample_id).reshape(-1).tolist())
            condition_means.extend(w.numpy().mean(axis=1).tolist())
    return (np.asarray(ys, dtype=np.float64), np.asarray(qs, dtype=np.float64),
            np.asarray(units, dtype=np.int64), np.asarray(sample_ids, dtype=np.int64),
            np.asarray(condition_means, dtype=np.float64))


def _dataset_loader(dataset, batch_size, shuffle=False, num_workers=0):
    if dataset is None:
        raise ValueError("dataset is required for quantile training/evaluation")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)


def _git_provenance():
    """Return reproducibility metadata without making git a runtime dependency."""
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True, capture_output=True, text=True).stdout.strip())
        return {"code_commit": commit, "code_worktree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"code_commit": "unknown", "code_worktree_dirty": None}


def _fit_grouper(bundle, model, train_loader, device):
    ys, qs, _, _, conditions = _predict(model, train_loader, device)
    grouper = ConditionStageQualityGrouper(random_state=int(bundle.metadata["split_seed"]))
    grouper.fit(conditions, ys)
    return grouper


def _aggregate_calibration_blocks(y, q, units, sample_ids, conditions, groups=None):
    """Reduce correlated endpoint cuts to one conservative score per unit."""
    scores = CQRCalibrator._scores(y, q[:, 0], q[:, 2])
    selected = []
    for unit in np.unique(units):
        indices = np.flatnonzero(units == unit)
        selected.append(indices[int(np.argmax(scores[indices]))])
    selected = np.asarray(selected, dtype=np.int64)
    reduced_groups = groups[selected] if groups is not None else None
    return (y[selected], q[selected], units[selected], sample_ids[selected],
            conditions[selected], reduced_groups, scores[selected])


def _corrupt_dataset(dataset, kind, severity, seed):
    if kind is None:
        return dataset, {}
    rng = np.random.default_rng(seed)
    functions = {"missing": inject_missing, "noise": inject_noise,
                 "bias": inject_bias, "drift": inject_drift}
    if kind not in functions:
        raise ValueError(f"unknown corruption {kind}")
    x, metadata = functions[kind](dataset.x, float(severity), rng)
    samples = [(x[i], dataset.w[i], float(dataset.y[i]), int(dataset.units[i]),
                int(dataset.sample_ids[i])) for i in range(len(dataset))]
    return WindowDataset(samples), {"kind": kind, "severity": float(severity), **metadata}


def _write_predictions(path, y, q, lower, upper, units, sample_ids):
    covered = (y >= lower) & (y <= upper)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["unit_id", "sample_id", "target_rul", "q10", "q50", "q90",
                         "lower", "upper", "covered", "interval_width"])
        for row in zip(units, sample_ids, y, q[:, 0], q[:, 1], q[:, 2], lower, upper,
                       covered, upper - lower):
            writer.writerow([int(row[0]), int(row[1]), *[float(x) for x in row[2:8]],
                             bool(row[8]), float(row[9])])


def fit_quantile_and_calibrate(bundle, model, output_dir, *, epochs=41,
                               batch_size=100, lr=1e-3, weight_decay=0.0,
                               patience=41, alpha=0.2, calibration_mode="global",
                               min_group_size=20, shrink_tau=20.0, seed=0,
                               device="auto", num_workers=0, corruption=None,
                               corruption_severity=None, config=None,
                               evaluate_test=True):
    if calibration_mode not in {"global", "hierarchical"}:
        raise ValueError("calibration_mode must be global or hierarchical")
    if abs(float(alpha) - 0.2) > 1e-8:
        raise ValueError("alpha must be 0.2 for the fixed q0.1/q0.5/q0.9 head")
    if corruption is not None and corruption_severity is None:
        raise ValueError("corruption_severity is required when corruption is enabled")
    if bundle.calibration_endpoint is None:
        raise ValueError("bundle must be built with calibration_ratio")
    set_seed(seed)
    device_obj = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                              if device == "auto" else device)
    model = model.to(device_obj)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = _dataset_loader(bundle.train, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = _dataset_loader(bundle.val_endpoint, batch_size, num_workers=num_workers)
    calibration_loader = _dataset_loader(bundle.calibration_endpoint, batch_size,
                                         num_workers=num_workers)
    test_dataset = bundle.test_endpoint if bundle.test_endpoint is not None else bundle.test
    test_dataset, corruption_metadata = _corrupt_dataset(
        test_dataset, corruption, corruption_severity if corruption_severity is not None else 0.0,
        seed)
    test_loader = _dataset_loader(test_dataset, batch_size, num_workers=num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    checkpoint = output_dir / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, w, y, _, _ in train_loader:
            x, w, y = x.to(device_obj), w.to(device_obj), y.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            q, regularization = model(x, w, return_aux=True)
            loss = pinball_loss(q, y) + regularization
            loss.backward()
            optimizer.step()
            total += float(loss.detach().item()) * len(y)
            count += len(y)
        train_loss = total / max(count, 1)
        val_y, val_q, _, _, _ = _predict(model, val_loader, device_obj)
        val_loss = float(np.mean((val_q[:, 1] - val_y) ** 2))
        val_rmse = rmse(val_y, val_q[:, 1])
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_q50_rmse": val_rmse})
        print(f"epoch:{epoch}, train_loss:{train_loss:.5f}, val_loss:{val_loss:.5f}, "
              f"val_q50_rmse:{val_rmse:.4f}, test_loss:deferred", flush=True)
        if val_rmse < best:
            best, best_epoch, stale = val_rmse, epoch, 0
            torch.save({"model_state": model.state_dict(), "best_epoch": epoch,
                        "selection": best, "selection_criterion": "val_q50_rmse"}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    if not checkpoint.exists():
        raise RuntimeError("quantile training did not produce a checkpoint")
    state = torch.load(checkpoint, map_location=device_obj, weights_only=False)
    model.load_state_dict(state["model_state"])

    grouper = _fit_grouper(bundle, model, train_loader, device_obj)
    cal_y, cal_q, cal_units, cal_ids, cal_conditions = _predict(model, calibration_loader, device_obj)
    groups = (grouper.groups(cal_conditions, cal_q[:, 1])
              if calibration_mode == "hierarchical" else None)
    cal_y, cal_q, cal_units, cal_ids, cal_conditions, groups, cal_scores = \
        _aggregate_calibration_blocks(cal_y, cal_q, cal_units, cal_ids,
                                      cal_conditions, groups)
    calibrator = CQRCalibrator(alpha=alpha, min_group_size=min_group_size,
                               shrink_tau=shrink_tau)
    calibrator.fit(cal_y, cal_q[:, 0], cal_q[:, 2], groups=groups)

    if not evaluate_test:
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "epoch", "train_loss", "val_loss", "val_q50_rmse"])
            writer.writeheader()
            writer.writerows(history)
        calibrator.save(output_dir / "calibrator.json")
        (output_dir / "grouping.json").write_text(json.dumps(grouper.state_dict(), indent=2),
                                                   encoding="utf-8")
        run_config = {"model": model.__class__.__name__, "seed": seed,
                      "epochs": epochs, "batch_size": batch_size, "lr": lr,
                      "weight_decay": weight_decay, "alpha": alpha,
                      "calibration_mode": calibration_mode,
                      "min_group_size": min_group_size, "shrink_tau": shrink_tau,
                      "data": bundle.metadata, "test_evaluated": False}
        if config:
            run_config.update(config)
        (output_dir / "config.json").write_text(json.dumps(run_config, indent=2),
                                                 encoding="utf-8")
        with (output_dir / "calibration_scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["unit_id", "sample_id", "score", "group"])
            writer.writerows(zip(cal_units, cal_ids, cal_scores,
                                 groups if groups is not None else ["global"] * len(cal_units)))
        result = {"dataset": bundle.metadata.get("dataset"), "best_epoch": int(best_epoch),
                  "val_q50_RMSE": float(best), "alpha": alpha,
                  "calibration_mode": calibration_mode, "seed": int(seed),
                  "n_calibration": int(len(cal_y)), "calibration_threshold": calibrator.global_threshold,
                  "test_evaluated": False}
        (output_dir / "calibration_only.json").write_text(json.dumps(result, indent=2),
                                                           encoding="utf-8")
        return result

    test_y, test_q, test_units, test_ids, test_conditions = _predict(model, test_loader, device_obj)
    test_groups = (grouper.groups(test_conditions, test_q[:, 1])
                   if calibration_mode == "hierarchical" else None)
    lower, upper = calibrator.transform(test_q[:, 0], test_q[:, 2], test_groups)
    point = regression_metrics(test_y, test_q[:, 1])
    if str(bundle.metadata.get("dataset", "")).startswith("DS"):
        point.pop("NASA_Score", None)
    point_interval = interval_metrics(test_y, lower, upper, alpha=alpha)
    result = {**point, **point_interval, "dataset": bundle.metadata.get("dataset"),
              "best_epoch": int(best_epoch), "val_q50_RMSE": float(best),
              "alpha": alpha, "calibration_mode": calibration_mode, "seed": int(seed),
              "device": str(device_obj), "n_calibration": int(len(cal_y)),
              "corruption": corruption_metadata}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "val_q50_rmse"])
        writer.writeheader()
        writer.writerows(history)
    _write_predictions(output_dir / "predictions.csv", test_y, test_q, lower, upper,
                       test_units, test_ids)
    with (output_dir / "calibration_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["unit_id", "sample_id", "score", "group"])
        writer.writerows(zip(cal_units, cal_ids, cal_scores,
                             groups if groups is not None else ["global"] * len(cal_units)))
    calibrator.save(output_dir / "calibrator.json")
    (output_dir / "grouping.json").write_text(json.dumps(grouper.state_dict(), indent=2),
                                               encoding="utf-8")
    run_config = {"model": model.__class__.__name__, "seed": seed,
                  "epochs": epochs, "batch_size": batch_size, "lr": lr,
                  "weight_decay": weight_decay, "alpha": alpha,
                  "calibration_mode": calibration_mode, "data": bundle.metadata}
    run_config["corruption"] = corruption_metadata
    if config:
        run_config.update(config)
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "dataset": bundle.metadata.get("dataset"),
        "split_seed": bundle.metadata.get("split_seed"),
        "train_units": bundle.metadata.get("train_units", []),
        "val_units": bundle.metadata.get("val_units", []),
        "calibration_units": bundle.metadata.get("calibration_units", []),
        "test_units": bundle.metadata.get("test_units", []),
        "alpha": alpha,
        "calibration_mode": calibration_mode,
        "n_calibration": int(len(cal_y)),
        "corruption": corruption_metadata,
        "scaler_mode": bundle.metadata.get("scaler_mode"),
        "scaler_fit_scope": "train_units_only",
        "scaler_fit_units": bundle.metadata.get("train_units", []),
        "test_evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_evaluated_once": True,
    }
    manifest.update(_git_provenance())
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                                encoding="utf-8")
    print(f"Q50_MAE: {point['MAE']:.2f} Q50_RMSE: {point['RMSE']:.2f} "
          f"PICP: {point_interval['PICP']:.4f} MPIW: {point_interval['MPIW']:.4f} "
          f"IntervalScore: {point_interval['IntervalScore']:.4f}", flush=True)
    return result
