from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import DataBundle
from .metrics import regression_metrics, rmse


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(model, loader, device):
    model.eval()
    losses, ys, ps, units, sample_ids = [], [], [], [], []
    criterion = nn.MSELoss()
    with torch.no_grad():
        for x, w, y, unit, sample_id in loader:
            x, w, y = x.to(device), w.to(device), y.to(device)
            prediction = model(x, w)
            losses.append(float(criterion(prediction, y).item()) * len(y))
            ys.extend(y.cpu().numpy().tolist())
            ps.extend(prediction.cpu().numpy().tolist())
            units.extend(np.asarray(unit).reshape(-1).tolist())
            sample_ids.extend(np.asarray(sample_id).reshape(-1).tolist())
    count = len(ys)
    return (float(sum(losses) / max(count, 1)), np.asarray(ys), np.asarray(ps),
            np.asarray(units, dtype=np.int64), np.asarray(sample_ids, dtype=np.int64))


def fit_and_evaluate(bundle: DataBundle, model: nn.Module, output_dir,
                     epochs: int = 20, batch_size: int = 64, lr: float = 1e-3,
                     weight_decay: float = 1e-4, patience: int = 10,
                     device: str = "auto", seed: int = 0, num_workers: int = 0,
                     train_drop_last: bool = False,
                     config: Optional[dict] = None,
                     optimizer_name: str = "adamw", use_scheduler: bool = True,
                     grad_clip: bool = True, evaluate_test: bool = True):
    if optimizer_name not in {"adamw", "adam"}:
        raise ValueError("optimizer_name must be 'adamw' or 'adam'")
    set_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device_obj)
    train_loader = DataLoader(bundle.train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=train_drop_last)
    val_loader = DataLoader(bundle.val, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    val_endpoint = getattr(bundle, "val_endpoint", None)
    val_endpoint_loader = (DataLoader(val_endpoint, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers)
                           if val_endpoint is not None else None)
    test_loader = DataLoader(bundle.test, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                            factor=0.5, patience=2)
                 if use_scheduler else None)
    criterion = nn.MSELoss()
    selection_criterion = "val_endpoint_rmse" if val_endpoint_loader is not None else "val_rmse_all"
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    checkpoint = output_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_total, train_count = 0.0, 0
        for x, w, y, _, _ in train_loader:
            x, w, y = x.to(device_obj), w.to(device_obj), y.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            prediction, regularization = model(x, w, return_aux=True)
            loss = criterion(prediction, y) + regularization
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_total += float(loss.detach().item()) * len(y)
            train_count += len(y)
        train_loss = train_total / max(train_count, 1)
        val_loss, val_y, val_p, _, _ = _evaluate(model, val_loader, device_obj)
        val_rmse_all = rmse(val_y, val_p) if len(val_y) else float("inf")
        if val_endpoint_loader is not None:
            _, ep_y, ep_p, _, _ = _evaluate(model, val_endpoint_loader, device_obj)
            selection = rmse(ep_y, ep_p) if len(ep_y) else float("inf")
        else:
            selection = val_rmse_all
        if scheduler is not None:
            scheduler.step(selection)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_rmse_all": val_rmse_all,
                        "val_endpoint_rmse": (selection if val_endpoint_loader is not None else None),
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"epoch:{epoch}, train_loss:{train_loss:.5f}, val_loss:{val_loss:.5f}, "
              f"val_rmse_all:{val_rmse_all:.4f}, val_endpoint_rmse:{selection:.4f}, "
              f"test_loss:deferred")
        if selection < best_val:
            best_val = selection
            best_epoch = epoch
            stale = 0
            torch.save({"model_state": model.state_dict(), "best_epoch": epoch,
                        "selection": best_val, "selection_criterion": selection_criterion},
                       checkpoint)
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping with best_ls:{best_val:.5f} ({selection_criterion})")
                break
    if not checkpoint.exists():
        torch.save({"model_state": model.state_dict(), "best_epoch": epochs,
                    "selection": best_val, "selection_criterion": selection_criterion},
                   checkpoint)
        best_epoch = epochs
    state = torch.load(checkpoint, map_location=device_obj, weights_only=False)
    model.load_state_dict(state["model_state"])
    if not evaluate_test:
        result = {"dataset": bundle.metadata.get("dataset"),
                  "best_epoch": int(best_epoch), "val_RMSE": float(best_val),
                  "selection_criterion": selection_criterion,
                  "seed": int(seed), "test_evaluated": False}
        (output_dir / "calibration_only.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss",
                                                   "val_rmse_all", "val_endpoint_rmse",
                                                   "learning_rate"])
            writer.writeheader()
            writer.writerows(history)
        return result
    test_loss, test_y, test_p, test_units, test_sample_ids = _evaluate(model, test_loader, device_obj)
    metrics = regression_metrics(test_y, test_p)
    result = {**metrics, "dataset": bundle.metadata.get("dataset"),
              "best_epoch": int(best_epoch), "val_RMSE": float(best_val),
              "selection_criterion": selection_criterion,
              "test_loss": float(test_loss), "seed": int(seed), "device": str(device_obj)}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["unit_id", "sample_id", "target_rul", "predicted_rul",
                         "absolute_error", "signed_error"])
        for unit, sample_id, y, p in zip(test_units, test_sample_ids, test_y, test_p):
            writer.writerow([int(unit), int(sample_id), float(y), float(p),
                             abs(float(p - y)), float(p - y)])
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss",
                                               "val_rmse_all", "val_endpoint_rmse",
                                               "learning_rate"])
        writer.writeheader()
        writer.writerows(history)
    run_config = {"model": model.__class__.__name__, "seed": seed,
                  "device": str(device_obj), "epochs": epochs, "batch_size": batch_size,
                  "lr": lr, "weight_decay": weight_decay, "patience": patience,
                  "train_drop_last": train_drop_last,
                  "data": bundle.metadata, "scaler": bundle.scaler.state_dict()}
    if config:
        run_config.update(config)
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"MAE: {metrics['MAE']:.2f} RMSE: {metrics['RMSE']:.2f} NASA_Score: {metrics['NASA_Score']:.2f}")
    return result
