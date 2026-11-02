from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _enable_dropout(module):
    # MultiheadAttention applies its dropout through functional code, so its
    # module-level training flag must be enabled explicitly as well.
    if isinstance(module, nn.Dropout) or (
            isinstance(module, nn.MultiheadAttention) and module.dropout > 0.0):
        module.train()


@torch.no_grad()
def predict_mc_dropout(model, loader, device, samples: int = 30):
    if samples < 2:
        raise ValueError("samples must be at least 2")
    device = torch.device(device)
    model.eval()
    predictions = []
    target = units = sample_ids = None
    for _ in range(samples):
        model.eval()
        model.apply(_enable_dropout)
        current, ys, us, ids = [], [], [], []
        for x, w, y, unit, sample_id in loader:
            current.extend(model(x.to(device), w.to(device)).detach().cpu().reshape(-1).tolist())
            ys.extend(y.reshape(-1).tolist())
            us.extend(np.asarray(unit).reshape(-1).tolist())
            ids.extend(np.asarray(sample_id).reshape(-1).tolist())
        predictions.append(np.asarray(current, dtype=np.float64))
        if target is None:
            target = np.asarray(ys, dtype=np.float64)
            units = np.asarray(us, dtype=np.int64)
            sample_ids = np.asarray(ids, dtype=np.int64)
    model.eval()
    draws = np.stack(predictions, axis=0)
    return {"target": target, "mean": draws.mean(axis=0),
            "lower": np.quantile(draws, 0.1, axis=0),
            "upper": np.quantile(draws, 0.9, axis=0),
            "units": units, "sample_ids": sample_ids, "draws": draws}
