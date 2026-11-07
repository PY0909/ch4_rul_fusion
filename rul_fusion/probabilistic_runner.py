from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import ndtr
from scipy.stats import t as student_t
from torch.utils.data import DataLoader

from .metrics import regression_metrics, rmse
from .mc_dropout import predict_mc_dropout
from .probabilistic import (BayesianCooperativeTransformerRUL,
                            EvidentialNIGRUL, HeteroscedasticRUL,
                            LSTMProbRUL, MSAUQRUL, PAPER_METHOD_DEFAULTS,
                            TCNTransformerMCDropoutRUL, bcpt_channel_loss,
                            gaussian_nll, nig_evidential_loss)
from .runner import set_seed

MC_DRAWS_KINDS = {"tcn_transformer_mc", "msauq"}
BCPT_KIND = "bcpt"
POINT_MSE_KINDS = {"tcn_transformer_mc", "msauq"}
Z_80 = 1.2815515655446004  # Phi^-1(0.9) for the 80% nominal band


def _predict(model, loader, device):
    model.eval()
    targets, units, sample_ids, params = [], [], [], []
    elapsed = 0.0
    with torch.no_grad():
        for x, w, y, unit, sample_id in loader:
            x, w = x.to(device), w.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = model(x, w)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - start
            params.append(output.cpu().numpy())
            targets.extend(y.numpy().tolist())
            units.extend(np.asarray(unit).reshape(-1).tolist())
            sample_ids.extend(np.asarray(sample_id).reshape(-1).tolist())
    return (np.asarray(targets, dtype=np.float64), np.concatenate(params),
            np.asarray(units, dtype=np.int64), np.asarray(sample_ids, dtype=np.int64), elapsed)


EDL_PAPER_SENSOR_IDS = (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 17)


def _edl_sensor_positions(metadata, sensor_dim):
    original_ids = metadata.get("sensor_indices")
    if original_ids is None:
        return tuple(range(sensor_dim))
    missing = sorted(set(EDL_PAPER_SENSOR_IDS) - set(original_ids))
    if missing:
        raise ValueError(f"dataset sensor selection lacks paper features: {missing}")
    return tuple(original_ids.index(sensor_id) for sensor_id in EDL_PAPER_SENSOR_IDS)


def _model(kind, sensor_dim, condition_dim, dropout, window_size,
           edl_hidden_dim=128, edl_sensor_indices=None):
    if kind in {"hnn", "ensemble"}:
        return HeteroscedasticRUL(sensor_dim, condition_dim, dropout=dropout,
                                  window_size=window_size)
    if kind == "edl_nig":
        return EvidentialNIGRUL(sensor_dim, condition_dim, dropout=dropout,
                                hidden_dim=edl_hidden_dim,
                                sensor_indices=edl_sensor_indices)
    if kind == "lstm_prob":
        return LSTMProbRUL(sensor_dim, condition_dim, dropout=dropout)
    if kind == "tcn_transformer_mc":
        return TCNTransformerMCDropoutRUL(sensor_dim, condition_dim,
                                          window_size=window_size, dropout=dropout)
    if kind == "msauq":
        return MSAUQRUL(sensor_dim, condition_dim, dropout=dropout)
    if kind == BCPT_KIND:
        return BayesianCooperativeTransformerRUL(sensor_dim, condition_dim,
                                                 window_size=window_size,
                                                 dropout=dropout)
    raise ValueError(f"unsupported probabilistic method: {kind}")


def _loss(kind, prediction, y, evidence_weight):
    if kind in {"hnn", "ensemble", "lstm_prob"}:
        return gaussian_nll(prediction, y)
    if kind in POINT_MSE_KINDS:
        return F.mse_loss(prediction.reshape(-1), y.reshape(-1))
    return nig_evidential_loss(prediction, y, evidence_weight)


def _bayesrul_weights_init(module):
    """Initialization copied from arthurviens/bayesrul weights_init()."""
    if isinstance(module, nn.Conv1d):
        torch.nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight)


def _mean(kind, params):
    return params[:, 0]


def _normal_crps(y, mean, scale):
    z = (y - mean) / scale
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return scale * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def _expected_abs_normal(mean, scale):
    z = mean / scale
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return scale * math.sqrt(2.0 / math.pi) * np.exp(-0.5 * z * z) + mean * (2 * ndtr(z) - 1)


def _mixture_crps(y, means, scales):
    first = np.mean(_expected_abs_normal(means - y[:, None], scales), axis=1)
    pair_mean = means[:, :, None] - means[:, None, :]
    pair_scale = np.sqrt(scales[:, :, None] ** 2 + scales[:, None, :] ** 2)
    second = 0.5 * np.mean(_expected_abs_normal(pair_mean, pair_scale), axis=(1, 2))
    return first - second


def _edl_distribution(params):
    gamma, nu, alpha, beta = params.T
    df = 2.0 * alpha
    scale = np.sqrt(beta * (nu + 1.0) / (alpha * nu))
    return gamma, df, scale


def _edl_crps(y, mean, df, scale, seed, samples=2048):
    rng = np.random.default_rng(seed)
    draws = mean[:, None] + scale[:, None] * rng.standard_t(df[:, None], size=(len(y), samples))
    first = np.mean(np.abs(draws - y[:, None]), axis=1)
    ordered = np.sort(draws, axis=1)
    coeff = (2.0 * np.arange(1, samples + 1) - samples - 1.0)
    second = np.sum(ordered * coeff[None, :], axis=1) / (samples ** 2)
    return first - second


def kde_quantiles(draws: np.ndarray, quantiles=(0.1, 0.9), grid_size: int = 512):
    """B6 predictive quantiles from a Gaussian-kernel KDE with Scott bandwidth.

    The paper constructs the predictive PDF by KDE over the MC draws and reads
    CIs from it; the bandwidth value is unstated, so Scott's rule is the
    registered choice. ``draws`` has shape [n_draws, n_samples].
    """
    draws = np.asarray(draws, dtype=np.float64)
    n = draws.shape[0]
    if n < 2:
        raise ValueError("KDE needs at least two draws")
    bandwidth = draws.std(axis=0, ddof=1) * n ** (-1.0 / 5.0)
    bandwidth = np.maximum(bandwidth, 1e-6)
    lo = draws.min(axis=0) - 3.0 * bandwidth
    hi = draws.max(axis=0) + 3.0 * bandwidth
    grid = np.linspace(lo, hi, grid_size)                        # [grid, n]
    step = (hi - lo) / (grid_size - 1)
    diff = grid[:, None, :] - draws[None, :, :]                  # [grid, n, n_s]
    log_pdf = (-0.5 * (diff / bandwidth[None, None, :]) ** 2
               - np.log(bandwidth[None, None, :]) - 0.5 * math.log(2.0 * math.pi))
    density = np.exp(log_pdf).mean(axis=1)                       # [grid, n_s]
    cdf = np.cumsum(density, axis=0) * step[None, :]
    cdf = cdf / np.maximum(cdf[-1], 1e-12)
    out = [np.empty(draws.shape[1]) for _ in quantiles]
    for col in range(draws.shape[1]):
        for index, q in enumerate(quantiles):
            out[index][col] = np.interp(q, cdf[:, col], grid[:, col])
    return np.stack(out, axis=0)


def crps_from_draws(y: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Monte-Carlo CRPS (energy form) from predictive draws [n_draws, n]."""
    first = np.abs(draws - y[None, :]).mean(axis=0)
    pairwise = np.abs(draws[:, :, None] - draws[:, None, :]).mean(axis=(0, 1))
    return first - 0.5 * pairwise


def gaussian_fit_outputs(target: np.ndarray, draws: np.ndarray,
                         quantiles: np.ndarray):
    """MC-draw outputs under the papers' Gaussian assumption: mean/std draws
    NLL, analytic-normal CRPS, and externally supplied q10/q90 bounds."""
    mean = draws.mean(axis=0)
    scale = np.maximum(draws.std(axis=0, ddof=1), 1e-6)
    z = (target - mean) / scale
    nll = (0.5 * z ** 2 + np.log(scale) + 0.5 * math.log(2.0 * math.pi))
    crps = _normal_crps(target, mean, scale)
    return mean, scale, quantiles[0], quantiles[1], nll, crps


@torch.no_grad()
def _bcpt_draw_components(model, loader, device, m: int):
    """Per-channel draws for one B5 model: full sampling (dropout + tau) means,
    dropout-only epistemic variance, tau-only attention MI, and mean eta^2.

    The separation of H(alpha) (tau sampling, dropout disabled) from sigma^2_PU
    (dropout sampling, tau disabled) is a registered approximation of the
    paper's Eqs. 12/18, which do not specify how the two sampling processes
    are disentangled.
    """
    device = torch.device(device)
    model.eval()
    full_means, epistemic, attention_mi, aleatoric = [], [], [], []
    targets, units, sample_ids = [], [], []
    for x, _w, y, unit, sample_id in loader:
        x = x.to(device)
        channels = x.shape[2]
        # Full sampling: dropout on, tau on.
        draws = []
        for _ in range(m):
            model.apply(_enable_dropout_modules)
            draws.append(model.forward_channels(x, sample_tau=True)["mean"])
        draws = torch.stack(draws)                               # [m, B, S]
        full_means.append(draws.mean(dim=0).cpu().numpy())
        # Dropout-only epistemic variance (Eq. 18).
        drop_draws = []
        for _ in range(m):
            model.apply(_enable_dropout_modules)
            drop_draws.append(model.forward_channels(x, sample_tau=False)["mean"])
        epistemic.append(torch.stack(drop_draws).var(dim=0).cpu().numpy())
        # Tau-only attention mutual information (Eq. 12): moment-matched
        # mixture entropy minus mean component entropy.
        model.eval()
        tau_means, tau_logvars = [], []
        for _ in range(m):
            out = model.forward_channels(x, sample_tau=True)
            tau_means.append(out["mean"])
            tau_logvars.append(out["log_var"])
        tau_means = torch.stack(tau_means)                       # [m, B, S]
        tau_logvars = torch.stack(tau_logvars)                   # [m, B, S]
        eta2 = torch.exp(tau_logvars)                            # [m, B, S]
        mixture_var = eta2.mean(dim=0) + tau_means.var(dim=0)
        entropy = 0.5 * torch.log(2.0 * math.pi * math.e * mixture_var.clamp_min(1e-12))
        mean_entropy = (0.5 * (eta2 * (2.0 * math.pi * math.e)
                               ).log()).mean(dim=0)
        attention_mi.append((entropy - mean_entropy).cpu().numpy())
        aleatoric.append(eta2.mean(dim=0).cpu().numpy())
        targets.extend(y.numpy().tolist())
        units.extend(np.asarray(unit).reshape(-1).tolist())
        sample_ids.extend(np.asarray(sample_id).reshape(-1).tolist())
        model.eval()
    model.eval()
    return {
        "mean": np.concatenate(full_means), "epistemic": np.concatenate(epistemic),
        "attention_mi": np.concatenate(attention_mi),
        "aleatoric": np.concatenate(aleatoric),
        "target": np.asarray(targets, dtype=np.float64),
        "units": np.asarray(units, dtype=np.int64),
        "sample_ids": np.asarray(sample_ids, dtype=np.int64),
    }


def _enable_dropout_modules(module):
    if isinstance(module, nn.Dropout):
        module.train()


@torch.no_grad()
def _bcpt_channel_weights(model, loader, device, m: int, high: bool):
    """Mean per-channel training weights (Eq. 22/23): 1 -/+ softmax(U)."""
    components = _bcpt_draw_components(model, loader, device, m)
    uncertainty = (components["epistemic"] + components["attention_mi"]
                   + components["aleatoric"])                   # [N, S]
    shifted = uncertainty - uncertainty.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    weights /= weights.sum(axis=1, keepdims=True)
    if high:
        return 1.0 + weights.mean(axis=0)
    return 1.0 - weights.mean(axis=0)


def _bcpt_fuse(lu, hu):
    """UGMIF fusion (Eqs. 24-26) from per-model component dicts."""
    u_lu = lu["epistemic"] + lu["attention_mi"] + lu["aleatoric"]
    u_hu = hu["epistemic"] + hu["attention_mi"] + hu["aleatoric"]
    shift = -np.maximum(u_lu, u_hu)
    omega = np.exp(u_lu + shift) / (np.exp(u_lu + shift) + np.exp(u_hu + shift))
    fused_mean = (1.0 - omega) * lu["mean"] + omega * hu["mean"]
    fused_u = (1.0 - omega) * u_lu + omega * u_hu
    shifted = fused_u - fused_u.max(axis=1, keepdims=True)
    channel_weights = np.exp(shifted)
    channel_weights /= channel_weights.sum(axis=1, keepdims=True)
    final_mean = (channel_weights * fused_mean).sum(axis=1)
    span = fused_u.max(axis=1) - fused_u.min(axis=1)
    normalizer = np.where(span < 1e-12, 1.0, span)
    total_u = np.where(span < 1e-12, fused_u.mean(axis=1),
                       (fused_u * (fused_u.max(axis=1, keepdims=True) - fused_u)
                        / normalizer[:, None]).sum(axis=1))
    total_u = np.maximum(total_u, 1e-9)
    return final_mean, total_u, fused_mean, fused_u


def _distribution_outputs(kind, params, target, seed):
    eps = 1e-9
    if kind in {"hnn", "lstm_prob", "ensemble"}:
        means, scales = params[:, :, 0], params[:, :, 1]
        if kind in {"hnn", "lstm_prob"}:
            mean, scale = means[:, 0], scales[:, 0]
            lower = mean + scale * student_t.ppf(0.1, df=np.inf)
            upper = mean + scale * student_t.ppf(0.9, df=np.inf)
            nll = 0.5 * ((target - mean) / scale) ** 2 + np.log(scale) + 0.5 * math.log(2 * math.pi)
            crps = _normal_crps(target, mean, scale)
        else:
            mean = means.mean(axis=1)
            scales = np.maximum(scales, eps)
            log_probs = (-0.5 * ((target[:, None] - means) / scales) ** 2
                         - np.log(scales) - 0.5 * math.log(2 * math.pi))
            nll = -(np.logaddexp.reduce(log_probs, axis=1) - math.log(means.shape[1]))
            crps = _mixture_crps(target, means, scales)
            rng = np.random.default_rng(seed)
            component = rng.integers(means.shape[1], size=(len(target), 4096))
            rows = np.arange(len(target))[:, None]
            draws = rng.normal(means[rows, component], scales[rows, component])
            lower, upper = np.quantile(draws, [0.1, 0.9], axis=1)
            scale = np.sqrt(np.mean(scales ** 2 + (means - mean[:, None]) ** 2, axis=1))
        return mean, scale, lower, upper, nll, crps

    nig = params[:, 0, :]
    gamma, df, scale = _edl_distribution(nig)
    lower = gamma + scale * student_t.ppf(0.1, df=df)
    upper = gamma + scale * student_t.ppf(0.9, df=df)
    nll = -student_t.logpdf(target, df=df, loc=gamma, scale=scale)
    crps = _edl_crps(target, gamma, df, scale, seed)
    return gamma, scale, lower, upper, nll, crps


def _save_history(path, history):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["member", "epoch", "train_loss",
                                                    "val_rmse", "val_selection_score",
                                                    "learning_rate"])
        writer.writeheader()
        writer.writerows(history)


def fit_probabilistic(bundle, kind, output_dir, *, seed=0, epochs=None, batch_size=None,
                      lr=None, weight_decay=None, patience=None,
                      dropout=None, ensemble_members=5, evidence_weight=1e-2,
                      edl_hidden_dim=128, mc_samples=None, bcpt_gamma=0.2,
                      bcpt_lam=0.1,
                      device="auto", num_workers=0, config=None,
                      test_dataset=None, defer_test=False, fixed_epochs=False):
    """Train a probabilistic baseline; evaluate test once."""
    supported = set(PAPER_METHOD_DEFAULTS)
    if kind not in supported:
        raise ValueError(f"kind must be one of {sorted(supported)}")
    if kind == "ensemble" and ensemble_members < 2:
        raise ValueError("ensemble_members must be at least 2")
    if fixed_epochs and kind == "ensemble":
        raise ValueError("fixed_epoch ensemble is not registered for N-CMAPSS")
    if bundle.val_endpoint is None and not fixed_epochs:
        raise ValueError("bundle must use fixed-cuts validation")
    default_epochs, default_batch, default_lr, default_wd, default_dropout, \
        _scaler, default_criterion, default_mc = PAPER_METHOD_DEFAULTS[kind]
    epochs = default_epochs if epochs is None else epochs
    batch_size = default_batch if batch_size is None else batch_size
    lr = default_lr if lr is None else lr
    weight_decay = default_wd if weight_decay is None else weight_decay
    patience = epochs if patience is None else patience
    dropout = default_dropout if dropout is None else dropout
    mc_samples = default_mc if mc_samples is None else int(mc_samples)
    set_seed(seed)
    device_obj = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                              if device == "auto" else device)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty result directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(bundle.train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = (None if fixed_epochs else DataLoader(
        bundle.val_endpoint, batch_size=batch_size, shuffle=False,
        num_workers=num_workers))
    test_dataset = (test_dataset if test_dataset is not None else
                    (getattr(bundle, "test_endpoint", None) or bundle.test))
    test_loader = (None if defer_test else DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers))
    member_count = ensemble_members if kind == "ensemble" else 1
    edl_sensor_indices = (_edl_sensor_positions(bundle.metadata, bundle.sensor_dim)
                          if kind == "edl_nig" else None)
    models, history, best_epochs = [], [], []
    start_train = time.perf_counter()

    if kind == "ensemble":
        optimizers = []
        for member in range(member_count):
            set_seed(seed + member * 1009)
            model = _model(kind, bundle.sensor_dim, bundle.condition_dim, dropout,
                           bundle.train[0][0].shape[0], edl_hidden_dim,
                           edl_sensor_indices)
            if kind != "edl_nig":
                model.apply(_bayesrul_weights_init)
            model.to(device_obj)
            models.append(model)
            optimizers.append(torch.optim.Adam(model.parameters(), lr=lr,
                                               weight_decay=weight_decay))
        best_score, stale = float("inf"), 0
        best_checkpoint = output_dir / "ensemble_best.pt"
        for epoch in range(1, epochs + 1):
            for model in models:
                model.train()
            loss_sums, counts = np.zeros(member_count), np.zeros(member_count, dtype=np.int64)
            for batch_index, (x, w, y, _, _) in enumerate(train_loader):
                member = (batch_index + 1) % member_count  # BayesRUL forward_one cycle
                x, w, y = x.to(device_obj), w.to(device_obj), y.to(device_obj)
                optimizers[member].zero_grad(set_to_none=True)
                prediction = models[member](x, w)
                loss = gaussian_nll(prediction, y)
                loss.backward()
                optimizers[member].step()
                loss_sums[member] += float(loss.detach()) * len(y)
                counts[member] += len(y)
            member_train_losses = loss_sums / np.maximum(counts, 1)
            val_predictions = [_predict(model, val_loader, device_obj) for model in models]
            val_y = val_predictions[0][0]
            val_params = np.stack([prediction[1] for prediction in val_predictions], axis=1)
            val_score = float(np.mean(_distribution_outputs(
                "ensemble", val_params, val_y, seed + epoch)[4]))
            val_mean = val_params[:, :, 0].mean(axis=1)
            val_rmse = rmse(val_y, val_mean)
            train_loss = float(member_train_losses.mean())
            history.append({"member": -1, "epoch": epoch, "train_loss": train_loss,
                            "val_rmse": val_rmse, "val_selection_score": val_score,
                            "learning_rate": lr})
            print(f"epoch:{epoch}, train_loss:{train_loss:.5f}, val_nll:{val_score:.5f}, "
                  f"val_rmse:{val_rmse:.4f}, test_loss:deferred", flush=True)
            if val_score < best_score:
                best_score, stale = val_score, 0
                torch.save({"model_states": [model.state_dict() for model in models],
                            "best_epoch": epoch, "selection": best_score,
                            "selection_criterion": "val_mixture_nll"}, best_checkpoint)
            else:
                stale += 1
                if stale >= patience:
                    break
        state = torch.load(best_checkpoint, map_location=device_obj, weights_only=False)
        for member, model in enumerate(models):
            model.load_state_dict(state["model_states"][member])
            best_epochs.append(int(state["best_epoch"]))
            torch.save({"model_state": model.state_dict(), "best_epoch": state["best_epoch"],
                        "selection": state["selection"],
                        "selection_criterion": state["selection_criterion"]},
                       output_dir / f"member_{member}.pt")

    def _weights_from_components(components, high):
        uncertainty = (components["epistemic"] + components["attention_mi"]
                       + components["aleatoric"])
        shifted = uncertainty - uncertainty.max(axis=1, keepdims=True)
        weights = np.exp(shifted)
        weights /= weights.sum(axis=1, keepdims=True)
        return 1.0 + weights.mean(axis=0) if high else 1.0 - weights.mean(axis=0)

    if kind == BCPT_KIND:
        lu = _model(kind, bundle.sensor_dim, bundle.condition_dim, dropout,
                    bundle.train[0][0].shape[0]).to(device_obj)
        hu = _model(kind, bundle.sensor_dim, bundle.condition_dim, dropout,
                    bundle.train[0][0].shape[0]).to(device_obj)
        optimizer_lu = torch.optim.Adam(lu.parameters(), lr=lr,
                                        weight_decay=weight_decay)
        optimizer_hu = torch.optim.Adam(hu.parameters(), lr=lr,
                                        weight_decay=weight_decay)
        # Paper states lr "0.005 -> 0.001"; halving-point decay is the
        # registered reconstruction of the unstated schedule.
        milestone = max(1, epochs // 2)
        scheduler_lu = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_lu, milestones=[milestone], gamma=0.2)
        scheduler_hu = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_hu, milestones=[milestone], gamma=0.2)
        best, best_epoch, stale = float("inf"), 0, 0
        checkpoint = output_dir / "best.pt"
        weight_loader = train_loader if fixed_epochs else val_loader
        for epoch in range(1, epochs + 1):
            lu_components = _bcpt_draw_components(lu, weight_loader, device_obj,
                                                  mc_samples)
            hu_components = _bcpt_draw_components(hu, weight_loader, device_obj,
                                                  mc_samples)
            weight_lu = torch.as_tensor(_weights_from_components(lu_components, False),
                                        dtype=torch.float32, device=device_obj)
            weight_hu = torch.as_tensor(_weights_from_components(hu_components, True),
                                        dtype=torch.float32, device=device_obj)
            epoch_losses = []
            for model, optimizer, channel_weights in ((lu, optimizer_lu, weight_lu),
                                                      (hu, optimizer_hu, weight_hu)):
                model.train()
                loss_sum, count = 0.0, 0
                for x, _w, y, _, _ in train_loader:
                    x, y = x.to(device_obj), y.to(device_obj)
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model.forward_channels(x, sample_tau=True)
                    loss = bcpt_channel_loss(outputs, y, channel_weights,
                                             gamma=bcpt_gamma, lam=bcpt_lam)
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(y)
                    count += len(y)
                epoch_losses.append(loss_sum / max(count, 1))
            train_loss = float(np.mean(epoch_losses))
            if fixed_epochs:
                history.append({"member": -1, "epoch": epoch,
                                "train_loss": train_loss, "val_rmse": None,
                                "val_selection_score": None, "learning_rate": lr})
                if epoch == epochs:
                    best_epoch = epoch
                    torch.save({"lu_state": lu.state_dict(),
                                "hu_state": hu.state_dict(),
                                "best_epoch": epoch, "selection": None,
                                "selection_criterion": "fixed_epoch"}, checkpoint)
                scheduler_lu.step()
                scheduler_hu.step()
                continue
            val_lu = _bcpt_draw_components(lu, val_loader, device_obj, mc_samples)
            val_hu = _bcpt_draw_components(hu, val_loader, device_obj, mc_samples)
            fused_mean, _total_u, _fused_c, _fused_u = _bcpt_fuse(val_lu, val_hu)
            val_score = rmse(val_lu["target"], fused_mean)
            history.append({"member": -1, "epoch": epoch, "train_loss": train_loss,
                            "val_rmse": val_score, "val_selection_score": val_score,
                            "learning_rate": lr})
            print(f"bcpt, epoch:{epoch}, train_loss:{train_loss:.5f}, "
                  f"val_fused_rmse:{val_score:.4f}, test_loss:deferred", flush=True)
            if val_score < best:
                best, best_epoch, stale = val_score, epoch, 0
                torch.save({"lu_state": lu.state_dict(), "hu_state": hu.state_dict(),
                            "best_epoch": epoch, "selection": best,
                            "selection_criterion": "val_fused_rmse"}, checkpoint)
            else:
                stale += 1
                if stale >= patience:
                    break
            scheduler_lu.step()
            scheduler_hu.step()
        state = torch.load(checkpoint, map_location=device_obj, weights_only=False)
        lu.load_state_dict(state["lu_state"])
        hu.load_state_dict(state["hu_state"])
        models.extend((lu, hu))
        best_epochs.append(int(state["best_epoch"]))
        selection_criterion = "fixed_epoch" if fixed_epochs else "val_fused_rmse"

    for member in range(member_count if kind not in {"ensemble", BCPT_KIND} else 0):
        member_seed = seed + member * 1009
        set_seed(member_seed)
        model = _model(kind, bundle.sensor_dim, bundle.condition_dim, dropout,
                       bundle.train[0][0].shape[0], edl_hidden_dim,
                       edl_sensor_indices)
        if kind in {"hnn", "ensemble"}:
            model.apply(_bayesrul_weights_init)
        model.to(device_obj)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        best = float("inf")
        best_epoch = 0
        stale = 0
        checkpoint = output_dir / (f"member_{member}.pt" if member_count > 1 else "best.pt")
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum, count = 0.0, 0
            for x, w, y, _, _ in train_loader:
                x, w, y = x.to(device_obj), w.to(device_obj), y.to(device_obj)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x, w)
                loss = _loss(kind, prediction, y, evidence_weight)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(y)
                count += len(y)
            train_loss = loss_sum / max(count, 1)
            if fixed_epochs:
                history.append({"member": member, "epoch": epoch,
                                "train_loss": train_loss, "val_rmse": None,
                                "val_selection_score": None, "learning_rate": lr})
                if epoch == epochs:
                    torch.save({"model_state": model.state_dict(),
                                "best_epoch": epoch, "selection": None,
                                "selection_criterion": "fixed_epoch"}, checkpoint)
                continue
            val_y, val_params, _, _, _ = _predict(model, val_loader, device_obj)
            val_rmse = rmse(val_y, _mean(kind, val_params))
            if kind == "ensemble":
                val_score = float(gaussian_nll(torch.as_tensor(val_params),
                                               torch.as_tensor(val_y)))
                selection_criterion = "val_mixture_nll"
            elif kind in {"edl_nig", "lstm_prob"} | POINT_MSE_KINDS:
                # B3's paper minimizes validation error (RMSE) in its BayesOpt;
                # point MC kinds track RMSE per their papers.
                val_score = val_rmse
                selection_criterion = "val_rmse"
            else:
                val_score = float(np.mean(_distribution_outputs(
                    kind, val_params[:, None, :], val_y, seed + epoch)[4]))
                selection_criterion = "val_nll"
            history.append({"member": member, "epoch": epoch, "train_loss": train_loss,
                            "val_rmse": val_rmse, "val_selection_score": val_score,
                            "learning_rate": lr})
            print(f"member:{member}, epoch:{epoch}, train_loss:{train_loss:.5f}, "
                  f"val_nll:{val_score:.5f}, val_rmse:{val_rmse:.4f}, test_loss:deferred", flush=True)
            if val_score < best:
                best, best_epoch, stale = val_score, epoch, 0
                torch.save({"model_state": model.state_dict(), "best_epoch": epoch,
                            "selection": best,
                            "selection_criterion": selection_criterion}, checkpoint)
            else:
                stale += 1
                if stale >= patience:
                    break
        state = torch.load(checkpoint, map_location=device_obj, weights_only=False)
        model.load_state_dict(state["model_state"])
        models.append(model)
        best_epochs.append(int(state["best_epoch"]))
        if fixed_epochs:
            selection_criterion = "fixed_epoch"

    train_seconds = time.perf_counter() - start_train
    if defer_test:
        _save_history(output_dir / "history.csv", history)
        return {"dataset": bundle.metadata.get("dataset"), "method": kind,
                "seed": int(seed), "best_epochs": best_epochs,
                "selection_criterion": selection_criterion,
                "train_seconds": train_seconds, "test_evaluated": False}
    inference_start = time.perf_counter()
    extra_columns: list = []
    extra_rows: list = []
    if kind == BCPT_KIND:
        lu_components = _bcpt_draw_components(models[0], test_loader, device_obj,
                                               mc_samples)
        hu_components = _bcpt_draw_components(models[1], test_loader, device_obj,
                                               mc_samples)
        for key in ("target", "units", "sample_ids"):
            if not np.array_equal(lu_components[key], hu_components[key]):
                raise RuntimeError("test loader order changed between LU/HU models")
        target = lu_components["target"]
        units = lu_components["units"]
        sample_ids = lu_components["sample_ids"]
        mean, total_u, fused_mean, fused_u = _bcpt_fuse(lu_components, hu_components)
        scale = np.sqrt(total_u)
        lower = mean - Z_80 * scale
        upper = mean + Z_80 * scale
        z = (target - mean) / scale
        nll = 0.5 * z ** 2 + np.log(scale) + 0.5 * math.log(2.0 * math.pi)
        crps = _normal_crps(target, mean, scale)
        extra_columns = ["total_uncertainty"]
        extra_rows = [total_u]
        np.savez(output_dir / "bcpt_test_arrays.npz",
                 fused_channel_mean=fused_mean, fused_channel_uncertainty=fused_u,
                 lu_mean=lu_components["mean"], hu_mean=hu_components["mean"],
                 lu_epistemic=lu_components["epistemic"],
                 hu_epistemic=hu_components["epistemic"],
                 lu_attention_mi=lu_components["attention_mi"],
                 hu_attention_mi=hu_components["attention_mi"],
                 lu_aleatoric=lu_components["aleatoric"],
                 hu_aleatoric=hu_components["aleatoric"])
    elif kind in MC_DRAWS_KINDS:
        mc = predict_mc_dropout(models[0], test_loader, device_obj, samples=mc_samples)
        target, units, sample_ids = mc["target"], mc["units"], mc["sample_ids"]
        draws = mc["draws"]
        bounds = (kde_quantiles(draws) if kind == "msauq"
                  else np.quantile(draws, [0.1, 0.9], axis=0))
        mean, scale, lower, upper, nll, crps = gaussian_fit_outputs(target, draws,
                                                                    bounds)
        np.save(output_dir / "mc_draws.npy", draws.astype(np.float32))
    else:
        predictions = [_predict(model, test_loader, device_obj) for model in models]
        target, units, sample_ids = predictions[0][0], predictions[0][2], predictions[0][3]
        if any(not np.array_equal(target, pred[0]) or not np.array_equal(units, pred[2])
               or not np.array_equal(sample_ids, pred[3]) for pred in predictions[1:]):
            raise RuntimeError("test loader order changed between ensemble members")
        params = np.stack([pred[1] for pred in predictions], axis=1)
        mean, scale, lower, upper, nll, crps = _distribution_outputs(kind, params, target, seed)
        if kind == "edl_nig":
            gamma_nig, nu_nig, alpha_nig, beta_nig = params[:, 0, :].T
            extra_columns = ["nig_gamma", "nig_nu", "nig_alpha", "nig_beta",
                             "epistemic_variance", "aleatoric_variance"]
            extra_rows = [gamma_nig, nu_nig, alpha_nig, beta_nig,
                          beta_nig / (nu_nig * (alpha_nig - 1.0)),
                          beta_nig / (alpha_nig - 1.0)]
    inference_seconds = time.perf_counter() - inference_start
    if not all(np.isfinite(a).all() for a in (mean, scale, lower, upper, nll, crps)):
        raise RuntimeError("probabilistic model emitted non-finite test predictions")
    if np.any(lower >= upper):
        raise RuntimeError("probabilistic interval bounds are invalid")
    point = regression_metrics(target, mean)
    if bundle.metadata.get("dataset") in {"DS03", "DS08a", "DS08A"}:
        point.pop("NASA_Score", None)
    if not fixed_epochs:
        selection_criterion = {"hnn": "val_nll", "edl_nig": "val_rmse",
                               "ensemble": "val_mixture_nll",
                               "lstm_prob": "val_rmse",
                               "tcn_transformer_mc": "val_rmse", "msauq": "val_rmse",
                               "bcpt": "val_fused_rmse"}[kind]
    interval = ((target >= lower) & (target <= upper)).mean()
    width = upper - lower
    alpha = 0.2
    interval_score = width + (2.0 / alpha) * (lower - target) * (target < lower) \
        + (2.0 / alpha) * (target - upper) * (target > upper)
    results = {**point, "NLL": float(np.mean(nll)), "CRPS": float(np.mean(crps)),
               "PICP": float(interval), "MPIW": float(np.mean(width)),
               "coverage_gap": float(abs(interval - (1.0 - alpha))),
               "IntervalScore": float(np.mean(interval_score)),
               "dataset": bundle.metadata.get("dataset"), "method": kind,
               "seed": int(seed), "device": str(device_obj),
               "best_epochs": best_epochs, "ensemble_members": member_count,
               "selection_criterion": selection_criterion,
               "train_seconds": train_seconds,
               "inference_seconds": float(inference_seconds),
               "mc_samples": (mc_samples if kind in MC_DRAWS_KINDS or kind == BCPT_KIND
                              else 0),
               "parameter_count": int(sum(sum(p.numel() for p in m.parameters()) for m in models))}
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        columns = ["unit_id", "sample_id", "target_rul", "predicted_mean", "predictive_scale",
                   "lower", "upper", "covered", "interval_width", "nll", "crps"]
        rows = [units, sample_ids, target, mean, scale, lower, upper,
                (target >= lower) & (target <= upper), width, nll, crps]
        columns.extend(extra_columns)
        rows.extend(extra_rows)
        writer.writerow(columns)
        writer.writerows(zip(*rows))
    _save_history(output_dir / "history.csv", history)
    config_data = {"model": kind, "seed": seed, "epochs": epochs,
                   "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
                   "patience": patience, "fixed_epochs": fixed_epochs,
                   "dropout": dropout,
                   "selection_criterion": selection_criterion,
                   "ensemble_members": member_count, "evidence_weight": evidence_weight,
                   "edl_hidden_dim": edl_hidden_dim,
                   "mc_samples": mc_samples, "bcpt_gamma": bcpt_gamma,
                   "bcpt_lam": bcpt_lam,
                   "edl_sensor_indices": (list(edl_sensor_indices)
                                          if edl_sensor_indices is not None else None),
                   "edl_sensor_ids_original": (list(EDL_PAPER_SENSOR_IDS)
                                               if edl_sensor_indices is not None
                                               and bundle.metadata.get("sensor_indices") is not None
                                               else None),
                   "interval_alpha": alpha,
                   "data": bundle.metadata, "scaler": bundle.scaler.state_dict()}
    if config:
        config_data.update(config)
    (output_dir / "config.json").write_text(json.dumps(config_data, indent=2), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain"],
                                    check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
        dirty = None
    source_files = [Path(__file__), Path(__file__).with_name("probabilistic.py")]
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_files}
    manifest = {"dataset": bundle.metadata.get("dataset"), "method": kind,
                "seed": seed, "split_seed": bundle.metadata.get("split_seed"),
                "selection_criterion": selection_criterion,
                "scaler_mode": bundle.metadata.get("scaler_mode"),
                "train_units": bundle.metadata.get("train_units", []),
                "val_units": bundle.metadata.get("val_units", []),
                "calibration_units": bundle.metadata.get("calibration_units", []),
                "test_units": bundle.metadata.get("test_units", []),
                "scaler_fit_scope": "train_units_only", "scaler_fit_units": bundle.metadata.get("train_units", []),
                "code_commit": commit, "code_dirty": dirty, "source_sha256": source_hashes,
                "test_evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "test_evaluated_once": True, "test_sample_count": int(len(target))}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    score_text = (f" NASA_Score: {point['NASA_Score']:.2f}"
                  if "NASA_Score" in point else "")
    print(f"MAE: {point['MAE']:.2f} RMSE: {point['RMSE']:.2f}{score_text} "
          f"NLL: {results['NLL']:.4f} CRPS: {results['CRPS']:.4f} "
          f"PICP: {results['PICP']:.4f} MPIW: {results['MPIW']:.4f}", flush=True)
    return results
