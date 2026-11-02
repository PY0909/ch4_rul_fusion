from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class InceptionModule(nn.Module):
    """BayesRUL's compact 1-D Inception block for temporal sensor windows."""

    def __init__(self, n_features, filters, activation=nn.ReLU, bias=True, dropout=0.0):
        super().__init__()
        f1, f3, f5, fp = filters
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(n_features, f1, 1, padding="same", bias=bias), activation()),
            nn.Sequential(nn.Conv1d(n_features, f3, 3, padding="same", bias=bias), activation()),
            nn.Sequential(nn.Conv1d(n_features, f5, 5, padding="same", bias=bias), activation()),
            nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                          nn.Conv1d(n_features, fp, 3, padding="same", bias=bias), activation()),
        ])
        if dropout > 0:
            for branch in self.branches:
                branch.append(nn.Dropout(dropout / 4))

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class InceptionModuleReducDim(nn.Module):
    """Inception block with bottleneck reductions, matching BayesRUL source."""

    def __init__(self, n_features, filters, activation=nn.ReLU, bias=True, dropout=0.0):
        super().__init__()
        f1, r3, f3, r5, f5, fp = filters
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(n_features, f1, 1, padding="same", bias=bias), activation()),
            nn.Sequential(nn.Conv1d(n_features, r3, 1, bias=bias), activation(),
                          nn.Conv1d(r3, f3, 3, padding="same", bias=bias), activation()),
            nn.Sequential(nn.Conv1d(n_features, r5, 1, bias=bias), activation(),
                          nn.Conv1d(r5, f5, 5, padding="same", bias=bias), activation()),
            nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                          nn.Conv1d(n_features, fp, 1, bias=bias), activation()),
        ])
        if dropout > 0:
            for branch in self.branches:
                branch.append(nn.Dropout(dropout / 4))

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class InceptionRUL(nn.Module):
    """Compact InceptionModel from the BayesRUL benchmark (out_size=2)."""

    def __init__(self, sensor_dim, condition_dim, window_size=50, dropout=0.0):
        super().__init__()
        n_features = int(sensor_dim + condition_dim)
        self.layers = nn.Sequential(
            InceptionModule(n_features, (27, 27, 27, 27), dropout=dropout),
            InceptionModuleReducDim(108, (8, 64, 8, 64, 8, 32), dropout=dropout),
            InceptionModuleReducDim(56, (4, 16, 4, 16, 4, 8), dropout=dropout),
            nn.Flatten(), nn.Linear(20 * int(window_size), 64), nn.ReLU(),
        )
        if dropout > 0:
            self.layers.append(nn.Dropout(dropout))
        self.last = nn.Linear(64, 2)

    def forward(self, x, condition):
        if x.ndim != 3 or condition.ndim != 3 or x.shape[:2] != condition.shape[:2]:
            raise ValueError("x and condition must be aligned [batch, time, channels]")
        features = self.forward_features(x, condition)
        return F.softplus(self.last(features)).clamp_min(1e-9)

    def forward_features(self, x, condition):
        if x.ndim != 3 or condition.ndim != 3 or x.shape[:2] != condition.shape[:2]:
            raise ValueError("x and condition must be aligned [batch, time, channels]")
        return self.layers(torch.cat((x, condition), dim=-1).transpose(1, 2))


class HeteroscedasticRUL(nn.Module):
    """BayesRUL heteroscedastic Gaussian Inception model."""

    def __init__(self, sensor_dim: int, condition_dim: int, dropout: float = 0.0,
                 window_size: int = 50):
        super().__init__()
        self.backbone = InceptionRUL(sensor_dim, condition_dim, window_size, dropout)

    def forward(self, x, condition):
        return self.backbone(x, condition)


class EvidentialNIGRUL(nn.Module):
    """Five-linear-layer MLP used for evidential C-MAPSS regression.

    The article specifies five layers and dropout=0.3, but does not report
    hidden widths. ``hidden_dim`` is therefore an explicit reconstruction
    choice, not a paper-reported hyperparameter.
    """

    def __init__(self, sensor_dim: int, condition_dim: int, dropout: float = 0.3,
                 epsilon: float = 1e-4, hidden_dim: int = 128,
                 sensor_indices=None):
        super().__init__()
        self.sensor_indices = tuple(range(sensor_dim) if sensor_indices is None
                                    else sensor_indices)
        if not self.sensor_indices:
            raise ValueError("EDL-NIG requires at least one selected sensor")
        if min(self.sensor_indices) < 0 or max(self.sensor_indices) >= sensor_dim:
            raise ValueError("EDL-NIG sensor selection is outside sensor_dim")
        layers = []
        in_features = len(self.sensor_indices)
        for _ in range(4):
            layers.extend((nn.Linear(in_features, hidden_dim), nn.ReLU(),
                           nn.Dropout(dropout)))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, 4))
        self.network = nn.Sequential(*layers)
        self.epsilon = float(epsilon)

    def forward(self, x, condition):
        if x.ndim != 3:
            raise ValueError("EDL-NIG sensor input must have shape [batch, time, channels]")
        # Paper's MLP consumes cycle-wise sensor attributes, not temporal
        # convolutions or operation-setting features. The current benchmark
        # dataset represents each cycle in a history window, so use its end.
        sensor = x[:, -1, list(self.sensor_indices)]
        raw = self.network(sensor)
        gamma = raw[:, 0]
        nu = F.softplus(raw[:, 1]) + self.epsilon
        alpha = F.softplus(raw[:, 2]) + 1.0 + self.epsilon
        beta = F.softplus(raw[:, 3]) + self.epsilon
        return torch.stack((gamma, nu, alpha, beta), dim=-1)


def gaussian_nll(parameters: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if parameters.ndim != 2 or parameters.shape[-1] != 2:
        raise ValueError("Gaussian parameters must have shape [batch, 2]")
    mean, scale = parameters.unbind(dim=-1)
    y = target.reshape(-1).to(dtype=mean.dtype)
    if len(y) != len(mean) or not torch.isfinite(parameters).all() or not torch.isfinite(y).all():
        raise ValueError("Gaussian NLL inputs must align and be finite")
    return (0.5 * ((y - mean) / scale).square() + torch.log(scale)
            + 0.5 * torch.log(mean.new_tensor(2.0 * torch.pi))).mean()


def nig_evidential_loss(parameters: torch.Tensor, target: torch.Tensor,
                        evidence_weight: float = 1e-2) -> torch.Tensor:
    if parameters.ndim != 2 or parameters.shape[-1] != 4:
        raise ValueError("NIG parameters must have shape [batch, 4]")
    gamma, nu, alpha, beta = parameters.unbind(dim=-1)
    y = target.reshape(-1).to(dtype=gamma.dtype)
    if len(y) != len(gamma) or not torch.isfinite(parameters).all() or not torch.isfinite(y).all():
        raise ValueError("NIG loss inputs must align and be finite")
    if torch.any(nu <= 0) or torch.any(alpha <= 1) or torch.any(beta <= 0):
        raise ValueError("NIG constraints require nu>0, alpha>1, beta>0")
    error2 = (y - gamma).square()
    omega = 2.0 * beta * (1.0 + nu)
    nll = (0.5 * torch.log(gamma.new_tensor(torch.pi) / nu)
           - alpha * torch.log(omega)
           + (alpha + 0.5) * torch.log(nu * error2 + omega)
           + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))
    regularizer = (y - gamma).abs() * (2.0 * nu + alpha)
    return (nll + float(evidence_weight) * regularizer).mean()
