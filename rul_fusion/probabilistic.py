from __future__ import annotations

import math

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


# ---------------------------------------------------------------------------
# B3–B6 first-round external baselines (chapter plan §8.5). Paper-reported
# values and reconstruction choices are registered in
# references/probabilistic_baselines.md; nothing here silently reinterprets a
# paper number.


def _sinusoidal_pe(length: int, dim: int, dtype, device) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32, device=device)[:, None]
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device)
                    * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div[: dim // 2])
    return pe.to(dtype=dtype)


class LSTMProbRUL(nn.Module):
    """B3: LSTM with a Normal-distribution output layer (Mohd Nor et al. 2022).

    Paper BayesOpt optimum on FD001: one LSTM layer (635 units), one dense
    layer (331), dropout 0.03; the output layer emits the mean and standard
    deviation of a Normal distribution (aleatoric only). The paper's HI
    target, cap 100, CUSUM degradation starts and random 20% split are
    replaced by the chapter protocol (direct RUL, cap 125, registered split);
    the mean is linear and the scale softplus (TFP distribution-layer
    convention).
    """

    def __init__(self, sensor_dim: int, condition_dim: int, hidden_units: int = 635,
                 dense_size: int = 331, dropout: float = 0.03, **_unused):
        super().__init__()
        self.lstm = nn.LSTM(int(sensor_dim + condition_dim), hidden_units,
                            batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_units, dense_size), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(dense_size, 2))

    def forward(self, x, condition):
        if x.ndim != 3 or condition.ndim != 3 or x.shape[:2] != condition.shape[:2]:
            raise ValueError("x and condition must be aligned [batch, time, channels]")
        out, _ = self.lstm(torch.cat((x, condition), dim=-1))
        raw = self.head(out[:, -1])
        scale = F.softplus(raw[:, 1]).clamp_min(1e-6)
        return torch.stack((raw[:, 0], scale), dim=-1)


class _Chomp(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = int(chomp)

    def forward(self, x):
        return x[..., : -self.chomp] if self.chomp > 0 else x


class _TCNBlock(nn.Module):
    """Causal residual TCN block with dropout inside (paper: dropout 0.2)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int,
                 dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation),
            _Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation),
            _Chomp(pad), nn.ReLU(), nn.Dropout(dropout))
        self.down = (nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch
                     else nn.Identity())

    def forward(self, x):
        return self.net(x) + self.down(x)


class TCNTransformerMCDropoutRUL(nn.Module):
    """B4: parallel TCN + Transformer with MC-dropout inference (Processes 2024).

    Paper values: TCN blocks filters 8/6/4, kernel 4, dilations 1/2/4;
    Transformer encoder d_model 14 (= sensor count), 2 heads, 2 layers,
    dropout 0.2, sinusoidal PE; point regressor, MC dropout at inference with
    unstated sample count (30 used). The channel fusion (last TCN step +
    mean-pooled Transformer output -> linear head) and the FFN width
    (4 x d_model) are reconstruction choices.
    """

    def __init__(self, sensor_dim: int, condition_dim: int, window_size: int = 50,
                 dropout: float = 0.2, filters=(8, 6, 4), kernel: int = 4,
                 dilations=(1, 2, 4), heads: int = 2, layers: int = 2,
                 ffn_mult: int = 4, **_unused):
        super().__init__()
        d_model = int(sensor_dim)
        if d_model % heads:
            # Paper pairs d_model 14 with 2 heads; datasets whose sensor count
            # is odd fall back to single-head attention rather than failing.
            heads = 1
        blocks, channels = [], int(sensor_dim)
        for width, dilation in zip(filters, dilations):
            blocks.append(_TCNBlock(channels, int(width), kernel, int(dilation),
                                    dropout))
            channels = int(width)
        self.tcn = nn.Sequential(*blocks)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * ffn_mult,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.d_model = d_model
        self.head = nn.Linear(channels + d_model, 1)

    def forward(self, x, condition=None):
        if x.ndim != 3:
            raise ValueError("sensor input must have shape [batch, time, channels]")
        tcn_feature = self.tcn(x.transpose(1, 2))[:, :, -1]
        pe = _sinusoidal_pe(x.shape[1], self.d_model, x.dtype, x.device)
        encoded = self.encoder(x + pe)
        fused = torch.cat((tcn_feature, encoded.mean(dim=1)), dim=-1)
        return self.head(fused)


class _ExponentialSmoothing(nn.Module):
    """Causal EMA S_t = a*x_t + (1-a)*S_{t-1}; B6 paper uses a = 0.2."""

    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x):
        smoothed, state = [], x[:, 0]
        for step in range(x.shape[1]):
            state = self.alpha * x[:, step] + (1.0 - self.alpha) * state
            smoothed.append(state)
        return torch.stack(smoothed, dim=1)


class _MultiScaleConv(nn.Module):
    """B6 module: parallel Conv1d {3,5,7} -> BN -> ReLU -> Dropout; concat;
    1x1 residual projection when channels change; final ReLU."""

    def __init__(self, in_ch: int, branch_ch: int, kernel_sizes=(3, 5, 7),
                 dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_ch, branch_ch, k, padding="same"),
                          nn.BatchNorm1d(branch_ch), nn.ReLU(),
                          nn.Dropout(dropout))
            for k in kernel_sizes])
        out_ch = branch_ch * len(kernel_sizes)
        self.project = (nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch
                        else nn.Identity())

    def forward(self, x):
        feats = torch.cat([branch(x) for branch in self.branches], dim=1)
        return F.relu(feats + self.project(x))


class _SingleHeadSelfAttention(nn.Module):
    """B6 self-attention (single head): LN + Dropout + residual around
    softmax(QK^T/sqrt(d))V."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = float(dim) ** -0.5

    def forward(self, x):
        scores = torch.matmul(self.query(x), self.key(x).transpose(-2, -1))
        attended = torch.softmax(scores * self.scale, dim=-1) @ self.value(x)
        return self.norm(x + self.dropout(attended))


class MSAUQRUL(nn.Module):
    """B6: multi-scale self-attention with UQ (Du & Fu 2025, ERX 7 035543).

    Paper values: N = 2 multi-scale conv modules with kernels {3,5,7},
    sinusoidal PE, one self-attention layer, GAP, Linear-ReLU-Linear head,
    80,225 parameters in total; grouped Z-score plus exponential smoothing
    a = 0.2 preprocessing; MC dropout with G = 100 draws and Gaussian-kernel
    KDE. Unstated in the paper: dropout rate (0.1 used), per-layer widths
    (branch width 24 and head width 384 sized to land near the 80,225 count),
    learning rate and batch size.
    """

    def __init__(self, sensor_dim: int, condition_dim=None, window_size: int = 50,
                 n_modules: int = 2, branch_ch: int = 24, dropout: float = 0.1,
                 fc_hidden: int = 384, smoothing_alpha: float = 0.2, **_unused):
        super().__init__()
        self.smoothing = _ExponentialSmoothing(smoothing_alpha)
        modules, channels = [], int(sensor_dim)
        for _ in range(int(n_modules)):
            modules.append(_MultiScaleConv(channels, int(branch_ch),
                                           dropout=dropout))
            channels = int(branch_ch) * 3
        self.conv_stack = nn.Sequential(*modules)
        self.attention = _SingleHeadSelfAttention(channels, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(channels, int(fc_hidden)), nn.ReLU(),
                                  nn.Linear(int(fc_hidden), 1))

    def forward(self, x, condition=None):
        if x.ndim != 3:
            raise ValueError("sensor input must have shape [batch, time, channels]")
        z = self.smoothing(x).transpose(1, 2)
        z = self.conv_stack(z).transpose(1, 2)
        z = z + _sinusoidal_pe(z.shape[1], z.shape[2], z.dtype, z.device)
        pooled = self.attention(z).mean(dim=1)
        return self.head(pooled)


class ProbabilisticAttention(nn.Module):
    """B5 PAM (Xie et al. 2025, Eqs. 9-11/20): attention logits alpha~N(mu, sigma^2)
    with mu/sigma from a 2-10-6-2 MLP on (s, |s|); the NLL term ties alpha to the
    deterministic logit s = Q_i.K_j/sqrt(d_k). tau sampling is toggled explicitly;
    input features (s, |s|) are a registered reconstruction of the unstated
    two-dimensional MLP input."""

    def __init__(self, d_model: int, heads: int, mlp_hidden=(10, 6),
                 dropout: float = 0.3):
        super().__init__()
        if d_model % heads:
            raise ValueError(f"d_model {d_model} not divisible by heads {heads}")
        self.heads = heads
        self.d_k = d_model // heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(nn.Linear(2, mlp_hidden[0]), nn.ReLU(),
                                 nn.Linear(mlp_hidden[0], mlp_hidden[1]), nn.ReLU(),
                                 nn.Linear(mlp_hidden[1], 2))
        self.dropout = nn.Dropout(dropout)
        self.scale = float(self.d_k) ** -0.5

    def forward(self, x, sample_tau: bool):
        batch, length, dim = x.shape

        def split(tensor):
            return (tensor.view(batch, length, self.heads, self.d_k)
                    .transpose(1, 2))

        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        predicted = self.mlp(torch.stack((logits, logits.abs()), dim=-1))
        mu, sigma = predicted[..., 0], F.softplus(predicted[..., 1]).clamp_min(1e-6)
        alpha = mu + torch.randn_like(mu) * sigma if sample_tau else mu
        attended = torch.softmax(alpha, dim=-1) @ v
        out = attended.transpose(1, 2).reshape(batch, length, dim)
        attention_nll = (0.5 * ((alpha - logits) / sigma) ** 2
                         + torch.log(sigma)).mean()
        return self.dropout(out), attention_nll


class _BCPTEncoderLayer(nn.Module):
    """One B5 encoder layer: PAM attention (residual + LN) and FFN d-64-d
    (residual + LN), dropout 0.3 (paper: FFN 30-64-30)."""

    def __init__(self, d_model: int, heads: int, ffn_hidden: int, dropout: float):
        super().__init__()
        self.attention = ProbabilisticAttention(d_model, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_hidden), nn.ReLU(),
                                 nn.Linear(ffn_hidden, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, sample_tau: bool):
        attended, attention_nll = self.attention(x, sample_tau)
        x = self.norm1(x + attended)
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, attention_nll


class BayesianCooperativeTransformerRUL(nn.Module):
    """B5 (Xie et al. 2025): per-channel Transformer + PAM + noise channel.

    Paper values: window 30, 5 heads, FFN 30-64-30, head FC 20/10, dropout
    0.3, gamma = 0.2, lambda = 0.1; every sensor channel is modelled as a
    univariate stream (operating settings unused); training uses LU/HU channel
    weights and inference the UGMIF fusion. Encoder layer count (1) and the
    embedding dimension (30, from the FFN spec) are unstated -> registered
    reconstruction choices. Window stays at the chapter protocol value.
    """

    def __init__(self, sensor_dim: int, condition_dim=None, window_size: int = 50,
                 d_model: int = 30, heads: int = 5, ffn_hidden: int = 64,
                 fc_hidden=(20, 10), dropout: float = 0.3, **_unused):
        super().__init__()
        self.sensor_dim = int(sensor_dim)
        self.d_model = int(d_model)
        self.embed = nn.Linear(1, d_model)
        self.layer = _BCPTEncoderLayer(d_model, heads, ffn_hidden, dropout)
        self.head = nn.Sequential(nn.Linear(d_model, fc_hidden[0]), nn.ReLU(),
                                  nn.Linear(fc_hidden[0], fc_hidden[1]), nn.ReLU(),
                                  nn.Linear(fc_hidden[1], 2))

    def forward_channels(self, x, sample_tau: bool = False):
        """x [batch, L, S] -> per-channel mean/log-variance [batch, S] plus the
        attention NLL scalar."""
        if x.ndim != 3:
            raise ValueError("sensor input must have shape [batch, time, channels]")
        batch, length, channels = x.shape
        flat = x.reshape(batch * channels, length, 1)
        z = self.embed(flat) + _sinusoidal_pe(length, self.d_model, x.dtype, x.device)
        z, attention_nll = self.layer(z, sample_tau)
        raw = self.head(z.mean(dim=1)).view(batch, channels, 2)
        return {"mean": raw[..., 0], "log_var": raw[..., 1],
                "attention_nll": attention_nll}

    def forward(self, x, condition=None):
        out = self.forward_channels(x)
        return torch.stack((out["mean"], out["log_var"]), dim=-1).reshape(
            x.shape[0] * x.shape[2], 2)


def bcpt_channel_loss(outputs, target: torch.Tensor, channel_weights: torch.Tensor,
                      gamma: float = 0.2, lam: float = 0.1) -> torch.Tensor:
    """B5 loss (Eqs. 21-23): heteroscedastic NLL with noise channel eta^2,
    gamma-weighted attention NLL and lambda-weighted L2 penalty on eta^2,
    summed with per-channel weights (1 -/+ softmax(U))."""
    mean, log_var = outputs["mean"], outputs["log_var"]
    if mean.shape != channel_weights.expand_as(mean).shape:
        raise ValueError("channel weights must align with [batch, channels]")
    y = target.reshape(-1, 1).to(dtype=mean.dtype)
    if len(y) != mean.shape[0]:
        raise ValueError("target batch must align with channel outputs")
    eta2 = torch.exp(log_var).clamp(1e-6, 1e6)
    nll = (0.5 * (y - mean) ** 2 / eta2 + 0.5 * log_var).mean(dim=0)
    penalty = (eta2 ** 2).mean(dim=0)
    attention = outputs["attention_nll"].expand_as(nll)
    per_channel = nll + float(gamma) * attention + float(lam) * penalty
    return (channel_weights * per_channel).sum()


PAPER_METHOD_DEFAULTS = {
    # kind: (epochs, batch_size, lr, weight_decay, dropout, scaler_mode,
    #        selection_criterion, mc_samples)
    "hnn": (41, 100, 1e-3, 1e-3, 0.0, "grouped", "val_nll", 0),
    "ensemble": (41, 100, 1e-3, 1e-3, 0.0, "grouped", "val_mixture_nll", 0),
    "edl_nig": (50, 32, 1e-3, 0.0, 0.3, "minmax", "val_rmse", 0),
    "lstm_prob": (14, 43, 5.613e-4, 0.0, 0.03, "minmax", "val_nll", 0),
    "tcn_transformer_mc": (50, 128, 1e-3, 0.0, 0.2, "minmax", "val_rmse", 30),
    "msauq": (50, 128, 1e-3, 0.0, 0.1, "zscore_grouped", "val_rmse", 100),
    "bcpt": (200, 256, 5e-3, 0.0, 0.3, "minmax", "val_fused_rmse", 30),
}
