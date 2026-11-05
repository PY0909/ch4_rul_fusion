from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

#: Configurable nonlinearities for the LOSO search. "legacy" keeps each
#: module's original mixed activation (relu conv stem / gelu router+graph)
#: so default behaviour is bit-identical; the named options switch every
#: nonlinearity uniformly.
_ACTIVATION_FUNCTIONS = {"relu": F.relu, "gelu": F.gelu, "silu": F.silu}


def _activation_function(name: str, legacy):
    if name == "legacy":
        return legacy
    if name in _ACTIVATION_FUNCTIONS:
        return _ACTIVATION_FUNCTIONS[name]
    raise ValueError(f"activation must be 'legacy' or one of "
                     f"{sorted(_ACTIVATION_FUNCTIONS)}")


def _activation_module(name: str, legacy: str):
    modules = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
    if name == "legacy":
        return modules[legacy]()
    if name in modules:
        return modules[name]()
    raise ValueError(f"activation must be 'legacy' or one of {sorted(modules)}")


def _as_coords(coords: Optional[torch.Tensor], length: int, reference: torch.Tensor) -> torch.Tensor:
    if coords is None:
        return torch.arange(length, device=reference.device, dtype=reference.dtype)
    coords = torch.as_tensor(coords, device=reference.device, dtype=reference.dtype)
    if coords.ndim != 1 or coords.numel() != length:
        raise ValueError("time coordinates must have shape [length]")
    return coords


def _endpoint_groups(length: int, factor: int) -> list[tuple[int, int]]:
    if length < 1:
        raise ValueError("sequence length must be positive")
    if factor < 1:
        raise ValueError("downsampling factor must be positive")
    remainder = length % factor
    groups = []
    start = 0
    if remainder:
        groups.append((0, remainder))
        start = remainder
    while start < length:
        groups.append((start, min(start + factor, length)))
        start += factor
    return groups


def _causal_anchors(length: int, factor: int) -> list[int]:
    anchors = list(range(max(factor - 1, 0), length, factor))
    if not anchors or anchors[-1] != length - 1:
        anchors.append(length - 1)
    return anchors


class EndpointAlignedAverageDownsample(nn.Module):
    """Average pooling with ceil output length and an included latest endpoint."""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = int(factor)

    def forward(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None):
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, sensors, time, features]")
        _, _, length, _ = x.shape
        coords = _as_coords(coords, length, x)
        values, output_coords = [], []
        for start, end in _endpoint_groups(length, self.factor):
            values.append(x[:, :, start:end, :].mean(dim=2))
            output_coords.append(coords[start:end].mean())
        return torch.stack(values, dim=2), torch.stack(output_coords)


class LearnedDownsample(nn.Module):
    """Causal learnable low-pass downsampling used by A3."""

    def __init__(self, d_model: int, factor: int = 2, window_size: int = 5):
        super().__init__()
        self.factor = int(factor)
        self.window_size = int(window_size)
        self.logits = nn.Parameter(torch.zeros(d_model, window_size))

    def forward(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None):
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, sensors, time, features]")
        _, _, length, _ = x.shape
        coords = _as_coords(coords, length, x)
        values, output_coords = [], []
        for anchor in _causal_anchors(length, self.factor):
            start = max(0, anchor - self.window_size + 1)
            count = anchor - start + 1
            window = x[:, :, start:anchor + 1, :]
            weights = F.softmax(self.logits[:, :count], dim=-1).transpose(0, 1)
            values.append((window * weights.view(1, 1, count, -1)).sum(dim=2))
            output_coords.append((coords[start:anchor + 1] * weights.mean(dim=1)).sum())
        return torch.stack(values, dim=2), torch.stack(output_coords)


class MomentConstrainedDownsample(nn.Module):
    """Learnable causal filter preserving constant and affine responses."""

    def __init__(self, d_model: int, factor: int = 2, window_size: int = 5,
                 eta_max: float = 0.25):
        super().__init__()
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self.factor = int(factor)
        self.window_size = int(window_size)
        self.eta_max = float(eta_max)
        self.theta = nn.Parameter(torch.randn(d_model, window_size) * 0.02)
        self.gamma = nn.Parameter(torch.zeros(()))
        self._projection_cache = {}

    def _projection(self, count: int, coords: torch.Tensor) -> torch.Tensor:
        # The constraint matrix depends only on the centered coordinate window,
        # which is identical for every anchor with the same valid count, so the
        # projector is computed once per count (identical to recomputing).
        key = (count, coords.device, coords.dtype)
        cached = self._projection_cache.get(key)
        if cached is None:
            centered = coords - coords.mean()
            constraints = torch.stack([torch.ones_like(centered), centered], dim=0)
            cached = torch.eye(count, device=coords.device, dtype=coords.dtype) \
                - torch.linalg.pinv(constraints) @ constraints
            self._projection_cache[key] = cached
        return cached

    def _weights(self, count: int, coords: torch.Tensor) -> torch.Tensor:
        base = torch.full((count,), 1.0 / count, device=coords.device, dtype=coords.dtype)
        projection = self._projection(count, coords)
        direction = projection @ self.theta[:, :count].to(coords.dtype).transpose(0, 1)
        norm = direction.abs().sum(dim=0, keepdim=True).clamp_min(1.0)
        delta = self.eta_max * torch.tanh(self.gamma) * direction / norm
        return base[:, None] + delta

    def forward(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None):
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, sensors, time, features]")
        _, _, length, _ = x.shape
        coords = _as_coords(coords, length, x)
        values, output_coords = [], []
        for anchor in _causal_anchors(length, self.factor):
            start = max(0, anchor - self.window_size + 1)
            window_coords = coords[start:anchor + 1]
            weights = self._weights(window_coords.numel(), window_coords)
            window = x[:, :, start:anchor + 1, :]
            values.append((window * weights.reshape(1, 1, -1, x.shape[-1])).sum(dim=2))
            base = torch.full_like(window_coords, 1.0 / window_coords.numel())
            output_coords.append((window_coords * base).sum())
        return torch.stack(values, dim=2), torch.stack(output_coords)


def compute_trend_stats(sequence: torch.Tensor, coords: torch.Tensor):
    """Return duration-scaled OLS slopes and per-sensor mean R2."""
    if sequence.ndim != 4:
        raise ValueError("sequence must have shape [batch, sensors, time, features]")
    coords = _as_coords(coords, sequence.shape[2], sequence)
    centered_t = coords - coords.mean()
    denom = centered_t.square().sum()
    if sequence.shape[2] < 2 or float(denom) <= 1e-12:
        shape = sequence.shape[:2] + (sequence.shape[-1],)
        return sequence.new_zeros(shape), sequence.new_zeros(sequence.shape[:2])
    mean_x = sequence.mean(dim=2, keepdim=True)
    centered_x = sequence - mean_x
    slope = (centered_x * centered_t.view(1, 1, -1, 1)).sum(dim=2) / denom
    trend = slope * (coords[-1] - coords[0]).abs()
    fitted = mean_x + slope.unsqueeze(2) * centered_t.view(1, 1, -1, 1)
    ss_res = (sequence - fitted).square().sum(dim=2)
    ss_tot = centered_x.square().sum(dim=2)
    r2 = (1.0 - ss_res / ss_tot.clamp_min(1e-8)).clamp(0.0, 1.0)
    reliable = (ss_tot > 1e-8).to(sequence.dtype)
    return trend, (r2 * reliable).mean(dim=-1)


class CrossScaleRouter(nn.Module):
    """Router attention over scale tokens for each sensor."""

    def __init__(self, d_model: int, num_heads: int = 4, num_routers: int = 4,
                 dropout: float = 0.1, activation: str = "legacy"):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.routers = nn.Parameter(torch.randn(num_routers, d_model) * 0.02)
        self.sender = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                             batch_first=True)
        self.receiver = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                              batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), _activation_module(activation, "gelu"),
            nn.Dropout(dropout), nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, s, d = x.shape
        tokens = x.reshape(b * c, s, d)
        routers = self.routers.unsqueeze(0).expand(b * c, -1, -1)
        summary, _ = self.sender(routers, tokens, tokens, need_weights=False)
        routed, _ = self.receiver(tokens, summary, summary, need_weights=False)
        out = self.norm1(tokens + self.dropout(routed))
        out = self.norm2(out + self.dropout(self.ffn(out)))
        return out.reshape(b, c, s, d)


class SensorGraphBlock(nn.Module):
    """One shared sensor graph with optional level/trend dynamic logits."""

    def __init__(self, sensor_dim: int, d_model: int, layers: int = 1,
                 dropout: float = 0.1, graph_mode: str = "static",
                 activation: str = "legacy"):
        super().__init__()
        if graph_mode not in {"static", "level", "trend"}:
            raise ValueError("graph_mode must be static, level, or trend")
        self.graph_mode = graph_mode
        self.act = _activation_function(activation, F.gelu)
        self.src = nn.Parameter(torch.randn(sensor_dim, d_model) * 0.02)
        self.dst = nn.Parameter(torch.randn(sensor_dim, d_model) * 0.02)
        self.updates = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        if graph_mode in {"level", "trend"}:
            self.query = nn.Linear(d_model, d_model, bias=False)
            self.key = nn.Linear(d_model, d_model, bias=False)
            self.dynamic_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, trend: Optional[torch.Tensor] = None,
                reliability: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = torch.matmul(self.src, self.dst.transpose(0, 1)) / math.sqrt(x.shape[-1])
        if self.graph_mode in {"level", "trend"}:
            source = x if self.graph_mode == "level" else trend
            if source is None:
                raise ValueError("trend graph requires trend features")
            q, k = self.query(source), self.key(source)
            dynamic = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(x.shape[-1])
            if self.graph_mode == "trend" and reliability is not None:
                gate = torch.sqrt(reliability.unsqueeze(2).clamp_min(0.0) *
                                  reliability.unsqueeze(1).clamp_min(0.0))
                dynamic = dynamic * gate
            logits = logits.unsqueeze(0) + self.dynamic_scale * dynamic
        else:
            logits = logits.unsqueeze(0).expand(x.shape[0], -1, -1)
        eye = torch.eye(logits.shape[-1], device=x.device, dtype=x.dtype).unsqueeze(0)
        adjacency = torch.softmax(logits + eye, dim=-1)
        out = x
        for update, norm in zip(self.updates, self.norms):
            msg = torch.einsum("bij,bjd->bid", adjacency, out)
            out = norm(out + self.dropout(self.act(update(msg))))
        return out


class SensorPatchStem(nn.Module):
    """Per-sensor temporal CNN stem."""

    def __init__(self, d_model: int, hidden: int = 8, activation: str = "legacy"):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden, kernel_size=2, padding="same")
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, d_model, kernel_size=2, padding="same")
        self.bn2 = nn.BatchNorm1d(d_model)
        self.act = _activation_function(activation, F.relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c = x.shape
        series = x.permute(0, 2, 1).reshape(b * c, 1, l)
        h = self.act(self.bn1(self.conv1(series)))
        h = self.act(self.bn2(self.conv2(h)))
        return h.permute(0, 2, 1).reshape(b, c, l, -1)


_VARIANTS = {
    "A0": {"downsample_mode": "legacy", "graph_mode": "static", "readout": "legacy", "use_trend_features": False, "disable_shared_specific": True},
    "A1": {"downsample_mode": "legacy", "graph_mode": "static", "readout": "single", "use_trend_features": False, "disable_shared_specific": True},
    "A2": {"downsample_mode": "avg_endpoint", "graph_mode": "static", "readout": "single", "use_trend_features": False, "disable_shared_specific": True},
    "A3": {"downsample_mode": "learned", "graph_mode": "static", "readout": "single", "use_trend_features": False, "disable_shared_specific": True},
    "A4": {"downsample_mode": "moment", "graph_mode": "static", "readout": "single", "use_trend_features": False, "disable_shared_specific": True},
    "A5": {"downsample_mode": "moment", "graph_mode": "static", "readout": "single", "use_trend_features": True, "disable_shared_specific": True},
    "A6": {"downsample_mode": "moment", "graph_mode": "level", "readout": "single", "use_trend_features": True, "disable_shared_specific": True},
    "A7": {"downsample_mode": "moment", "graph_mode": "trend", "readout": "single", "use_trend_features": True, "disable_shared_specific": True},
}


class MultiScalePhaseGraphRUL(nn.Module):
    """Multi-scale patch-CNN RUL model with A1-A7 research variants."""

    def __init__(self, sensor_dim: int, condition_dim: int, d_model: int = 64,
                 num_scales: int = 3, num_heads: int = 4, num_routers: int = 4,
                 graph_layers: int = 1, dropout: float = 0.1,
                 disable_router: bool = False, disable_graph: bool = False,
                 disable_shared_specific: bool = False, alpha_shared: float = 0.0,
                 alpha_specific: float = 0.0, encoder: str = "linear",
                 input_norm: str = "none", variant: Optional[str] = None,
                 downsample_mode: str = "legacy", graph_mode: str = "static",
                 readout: str = "legacy", use_trend_features: bool = False,
                 trend_window_size: int = 5, condition_mode: str = "full",
                 activation: str = "legacy"):
        super().__init__()
        if sensor_dim < 1 or condition_dim < 0:
            raise ValueError("sensor_dim must be positive and condition_dim non-negative")
        if num_scales < 1:
            raise ValueError("num_scales must be positive")
        if activation not in {"legacy", *_ACTIVATION_FUNCTIONS}:
            raise ValueError(f"activation must be 'legacy' or one of "
                             f"{sorted(_ACTIVATION_FUNCTIONS)}")
        if encoder not in {"linear", "patch_cnn"}:
            raise ValueError("encoder must be 'linear' or 'patch_cnn'")
        if input_norm not in {"none", "bn"}:
            raise ValueError("input_norm must be 'none' or 'bn'")
        if downsample_mode not in {"legacy", "avg_endpoint", "learned", "moment"}:
            raise ValueError("invalid downsample_mode")
        if graph_mode not in {"static", "level", "trend"}:
            raise ValueError("invalid graph_mode")
        if readout not in {"legacy", "single"}:
            raise ValueError("readout must be legacy or single")
        if condition_mode not in {"full", "mean", "none"}:
            raise ValueError("condition_mode must be full, mean, or none")
        if variant is not None:
            variant = variant.upper()
            if variant not in _VARIANTS:
                raise ValueError(f"unknown variant {variant}; choose A0-A7")
            settings = _VARIANTS[variant]
            downsample_mode = settings["downsample_mode"]
            graph_mode = settings["graph_mode"]
            readout = settings["readout"]
            use_trend_features = settings["use_trend_features"]
            disable_shared_specific = settings["disable_shared_specific"]
        self.sensor_dim, self.condition_dim = sensor_dim, condition_dim
        self.d_model, self.num_scales = d_model, num_scales
        self.disable_router, self.disable_graph = disable_router, disable_graph
        self.disable_shared_specific = disable_shared_specific
        self.alpha_shared, self.alpha_specific = alpha_shared, alpha_specific
        self.encoder, self.input_norm = encoder, input_norm
        self.variant, self.downsample_mode = variant, downsample_mode
        self.graph_mode, self.readout = graph_mode, readout
        self.use_trend_features, self.condition_mode = use_trend_features, condition_mode

        if encoder == "patch_cnn":
            self.stem = SensorPatchStem(d_model, activation=activation)
            self.cond_proj = nn.Linear(max(condition_dim, 1), d_model)
        else:
            self.input_proj = nn.Linear(1 + condition_dim, d_model)
        self.input_bn = nn.BatchNorm1d(d_model) if input_norm == "bn" else None
        self.scale_convs = nn.ModuleList([nn.Conv1d(d_model, d_model, 3, padding=1) for _ in range(num_scales)])
        self.scale_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_scales)])
        self.scale_embedding = nn.Parameter(torch.randn(num_scales, d_model) * 0.02)
        self.router = CrossScaleRouter(d_model, num_heads, num_routers, dropout,
                                       activation=activation)
        self.graph = SensorGraphBlock(sensor_dim, d_model, graph_layers, dropout,
                                      graph_mode, activation=activation)
        self.shared_gate = nn.Linear(d_model, d_model)
        self.condition_pool = nn.Sequential(nn.Linear(max(condition_dim, 1), d_model), nn.GELU())
        head_dim = 5 * d_model if readout == "legacy" else d_model
        self.head = nn.Sequential(nn.Linear(head_dim, 2 * d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(2 * d_model, 1), nn.Softplus())
        self.trend_proj = nn.Linear(d_model, d_model) if use_trend_features else None
        self.downsamplers = nn.ModuleList()
        for _ in range(max(num_scales - 1, 0)):
            if downsample_mode == "avg_endpoint":
                self.downsamplers.append(EndpointAlignedAverageDownsample(2))
            elif downsample_mode == "learned":
                self.downsamplers.append(LearnedDownsample(d_model, 2, trend_window_size))
            elif downsample_mode == "moment":
                self.downsamplers.append(MomentConstrainedDownsample(d_model, 2, trend_window_size))
        self._last_regularization = torch.tensor(0.0)

    def _condition_input(self, condition: torch.Tensor) -> torch.Tensor:
        if self.condition_mode == "none":
            return torch.zeros_like(condition)
        if self.condition_mode == "mean":
            return condition.mean(dim=1, keepdim=True).expand_as(condition)
        return condition

    def _make_scales(self, features: torch.Tensor):
        result, trends, reliabilities = [], [], []
        b, c, _, d = features.shape
        coords = torch.arange(features.shape[2], device=features.device, dtype=features.dtype)
        current, current_coords = features, coords
        for idx in range(self.num_scales):
            if idx > 0:
                if self.downsample_mode == "legacy":
                    flat = current.reshape(b * c, current.shape[2], d).transpose(1, 2)
                    stride = min(2, current.shape[2])
                    current = F.avg_pool1d(flat, kernel_size=stride, stride=stride,
                                           ceil_mode=False).transpose(1, 2).reshape(b, c, -1, d)
                    current_coords = F.avg_pool1d(current_coords[None, None], kernel_size=stride,
                                                  stride=stride, ceil_mode=False).flatten()
                else:
                    current, current_coords = self.downsamplers[idx - 1](current, current_coords)
            trend, reliability = compute_trend_stats(current, current_coords)
            conv = self.scale_convs[idx](current.reshape(b * c, current.shape[2], d).transpose(1, 2))
            pooled = conv.mean(dim=-1).reshape(b, c, d)
            if self.trend_proj is not None:
                pooled = pooled + self.trend_proj(trend)
            result.append(self.scale_norms[idx](pooled) + self.scale_embedding[idx])
            trends.append(trend)
            reliabilities.append(reliability)
        return result, trends, reliabilities

    def _regularization(self, shared: torch.Tensor, specific: torch.Tensor) -> torch.Tensor:
        if self.disable_shared_specific or (self.alpha_shared == 0 and self.alpha_specific == 0):
            return shared.new_zeros(())
        sh, sp = F.normalize(shared.flatten(0, 2), dim=-1), F.normalize(specific.flatten(0, 2), dim=-1)
        orthogonal = ((sh * sp).sum(-1)).square().mean()
        return self.alpha_shared * orthogonal + self.alpha_specific * orthogonal

    def encode_features(self, x: torch.Tensor,
                        condition: Optional[torch.Tensor] = None):
        """Return the single-readout A6 representation before the point head."""
        if self.readout != "single":
            raise ValueError("encode_features requires the single-readout A6 path")
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, sensors]")
        b, l, c = x.shape
        if c != self.sensor_dim:
            raise ValueError(f"expected {self.sensor_dim} sensors, got {c}")
        if condition is None:
            condition = x.new_zeros((b, l, self.condition_dim))
        if condition.ndim != 3 or condition.shape[:2] != (b, l) or \
                condition.shape[-1] != self.condition_dim:
            raise ValueError("condition must have shape [batch, time, conditions]")
        condition = self._condition_input(condition)
        if self.encoder == "patch_cnn":
            features = self.stem(x)
            if self.condition_dim:
                features = features + self.cond_proj(condition).unsqueeze(1)
        else:
            cond_for_sensor = condition.unsqueeze(2).expand(-1, -1, c, -1)
            features = self.input_proj(torch.cat([x.unsqueeze(-1), cond_for_sensor], dim=-1))
            features = features.permute(0, 2, 1, 3)
        if self.input_bn is not None:
            b_, c_, l_, d_ = features.shape
            features = self.input_bn(features.reshape(-1, d_)).reshape(b_, c_, l_, d_)
        scale_list, trend_list, reliability_list = self._make_scales(features)
        routed = torch.stack(scale_list, dim=2)
        if not self.disable_router:
            routed = self.router(routed)
        sensor_base = routed.mean(dim=2)
        graph_trend = torch.stack(trend_list, dim=2).mean(dim=2)
        graph_reliability = torch.stack(reliability_list, dim=2).mean(dim=2)
        graph_features = sensor_base if self.disable_graph else self.graph(
            sensor_base, graph_trend, graph_reliability)
        enriched = routed + graph_features.unsqueeze(2)
        pooled = enriched.mean(dim=(1, 2))
        if self.condition_dim and self.condition_mode != "none":
            pooled = pooled + self.condition_pool(condition.mean(dim=1))
        return pooled, pooled.new_zeros(())

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None,
                return_aux: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if self.readout == "single":
            pooled, regularization = self.encode_features(x, condition)
            prediction = self.head(pooled).squeeze(-1)
            self._last_regularization = regularization.detach()
            return (prediction, regularization) if return_aux else prediction
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, sensors]")
        b, l, c = x.shape
        if c != self.sensor_dim:
            raise ValueError(f"expected {self.sensor_dim} sensors, got {c}")
        if condition is None:
            condition = x.new_zeros((b, l, self.condition_dim))
        if condition.ndim != 3 or condition.shape[:2] != (b, l) or condition.shape[-1] != self.condition_dim:
            raise ValueError("condition must have shape [batch, time, conditions]")
        condition = self._condition_input(condition)
        if self.encoder == "patch_cnn":
            features = self.stem(x)
            if self.condition_dim:
                features = features + self.cond_proj(condition).unsqueeze(1)
        else:
            cond_for_sensor = condition.unsqueeze(2).expand(-1, -1, c, -1)
            features = self.input_proj(torch.cat([x.unsqueeze(-1), cond_for_sensor], dim=-1)).permute(0, 2, 1, 3)
        if self.input_bn is not None:
            b_, c_, l_, d_ = features.shape
            features = self.input_bn(features.reshape(-1, d_)).reshape(b_, c_, l_, d_)
        scale_list, trend_list, reliability_list = self._make_scales(features)
        routed = torch.stack(scale_list, dim=2)
        if not self.disable_router:
            routed = self.router(routed)
        sensor_base = routed.mean(dim=2)
        graph_trend = torch.stack(trend_list, dim=2).mean(dim=2)
        graph_reliability = torch.stack(reliability_list, dim=2).mean(dim=2)
        graph_features = sensor_base if self.disable_graph else self.graph(sensor_base, graph_trend, graph_reliability)
        enriched = routed + graph_features.unsqueeze(2)
        if self.disable_shared_specific:
            shared_tokens = specific_tokens = enriched
        else:
            gate = torch.sigmoid(self.shared_gate(enriched))
            shared_tokens, specific_tokens = gate * enriched, (1.0 - gate) * enriched
        shared, specific = shared_tokens.mean(dim=(1, 2)), specific_tokens.mean(dim=(1, 2))
        graph_pool, last_state = graph_features.mean(dim=1), enriched[:, :, 0, :].mean(dim=1)
        cond_summary = self.condition_pool(condition.mean(dim=1)) if self.condition_dim else self.condition_pool(x.new_zeros((b, 1)))
        prediction = self.head(torch.cat([shared, specific, graph_pool, last_state, cond_summary], dim=-1)).squeeze(-1)
        regularization = self._regularization(shared_tokens, specific_tokens)
        self._last_regularization = regularization.detach()
        return (prediction, regularization) if return_aux else prediction
