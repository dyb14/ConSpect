"""Standalone model with shared sample state and bounded spectral diffusion."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralDiffusion2D(nn.Module):
    """Heat operator with one learnable diffusion scale on a fixed grid."""

    def __init__(self, height: int, width: int, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.height, self.width = int(height), int(width)
        self.dim, self.hidden_dim = int(dim), int(hidden_dim)
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"invalid grid size: {(self.height, self.width)}")

        self.dwconv = nn.Conv2d(
            dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        # softplus(raw_base_tau) starts at 1.0 and remains strictly positive.
        initial_tau = torch.tensor(1.0)
        self.raw_base_tau = nn.Parameter(torch.log(torch.expm1(initial_tau)))

        # Fixed tensors follow model.to(device) but do not enter checkpoints.
        self.register_buffer(
            "dct_height", self._make_dct_matrix(height), persistent=False
        )
        self.register_buffer(
            "dct_width", self._make_dct_matrix(width), persistent=False
        )
        self.register_buffer(
            "laplacian_eigenvalues",
            self._make_laplacian_eigenvalues(height, width),
            persistent=False,
        )

    @staticmethod
    def _make_dct_matrix(size: int) -> torch.Tensor:
        positions = (torch.linspace(0, size - 1, size).view(1, -1) + 0.5) / size
        frequencies = torch.linspace(0, size - 1, size).view(-1, 1)
        weight = torch.cos(frequencies * positions * torch.pi) * math.sqrt(2 / size)
        weight[0] /= math.sqrt(2)
        return weight

    @staticmethod
    def _make_laplacian_eigenvalues(height: int, width: int) -> torch.Tensor:
        row = torch.arange(height, dtype=torch.float32).view(-1, 1)
        col = torch.arange(width, dtype=torch.float32).view(1, -1)
        row_eigenvalues = 4.0 * torch.sin(torch.pi * row / (2.0 * height)).square()
        col_eigenvalues = 4.0 * torch.sin(torch.pi * col / (2.0 * width)).square()
        return row_eigenvalues + col_eigenvalues

    @property
    def base_tau(self) -> torch.Tensor:
        """Positive scalar diffusion time shared by every frequency and channel."""
        return F.softplus(self.raw_base_tau) + torch.finfo(
            self.raw_base_tau.dtype
        ).tiny

    def spectral_kernel(self, tau: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return exp(-tau * lambda) without channel-specific parameters."""
        if tau is None:
            tau = self.base_tau
        tau = torch.as_tensor(
            tau,
            device=self.laplacian_eigenvalues.device,
            dtype=self.laplacian_eigenvalues.dtype,
        )
        if tau.numel() != 1:
            raise ValueError(f"expected scalar tau, got shape {tuple(tau.shape)}")
        return torch.exp(-tau.reshape(()) * self.laplacian_eigenvalues)

    def effective_tau(
        self, state_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Combine this unit's scalar base tau with an optional sample scale."""
        if state_scale is None:
            return self.base_tau
        state_scale = torch.as_tensor(
            state_scale,
            device=self.raw_base_tau.device,
            dtype=self.raw_base_tau.dtype,
        )
        if state_scale.ndim == 2 and state_scale.shape[1] == 1:
            state_scale = state_scale[:, 0]
        if state_scale.ndim != 1:
            raise ValueError(
                "expected state_scale shape (batch,) or (batch, 1), "
                f"got {tuple(state_scale.shape)}"
            )
        if not torch.isfinite(state_scale).all() or not (state_scale > 0.0).all():
            raise ValueError("state_scale must contain finite positive values")
        return self.base_tau * state_scale

    def _validate_shape(self, x: torch.Tensor) -> None:
        expected_x = (self.dim, self.height, self.width)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected_x:
            raise ValueError(
                f"expected x shape (batch, {expected_x}), got {tuple(x.shape)}"
            )

    def dct_2d(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the cached orthonormal DCT-II to a channels-last tensor."""
        x_freq = torch.matmul(self.dct_height, features.transpose(1, 2))
        return torch.matmul(self.dct_width, x_freq.transpose(1, 2))

    def idct_2d(self, x_freq: torch.Tensor) -> torch.Tensor:
        """Apply the inverse of :meth:`dct_2d`."""
        output = torch.matmul(self.dct_height.t(), x_freq.transpose(1, 2))
        return torch.matmul(self.dct_width.t(), output.transpose(1, 2))

    def _dct_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_shape(x)
        features = self.linear(self.dwconv(x).permute(0, 2, 3, 1))
        features, gate = features.chunk(2, dim=-1)
        return self.dct_2d(features), gate

    def _idct_output(self, x_freq: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        output = self.idct_2d(x_freq)
        output = self.out_linear(self.out_norm(output) * F.silu(gate))
        return output.permute(0, 3, 1, 2).contiguous()

    def _forward_impl(
        self, x: torch.Tensor, state_scale: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_freq, gate = self._dct_features(x)
        effective_tau = self.effective_tau(state_scale)
        if effective_tau.ndim == 0:
            heat_kernel = torch.exp(
                -effective_tau * self.laplacian_eigenvalues
            ).unsqueeze(-1)
        else:
            if effective_tau.shape[0] != x.shape[0]:
                raise ValueError(
                    f"state_scale batch {effective_tau.shape[0]} "
                    f"does not match input batch {x.shape[0]}"
                )
            heat_kernel = torch.exp(
                -effective_tau[:, None, None, None]
                * self.laplacian_eigenvalues[None, :, :, None]
            )
        return self._idct_output(x_freq * heat_kernel, gate), effective_tau

    def forward(
        self, x: torch.Tensor, state_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output, _ = self._forward_impl(x, state_scale)
        return output

    def forward_with_debug(
        self, x: torch.Tensor, state_scale: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output, effective_tau = self._forward_impl(x, state_scale)
        return output, {
            "base_tau": self.base_tau,
            "effective_tau": effective_tau,
        }


class StateController(nn.Module):
    """One shared sample-level controller for every diffusion unit."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 16,
        log_scale_limit: float = math.log(2.0),
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if log_scale_limit <= 0.0:
            raise ValueError("log_scale_limit must be positive")
        self.feature_dim = int(feature_dim)
        self.log_scale_limit = float(log_scale_limit)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, pooled_feature: torch.Tensor) -> torch.Tensor:
        if pooled_feature.ndim != 2 or pooled_feature.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected pooled feature shape (batch, {self.feature_dim}), "
                f"got {tuple(pooled_feature.shape)}"
            )
        delta = self.log_scale_limit * torch.tanh(
            self.mlp(pooled_feature).squeeze(-1)
        )
        return torch.exp(delta)


class BoundedHeatResUnit(nn.Module):
    """Sigmoid-bounded interpolation of local and spectral-diffusion paths."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.hco = SpectralDiffusion2D(height, width, out_channels, out_channels)
        self.heat_scale_logit = nn.Parameter(torch.logit(torch.tensor(0.1)))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def _fuse(
        self, residual: torch.Tensor, local: torch.Tensor, global_out: torch.Tensor
    ) -> torch.Tensor:
        alpha = torch.sigmoid(self.heat_scale_logit)
        fused = (1.0 - alpha) * local + alpha * global_out
        return self.conv2(F.relu(self.bn2(fused))) + residual

    def forward(
        self, x: torch.Tensor, state_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        local = self.conv1(F.relu(self.bn1(x)))
        global_out = self.hco(local, state_scale)
        return self._fuse(x, local, global_out)

    def forward_with_debug(
        self, x: torch.Tensor, state_scale: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        local = self.conv1(F.relu(self.bn1(x)))
        global_out, debug = self.hco.forward_with_debug(local, state_scale)
        debug["fusion_alpha"] = torch.sigmoid(self.heat_scale_logit)
        output = self._fuse(x, local, global_out)
        return output, debug


class ConSpect(nn.Module):
    """C/P/T fusion followed by one shared adaptive diffusion backbone."""

    def __init__(
        self,
        len_closeness: int = 3,
        len_period: int = 1,
        len_trend: int = 1,
        external_dim: Optional[int] = None,
        nb_flow: int = 2,
        map_height: int = 32,
        map_width: int = 32,
        nb_residual_unit: int = 4,
        controller_hidden_dim: int = 16,
        log_scale_limit: float = math.log(2.0),
    ) -> None:
        super().__init__()
        if any(length <= 0 for length in (len_closeness, len_period, len_trend)):
            raise ValueError("C/P/T lengths must all be positive")
        if nb_residual_unit <= 0:
            raise ValueError("nb_residual_unit must be positive")
        self.len_closeness = int(len_closeness)
        self.len_period = int(len_period)
        self.len_trend = int(len_trend)
        self.external_dim = external_dim
        self.nb_flow = nb_flow
        self.map_height, self.map_width = map_height, map_width
        self.nb_residual_unit = nb_residual_unit

        feature_dim = 64
        self.closeness_stem = nn.Conv2d(
            self.len_closeness * nb_flow, feature_dim, kernel_size=3, padding=1
        )
        self.period_stem = nn.Conv2d(
            self.len_period * nb_flow, feature_dim, kernel_size=3, padding=1
        )
        self.trend_stem = nn.Conv2d(
            self.len_trend * nb_flow, feature_dim, kernel_size=3, padding=1
        )
        self.fusion = nn.Conv2d(3 * feature_dim, feature_dim, kernel_size=1)
        self.residual_units = nn.ModuleList(
            BoundedHeatResUnit(feature_dim, feature_dim, map_height, map_width)
            for _ in range(nb_residual_unit)
        )
        self.prediction_head = nn.Conv2d(feature_dim, nb_flow, kernel_size=3, padding=1)
        self.state_controller = StateController(
            feature_dim=feature_dim,
            hidden_dim=controller_hidden_dim,
            log_scale_limit=log_scale_limit,
        )

        if external_dim:
            self.e_net = nn.Sequential(
                nn.Linear(external_dim, 10),
                nn.ReLU(inplace=True),
                nn.Linear(10, nb_flow * map_height * map_width),
            )

    def _validate_inputs(
        self,
        x_c: torch.Tensor,
        x_p: torch.Tensor,
        x_t: torch.Tensor,
        ext: Optional[torch.Tensor],
    ) -> None:
        expected_inputs = (
            ("x_c", x_c, self.len_closeness),
            ("x_p", x_p, self.len_period),
            ("x_t", x_t, self.len_trend),
        )
        batch = x_c.shape[0] if x_c.ndim > 0 else 0
        for name, tensor, length in expected_inputs:
            expected = (
                length * self.nb_flow,
                self.map_height,
                self.map_width,
            )
            if tensor.ndim != 4 or tuple(tensor.shape) != (batch, *expected):
                raise ValueError(
                    f"expected {name} shape (batch, {expected}), "
                    f"got {tuple(tensor.shape)}"
                )
        if ext is not None:
            expected_ext = (batch, self.external_dim)
            if not self.external_dim or ext.ndim != 2 or tuple(ext.shape) != expected_ext:
                raise ValueError(
                    f"expected ext shape {expected_ext}, got {tuple(ext.shape)}"
                )

    def _fused_feature(
        self, x_c: torch.Tensor, x_p: torch.Tensor, x_t: torch.Tensor
    ) -> torch.Tensor:
        stem_features = (
            self.closeness_stem(x_c), self.period_stem(x_p), self.trend_stem(x_t)
        )
        return self.fusion(torch.cat(stem_features, dim=1))

    def _state_scale(self, stem_feature: torch.Tensor) -> torch.Tensor:
        return self.state_controller(stem_feature.mean(dim=(2, 3)))

    def _finish_output(
        self, output: torch.Tensor, ext: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if hasattr(self, "e_net") and ext is not None:
            output = output + self.e_net(ext).view(
                -1, self.nb_flow, self.map_height, self.map_width
            )
        return torch.tanh(output)

    def forward(
        self,
        x_c: torch.Tensor,
        x_p: torch.Tensor,
        x_t: torch.Tensor,
        ext: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_inputs(x_c, x_p, x_t, ext)
        x = self._fused_feature(x_c, x_p, x_t)
        state_scale = self._state_scale(x)
        for layer in self.residual_units:
            x = layer(x, state_scale)
        return self._finish_output(self.prediction_head(x), ext)

    def forward_with_debug(
        self,
        x_c: torch.Tensor,
        x_p: torch.Tensor,
        x_t: torch.Tensor,
        ext: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return prediction plus shared-state diagnostics for every unit."""
        self._validate_inputs(x_c, x_p, x_t, ext)
        x = self._fused_feature(x_c, x_p, x_t)
        state_scale = self._state_scale(x)
        all_unit_debug = []
        for layer in self.residual_units:
            x, unit_debug = layer.forward_with_debug(x, state_scale)
            all_unit_debug.append(unit_debug)
        output = self._finish_output(self.prediction_head(x), ext)
        reported_state_scale = state_scale
        effective_tau = []
        for unit_debug in all_unit_debug:
            value = unit_debug["effective_tau"]
            effective_tau.append(
                value.expand(x_c.shape[0]) if value.ndim == 0 else value
            )
        debug = {
            "state_scale": reported_state_scale,
            "base_tau": torch.stack(
                [unit_debug["base_tau"] for unit_debug in all_unit_debug]
            ),
            "effective_tau": torch.stack(effective_tau, dim=1),
            "fusion_alpha": torch.stack(
                [unit_debug["fusion_alpha"] for unit_debug in all_unit_debug]
            ),
        }
        return output, debug

    def state_controller_parameters(self):
        yield from self.state_controller.parameters()

    def diffusion_units(self):
        for module in self.residual_units:
            yield module.hco

    def fusion_parameters(self):
        for module in self.residual_units:
            yield module.heat_scale_logit

    @torch.no_grad()
    def effective_fusion_coefficients(self) -> torch.Tensor:
        return torch.stack([torch.sigmoid(p) for p in self.fusion_parameters()])
