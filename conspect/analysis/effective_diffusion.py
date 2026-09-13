"""Offline spatial-range analysis for the model's spectral diffusion kernel."""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Sequence
from typing import Optional

import torch

from conspect.model import SpectralDiffusion2D

Anchor = tuple[int, int]
DEFAULT_TAUS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def default_anchors(height: int, width: int) -> list[Anchor]:
    """Return center, near-center, top-edge, left-edge, and corner anchors."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    center = (height // 2, width // 2)
    candidates = [
        center,
        (max(center[0] - 1, 0), min(center[1] + 1, width - 1)),
        (0, width // 2),
        (height // 2, 0),
        (0, 0),
    ]
    return list(dict.fromkeys(candidates))


def grid_anchors(
    height: int, width: int, rows: int = 3, columns: int = 3
) -> list[Anchor]:
    """Return a regular grid of anchors including boundaries."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive")
    row_indices = (
        torch.linspace(0, height - 1, steps=rows).round().to(torch.int64).tolist()
    )
    column_indices = (
        torch.linspace(0, width - 1, steps=columns).round().to(torch.int64).tolist()
    )
    return list(dict.fromkeys((row, col) for row in row_indices for col in column_indices))


def _summary(values: Sequence[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
        "median": tensor.median().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
    }


class EffectiveDiffusionAnalyzer:
    """Compute impulse-response range metrics without entering model forward."""

    def __init__(
        self,
        height: int = 32,
        width: int = 32,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating point")
        self.height, self.width = int(height), int(width)
        self.dtype = dtype
        self.operator = SpectralDiffusion2D(
            self.height, self.width, dim=1, hidden_dim=1
        ).to(device="cpu", dtype=dtype)
        self.operator.eval()
        self._distance_cache: dict[Anchor, torch.Tensor] = {}
        self._metric_cache: dict[
            tuple[float, Anchor, float], tuple[float, int]
        ] = {}

    def _validate_anchor(self, anchor: Anchor) -> Anchor:
        if len(anchor) != 2:
            raise ValueError(f"expected (row, col) anchor, got {anchor}")
        row, col = int(anchor[0]), int(anchor[1])
        if not 0 <= row < self.height or not 0 <= col < self.width:
            raise ValueError(
                f"anchor {(row, col)} is outside {(self.height, self.width)}"
            )
        return row, col

    @staticmethod
    def _validate_tau(tau: float) -> float:
        tau = float(tau)
        if not math.isfinite(tau) or tau < 0.0:
            raise ValueError(f"tau must be finite and non-negative, got {tau}")
        return tau

    @staticmethod
    def _validate_rho(rho: float) -> float:
        rho = float(rho)
        if not math.isfinite(rho) or not 0.0 < rho <= 1.0:
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        return rho

    def chebyshev_distances(self, anchor: Anchor) -> torch.Tensor:
        """Return integer d_inf distance from anchor to every grid cell."""
        anchor = self._validate_anchor(anchor)
        if anchor not in self._distance_cache:
            row = torch.arange(self.height).view(-1, 1)
            col = torch.arange(self.width).view(1, -1)
            self._distance_cache[anchor] = torch.maximum(
                (row - anchor[0]).abs(), (col - anchor[1]).abs()
            )
        return self._distance_cache[anchor]

    @torch.no_grad()
    def normalized_impulse_response(
        self, tau: float, anchor: Anchor
    ) -> torch.Tensor:
        """Return normalized abs(IDCT(exp(-tau*lambda) * DCT(e_anchor)))."""
        tau = self._validate_tau(tau)
        anchor = self._validate_anchor(anchor)
        impulse = torch.zeros(
            1, self.height, self.width, 1, dtype=self.dtype, device="cpu"
        )
        impulse[0, anchor[0], anchor[1], 0] = 1.0
        frequency_response = (
            self.operator.dct_2d(impulse)
            * self.operator.spectral_kernel(tau).unsqueeze(-1)
        )
        response = self.operator.idct_2d(frequency_response)[0, :, :, 0].abs()
        total = response.sum()
        if not torch.isfinite(response).all() or not torch.isfinite(total):
            raise RuntimeError("impulse response contains NaN or Inf")
        if total.item() <= 0.0:
            raise RuntimeError("impulse response has zero total mass")
        return response / total

    @torch.no_grad()
    def anchor_metrics(
        self, tau: float, anchor: Anchor, rho: float = 0.95
    ) -> dict[str, object]:
        """Return mean distance and effective ring for one tau/anchor pair."""
        tau = self._validate_tau(tau)
        anchor = self._validate_anchor(anchor)
        rho = self._validate_rho(rho)
        cache_key = (tau, anchor, rho)
        if cache_key not in self._metric_cache:
            response = self.normalized_impulse_response(tau, anchor)
            distances = self.chebyshev_distances(anchor)
            mean_distance = (
                response * distances.to(dtype=response.dtype)
            ).sum().item()
            ring_mass = torch.bincount(
                distances.reshape(-1),
                weights=response.reshape(-1),
                minlength=int(distances.max().item()) + 1,
            )
            cumulative_mass = ring_mass.cumsum(dim=0)
            target_mass = rho * ring_mass.sum()
            effective_ring = int(
                torch.searchsorted(cumulative_mass, target_mass).item()
            )
            self._metric_cache[cache_key] = (mean_distance, effective_ring)
        mean_distance, effective_ring = self._metric_cache[cache_key]
        return {
            "anchor": anchor,
            "mean_distance": mean_distance,
            "effective_ring": effective_ring,
        }

    def analyze_tau(
        self,
        tau: float,
        anchors: Optional[Iterable[Anchor]] = None,
        rho: float = 0.95,
    ) -> dict[str, object]:
        """Analyze all anchors and summarize boundary-dependent metrics."""
        tau = self._validate_tau(tau)
        rho = self._validate_rho(rho)
        selected_anchors = (
            default_anchors(self.height, self.width)
            if anchors is None
            else list(anchors)
        )
        if not selected_anchors:
            raise ValueError("anchors must not be empty")
        per_anchor = [
            self.anchor_metrics(tau, anchor, rho) for anchor in selected_anchors
        ]
        mean_distances = [float(item["mean_distance"]) for item in per_anchor]
        effective_rings = [float(item["effective_ring"]) for item in per_anchor]
        return {
            "tau": tau,
            "rho": rho,
            "per_anchor": per_anchor,
            "summary": {
                "mean_distance": _summary(mean_distances),
                "effective_ring": _summary(effective_rings),
            },
        }

    def convert_effective_tau(
        self,
        effective_tau: torch.Tensor,
        anchor: Optional[Anchor] = None,
        rho: float = 0.95,
    ) -> dict[str, torch.Tensor]:
        """Convert any effective-tau tensor by exact cached offline evaluation."""
        if not torch.is_tensor(effective_tau) or not effective_tau.is_floating_point():
            raise ValueError("effective_tau must be a floating-point tensor")
        if not torch.isfinite(effective_tau).all() or not (effective_tau >= 0).all():
            raise ValueError("effective_tau must contain finite non-negative values")
        anchor = (
            (self.height // 2, self.width // 2)
            if anchor is None
            else self._validate_anchor(anchor)
        )
        rho = self._validate_rho(rho)
        source_device = effective_tau.device
        source_dtype = effective_tau.dtype
        flat_tau = effective_tau.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        unique_tau, inverse = torch.unique(flat_tau, sorted=False, return_inverse=True)
        unique_mean = []
        unique_ring = []
        for tau in unique_tau.tolist():
            metrics = self.anchor_metrics(tau, anchor, rho)
            unique_mean.append(float(metrics["mean_distance"]))
            unique_ring.append(int(metrics["effective_ring"]))
        mean_values = torch.tensor(unique_mean, dtype=torch.float64)[inverse]
        ring_values = torch.tensor(unique_ring, dtype=torch.int64)[inverse]
        mean_dtype = torch.float64 if source_dtype == torch.float64 else torch.float32
        return {
            "mean_diffusion_distance": mean_values.reshape(effective_tau.shape).to(
                device=source_device, dtype=mean_dtype
            ),
            "effective_ring": ring_values.reshape(effective_tau.shape).to(
                device=source_device
            ),
        }


def build_tau_mapping(
    analyzer: EffectiveDiffusionAnalyzer,
    taus: Sequence[float] = DEFAULT_TAUS,
    anchors: Optional[Iterable[Anchor]] = None,
    rho: float = 0.95,
) -> list[dict[str, float]]:
    """Build the requested tau-to-spatial-range summary table."""
    rows = []
    for tau in taus:
        analysis = analyzer.analyze_tau(tau, anchors=anchors, rho=rho)
        mean_summary = analysis["summary"]["mean_distance"]
        ring_summary = analysis["summary"]["effective_ring"]
        rows.append(
            {
                "tau": float(tau),
                "mean_distance_mean": mean_summary["mean"],
                "mean_distance_std": mean_summary["std"],
                "effective_ring_median": ring_summary["median"],
                "effective_ring_min": ring_summary["min"],
                "effective_ring_max": ring_summary["max"],
            }
        )
    return rows


def extend_debug_with_diffusion_metrics(
    debug: dict[str, torch.Tensor],
    analyzer: EffectiveDiffusionAnalyzer,
    anchor: Optional[Anchor] = None,
    rho: float = 0.95,
) -> dict[str, torch.Tensor]:
    """Return a copy of model debug data extended with offline range metrics."""
    if "effective_tau" not in debug:
        raise KeyError("debug data does not contain effective_tau")
    metrics = analyzer.convert_effective_tau(
        debug["effective_tau"], anchor=anchor, rho=rho
    )
    extended = dict(debug)
    extended["mean_diffusion_distance"] = metrics["mean_diffusion_distance"]
    extended[f"effective_ring_{round(100 * rho)}"] = metrics["effective_ring"]
    return extended


def _print_mapping(rows: Sequence[dict[str, float]]) -> None:
    columns = (
        "tau",
        "mean_distance_mean",
        "mean_distance_std",
        "effective_ring_median",
        "effective_ring_min",
        "effective_ring_max",
    )
    print(" ".join(f"{column:>23}" for column in columns))
    for row in rows:
        print(
            f"{row['tau']:23.4f} "
            f"{row['mean_distance_mean']:23.6f} "
            f"{row['mean_distance_std']:23.6f} "
            f"{row['effective_ring_median']:23.0f} "
            f"{row['effective_ring_min']:23.0f} "
            f"{row['effective_ring_max']:23.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--grid-size", type=int, default=0)
    args = parser.parse_args()
    analyzer = EffectiveDiffusionAnalyzer(args.height, args.width)
    anchors = (
        grid_anchors(args.height, args.width, args.grid_size, args.grid_size)
        if args.grid_size
        else default_anchors(args.height, args.width)
    )
    _print_mapping(build_tau_mapping(analyzer, anchors=anchors, rho=args.rho))


if __name__ == "__main__":
    main()
