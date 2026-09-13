"""Formal offline analysis bridge for final-model effective diffusion times."""

from typing import Optional

import torch

from .effective_diffusion import EffectiveDiffusionAnalyzer, grid_anchors

FORMAL_ANCHOR_GRID = 4
FORMAL_RHO = 0.95


def convert_final_effective_tau(
    effective_tau: torch.Tensor,
    height: int = 32,
    width: int = 32,
    analyzer: Optional[EffectiveDiffusionAnalyzer] = None,
) -> dict[str, torch.Tensor]:
    """Apply the frozen 4x4-anchor, Chebyshev, rho=.95 P2 protocol offline."""
    if not torch.is_tensor(effective_tau) or not effective_tau.is_floating_point():
        raise ValueError("effective_tau must be a floating-point tensor")
    if not torch.isfinite(effective_tau).all() or not (effective_tau >= 0).all():
        raise ValueError("effective_tau must contain finite non-negative values")

    analyzer = analyzer or EffectiveDiffusionAnalyzer(height=height, width=width)
    if (analyzer.height, analyzer.width) != (height, width):
        raise ValueError("analyzer dimensions do not match the requested grid")
    anchors = grid_anchors(
        height, width, rows=FORMAL_ANCHOR_GRID, columns=FORMAL_ANCHOR_GRID
    )

    source_device, source_dtype = effective_tau.device, effective_tau.dtype
    flat = effective_tau.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    unique_tau, inverse = torch.unique(flat, sorted=False, return_inverse=True)
    mean_distance, effective_ring = [], []
    for tau in unique_tau.tolist():
        result = analyzer.analyze_tau(tau, anchors=anchors, rho=FORMAL_RHO)
        mean_distance.append(result["summary"]["mean_distance"]["mean"])
        effective_ring.append(result["summary"]["effective_ring"]["median"])

    mean_dtype = torch.float64 if source_dtype == torch.float64 else torch.float32
    mean_values = torch.tensor(mean_distance, dtype=torch.float64)[inverse]
    ring_values = torch.tensor(effective_ring, dtype=torch.int64)[inverse]
    return {
        "mean_diffusion_distance": mean_values.reshape(effective_tau.shape).to(
            device=source_device, dtype=mean_dtype
        ),
        "effective_ring_95": ring_values.reshape(effective_tau.shape).to(
            device=source_device
        ),
    }


def extend_final_debug(
    debug: dict[str, torch.Tensor],
    height: int = 32,
    width: int = 32,
    analyzer: Optional[EffectiveDiffusionAnalyzer] = None,
) -> dict[str, torch.Tensor]:
    """Return copied debug data with formal offline spatial-range metrics."""
    if "effective_tau" not in debug:
        raise KeyError("debug data does not contain effective_tau")
    extended = dict(debug)
    extended.update(
        convert_final_effective_tau(
            debug["effective_tau"], height=height, width=width, analyzer=analyzer
        )
    )
    return extended
