"""Stable routing metrics for SSC training/evaluation logs."""

from __future__ import annotations

import torch

from .mc_ssc import SSCOutput


def routing_metrics(result: SSCOutput) -> dict[str, float]:
    """Return JSON-serializable SSC utilization metrics."""
    valid = result.route_indices >= 0
    selected = result.route_weights.masked_select(valid)
    entropy_terms = torch.where(
        result.route_weights > 0,
        -result.route_weights.float() * result.route_weights.float().log(),
        torch.zeros_like(result.route_weights, dtype=torch.float32),
    )
    return {
        "online_weight_mean": result.online_weight.float().mean().item(),
        "cached_weight_mean_valid": selected.float().mean().item() if selected.numel() else 0.0,
        "cached_route_entropy_mean": entropy_terms.sum(dim=-1).mean().item(),
        "valid_cached_routes_per_token": valid.float().sum(dim=-1).mean().item(),
        "cached_output_norm": result.cached_output.float().norm(dim=(-2, -1)).mean().item(),
        "output_norm": result.output.float().norm(dim=(-2, -1)).mean().item(),
    }
