"""GDN-2 adapters for dense GRM and its linear Memory Soup equivalent."""

from __future__ import annotations

import torch

from dsc.mc_baseline.mc_grm import GatedResidualMemory, LinearMemorySoup
from dsc.mc_baseline.mc_ssc import SSCOutput

from .ssc import _segment_gdn2_batched


class GDN2GRM(GatedResidualMemory):
    """GRM configured for GDN-2's normalized matrix-state read."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_qk_dim: int,
        *,
        chunk_size: int = 256,
        route_block_size: int = 16,
    ) -> None:
        super().__init__(
            hidden_size,
            num_heads,
            head_qk_dim,
            chunk_size=chunk_size,
            normalize_queries=True,
            route_block_size=route_block_size,
        )


class GDN2MemorySoup(LinearMemorySoup):
    """Named Memory Soup adapter; exactly the same GDN-2 math as GDN2GRM."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_qk_dim: int,
        *,
        chunk_size: int = 256,
        route_block_size: int = 16,
    ) -> None:
        super().__init__(
            hidden_size,
            num_heads,
            head_qk_dim,
            chunk_size=chunk_size,
            normalize_queries=True,
            route_block_size=route_block_size,
        )


def gdn2_grm_forward(
    aggregator: GDN2GRM | GDN2MemorySoup,
    hidden_states: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    *,
    checkpoint_mode: str = "independent",
    chunk_gdn2_fn=None,
) -> SSCOutput:
    """Build independent GDN-2 segment memories, then apply dense MC."""
    if checkpoint_mode != "independent":
        raise ValueError(
            "GDN-2 GRM/Memory Soup v2 supports independent compressors only"
        )
    if chunk_gdn2_fn is None:
        from dsc.lit_gpt.gdn2_ops.chunk_gdn2 import (
            chunk_gdn2 as chunk_gdn2_fn,
        )

    online_output, memories = _segment_gdn2_batched(
        q,
        k,
        v,
        g,
        b,
        w,
        chunk_size=aggregator.chunk_size,
        chunk_gdn2_fn=chunk_gdn2_fn,
    )
    return aggregator(hidden_states, q, k, online_output, memories)


# The named API makes the paper terminology discoverable while preserving one
# canonical numerical path for linear GDN-2 memories.
gdn2_memory_soup_forward = gdn2_grm_forward
