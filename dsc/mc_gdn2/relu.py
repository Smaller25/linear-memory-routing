"""GDN-2 adapter for ReLU Dynamic Selection (fluid multi-state routing).

Mirrors ``grm.py`` but hosts ``mc_baseline.mc_relu.ReLUSelectiveCaching``.
The segment scan is the same batched independent-compressor call every MC
arm uses (``_segment_gdn2_batched``); routing keys are L2-normalized exactly
like the Hard Top-k SSC adapter (``ssc.gdn2_ssc_forward``) so the ONLY
difference between the ReLU arm and the Hard Top-k arms is the gate.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from dsc.mc_baseline.mc_relu import ReLUSelectiveCaching, ReLUSSCOutput

from .ssc import _segment_gdn2_batched


class GDN2ReLUSelectiveCaching(ReLUSelectiveCaching):
    """ReLU gate configured for GDN-2's normalized matrix-state read.

    ``normalize_queries=True`` matches ``GDN2SSC``: chunk_gdn2 L2-normalizes q
    inside its kernel, so the read path must score with the same q the memory
    actually consumes.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_qk_dim: int,
        *,
        chunk_size: int = 256,
        route_block_size: int = 16,
        normalize_gate: bool = True,
    ) -> None:
        super().__init__(
            hidden_size,
            num_heads,
            head_qk_dim,
            chunk_size=chunk_size,
            normalize_queries=True,
            route_block_size=route_block_size,
            normalize_gate=normalize_gate,
        )


def gdn2_relu_forward(
    aggregator: GDN2ReLUSelectiveCaching,
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
) -> ReLUSSCOutput:
    """Build independent GDN-2 segment memories, then apply the ReLU gate."""
    if checkpoint_mode != "independent":
        raise ValueError(
            "GDN-2 ReLU Dynamic Selection supports independent compressors "
            f"only, got checkpoint_mode={checkpoint_mode!r}"
        )
    if chunk_gdn2_fn is None:
        from dsc.lit_gpt.gdn2_ops.chunk_gdn2 import (
            chunk_gdn2 as chunk_gdn2_fn,
        )

    online_output, memories = _segment_gdn2_batched(
        q, k, v, g, b, w,
        chunk_size=aggregator.chunk_size, chunk_gdn2_fn=chunk_gdn2_fn,
    )
    # Same routing-key convention as the Hard Top-k SSC adapter: chunk_gdn2
    # normalizes keys in-kernel, so descriptors are built from the normalized
    # keys.  Keeping this identical to SSC isolates the gate as the only
    # changed factor between the ReLU arm and the Top-k arms.
    routing_keys = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)
    return aggregator(hidden_states, q, routing_keys, online_output, memories)
