"""Dense Memory Caching for linear recurrent memories.

This module implements Gated Residual Memory (GRM), equations (9)--(10) of
*Memory Caching: RNNs with Growing Memory* (arXiv:2602.24281v1).  It also
exposes the paper's Memory Soup name.  For a matrix-valued memory whose read is
linear in the state, Memory Soup and GRM are exactly the same computation:

    (sum_i gamma_i M_i)(q) == sum_i gamma_i M_i(q)

The GDN-2 adapter therefore uses one canonical implementation for both names.
Unlike SSC, dense Memory Caching reads every completed segment; there is no
top-k selection.

Tensor convention:
    hidden_states: [B, T, D]
    queries/keys:  [B, T, H, K]
    online_output: [B, T, H, V]
    memories:      [B, N, H, K, V]

Paper-faithful v3 (2026-07-27 fix):
    • MeanPooling(S^(i)) uses **keys** (Σ k_j), not hidden_states.
      SSC v2 already implemented this correctly; v3 reuses SSC's helpers.
    • Connector W_u projects to per-head dim (num_heads * head_qk_dim),
      matching SSC's interpretation of Eq. (10) and giving per-head scoring
      that is then summed over heads.
"""

from __future__ import annotations

import torch
from torch import nn

from .cached_memory_read import ssc_gather_read
from .mc_ssc import SSCOutput, segment_key_sums, causal_online_key_sums


def dense_cached_memory_read(
    queries: torch.Tensor,
    memories: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    *,
    scale: float,
    normalize_queries: bool,
    route_block_size: int = 16,
) -> torch.Tensor:
    """Read all dense routes with the proven SSC-v2 fused kernel.

    The v2 Triton kernel specializes its route count at compile time.  Long
    contexts can contain hundreds of segments, so routes are processed in
    bounded blocks instead of compiling one enormous unrolled kernel.  The
    block outputs are summed, which is exactly equivalent by linearity.
    """
    if route_block_size <= 0:
        raise ValueError(
            f"route_block_size must be positive, got {route_block_size}"
        )
    if indices.shape != weights.shape:
        raise ValueError("indices and weights must have matching [B,T,N] shapes")
    if indices.ndim != 3:
        raise ValueError("indices and weights must be [B,T,N]")

    route_count = indices.shape[-1]
    if route_count == 0:
        return queries.new_zeros(
            *queries.shape[:-1],
            memories.shape[-1],
        )

    output = None
    for start in range(0, route_count, route_block_size):
        stop = min(start + route_block_size, route_count)
        block = ssc_gather_read(
            queries,
            memories,
            indices[:, :, start:stop].contiguous(),
            weights[:, :, start:stop].contiguous(),
            scale=scale,
            normalize_queries=normalize_queries,
        )
        output = block if output is None else output + block
    assert output is not None
    return output


class GatedResidualMemory(nn.Module):
    """Paper GRM with dense softmax over online and all completed memories.

    Equation (10) (paper-faithful v3): ``u_t = x_t W_u`` is per-head (output
    dim = num_heads * head_qk_dim), and compared with mean-pooled L2-normalized
    segment **keys** ``MeanPooling(S^(i)) = Σ k_j`` (per-segment, per-head).
    The current segment uses only its causal prefix; cached entries are
    restricted to completed segments.  All eligible past segments contribute
    to the dense softmax (no top-k).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_qk_dim: int,
        *,
        chunk_size: int = 256,
        normalize_queries: bool = False,
        read_scale: float | None = None,
        route_block_size: int = 16,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_qk_dim <= 0:
            raise ValueError(
                f"head_qk_dim must be positive, got {head_qk_dim}"
            )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if route_block_size <= 0:
            raise ValueError(
                f"route_block_size must be positive, got {route_block_size}"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_qk_dim = head_qk_dim
        self.chunk_size = chunk_size
        self.normalize_queries = normalize_queries
        self.read_scale = (
            head_qk_dim ** -0.5 if read_scale is None else read_scale
        )
        self.route_block_size = route_block_size
        # Eq. (10) per-head connector: u_t = x_t W_u lives in the same per-head
        # space as the keys, matching SSC's interpretation.
        self.connector = nn.Linear(
            hidden_size, num_heads * head_qk_dim, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        queries: torch.Tensor,
        keys: torch.Tensor,
        online_output: torch.Tensor,
        memories: torch.Tensor,
    ) -> SSCOutput:
        """Apply equations (9)--(10) with strict causal segment eligibility."""
        if hidden_states.ndim != 3 or queries.ndim != 4:
            raise ValueError("expected hidden [B,T,D] and queries [B,T,H,K]")
        if keys.shape != queries.shape:
            raise ValueError("keys must match queries shape [B,T,H,K]")

        batch, length, hidden_size = hidden_states.shape
        query_batch, query_length, heads, key_dim = queries.shape
        if (query_batch, query_length) != (batch, length):
            raise ValueError("hidden_states and queries must share [B,T]")
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"hidden size {hidden_size} != configured {self.hidden_size}"
            )
        if (heads, key_dim) != (self.num_heads, self.head_qk_dim):
            raise ValueError(
                f"query heads/dim {(heads, key_dim)} != configured "
                f"{(self.num_heads, self.head_qk_dim)}"
            )
        if online_output.shape[:3] != (batch, length, heads):
            raise ValueError("online_output must be [B,T,H,V]")

        expected_segments = (length + self.chunk_size - 1) // self.chunk_size
        if (
            memories.ndim != 5
            or memories.shape[:2] != (batch, expected_segments)
            or memories.shape[2:4] != (heads, key_dim)
            or memories.shape[-1] != online_output.shape[-1]
        ):
            raise ValueError(
                "memories must contain one [H,K,V] state per sequence segment"
            )

        num_segments = memories.shape[1]
        # Eq. (10) u_t in per-head space [B, T, H, K].
        u = self.connector(hidden_states).view(
            batch, length, heads, key_dim
        )
        # MeanPooling(S^(i)) over keys, per SSC's paper-faithful implementation.
        summaries = segment_key_sums(keys, self.chunk_size)
        # Score is per-head then summed over heads -> [B, T, N].
        all_scores = torch.einsum(
            "bthk,bnhk->btn",
            u.float(),
            summaries.float(),
        )

        segment_ids = (
            torch.arange(length, device=queries.device) // self.chunk_size
        )
        memory_ids = torch.arange(num_segments, device=queries.device)
        eligible = memory_ids[None, :] < segment_ids[:, None]
        past_scores = all_scores.masked_fill(
            ~eligible.unsqueeze(0),
            -torch.inf,
        )

        online_summary = causal_online_key_sums(keys, self.chunk_size)
        online_score = torch.einsum(
            "bthk,bthk->bt",
            u.float(),
            online_summary.float(),
        )
        gate_logits = torch.cat(
            [online_score.unsqueeze(-1), past_scores],
            dim=-1,
        )
        gates = torch.softmax(gate_logits, dim=-1).to(online_output.dtype)
        online_weight = gates[..., :1]
        route_weights = gates[..., 1:]

        route_indices = memory_ids.view(1, 1, num_segments).expand(
            batch,
            length,
            num_segments,
        )
        route_valid = eligible.unsqueeze(0).expand(
            batch,
            length,
            num_segments,
        )
        safe_indices = route_indices.masked_fill(~route_valid, 0)

        cached_output = dense_cached_memory_read(
            queries,
            memories,
            safe_indices,
            route_weights,
            scale=self.read_scale,
            normalize_queries=self.normalize_queries,
            route_block_size=self.route_block_size,
        ).to(online_output.dtype)
        output = online_weight.unsqueeze(-1) * online_output + cached_output

        return SSCOutput(
            output=output,
            online_output=online_output,
            cached_output=cached_output,
            route_indices=route_indices.masked_fill(~route_valid, -1),
            route_weights=route_weights,
            online_weight=online_weight,
            route_scores=past_scores,
        )


class LinearMemorySoup(GatedResidualMemory):
    """Memory Soup for a linear matrix memory.

    This intentionally inherits GRM without overriding ``forward``.  GDN-2's
    read ``S^T q`` is linear in ``S``, so a second numerical path would merely
    duplicate an exactly equivalent computation and invite drift.
    """


# Concise paper names for public imports.
GRM = GatedResidualMemory
MemorySoup = LinearMemorySoup
