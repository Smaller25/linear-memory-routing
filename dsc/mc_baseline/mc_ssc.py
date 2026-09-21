"""Paper-faithful Sparse Selective Caching (SSC) for linear RNN memories.

This module implements equations (16)--(17) of *Memory Caching: RNNs with
Growing Memory* (arXiv:2602.24281).  It deliberately contains no GDN kernel
code; GDN-1 and GDN-2 adapters live in their own canonical directories.

Tensor convention used here:
    hidden_states: [B, T, D]
    queries/keys:  [B, T, H, K]
    online_output: [B, T, H, V]
    memories:      [B, N, H, K, V]

The paper's ``MeanPooling(S^(i))`` (SSC §3.1) is implemented as a true mean
over L2-normalized segment keys, not the raw sum that the equation notation
suggests.  Raw sum gives descriptor magnitude sqrt(chunk_size)=16 which
collapses softmax(gate_logits) to one-hot at init and causes training to
diverge (loss 5.3 → 7.2 self-amplifying).  Mean-pool keeps magnitude ≈ 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SSCOutput:
    """SSC output plus routing diagnostics needed for reproducible evaluation."""

    output: torch.Tensor
    online_output: torch.Tensor
    cached_output: torch.Tensor
    route_indices: torch.Tensor
    route_weights: torch.Tensor
    online_weight: torch.Tensor
    route_scores: torch.Tensor


def segment_key_sums(keys: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Return mean of L2-normalized keys per segment (paper-faithful MeanPooling).

    Paper Eq 16 descriptor is conceptually ``MeanPooling(S^(i))`` — using the raw
    sum with chunk_size=256 L2-normalized keys gives magnitude sqrt(256)=16,
    which collapses softmax(gate_logits) to one-hot at init and causes training
    divergence. We mean-pool to keep descriptor magnitude ≈ 1 (= L2-normalized
    key magnitude), matching paper §3.1 "we also normalize γ_t^(i) using softmax".
    """
    if keys.ndim != 4:
        raise ValueError(f"keys must be [B,T,H,K], got {tuple(keys.shape)}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    _, length, _, _ = keys.shape
    out = []
    for start in range(0, length, chunk_size):
        seg = keys[:, start : min(start + chunk_size, length)]
        out.append(seg.mean(dim=1))
    return torch.stack(out, dim=1)


def causal_online_key_sums(keys: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Causal **mean**-pool descriptor within current chunk.

    Plain cumsum causes magnitude to grow 0→sqrt(chunk_size)=16 inside every
    chunk → online_score becomes position-dependent (chunk start = pure online,
    chunk end = pure cached). Mean-pool by current position count keeps
    magnitude ≈ 1 everywhere, so the online-vs-cached gate depends on token
    content rather than intra-chunk position.
    """
    _, length, _, _ = keys.shape
    out = torch.empty_like(keys)
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        seg = keys[:, start:stop]
        n = torch.arange(1, stop - start + 1, device=keys.device, dtype=keys.dtype)
        out[:, start:stop] = seg.cumsum(dim=1) / n.view(1, -1, 1, 1)
    return out


def linear_memory_read(
    queries: torch.Tensor,
    memories: torch.Tensor,
    *,
    scale: float,
    normalize_queries: bool,
) -> torch.Tensor:
    """Evaluate selected matrix-valued memories ``M_i(q_t)`` directly.

    Args:
        queries: [B, T, H, K]
        memories: [B, T, R, H, K, V]
    Returns:
        [B, T, R, H, V]

    Implementation note: the naive einsum path broadcasts the query to
    ``[B,T,R,H,K]`` and materialises a ``[B,T,R,H,K,V]`` intermediate (~1 GB
    per layer on bf16 at B=8 T=4096 R=2).  We instead contract via batched
    ``matmul`` on flattened (B,T,R,H) batches, which keeps peak memory to the
    ~32 MB result tensor plus a single 256 MB query broadcast.
    """
    b, t, h, k = queries.shape
    r = memories.shape[2]
    v = memories.shape[-1]
    q = F.normalize(queries.float(), p=2, dim=-1) if normalize_queries else queries.float()
    # (B,T,H,K) -> (B,T,R,H,K) -> (B*T*R*H, 1, K)
    q_exp = q.unsqueeze(2).expand(b, t, r, h, k).reshape(b * t * r * h, 1, k)
    # (B,T,R,H,K,V) -> (B*T*R*H, K, V)
    m = memories.float().reshape(b * t * r * h, k, v)
    # batched matmul -> (B*T*R*H, 1, V) -> (B,T,R,H,V)
    out = torch.bmm(q_exp * scale, m).reshape(b, t, r, h, v)
    return out


class SparseSelectiveCaching(nn.Module):
    """Equation (16)--(17) SSC router and aggregator.

    ``W_u`` is learnable, as specified by ``u_t = x_t W_u`` in the paper.
    The current online memory is always included.  Top-k selection applies
    only to completed past segments.  Gating is normalized jointly over the
    online memory and selected cached memories.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_qk_dim: int,
        *,
        topk: int = 2,
        chunk_size: int = 256,
        normalize_queries: bool = False,
        read_scale: float | None = None,
        read_block_size: int = 256,
    ) -> None:
        super().__init__()
        if topk < 0:
            raise ValueError(f"topk must be non-negative, got {topk}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_qk_dim = head_qk_dim
        self.topk = topk
        self.chunk_size = chunk_size
        self.normalize_queries = normalize_queries
        self.read_scale = head_qk_dim ** -0.5 if read_scale is None else read_scale
        if read_block_size <= 0:
            raise ValueError(f"read_block_size must be positive, got {read_block_size}")
        self.read_block_size = read_block_size
        self.connector = nn.Linear(hidden_size, num_heads * head_qk_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        queries: torch.Tensor,
        keys: torch.Tensor,
        online_output: torch.Tensor,
        memories: torch.Tensor,
    ) -> SSCOutput:
        """Aggregate online and cached memories with strict causal routing."""
        if hidden_states.ndim != 3 or queries.ndim != 4 or keys.shape != queries.shape:
            raise ValueError("expected hidden [B,T,D] and matching query/key [B,T,H,K]")
        batch, length, heads, key_dim = queries.shape
        if (heads, key_dim) != (self.num_heads, self.head_qk_dim):
            raise ValueError(
                f"query heads/dim {(heads, key_dim)} != configured "
                f"{(self.num_heads, self.head_qk_dim)}"
            )
        if online_output.shape[:3] != (batch, length, heads):
            raise ValueError("online_output must be [B,T,H,V]")
        if memories.ndim != 5 or memories.shape[:2] != (batch, (length + self.chunk_size - 1) // self.chunk_size):
            raise ValueError("memories must contain one [H,K,V] state per sequence segment")

        num_segments = memories.shape[1]
        u = self.connector(hidden_states).view(batch, length, heads, key_dim)
        summaries = segment_key_sums(keys, self.chunk_size)
        # Equation (16): r_t^i = <u_t, sum_{j in S_i} k_j>.
        all_scores = torch.einsum("bthk,bnhk->btn", u.float(), summaries.float())

        segment_ids = torch.arange(length, device=queries.device) // self.chunk_size
        eligible = torch.arange(num_segments, device=queries.device)[None, :] < segment_ids[:, None]
        past_scores = all_scores.masked_fill(~eligible.unsqueeze(0), -torch.inf)

        route_count = min(self.topk, num_segments)
        if route_count:
            top_scores, top_indices = torch.topk(past_scores, k=route_count, dim=-1)
            valid = torch.isfinite(top_scores)
            safe_indices = top_indices.masked_fill(~valid, 0)
        else:
            top_scores = all_scores.new_empty(batch, length, 0)
            top_indices = torch.empty(batch, length, 0, device=queries.device, dtype=torch.long)
            valid = torch.empty(batch, length, 0, device=queries.device, dtype=torch.bool)
            safe_indices = top_indices

        # The online descriptor is the current segment's causal prefix.  This
        # is the causal realization of gamma_t^(s); using the full current
        # segment would expose future tokens.
        online_summary = causal_online_key_sums(keys, self.chunk_size)
        online_score = torch.einsum("bthk,bthk->bt", u.float(), online_summary.float())
        gate_logits = torch.cat([online_score.unsqueeze(-1), top_scores], dim=-1)
        gate_valid = torch.cat(
            [torch.ones(batch, length, 1, device=queries.device, dtype=torch.bool), valid], dim=-1
        )
        gate_logits = gate_logits.masked_fill(~gate_valid, -torch.inf)
        gates = torch.softmax(gate_logits, dim=-1).to(online_output.dtype)

        online_weight = gates[..., :1]
        route_weights = gates[..., 1:]
        # Equation (17) cached-memory term, computed by the Triton-fused
        # gather+matmul kernel (``cached_memory_read.ssc_gather_read``).
        # That kernel reads selected segment memories on the fly inside the
        # matmul, so the [B, T, R, H, K, V] ~64 GB intermediate never
        # materialises.  At B=8 T=4096 H=16 K=V=128 R=2 on H200 this is
        # ~17 ms per layer fwd (~0.8 s for 16 layers x 3 passes/iter) vs ~4 s
        # for the PyTorch gather+checkpoint path it replaces.
        if route_count:
            # Kernel version dispatch (env var MC_KERNEL_VERSION):
            #   v2  (default): original implementation (mc_baseline/cached_memory_read)
            #   v3a : cache hints + BLOCK_T=2, bit-exact vs v2
            #   v3c : segment-conditional load + tl.dot TensorCores, max ~5% rel diff vs v2
            #        v3c uses bf16 TensorCore in bwd_mem → 7.7% PPL regression at 1B tokens
            #   v4  : v3c fwd/bwd_q/bwd_w + TF32 TensorCore bwd_mem (precision fix over v3.2)
            #        grad_mem 6.2x more precise than v3.2, recovers v2 quality
            #   v5  : all TF32 (fwd + bwd_q + bwd_w + bwd_mem). 1.6-1.8x tighter fwd
            #        than v3/v4 (the only version with TF32 fwd, others reuse v3c bf16).
            import os as _os
            _kv = _os.environ.get("MC_KERNEL_VERSION", "v2")
            if _kv == "v2":
                from dsc.mc_baseline.cached_memory_read import ssc_gather_read as _ssc_fn
            elif _kv == "v3a":
                from dsc.mc_v3 import ssc_gather_read_v3a as _ssc_fn
            elif _kv == "v3c":
                from dsc.mc_v3 import ssc_gather_read_v3c as _ssc_fn
            elif _kv == "v4":
                from dsc.mc_v4.cached_memory_read_v4 import ssc_gather_read_v4 as _ssc_fn
            elif _kv == "v5":
                from dsc.mc_v5.cached_memory_read_v5 import ssc_gather_read_v5 as _ssc_fn
            else:
                raise ValueError(f"unknown MC_KERNEL_VERSION={_kv!r} (expected v2|v3a|v3c|v4|v5)")
            cached_output = _ssc_fn(
                queries, memories, safe_indices, route_weights,
                scale=self.read_scale, normalize_queries=self.normalize_queries,
            ).to(online_output.dtype)
        else:
            cached_output = torch.zeros_like(online_output)
        output = online_weight.unsqueeze(-1) * online_output + cached_output
        return SSCOutput(
            output=output,
            online_output=online_output,
            cached_output=cached_output,
            route_indices=top_indices.masked_fill(~valid, -1),
            route_weights=route_weights,
            online_weight=online_weight,
            route_scores=top_scores.masked_fill(~valid, -torch.inf),
        )


# Backward-compatible name for callers that imported the old class.  Its
# constructor is intentionally the new paper-faithful signature.
MCSSC = SparseSelectiveCaching
