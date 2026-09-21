"""GDN-2 adapter for the paper-faithful Memory Caching SSC core.

The adapter changes only the aggregation/read path.  All GDN-2 segments are
scanned in ONE batched ``chunk_gdn2`` call (paper Section 3.4 "Independent
Compressors" — segments are independent so they parallelise across the batch
dimension).  The final matrix state per segment is then cached and evaluated
directly as ``M_i(q_t)``.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from dsc.mc_baseline.mc_ssc import SSCOutput, SparseSelectiveCaching


class GDN2SSC(SparseSelectiveCaching):
    """SSC connector configured for GDN-2's time-first state convention."""

    def __init__(self, hidden_size: int, num_heads: int, head_qk_dim: int, *, topk: int = 2,
                 chunk_size: int = 256, read_block_size: int = 1024) -> None:
        # chunk_gdn2 L2-normalizes q inside its kernel before applying 1/sqrt(K).
        # read_block_size kept for API stability; the Triton kernel handles blocking.
        super().__init__(hidden_size, num_heads, head_qk_dim, topk=topk,
                         chunk_size=chunk_size, normalize_queries=True,
                         read_block_size=read_block_size)


def _segment_gdn2_batched(q, k, v, g, b, w, *, chunk_size, chunk_gdn2_fn):
    """Run all GDN-2 segments in ONE batched call (paper's Independent Compressors).

    Reshapes [B, T, H, *] -> [B*N, chunk_size, H, *], runs chunk_gdn2 once on
    the B*N independent sequences (each starting from a zero state), then
    reshapes the per-segment outputs back to [B, T, H, *] and stacks the
    per-segment final states into [B, N, H, K, V].

    On H200 at B=8 T=4096 chunk_size=256 H=16 K=V=128:
      Old (16 sequential chunk_gdn2 calls): ~15 ms / layer
      New (1 batched call, B*N=128 sequences): ~4.4 ms / layer  -> 3.5x speedup

    Falls back to a sequential per-segment loop when T is not divisible by
    chunk_size (small unit-test shapes only; training always uses divisible T).
    """
    B, T = q.shape[0], q.shape[1]
    if T % chunk_size != 0:
        return _segment_gdn2_sequential(q, k, v, g, b, w, chunk_size=chunk_size,
                                        chunk_gdn2_fn=chunk_gdn2_fn)
    N = T // chunk_size

    def seg(x: torch.Tensor) -> torch.Tensor:
        # [B, T, H, *] -> [B, N, chunk_size, H, *] -> [B*N, chunk_size, H, *]
        return x.view(B, N, chunk_size, *x.shape[2:]).reshape(B * N, chunk_size, *x.shape[2:])

    out_bn, state_bn = chunk_gdn2_fn(
        q=seg(q), k=seg(k), v=seg(v), g=seg(g), b=seg(b), w=seg(w),
        initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False,
        cu_seqlens=None,
    )
    # out_bn: [B*N, chunk_size, H, V] -> [B, N, chunk_size, H, V] -> [B, T, H, V]
    online_output = out_bn.view(B, N, chunk_size, *out_bn.shape[2:]).reshape(B, T, *out_bn.shape[2:])
    # state_bn: [B*N, H, K, V] -> [B, N, H, K, V]
    memories = state_bn.view(B, N, *state_bn.shape[1:])
    return online_output, memories


def _segment_gdn2_sequential(q, k, v, g, b, w, *, chunk_size, chunk_gdn2_fn):
    """Fallback path used when T is not divisible by chunk_size."""
    length = q.shape[1]
    outputs, states = [], []
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        out, state = chunk_gdn2_fn(
            q=q[:, start:stop], k=k[:, start:stop], v=v[:, start:stop],
            g=g[:, start:stop], b=b[:, start:stop], w=w[:, start:stop],
            initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False,
            cu_seqlens=None,
        )
        outputs.append(out)
        states.append(state)
    return torch.cat(outputs, dim=1), torch.stack(states, dim=1)


def gdn2_ssc_forward(
    ssc: GDN2SSC,
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
    """Run GDN-2 recurrence in one batched call, then apply paper SSC equations.

    Inputs use GDN-2's native ``[B,T,H,*]`` layout and are expected to be the
    already activated projections produced by ``GatedDeltaNet2.forward``.

    Only ``checkpoint_mode="independent"`` is implemented: both segment scans
    below hardcode ``initial_state=None``.  The chained ``"checkpoint"`` mode of
    paper section 3.4 is rejected rather than silently ignored; ``mc_gdn1`` still
    offers it via ``scan_segments`` if the ablation is needed.
    """
    if checkpoint_mode != "independent":
        raise ValueError(
            "GDN-2 SSC supports independent compressors only, got "
            f"checkpoint_mode={checkpoint_mode!r}"
        )
    if chunk_gdn2_fn is None:
        from dsc.lit_gpt.gdn2_ops.chunk_gdn2 import chunk_gdn2 as chunk_gdn2_fn

    online_output, memories = _segment_gdn2_batched(
        q, k, v, g, b, w,
        chunk_size=ssc.chunk_size, chunk_gdn2_fn=chunk_gdn2_fn,
    )
    # Equation (16) uses the same k_j that the memory update consumes.
    # chunk_gdn2 normalizes keys inside the kernel, so normalize the router
    # descriptors identically instead of routing on the pre-kernel projection.
    routing_keys = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)
    return ssc(hidden_states, q, routing_keys, online_output, memories)
