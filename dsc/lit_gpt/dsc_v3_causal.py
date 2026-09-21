"""DSC v3: causal seg_query fix for the chunk-mean leak.

Leak (v2): `dsc.py:144`
    seg_query = normed_chunks.mean(dim=2)              # [B, N, H]   ← LEAK
    u = self.dsc_router(seg_query)
The chunk-level mean aggregates positions 0..c-1 of chunk i into the routing
decision for chunk i. Position t < c-1 in chunk i is then routed using
positions t+1..c-1 of chunk i — a future leak.

Fix (v3, Option 2 — drop-in replacement, smallest delta):
    last_pos_per_chunk = normed_chunks[:, :, c - 1, :]              # [B, N, H]
    seg_query = torch.zeros_like(last_pos_per_chunk)                # [B, N, H]
    seg_query[:, 1:] = last_pos_per_chunk[:, :-1]                   # chunk i>=1 gets prev's last
    u = self.dsc_router(seg_query)

Chunk i's routing decision uses ONLY position (i-1)*c + (c-1) = i*c - 1 of the
input — the last position of chunk i-1, which is in the strict past for all
positions of chunk i. Zero leak surface.

Option 1 (per-position causal cumulative mean) requires redesigning
chunk_gdn2 invocation (per-position initial state is not well-defined for
a chunk-level recurrence), so it's NOT a drop-in fix. Documented in the
postmortem as future work. We ship Option 2 first.

This module is NOT imported by training or eval — we don't want to perturb
the running v2 pretrain. After v2 ckpt eval completes, apply the patch from
`apply_v3_option2_patch()` to `dsc/lit_gpt/dsc.py` and restart pretrain.

Verification (after patching):
    python dsc/scripts/smoke_test_routing_invariance.py   # MUST PASS
    python dsc/scripts/smoke_test_causality_strict.py     # MUST PASS
    python dsc/scripts/smoke_test_causality.py            # MUST PASS

If any test fails, do NOT launch v3 pretrain. Investigate Option 1
(cumulative mean + per-position memory scan) or Option 3 (drop seg_query
entirely, use position t's own query) instead.
"""

from __future__ import annotations

import torch

from .gdn2_ops.chunk_gdn2 import chunk_gdn2


def _dsc_memory_combine_v3_option2(
    self,
    q, k, g, b, w, cached_s, n_full, c, normed,
):
    """v3 Option 2: route chunk i using ONLY the last position of chunk i-1.

    Returns: [B, T_full, HV, V] memory contribution tensor.
    Identical to v2 except at the seg_query computation (lines equivalent to
    dsc.py:142-146).
    """
    B, T, HV, K = q.shape
    V = self.head_v_dim
    device = q.device
    topk = self.dsc_topk

    if n_full == 0 or topk == 0:
        return torch.zeros(B, T, HV, V, device=device, dtype=q.dtype)

    N = n_full

    # 1) Multi-res descriptor per past chunk (unchanged from v2).
    desc_list = []
    for s in cached_s:
        mean_v = s.mean(dim=3)
        max_v = s.max(dim=3).values
        weights = s.softmax(dim=3)
        attn_v = (s * weights).sum(dim=3)
        desc = torch.stack([mean_v, max_v, attn_v], dim=-1).flatten(start_dim=1)
        desc_list.append(desc.to(q.dtype))
    desc_stack = torch.stack(desc_list, dim=0)                          # [N, B, dd]
    desc_bjd = desc_stack.permute(1, 0, 2)                              # [B, N, dd]

    # 2) v3 CAUSAL seg_query: last position of PREVIOUS chunk per current chunk.
    # For chunk i >= 1: use normed[i*c - 1] (last pos of chunk i-1, strict past).
    # For chunk 0: no previous chunk -> zero (mask will zero its row anyway).
    normed_chunks = normed[:, :N * c].view(B, N, c, -1)                 # [B, N, c, H]
    last_pos_per_chunk = normed_chunks[:, :, c - 1, :]                  # [B, N, H]
    seg_query = torch.zeros_like(last_pos_per_chunk)                    # [B, N, H]
    seg_query[:, 1:] = last_pos_per_chunk[:, :-1]                       # shift by 1

    # 3) Router (same shape as v2).
    u = self.dsc_router(seg_query)                                      # [B, N, dd]

    # 4) Score + causal mask (unchanged from v2).
    logits = torch.einsum('bid,bjd->bij', u, desc_bjd)                  # [B, i=N, j=N]
    causal_mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=0)
    logits = logits.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

    # 5) Top-k (unchanged from v2).
    k_eff = min(topk, N - 1)
    if k_eff <= 0:
        return torch.zeros(B, T, HV, V, device=device, dtype=q.dtype)
    topk_vals, topk_idx = torch.topk(logits, k=k_eff, dim=-1)
    topk_weights = torch.softmax(topk_vals, dim=-1)
    topk_weights = torch.where(torch.isnan(topk_weights),
                                torch.zeros_like(topk_weights), topk_weights)

    # 6) Gather selected states (v2 gather fix preserved).
    cached_stack = torch.stack(cached_s, dim=0)                         # [N, B, HV, K, V]
    cached_bN = cached_stack.permute(1, 0, 2, 3, 4)                     # [B, N, HV, K, V]
    idx_exp = topk_idx.view(B, N, k_eff, 1, 1, 1).expand(B, N, k_eff, *cached_bN.shape[2:])
    gathered = torch.gather(
        cached_bN.unsqueeze(1).expand(B, N, N, *cached_bN.shape[2:]),
        dim=2, index=idx_exp,
    )                                                                   # [B, N, k_eff, HV, K, V]

    # 7) Memory scan per k (unchanged from v2 — chunk_gdn2 with v=0).
    q_chunks = q[:, :N * c].view(B, N, c, HV, K)
    k_chunks = k[:, :N * c].view(B, N, c, HV, K)
    g_chunks = g[:, :N * c].view(B, N, c, HV, K)
    b_chunks = b[:, :N * c].view(B, N, c, HV, K)
    w_chunks = w[:, :N * c].view(B, N, c, HV, V)

    q_flat = q_chunks.reshape(B * N, c, HV, K)
    k_flat = k_chunks.reshape(B * N, c, HV, K)
    g_flat = g_chunks.reshape(B * N, c, HV, K)
    b_flat = b_chunks.reshape(B * N, c, HV, K)
    w_flat = w_chunks.reshape(B * N, c, HV, V)
    v_zero = torch.zeros(B * N, c, HV, V, device=device, dtype=q.dtype)

    o_mem_list = []
    for k_idx in range(k_eff):
        gathered_k = gathered[:, :, k_idx]                              # [B, N, HV, K, V]
        gathered_k_flat = gathered_k.reshape(B * N, HV, K, V)
        o_k, _ = chunk_gdn2(
            q=q_flat, k=k_flat, v=v_zero, g=g_flat, b=b_flat, w=w_flat,
            A_log=self.A_log, dt_bias=self.dt_bias,
            initial_state=gathered_k_flat, output_final_state=False,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False,
            cu_seqlens=None,
        )
        o_mem_list.append(o_k)
    o_mem = torch.stack(o_mem_list, dim=1).view(B, N, k_eff, c, HV, V)

    # 8) Weighted combine + zero-pad (unchanged from v2).
    combined = torch.einsum('bnk,bnkchv->bnchv', topk_weights.to(o_mem.dtype), o_mem)
    combined_flat = combined.reshape(B, N * c, HV, V)
    if T > N * c:
        pad = torch.zeros(B, T - N * c, HV, V, device=device, dtype=combined_flat.dtype)
        combined_flat = torch.cat([combined_flat, pad], dim=1)
    return combined_flat


PATCH_INSTRUCTIONS = """
To apply v3 Option 2 (last-position-of-prev-chunk) fix:

In dsc/lit_gpt/dsc.py, replace lines 142-146:

    # 2) Per-chunk pooled query (from the normed block input).
    normed_chunks = normed[:, :N * c].view(B, N, c, -1)                 # [B, N, c, H]
    seg_query = normed_chunks.mean(dim=2)                               # [B, N, H]
    # Router: [B, N, H] @ [H, dd] -> [B, N, dd]
    u = self.dsc_router(seg_query)                                      # [B, N, dd]

with:

    # 2) v3 CAUSAL seg_query: last position of PREVIOUS chunk per current chunk.
    # Fixes the chunk-mean leak at v2:144 (future positions in chunk i could
    # flip topk_idx for chunk i's earlier positions).
    # For chunk i >= 1: use normed[i*c - 1] (last pos of chunk i-1, strict past).
    # For chunk 0: no previous chunk -> zero (mask will zero its row anyway).
    normed_chunks = normed[:, :N * c].view(B, N, c, -1)                 # [B, N, c, H]
    last_pos_per_chunk = normed_chunks[:, :, c - 1, :]                  # [B, N, H]
    seg_query = torch.zeros_like(last_pos_per_chunk)                    # [B, N, H]
    seg_query[:, 1:] = last_pos_per_chunk[:, :-1]                       # chunk i>=1 gets prev's last
    # Router: [B, N, H] @ [H, dd] -> [B, N, dd]
    u = self.dsc_router(seg_query)                                      # [B, N, dd]

After patching, run all 3 tests (must all PASS before launching v3 pretrain):
    python dsc/scripts/smoke_test_routing_invariance.py
    python dsc/scripts/smoke_test_causality_strict.py
    python dsc/scripts/smoke_test_causality.py

If any test fails, do NOT launch v3 pretrain. Investigate Option 1
(cumulative mean + per-position memory scan, requires architecture redesign)
or Option 3 (drop seg_query entirely, use position t's own query).
"""


if __name__ == "__main__":
    print(PATCH_INSTRUCTIONS)
