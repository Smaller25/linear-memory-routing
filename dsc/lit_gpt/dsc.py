"""DSC (Dynamic Sparse Caching) for GDN-2 — v4.

v4 = v3 + two HiLS/BlockSearch-inspired additions (see docs/DSC_IDEAS_FROM_HILS_ARXIV_KO.md):

(Idea 1) Learnable chunk summary (4th descriptor view)
    v3 used 3 FIXED views (mean / max / softmax-attn pool over V axis). v4 ADDS a 4th
    view produced by a learnable attention pool. Adds HV*K*V params per layer
    (~262K at HV=16, K=V=128), <0.1% of 370M. Initialized small so v4 starts close to v3.

(Idea 2) Entropy bias on routing logits
    v3 router logit(i,j) = <seg_query_i, descriptor_j>. v4 adds ent_bias_scale * H(state_j)
    where H = entropy of softmax(state_j, dim=V) averaged over (HV, K). This lets the
    router distinguish concentrated chunks (low entropy) from diffuse chunks (high entropy).
    ent_bias_scale is a learnable scalar init=0, so v4 starts EXACTLY at v3 behavior.

v3 leak-fix preserved:
    seg_query = zeros-like + last-pos-of-prev-chunk fill (no future leak).
    causal_mask = triu(diagonal=0) (strict past, self NOT allowed).

The result is added to the main GDN-2 output before o_norm + o_proj, so DSC sits parallel
to the existing recurrence rather than replacing it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .gdn2 import GatedDeltaNet2
from .gdn2_ops.chunk_gdn2 import chunk_gdn2


def _multi_res_descriptor(state: torch.Tensor) -> torch.Tensor:
    """[B, HV, K, V] -> [B, 3*HV*K] via mean / max / softmax-attn pool over V."""
    mean_v = state.mean(dim=3)                                   # [B, HV, K]
    max_v = state.max(dim=3).values                              # [B, HV, K]
    weights = state.softmax(dim=3)
    attn_v = (state * weights).sum(dim=3)                        # [B, HV, K]
    desc = torch.stack([mean_v, max_v, attn_v], dim=-1)          # [B, HV, K, 3]
    return desc.flatten(start_dim=1)                             # [B, 3*HV*K]


def _chunk_entropy(state: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """[B, HV, K, V] -> [B] per-chunk entropy of the softmax(state, dim=V) distribution.

    For each (HV, K) position we treat the V-vector as a categorical distribution after
    softmax and compute its Shannon entropy in nats. We then average over (HV, K) to get
    a single scalar per chunk. Range: [0, log(V)].

    High entropy = state is diffuse (no single value dominates). Low entropy = concentrated.
    """
    # Cast to fp32 for stable softmax + log.
    s = state.float()
    p = s.softmax(dim=-1)
    ent_per_kv = -(p * (p + eps).log()).sum(dim=-1)              # [B, HV, K]
    return ent_per_kv.mean(dim=[1, 2])                           # [B]


def _slice_chunk(t: torch.Tensor, c: int, i: int):
    """Slice one chunk along dim=1 (T axis)."""
    return t[:, i * c:(i + 1) * c]


class DSCGatedDeltaNet2(GatedDeltaNet2):
    """GDN-2 with Dynamic Sparse Caching (DSC) — v4.

    v4 additions over v3 (all v3 leak-fixes preserved):
      * desc_attn_v: learnable attention query [HV, K, V] producing a 4th descriptor
        view. Concat with v3's 3 fixed views (mean/max/softmax-attn) → 4*HV*K descriptor.
        Init ~ N(0, 1/V) → output ≈ mean_v at step 0.
      * ent_bias_scale: learnable scalar (init=0) gating an entropy-of-state term
        added to router logits. lets router distinguish diffuse vs concentrated chunks.
        Init=0 → exactly v3 routing at step 0.

    Args:
        dsc_chunk_size: # tokens per chunk for state caching (default 256).
        dsc_topk: # cached states each chunk reads from (default 2).
        dsc_combine_alpha: learnable scalar (init 0.5) gating the memory contribution.
    """

    def __init__(self, hidden_size: int, head_dim: int = 128, num_heads: int = 16,
                 num_v_heads: int | None = None, **kwargs) -> None:
        super().__init__(hidden_size=hidden_size, head_dim=head_dim,
                         num_heads=num_heads, num_v_heads=num_v_heads, **kwargs)
        # Default DSC config; override via assign-after-init (set by Block when config sets it).
        self.dsc_chunk_size: int = 256
        self.dsc_topk: int = 2

        # === v4 additions (HiLS/BlockSearch-inspired) ============================
        HV, K, V = self.num_v_heads, self.head_k_dim, self.head_v_dim
        # (Idea 1) Learnable attention query for the 4th descriptor view.
        # Shape [HV, K, V] — for each (head, key-slot), a learnable vector that scores
        # the V axis of the state. Produces one scalar per (HV, K) via dot product,
        # then softmax over K gives the attention pool weights.
        # Scaled init so the 4th view starts near zero (v4 ≈ v3 at step 0).
        scale = 1.0 / math.sqrt(V)
        self.desc_attn_v = nn.Parameter(torch.randn(HV, K, V) * scale)
        # (Idea 2) Entropy bias: learnable scalar (init=0 → exactly v3 at step 0).
        self.ent_bias_scale = nn.Parameter(torch.tensor([0.0]))
        # ==========================================================================

        # v4 descriptor = 3-view (mean/max/attn) + 1 learnable view = 4 views.
        desc_dim = 4 * self.num_v_heads * self.head_k_dim
        # Router: linear from chunk-pooled query to descriptor dim, scored by dot product.
        self.dsc_router = nn.Linear(hidden_size, desc_dim, bias=False)
        # Learnable scale on the memory contribution; init so DSC contribution is meaningful
        # at the start of training but does not dominate the main scan.
        # 1D tensor (not scalar) — FSDP requires numel >= 1, not 0-dim.
        self.dsc_combine_alpha = nn.Parameter(torch.tensor([0.5]))

    def _learnable_attn_view(self, state: torch.Tensor) -> torch.Tensor:
        """[B, HV, K, V] -> [B, HV, K] via learnable attention pool over V.

        For each (B, HV, K), compute attention weights over V via a learnable per-(HV,K)
        query vector (`desc_attn_v`), then weighted-sum the V axis. Output is a [B, HV, K]
        summary — one more view to concat with the 3 fixed views from `_multi_res_descriptor`.

        Init: `desc_attn_v ~ N(0, 1/V)` so the pointwise logits `state * q` are small →
        softmax over V is near-uniform → output ≈ `mean_v`. This means v4 at step 0 is
        very close to v3 with a (near-zero) extra feature; gradients decide whether to
        specialize away from mean-pool.
        """
        # state: [B, HV, K, V], self.desc_attn_v: [HV, K, V]
        logits_v = state * self.desc_attn_v                         # [B, HV, K, V]
        weights_v = logits_v.softmax(dim=-1)                        # softmax over V
        summary = (state * weights_v).sum(dim=-1)                   # [B, HV, K]
        return summary

    # ---- DSC forward scan: per-chunk state propagation, returns main output + cached states ----
    def _dsc_forward_scan(self, q, k, v, g, b, w):
        """Run chunk_gdn2 sequentially over chunks of size dsc_chunk_size.

        Each call propagates state from the previous chunk; we cache the final state per chunk.
        Returns:
            o_main:    [B, T, HV, V]  -- concatenated main-scan outputs (covers all T positions)
            cached_s:  list of [B, HV, K, V] tensors, length n_full (one per full chunk)
        """
        B, T, HV, K = q.shape
        c = self.dsc_chunk_size
        n_full = T // c
        V = v.shape[3]
        o_chunks: list[torch.Tensor] = []
        cached_s: list[torch.Tensor] = []
        state = None
        for i in range(n_full):
            q_c = _slice_chunk(q, c, i)
            k_c = _slice_chunk(k, c, i)
            v_c = _slice_chunk(v, c, i)
            g_c = _slice_chunk(g, c, i)
            b_c = _slice_chunk(b, c, i)
            w_c = _slice_chunk(w, c, i)
            o_c, state = chunk_gdn2(
                q=q_c, k=k_c, v=v_c, g=g_c, b=b_c, w=w_c,
                A_log=self.A_log, dt_bias=self.dt_bias,
                initial_state=state, output_final_state=True,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False,
                cu_seqlens=None,
            )
            o_chunks.append(o_c)
            cached_s.append(state)
        # Tail: if T is not a multiple of c, run chunk_gdn2 over the remaining
        # T - n_full*c positions with initial_state from the last full chunk.
        # This handles RULER's variable-length inputs (e.g., len 3979 with c=256).
        # The tail is NOT cached (no descriptor / routing for partial chunks).
        tail = T - n_full * c
        if tail > 0:
            q_t = q[:, n_full * c:]
            k_t = k[:, n_full * c:]
            v_t = v[:, n_full * c:]
            g_t = g[:, n_full * c:]
            b_t = b[:, n_full * c:]
            w_t = w[:, n_full * c:]
            o_t, _ = chunk_gdn2(
                q=q_t, k=k_t, v=v_t, g=g_t, b=b_t, w=w_t,
                A_log=self.A_log, dt_bias=self.dt_bias,
                initial_state=state, output_final_state=False,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False,
                cu_seqlens=None,
            )
            o_chunks.append(o_t)
        o_main = torch.cat(o_chunks, dim=1)
        return o_main, cached_s, n_full, c

    def _dsc_memory_combine(self, q, k, g, b, w, cached_s, n_full, c, normed):
        """Top-k routing + batched memory scan + weighted combine.

        For each chunk i, route to pick top-k from cached_s[:i] (causal). Stack the (i, k)
        selected states into the batch dim and run ONE chunk_gdn2 call (v=0) to get their
        contributions. An einsum combines them with softmaxed router scores.

        Returns: [B, T_full, HV, V] memory contribution tensor (zeros where no top-k available).
        """
        B, T, HV, K = q.shape
        V = self.head_v_dim
        device = q.device
        topk = self.dsc_topk

        if n_full == 0 or topk == 0:
            return torch.zeros(B, T, HV, V, device=device, dtype=q.dtype)

        # 1) v4 descriptor per chunk: [N, B, 4*HV*K]
        # 3 fixed views (mean/max/softmax-attn) from `_multi_res_descriptor` +
        # 1 learnable attention view from `_learnable_attn_view`.
        # States are fp32 (kernel-returned); cast to q.dtype for routing math.
        desc_fixed = [_multi_res_descriptor(s).to(q.dtype) for s in cached_s]   # list of [B, 3*HV*K]
        desc_learn = [self._learnable_attn_view(s).to(q.dtype).flatten(start_dim=1) for s in cached_s]  # list of [B, HV*K]
        desc_list = [torch.cat([f, l], dim=-1) for f, l in zip(desc_fixed, desc_learn)]  # list of [B, 4*HV*K]
        desc_stack = torch.stack(desc_list, dim=0)                          # [N, B, dd]
        N = n_full

        # 1b) v4 entropy bias per chunk: [N, B]
        # ent_bias_scale init=0 → exactly v3 at step 0. As router learns, entropy term
        # lets it distinguish diffuse (high H) vs concentrated (low H) chunks. This is
        # independent of the seg_query · descriptor dot product.
        ent_list = [_chunk_entropy(s).to(q.dtype) for s in cached_s]        # list of [B]
        ent_stack = torch.stack(ent_list, dim=0)                            # [N, B]

        # 2) Per-chunk pooled query (v3 CAUSAL: last-pos-of-prev-chunk, no leak).
        # v2 used normed_chunks.mean(dim=2), which leaked future-within-chunk:
        # positions i*c+1..(i+1)*c-1 influenced chunk i's routing decision, and
        # that decision was used to read memory for position i*c itself.
        # v3 fix: for chunk i>=1, use the last position of chunk i-1 (strictly
        # causal — all of chunk i is future). Chunk 0 has no previous chunk;
        # use its own first position (chunk 0's routing is always masked).
        normed_chunks = normed[:, :N * c].view(B, N, c, -1)                 # [B, N, c, H]
        seg_query = torch.zeros_like(normed_chunks[:, :, 0])                # [B, N, H]
        seg_query[:, 1:] = normed_chunks[:, :-1, -1]                        # chunk i>=1
        seg_query[:, 0] = normed_chunks[:, 0, 0]                            # chunk 0 degenerate
        # Router: [B, N, H] @ [H, dd] -> [B, N, dd]
        u = self.dsc_router(seg_query)                                      # [B, N, dd]

        # 3) Score every (chunk, prev_chunk) pair via dot product on descriptors.
        # u: [B, N, dd], desc: [N, B, dd] -> permute to [B, N, dd], einsum to [B, i, j].
        desc_bjd = desc_stack.permute(1, 0, 2)                              # [B, N, dd]
        logits = torch.einsum('bid,bjd->bij', u, desc_bjd)                 # [B, i=N, j=N]
        # v4 entropy bias: add ent_bias_scale * H(state_j) to logit(i, j).
        # ent_stack: [N, B] -> permute to [B, N] -> unsqueeze to [B, 1, N] broadcast over i.
        ent_bj = ent_stack.permute(1, 0)                                   # [B, N]
        logits = logits + self.ent_bias_scale * ent_bj.unsqueeze(1)        # broadcast over i
        # Causal mask: chunk i can only read from chunks j < i (strict past — self NOT allowed,
        # because cached_s[i] aggregates positions i*c..(i+1)*c-1 and would leak future within chunk i).
        causal_mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=0)
        logits = logits.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
        # For chunk 0 (no prev), all logits are -inf; mark valid=False.

        # 4) Top-k selection: pick top-k previous chunks per query chunk.
        # For chunks with fewer than topk visible states, pad with 0 + mask.
        k_eff = min(topk, N - 1)
        if k_eff <= 0:
            return torch.zeros(B, T, HV, V, device=device, dtype=q.dtype)
        # We take topk over the FULL row; rows where fewer than topk are valid get -inf in the
        # masked positions and the softmax weights will be 0 for them. To make the indices safe
        # for gather, replace -inf positions' indices with 0 and track validity via weights.
        topk_vals, topk_idx = torch.topk(logits, k=k_eff, dim=-1)          # [B, N, k_eff]
        # When the entire row is -inf (chunk 0), topk_vals = -inf, weights -> nan. Fix at combine.
        topk_weights = torch.softmax(topk_vals, dim=-1)                    # [B, N, k_eff]
        topk_weights = torch.where(torch.isnan(topk_weights),
                                    torch.zeros_like(topk_weights), topk_weights)

        # 5) Gather selected states -> [B, N, k_eff, HV, K, V], then flatten to [B*N*k_eff, HV, K, V]
        # cached_s[i] has shape [B, HV, K, V]; stack into [N, B, HV, K, V] for indexing.
        cached_stack = torch.stack(cached_s, dim=0)                        # [N, B, HV, K, V]
        # Gather along chunk axis j. topk_idx: [B, N, k_eff].
        # expand_to [B, N, k_eff, HV, K, V] via index_select on the right axis.
        idx = topk_idx                                                      # [B, N, k_eff]
        # For each (b, n, k), pick cached_stack[idx[b,n,k], b].
        # Vectorized: bring b to last axis, gather, then move back.
        cached_bN = cached_stack.permute(1, 0, 2, 3, 4)                     # [B, N, HV, K, V]
        # We need to gather along dim=1 (N) using idx[b,n,k] -> result[b,n,k,hv,k,v].
        # Expand idx to the right shape for gather.
        idx_exp = idx.view(B, N, k_eff, 1, 1, 1).expand(B, N, k_eff, *cached_bN.shape[2:])
        # IMPORTANT: unsqueeze(1) — source is [B, 1, N, HV, K, V] expanded to [B, N, N, HV, K, V]
        # so that source[b, i, j, ...] = cached_bN[b, j, ...]. Gather along dim=2 then picks
        # cached_bN[b, idx[b, i, k], ...] — i.e. the ROUTED past chunk's state, not chunk i's own.
        # (unsqueeze(2) here previously replicated source across the gather dim and made the gather
        #  return cached_bN[b, i] regardless of idx — a future-information leak via cached_s[i].)
        gathered = torch.gather(
            cached_bN.unsqueeze(1).expand(B, N, N, *cached_bN.shape[2:]),
            dim=2, index=idx_exp,
        )                                                                  # [B, N, k_eff, HV, K, V]

        # 6) Loop over k_eff (only 2 iterations — minimal Python overhead).
        # This avoids materializing the B*N*k_eff tile of q/k/g/b/w which would 32x blow up VRAM.
        # Each iteration runs chunk_gdn2 with batch dim B*N — same as a single chunked scan.
        q_chunks = q[:, :N * c].view(B, N, c, HV, K)                       # [B, N, c, HV, K]
        k_chunks = k[:, :N * c].view(B, N, c, HV, K)
        g_chunks = g[:, :N * c].view(B, N, c, HV, K)
        b_chunks = b[:, :N * c].view(B, N, c, HV, K)
        w_chunks = w[:, :N * c].view(B, N, c, HV, V)

        # Flatten per-chunk tensors to [B*N, c, ...] ONCE (reused across k_eff iters).
        q_flat = q_chunks.reshape(B * N, c, HV, K)
        k_flat = k_chunks.reshape(B * N, c, HV, K)
        g_flat = g_chunks.reshape(B * N, c, HV, K)
        b_flat = b_chunks.reshape(B * N, c, HV, K)
        w_flat = w_chunks.reshape(B * N, c, HV, V)
        v_zero = torch.zeros(B * N, c, HV, V, device=device, dtype=q.dtype)

        # gathered: [B, N, k_eff, HV, K, V] from step 5 above.
        o_mem_list = []
        for k_idx in range(k_eff):
            # k-th selected state per (B, N) chunk.
            gathered_k = gathered[:, :, k_idx]                              # [B, N, HV, K, V]
            gathered_k_flat = gathered_k.reshape(B * N, HV, K, V)
            o_k, _ = chunk_gdn2(
                q=q_flat, k=k_flat, v=v_zero, g=g_flat, b=b_flat, w=w_flat,
                A_log=self.A_log, dt_bias=self.dt_bias,
                initial_state=gathered_k_flat, output_final_state=False,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False,
                cu_seqlens=None,
            )                                                              # [B*N, c, HV, V]
            o_mem_list.append(o_k)
        # Stack along new k_eff axis: [B*N, k_eff, c, HV, V] -> [B, N, k_eff, c, HV, V]
        o_mem = torch.stack(o_mem_list, dim=1).view(B, N, k_eff, c, HV, V)

        # 7) Weighted combine over k_eff axis using topk_weights [B, N, k_eff].
        combined = torch.einsum('bnk,bnkchv->bnchv', topk_weights.to(o_mem.dtype), o_mem)  # [B, N, c, HV, V]
        combined_flat = combined.reshape(B, N * c, HV, V)                  # [B, N*c, HV, V]

        # Zero-pad to T to align with main scan output.
        if T > N * c:
            pad = torch.zeros(B, T - N * c, HV, V, device=device, dtype=combined_flat.dtype)
            combined_flat = torch.cat([combined_flat, pad], dim=1)
        return combined_flat

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, output_attentions=False, **kwargs):
        """DSC forward: chunk-cache + multi-res descriptor + top-k routing + batched memory scan.

        Same I/O as GatedDeltaNet2.forward; the memory contribution is folded into the
        recurrent output before o_norm + o_proj.
        """
        if attention_mask is not None:
            # Keep simple: DSC v0 does not support padded sequences (training-time only).
            assert len(attention_mask.shape) == 2, "Expected 2D attention mask."
        if past_key_values is not None:
            # Incremental decode: DSC v0 is training-only; fall back to parent.
            return super().forward(hidden_states, attention_mask=attention_mask,
                                   past_key_values=past_key_values, use_cache=use_cache,
                                   output_attentions=output_attentions, **kwargs)

        B, T, _ = hidden_states.shape

        # === Project q/k/v/g/b/w (same as parent) ===
        if self.use_short_conv:
            q, _ = self.q_conv1d(x=self.q_proj(hidden_states), cache=None, output_final_state=False)
            k, _ = self.k_conv1d(x=self.k_proj(hidden_states), cache=None, output_final_state=False)
            v, _ = self.v_conv1d(x=self.v_proj(hidden_states), cache=None, output_final_state=False)
        else:
            q = torch.nn.functional.silu(self.q_proj(hidden_states))
            k = torch.nn.functional.silu(self.k_proj(hidden_states))
            v = torch.nn.functional.silu(self.v_proj(hidden_states))
        b = self.b_proj(hidden_states).sigmoid()
        w = self.w_proj(hidden_states).sigmoid()
        g = (
            -self.A_log.float().exp().repeat_interleave(self.head_k_dim)
            * torch.nn.functional.softplus(self.f_proj(hidden_states).float() + self.dt_bias)
        )
        q, k, g = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k, g))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        b = rearrange(b, "... (h d) -> ... h d", d=self.head_k_dim)
        w = rearrange(w, "... (h d) -> ... h d", d=self.head_v_dim)
        if self.num_v_heads > self.num_heads:
            q, k, g, b = (
                repeat(x, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads)
                for x in (q, k, g, b)
            )
        if self.allow_neg_eigval:
            b = b * 2.0

        # === (a) Forward scan with chunk-boundary state cache ===
        o_main, cached_s, n_full, c = self._dsc_forward_scan(q, k, v, g, b, w)

        # === (b)+(c) Multi-res descriptor, top-k routing, batched memory scan, weighted combine ===
        o_mem = self._dsc_memory_combine(q, k, g, b, w, cached_s, n_full, c, hidden_states)

        # === Combine: main + alpha * memory, then o_norm + o_proj (parent's tail) ===
        o = o_main + self.dsc_combine_alpha * o_mem
        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        return o, None, None
