# -*- coding: utf-8 -*-
"""DynMC layer: GatedDeltaNet with independent-compressor segments + MC-GRM read.

Wrapper-level implementation (plan §1 kernel constraint — no kernel changes):
  - segment state reset via `cu_seqlens` varlen packing (each segment is an
    independent sequence for the chunkwise kernel; boundaries are multiples of
    64 by construction, see segmenting.py)
  - per-segment final states come from `output_final_state=True`; the kernel's
    backward accepts `dht`, so gradients flow through cached states (plan §2.2)
  - MC-GRM read (plan §2.2): output-space mixing
        y_t = γ_t^(cur) · o_t + Σ_{i<cur, same doc} γ_t^(i) · (S^(i) q̂_t)
        γ_t = softmax([c, ⟨u_t, d_i⟩ · dd^{-1/2}])
    where q̂_t = l2norm(q_t) · head_k_dim^{-1/2} (identical to the kernel's
    in-kernel q processing, so cached reads live in the same output space),
    d_i = MeanPool_V(S^(i)) ∈ R^{H·K} (mosc convention `gdn_meanpool_state`:
    average over the non-contracted V axis, flatten (H,K)), and c is a learnable
    "current segment" logit (the running S_t has no materialized descriptor
    mid-chunk; init 2.0 so early training is dominated by the online path).
  - training-time cache budget: sliding window of the most recent
    `cache_budget` same-document segments (fifo). Signal-based drop policies
    are inference-time concerns (plan §2.6) handled by the eval driver.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.modules.l2norm import l2norm
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


class DynMCGatedDeltaNet(GatedDeltaNet):
    """GatedDeltaNet + segment cache + MC-GRM read (training, packed varlen)."""

    def __init__(self, *args, cache_budget: int = 32, cur_logit_init: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_budget = cache_budget
        # read projection u_t (plan §2.2); descriptor dim = H·K per mosc convention
        self.descriptor_dim = self.num_v_heads * self.head_k_dim
        self.u_proj = nn.Linear(self.hidden_size, self.descriptor_dim, bias=False)
        self.cur_logit = nn.Parameter(torch.tensor(float(cur_logit_init)))

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, output_attentions=False, **kwargs):
        segs = kwargs.pop("dynmc_segs", None)
        if segs is None:
            # plain GDN behaviour (e.g. debugging, or reuse as anchor)
            return super().forward(hidden_states, attention_mask, past_key_values,
                                   use_cache, output_attentions, **kwargs)

        assert hidden_states.shape[0] == 1, "DynMC training expects packed [1, T, d]"
        assert attention_mask is None and past_key_values is None

        cu_seqlens = segs["cu_seqlens"]          # [S+1] int32, on device
        seg_doc_start = segs["seg_doc_start"]    # [S] long: first segment idx of the doc

        # --- projections + short conv (identical to parent, varlen-aware) ---
        if self.use_short_conv:
            q, _ = self.q_conv1d(x=self.q_proj(hidden_states), cu_seqlens=cu_seqlens)
            k, _ = self.k_conv1d(x=self.k_proj(hidden_states), cu_seqlens=cu_seqlens)
            v, _ = self.v_conv1d(x=self.v_proj(hidden_states), cu_seqlens=cu_seqlens)
        else:
            q = torch.nn.functional.silu(self.q_proj(hidden_states))
            k = torch.nn.functional.silu(self.k_proj(hidden_states))
            v = torch.nn.functional.silu(self.v_proj(hidden_states))

        q, k = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        beta = self.b_proj(hidden_states)

        # --- segment-independent recurrence; final state per segment ---
        o, states = chunk_gated_delta_rule(
            q=q, k=k, v=v,
            g=self.a_proj(hidden_states),
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=self.allow_neg_eigval,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
        )
        # o: [1, T, HV, V] / states: [S, HV, V, K] (fp32)

        # activation checkpointing: read의 z 텐서([T_j,E,H,V] × S)를 backward용으로
        # 저장하면 수십 GB → forward 재계산으로 교환 (read 자체가 ~20% 오버헤드)
        if self.training:
            o = torch.utils.checkpoint.checkpoint(
                self._grm_read, hidden_states, q, o, states, cu_seqlens, seg_doc_start,
                use_reentrant=False)
        else:
            o = self._grm_read(hidden_states, q, o, states, cu_seqlens, seg_doc_start)

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        return o, None, past_key_values

    def _grm_read(self, h_in, q, o, states, cu_seqlens, seg_doc_start):
        """Mix cached segment states into the online output (plan §2.2)."""
        S = states.shape[0]
        K_budget = self.cache_budget

        # q̂ exactly as the kernel treats q (l2norm + scale), GVA-broadcast to HV
        qn = l2norm(q) * (self.head_k_dim ** -0.5)
        if self.num_v_heads > self.num_heads:
            qn = qn.repeat_interleave(self.num_v_heads // self.num_heads, dim=-2)
        qn = qn.squeeze(0).float()                              # [T, HV, K]

        desc = states.mean(dim=2).reshape(S, -1)                # [S, HV*K] fp32
        d_scale = self.descriptor_dim ** -0.5
        u = self.u_proj(h_in).squeeze(0).float()                # [T, HV*K]

        o_flat = o.squeeze(0)                                   # [T, HV, V]
        out = o_flat.clone()
        cu = cu_seqlens.tolist()
        doc_start = seg_doc_start.tolist()
        for j in range(S):
            lo = max(doc_start[j], j - K_budget)
            if lo >= j:
                continue  # no eligible cached segment: γ_cur = 1, out = o
            t0, t1 = cu[j], cu[j + 1]
            s_e = states[lo:j]                                  # [E, HV, V, K] fp32
            logits = u[t0:t1] @ desc[lo:j].t() * d_scale        # [T_j, E]
            cur = self.cur_logit.expand(t1 - t0, 1)
            gam = torch.softmax(torch.cat([cur, logits], dim=-1), dim=-1)  # [T_j, 1+E]
            z = torch.einsum("ehvk,thk->tehv", s_e, qn[t0:t1])  # [T_j, E, HV, V]
            mix = torch.einsum("te,tehv->thv", gam[:, 1:], z)
            out[t0:t1] = (gam[:, 0].view(-1, 1, 1) * o_flat[t0:t1].float() + mix).to(o_flat.dtype)
        return out.unsqueeze(0)
