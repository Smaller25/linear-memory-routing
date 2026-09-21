"""Training-ready GDN-2 layer wrapper with paper SSC aggregation."""

from __future__ import annotations

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from .ssc import GDN2SSC, gdn2_ssc_forward


class MemoryCachingGDN2Layer(nn.Module):
    """Wrap an existing GDN-2 layer while preserving its projection/output path.

    The base layer is retained as ``base``; SSC adds only connector ``W_u``.
    This full-sequence training path intentionally rejects cache/padding modes
    until their segment metadata is explicitly supplied.
    """

    def __init__(self, base: nn.Module, *, topk: int = 2, chunk_size: int = 256,
                 checkpoint_mode: str = "independent", chunk_gdn2_fn=None) -> None:
        super().__init__()
        self.base = base
        self.ssc = GDN2SSC(base.hidden_size, base.num_v_heads, base.head_k_dim,
                           topk=topk, chunk_size=chunk_size)
        self.checkpoint_mode = checkpoint_mode
        self.chunk_gdn2_fn = chunk_gdn2_fn

    def _project(self, hidden_states: torch.Tensor):
        if self.base.use_short_conv:
            q, _ = self.base.q_conv1d(x=self.base.q_proj(hidden_states), cache=None, output_final_state=False)
            k, _ = self.base.k_conv1d(x=self.base.k_proj(hidden_states), cache=None, output_final_state=False)
            v, _ = self.base.v_conv1d(x=self.base.v_proj(hidden_states), cache=None, output_final_state=False)
        else:
            q, k, v = (F.silu(proj(hidden_states))
                       for proj in (self.base.q_proj, self.base.k_proj, self.base.v_proj))
        g = (-self.base.A_log.float().exp().repeat_interleave(self.base.head_k_dim)
             * F.softplus(self.base.f_proj(hidden_states).float() + self.base.dt_bias))
        b = self.base.b_proj(hidden_states).sigmoid()
        w = self.base.w_proj(hidden_states).sigmoid()
        q, k, g = (rearrange(x, "... (h d) -> ... h d", d=self.base.head_k_dim)
                   for x in (q, k, g))
        v = rearrange(v, "... (h d) -> ... h d", d=self.base.head_v_dim)
        b = rearrange(b, "... (h d) -> ... h d", d=self.base.head_k_dim)
        w = rearrange(w, "... (h d) -> ... h d", d=self.base.head_v_dim)
        if self.base.num_v_heads > self.base.num_heads:
            q, k, g, b = (repeat(x, "... h d -> ... (h group) d",
                                 group=self.base.num_v_heads // self.base.num_heads)
                          for x in (q, k, g, b))
        if self.base.allow_neg_eigval:
            b = b * 2.0
        return q, k, v, g, b, w

    def forward_with_diagnostics(self, hidden_states: torch.Tensor):
        q, k, v, g, b, w = self._project(hidden_states)
        result = gdn2_ssc_forward(
            self.ssc, hidden_states, q, k, v, g, b, w,
            checkpoint_mode=self.checkpoint_mode,
            chunk_gdn2_fn=self.chunk_gdn2_fn,
        )
        output = self.base.o_norm(
            result.output,
            rearrange(self.base.g_proj(hidden_states), "... (h d) -> ... h d", d=self.base.head_v_dim),
        )
        return self.base.o_proj(rearrange(output, "b t h d -> b t (h d)")), result

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, output_attentions=False, **kwargs):
        if attention_mask is not None or past_key_values is not None or use_cache or kwargs.get("cu_seqlens") is not None:
            raise NotImplementedError("MC SSC wrapper currently supports unpadded full-sequence training/eval only")
        output, _ = self.forward_with_diagnostics(hidden_states)
        return output, None, past_key_values
