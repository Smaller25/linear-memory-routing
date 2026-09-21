"""Isolated GDN-2 wrapper for dense GRM and Memory Soup.

This wrapper intentionally does not modify or subclass the production SSC-v2
``MemoryCachingGDN2Layer``.  The projection and output operations mirror that
wrapper, while the aggregation path is kept in a separate module.
"""

from __future__ import annotations

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from .grm import GDN2GRM, GDN2MemorySoup, gdn2_grm_forward
from .relu import GDN2ReLUSelectiveCaching, gdn2_relu_forward


class DenseMemoryCachingGDN2Layer(nn.Module):
    """Wrap GDN-2 with GRM, Memory Soup, or ReLU Dynamic Selection.

    All three variants share the dense read path (SSC-v2 kernel in bounded
    route blocks); ``relu`` swaps only the gate (see ``mc_baseline.mc_relu``).
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        variant: str = "grm",
        chunk_size: int = 256,
        checkpoint_mode: str = "independent",
        chunk_gdn2_fn=None,
        route_block_size: int = 16,
    ) -> None:
        super().__init__()
        normalized_variant = variant.lower().replace("-", "_")
        if normalized_variant == "grm":
            aggregator_class = GDN2GRM
        elif normalized_variant in {"memory_soup", "soup"}:
            normalized_variant = "memory_soup"
            aggregator_class = GDN2MemorySoup
        elif normalized_variant == "relu":
            aggregator_class = GDN2ReLUSelectiveCaching
        elif normalized_variant == "relu_raw":
            # Same module, unnormalized gate (ReMoE original) — parameter-free
            # difference, checkpoints interchangeable with 'relu'.
            aggregator_class = GDN2ReLUSelectiveCaching
        else:
            raise ValueError(
                f"unknown dense Memory Caching variant {variant!r}; "
                "expected 'grm', 'memory_soup', 'relu', or 'relu_raw'"
            )

        aggregator_kwargs = {}
        if normalized_variant == "relu_raw":
            aggregator_kwargs["normalize_gate"] = False
        self.base = base
        self.aggregator = aggregator_class(
            base.hidden_size,
            base.num_v_heads,
            base.head_k_dim,
            chunk_size=chunk_size,
            route_block_size=route_block_size,
            **aggregator_kwargs,
        )
        self.variant = normalized_variant
        self.checkpoint_mode = checkpoint_mode
        self.chunk_gdn2_fn = chunk_gdn2_fn

    def enable_active_state_logging(self, enabled: bool = True) -> None:
        """Toggle eval-time active-state diagnostics (ReLU variant only)."""
        if self.variant not in ("relu", "relu_raw"):
            raise ValueError(
                "active-state logging is only meaningful for variant='relu'"
            )
        self.aggregator.log_active_states = enabled

    def _project(self, hidden_states: torch.Tensor):
        if self.base.use_short_conv:
            q, _ = self.base.q_conv1d(
                x=self.base.q_proj(hidden_states),
                cache=None,
                output_final_state=False,
            )
            k, _ = self.base.k_conv1d(
                x=self.base.k_proj(hidden_states),
                cache=None,
                output_final_state=False,
            )
            v, _ = self.base.v_conv1d(
                x=self.base.v_proj(hidden_states),
                cache=None,
                output_final_state=False,
            )
        else:
            q, k, v = (
                F.silu(projection(hidden_states))
                for projection in (
                    self.base.q_proj,
                    self.base.k_proj,
                    self.base.v_proj,
                )
            )

        g = (
            -self.base.A_log.float().exp().repeat_interleave(
                self.base.head_k_dim
            )
            * F.softplus(
                self.base.f_proj(hidden_states).float()
                + self.base.dt_bias
            )
        )
        b = self.base.b_proj(hidden_states).sigmoid()
        w = self.base.w_proj(hidden_states).sigmoid()
        q, k, g = (
            rearrange(x, "... (h d) -> ... h d", d=self.base.head_k_dim)
            for x in (q, k, g)
        )
        v = rearrange(
            v,
            "... (h d) -> ... h d",
            d=self.base.head_v_dim,
        )
        b = rearrange(
            b,
            "... (h d) -> ... h d",
            d=self.base.head_k_dim,
        )
        w = rearrange(
            w,
            "... (h d) -> ... h d",
            d=self.base.head_v_dim,
        )
        if self.base.num_v_heads > self.base.num_heads:
            q, k, g, b = (
                repeat(
                    x,
                    "... h d -> ... (h group) d",
                    group=self.base.num_v_heads // self.base.num_heads,
                )
                for x in (q, k, g, b)
            )
        if self.base.allow_neg_eigval:
            b = b * 2.0
        return q, k, v, g, b, w

    def forward_with_diagnostics(self, hidden_states: torch.Tensor):
        q, k, v, g, b, w = self._project(hidden_states)
        forward_fn = (
            gdn2_relu_forward if self.variant in ("relu", "relu_raw")
            else gdn2_grm_forward
        )
        result = forward_fn(
            self.aggregator,
            hidden_states,
            q,
            k,
            v,
            g,
            b,
            w,
            checkpoint_mode=self.checkpoint_mode,
            chunk_gdn2_fn=self.chunk_gdn2_fn,
        )
        output = self.base.o_norm(
            result.output,
            rearrange(
                self.base.g_proj(hidden_states),
                "... (h d) -> ... h d",
                d=self.base.head_v_dim,
            ),
        )
        return (
            self.base.o_proj(
                rearrange(output, "b t h d -> b t (h d)")
            ),
            result,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        if (
            attention_mask is not None
            or past_key_values is not None
            or use_cache
            or kwargs.get("cu_seqlens") is not None
        ):
            raise NotImplementedError(
                "dense MC wrapper supports unpadded full-sequence "
                "training/eval only"
            )
        output, _ = self.forward_with_diagnostics(hidden_states)
        return output, None, past_key_values
