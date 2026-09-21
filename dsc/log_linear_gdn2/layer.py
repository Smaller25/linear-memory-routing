"""Training wrapper that lifts an existing GDN-2 layer to Log-Linear GDN-2."""

from __future__ import annotations

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from .core import LogLinearGDN2Result, log_linear_gdn2_chunkwise, required_num_levels


class LogLinearGDN2Layer(nn.Module):
    """Preserve the GDN-2 projections/output path and add hierarchical memory.

    The only new learned components are a per-token/per-head lambda projection
    and a per-head/per-level scale, matching the positive parameterization in
    the authors' released Log-Linear Gated DeltaNet model.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        max_sequence_length: int,
        lambda_mode: str = "positive",
        checkpoint_levels: bool = True,
        chunk_gdn2_fn=None,
    ) -> None:
        super().__init__()
        if lambda_mode != "positive":
            raise ValueError(
                "the paper-matched GDN2 implementation currently supports "
                "lambda_mode='positive' only"
            )
        self.base = base
        self.max_sequence_length = max_sequence_length
        self.max_num_levels = required_num_levels(max_sequence_length)
        self.lambda_mode = lambda_mode
        self.checkpoint_levels = checkpoint_levels
        self.chunk_gdn2_fn = chunk_gdn2_fn

        self.l_proj = nn.Linear(
            base.hidden_size,
            base.num_v_heads * self.max_num_levels,
            bias=False,
        )
        nn.init.xavier_uniform_(self.l_proj.weight, gain=2**-2.5)
        self.L = nn.Parameter(
            torch.ones(base.num_v_heads, self.max_num_levels, dtype=torch.float32)
        )
        self.L._no_weight_decay = True

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

        g = -self.base.A_log.float().exp().repeat_interleave(
            self.base.head_k_dim
        ) * F.softplus(self.base.f_proj(hidden_states).float() + self.base.dt_bias)
        b = self.base.b_proj(hidden_states).sigmoid()
        w = self.base.w_proj(hidden_states).sigmoid()
        q, k, g = (
            rearrange(x, "... (h d) -> ... h d", d=self.base.head_k_dim)
            for x in (q, k, g)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=self.base.head_v_dim)
        b = rearrange(b, "... (h d) -> ... h d", d=self.base.head_k_dim)
        w = rearrange(w, "... (h d) -> ... h d", d=self.base.head_v_dim)
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

    def _compute_lambdas(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dl = rearrange(
            self.l_proj(hidden_states),
            "b t (h level) -> b t h level",
            h=self.base.num_v_heads,
            level=self.max_num_levels,
        )
        lambdas = F.softplus(rearrange(self.L, "h level -> 1 1 h level") * dl)
        return lambdas

    def forward_with_diagnostics(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, LogLinearGDN2Result, torch.Tensor]:
        if hidden_states.shape[1] > self.max_sequence_length:
            raise ValueError(
                f"sequence length {hidden_states.shape[1]} exceeds configured "
                f"maximum {self.max_sequence_length}"
            )
        q, k, v, g, b, w = self._project(hidden_states)
        lambdas = self._compute_lambdas(hidden_states).to(dtype=q.dtype)
        result = log_linear_gdn2_chunkwise(
            q,
            k,
            v,
            g,
            b,
            w,
            lambdas,
            chunk_gdn2_fn=self.chunk_gdn2_fn,
            checkpoint_levels=self.checkpoint_levels and self.training,
        )
        output = self.base.o_norm(
            result.output,
            rearrange(
                self.base.g_proj(hidden_states),
                "... (h d) -> ... h d",
                d=self.base.head_v_dim,
            ),
        )
        output = self.base.o_proj(rearrange(output, "b t h d -> b t (h d)"))
        return output, result, lambdas

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
                "Log-Linear GDN2 currently supports unpadded full-sequence "
                "pretraining/evaluation only"
            )
        output, _, _ = self.forward_with_diagnostics(hidden_states)
        return output, None, past_key_values
