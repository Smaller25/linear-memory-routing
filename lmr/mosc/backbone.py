# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""From-scratch GDN2 backbone LM — the vanilla baseline for the Dynamic-MoSC track.

A minimal causal LM whose token mixer is FLA's :class:`GatedDeltaNet2` (GDN-2 generalises KDA and
Gated DeltaNet v1: scalar gate -> GDN-v1, vector gate -> KDA). Same embed -> N x (RMSNorm + mixer +
residual) -> norm -> lm_head shape as the existing ``TwoLayerMoCM`` trainer, so it drops into the
MQAR harness unchanged. This is both the headline backbone and the Phase-0 ``vanilla`` comparison.

GDN-2 is pure-Triton (no mamba_ssm / tilelang), so it trains from scratch on Blackwell/sm_120
(verified via scripts/sh_check_backbones.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fla.layers.gdn2 import GatedDeltaNet2


class GDN2LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 2,
        head_dim: int = 64,
        num_heads: int = 4,
        expand_v: float = 1.0,
        mlp_ratio: int = 0,  # 0 = token-mixer only (matches TwoLayerMoCM); >0 adds a GLU MLP block
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.mix_norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(n_layers)])
        self.mixers = nn.ModuleList([
            GatedDeltaNet2(hidden_size=d_model, head_dim=head_dim, num_heads=num_heads, expand_v=expand_v)
            for _ in range(n_layers)
        ])
        if mlp_ratio:
            self.mlp_norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(n_layers)])
            self.mlps = nn.ModuleList([
                nn.Sequential(nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
                              nn.Linear(mlp_ratio * d_model, d_model))
                for _ in range(n_layers)
            ])
        else:
            self.mlp_norms = self.mlps = None
        for i, m in enumerate(self.mixers):
            m.layer_idx = i  # required for fla's per-layer Cache (segment-wise state threading)
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # flattened recurrent-state width [H*K*V] of one GDN2 layer (for the true-state read-out)
        self.state_dim = num_heads * head_dim * int(head_dim * expand_v)

    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False,
                salience_gate: torch.Tensor | None = None) -> torch.Tensor:
        """salience_gate [B, T] in [0,1] (optional): scales each token's mixer INPUT, so low-salience
        (filler) tokens write weakly into the fixed recurrent state while the residual stream still
        carries them for prediction. This is the constant-memory salience-gated-retention test — no
        cache: keep the needle in a fixed state by not letting filler overwrite it."""
        h = self.embed(input_ids)
        g = None if salience_gate is None else salience_gate[..., None].to(h.dtype)  # [B,T,1]
        for i, (norm, mixer) in enumerate(zip(self.mix_norms, self.mixers)):
            mix_in = norm(h)
            if g is not None:
                mix_in = mix_in * g                       # attenuate low-salience writes to the state
            h = h + mixer(hidden_states=mix_in)[0]
            if self.mlps is not None:
                h = h + self.mlps[i](self.mlp_norms[i](h))
        h = self.norm_f(h)
        return h if return_hidden else self.lm_head(h)

    def run_segmented(self, input_ids: torch.Tensor, seg_bounds):
        """Run segment-by-segment with state threading (fla Cache), capturing each segment's TRUE
        last-layer recurrent state. ``seg_bounds``: list of (start, end) slices partitioning [0, T],
        uniform across the batch. Returns ``(hidden_full [B,T,d], states [B, N, state_dim])``.
        Verified equivalent to a full forward (state threads correctly)."""
        from fla.models.utils import Cache
        cache = Cache.from_legacy_cache(None)
        outs, states = [], []
        last = len(self.mixers) - 1
        for s, e in seg_bounds:
            h = self.embed(input_ids[:, s:e])
            for i, (norm, mixer) in enumerate(zip(self.mix_norms, self.mixers)):
                o, _, cache = mixer(hidden_states=norm(h), past_key_values=cache, use_cache=True)
                h = h + o
                if self.mlps is not None:
                    h = h + self.mlps[i](self.mlp_norms[i](h))
            outs.append(h)
            states.append(cache[last]["recurrent_state"].flatten(1))   # [B, H*K*V]
        return self.norm_f(torch.cat(outs, dim=1)), torch.stack(states, dim=1)
