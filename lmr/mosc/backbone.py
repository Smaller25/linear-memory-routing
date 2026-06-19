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
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        h = self.embed(input_ids)
        for i, (norm, mixer) in enumerate(zip(self.mix_norms, self.mixers)):
            h = h + mixer(hidden_states=norm(h))[0]
            if self.mlps is not None:
                h = h + self.mlps[i](self.mlp_norms[i](h))
        h = self.norm_f(h)
        return h if return_hidden else self.lm_head(h)
