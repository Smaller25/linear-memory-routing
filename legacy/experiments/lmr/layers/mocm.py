# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""MoCM mixer — the parallel-memory core (MoM axis) of Mixture-of-Cached-Memories.

A drop-in linear-recurrent mixer with ``M`` parallel gated-delta memories + a write-router (top-k_w)
+ an optional always-on shared memory. This is the *parallel axis* (interference handling); the
*temporal axis* (per-segment caching + read-router) is added by the segment runner around it.
Designed for from-scratch training (2-layer MQAR validation): set ``num_memories=1`` to recover a
plain single-memory gated-delta layer (the M=1 ablation), ``num_memories=M`` for MoM-style.

Per memory: independent k/v/β/g (query is shared). Write-routing zeroes the value for non-selected
memories per token (MoM: non-activated memories stay unchanged). Read-out = Σ_m g_t^m (q_t·h_t^m)
+ shared, normed + projected. Uses FLA's ``chunk_gated_delta_rule`` per memory.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _init_A_log(num_heads):
    A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
    return nn.Parameter(torch.log(A))


def _init_dt_bias(num_heads, dt_min=1e-3, dt_max=0.1, dt_floor=1e-4):
    dt = torch.exp(torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
    dt = torch.clamp(dt, min=dt_floor)
    return nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus


class MoCMMixer(nn.Module):
    def __init__(self, hidden_size: int, num_memories: int = 4, topk_w: int = 2,
                 head_dim: int = 64, expand_v: float = 1.0, shared_mem: bool = True,
                 aux_scale: float = 1e-3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_memories = num_memories
        self.topk_w = min(topk_w, num_memories)
        self.head_dim = head_dim
        self.shared_mem = shared_mem
        self.aux_scale = aux_scale
        self.num_heads = hidden_size // head_dim
        self.v_dim = int(hidden_size * expand_v)
        self.v_head_dim = self.v_dim // self.num_heads

        n_mem_total = num_memories + (1 if shared_mem else 0)
        # shared query; per-memory k/v/beta/gate projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(n_mem_total)])
        self.v_proj = nn.ModuleList([nn.Linear(hidden_size, self.v_dim, bias=False) for _ in range(n_mem_total)])
        self.b_proj = nn.ModuleList([nn.Linear(hidden_size, self.num_heads, bias=False) for _ in range(n_mem_total)])
        self.a_proj = nn.ModuleList([nn.Linear(hidden_size, self.num_heads, bias=False) for _ in range(n_mem_total)])
        # Proper gated-delta init: slow forgetting (decay ~1) so memory actually retains — A_log=0/
        # dt_bias=0 gives ~0.5 decay/token => state vanishes in a few tokens => no recall.
        self.A_log = nn.ParameterList([_init_A_log(self.num_heads) for _ in range(n_mem_total)])
        self.dt_bias = nn.ParameterList([_init_dt_bias(self.num_heads) for _ in range(n_mem_total)])
        # write-router over the M selectable memories (shared memory is always on, not routed)
        self.router = nn.Linear(hidden_size, num_memories, bias=False)
        self.o_norm = nn.RMSNorm(self.v_head_dim)
        self.o_proj = nn.Linear(self.v_dim, hidden_size, bias=False)

    def _mem_out(self, idx, x, q, gate_t):
        """Run memory ``idx`` over the segment; gate_t [b,l] (write weight, 0 for unselected)."""
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        k = self.k_proj[idx](x)
        v = self.v_proj[idx](x)
        if gate_t is not None:
            v = v * gate_t[..., None]          # write-routing: scale the value path by the gate
        beta = self.b_proj[idx](x).sigmoid()
        g = self.a_proj[idx](x)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_dim)
        v = rearrange(v, "b l (h d) -> b l h d", d=self.v_head_dim)
        o, _ = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, output_final_state=False,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            A_log=self.A_log[idx], dt_bias=self.dt_bias[idx],
        )
        return o                                # [b, l, h, v_head_dim]

    def forward(self, x):
        b, L, _ = x.shape
        q = rearrange(self.q_proj(x), "b l (h d) -> b l h d", d=self.head_dim)

        # write-router: top-k_w over the M selectable memories, renormalised softmax
        logits = self.router(x)                                 # [b, l, M]
        probs = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, self.topk_w, dim=-1)
        topv = topv / (topv.sum(-1, keepdim=True) + 1e-9)
        gate = torch.zeros_like(probs).scatter_(-1, topi, topv)  # [b, l, M] (0 for unselected)

        y = 0
        for m in range(self.num_memories):
            o = self._mem_out(m, x, q, gate[..., m])
            y = y + gate[..., m][..., None, None] * o            # read-out weighted by the same gate
        if self.shared_mem:
            y = y + self._mem_out(self.num_memories, x, q, None)  # always-on shared memory

        y = self.o_norm(y)
        y = rearrange(y, "b l h d -> b l (h d)")
        out = self.o_proj(y)

        # Switch-style load-balance aux over the M routed memories
        mask = torch.zeros_like(probs).scatter_(-1, topi, 1.0)
        aux = self.aux_scale * self.num_memories * (mask.mean((0, 1)) * probs.mean((0, 1))).sum()
        return out, aux
