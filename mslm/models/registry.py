"""Plug points for Memory Caching ideas.

Every MC layer is assembled from five named parts, each chosen by a config
variable. To try an idea, register a new entry here (or in your own module
imported from ``mslm/models/__init__.py``) and point the variable at its name.
Nothing else has to change.

| kind        | variable        | signature                                                        |
|-------------|-----------------|------------------------------------------------------------------|
| descriptor  | MC_DESCRIPTOR   | fn(keys[B,T,H,K], chunk) -> (seg[B,N,H,K], online[B,T,H,K])      |
| router      | MC_ROUTER       | nn.Module(d_model, n_head, head_dim, mc_cfg);                    |
|             |                 |   forward(x[B,T,D], seg, online) -> (scores[B,T,N], online[B,T]) |
| gate        | MC_GATE         | fn(online_score[B,T], sel_scores[B,T,R], valid[B,T,R])           |
|             |                 |   -> weights[B,T,1+R]  (column 0 = online memory)                |
| connect     | MC_CONNECT      | fn(q,k,v,g,b,w, chunk, gdn_fn) -> (online_out[B,T,H,V],          |
|             |                 |   memories[B,N,H,K,V])   (T is a multiple of chunk here)         |
| aux_loss    | MC_AUX_LOSS     | fn(rec: RoutingRecord, batch: dict) -> scalar tensor             |

``keys`` given to descriptors are already L2-normalised (the GDN-2 kernel
normalises them before the memory update, so routing sees the same keys).
``RoutingRecord`` (mslm/models/mc.py) carries scores, chosen indices, gate
weights and the per-token segment ids; ``batch`` carries whatever the
training/eval loop knows (e.g. ``gold_segments`` for RULER items).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

DESCRIPTORS: dict[str, Callable] = {}
ROUTERS: dict[str, type] = {}
GATES: dict[str, Callable] = {}
CONNECTORS: dict[str, Callable] = {}
AUX_LOSSES: dict[str, Callable] = {}

_KINDS = {"descriptor": DESCRIPTORS, "router": ROUTERS, "gate": GATES,
          "connect": CONNECTORS, "aux_loss": AUX_LOSSES}


def register(kind: str, name: str):
    table = _KINDS[kind]

    def deco(obj):
        if name in table:
            raise KeyError(f"{kind} {name!r} already registered")
        table[name] = obj
        return obj
    return deco


def get(kind: str, name: str):
    table = _KINDS[kind]
    if name not in table:
        raise KeyError(f"unknown {kind} {name!r}; registered: {sorted(table)}")
    return table[name]


# --------------------------------------------------------------------------- descriptors
@register("descriptor", "meanpool")
def meanpool_descriptor(keys: torch.Tensor, chunk: int):
    """Mean of normalised keys per segment, and the causal running mean inside the
    current segment for the online memory (MC paper Eq. 16 with mean instead of sum:
    the sum has norm ~sqrt(chunk) and collapses the softmax gate at init)."""
    B, T, H, K = keys.shape
    N = T // chunk
    kc = keys.float().view(B, N, chunk, H, K)
    seg = kc.mean(dim=2)
    count = torch.arange(1, chunk + 1, device=keys.device, dtype=torch.float32).view(1, 1, chunk, 1, 1)
    online = (kc.cumsum(dim=2) / count).view(B, T, H, K)
    return seg, online


# --------------------------------------------------------------------------- routers
@register("router", "dot")
class DotRouter(nn.Module):
    """r_t^i = sum_h <u_t^h, c_i^h> with u_t = x_t W_u (paper Eq. 16)."""

    def __init__(self, d_model: int, n_head: int, head_dim: int, mc_cfg=None):
        super().__init__()
        self.n_head, self.head_dim = n_head, head_dim
        self.connector = nn.Linear(d_model, n_head * head_dim, bias=False)

    def forward(self, x, seg, online):
        B, T, _ = x.shape
        u = self.connector(x).view(B, T, self.n_head, self.head_dim).float()
        scores = torch.einsum("bthk,bnhk->btn", u, seg.float())
        online_score = torch.einsum("bthk,bthk->bt", u, online.float())
        return scores, online_score


# --------------------------------------------------------------------------- gates
@register("gate", "softmax")
def softmax_gate(online_score, sel_scores, valid):
    """Joint softmax over the online memory and the selected cached memories."""
    logits = torch.cat([online_score.unsqueeze(-1), sel_scores.masked_fill(~valid, -torch.inf)], dim=-1)
    return torch.softmax(logits, dim=-1)


# --------------------------------------------------------------------------- connectors
@register("connect", "independent")
def independent_segments(q, k, v, g, b, w, chunk, gdn_fn):
    """Paper §3.4 independent compressors: every segment starts from a zero state,
    all segments run as one batched call over [B*N, chunk]."""
    B, T = q.shape[:2]
    N = T // chunk

    def seg(x):
        return x.reshape(B * N, chunk, *x.shape[2:])
    o, s = gdn_fn(q=seg(q), k=seg(k), v=seg(v), g=seg(g), b=seg(b), w=seg(w),
                  initial_state=None, output_final_state=True,
                  use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False, cu_seqlens=None)
    return o.reshape(B, T, *o.shape[2:]), s.view(B, N, *s.shape[1:])


@register("connect", "chained")
def chained_segments(q, k, v, g, b, w, chunk, gdn_fn):
    """Paper §3.4 checkpointed memory: each segment continues from the previous
    state, so the cached state i covers segments 0..i (sequential, slower)."""
    outs, states, state = [], [], None
    for s in range(0, q.shape[1], chunk):
        sl = slice(s, s + chunk)
        o, state = gdn_fn(q=q[:, sl], k=k[:, sl], v=v[:, sl], g=g[:, sl], b=b[:, sl], w=w[:, sl],
                          initial_state=state, output_final_state=True,
                          use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False, cu_seqlens=None)
        outs.append(o)
        states.append(state)
    return torch.cat(outs, dim=1), torch.stack(states, dim=1)


# --------------------------------------------------------------------------- aux losses
@register("aux_loss", "none")
def no_aux_loss(rec, batch):
    return rec.scores.new_zeros(())


@register("aux_loss", "gold_ce")
def gold_segment_ce(rec, batch):
    """Example hook: cross-entropy of the router scores toward the gold segment at
    the query positions. Needs ``batch['gold_segments']`` [B,G] (-1 padded) and
    ``batch['query_pos']`` [B]; returns 0 when they are absent (plain LM data)."""
    gold, pos = batch.get("gold_segments"), batch.get("query_pos")
    if gold is None or pos is None:
        return rec.scores.new_zeros(())
    idx = torch.arange(gold.shape[0], device=gold.device)
    logits = rec.scores[idx, pos].float()                       # [B,N], -inf = not eligible
    logp = F.log_softmax(logits.masked_fill(~torch.isfinite(logits), -1e4), dim=-1)
    g = gold.clamp(min=0)
    lp = logp.gather(-1, g).masked_fill(gold < 0, -torch.inf)
    return -torch.logsumexp(lp, dim=-1).mean()                 # any gold counts
