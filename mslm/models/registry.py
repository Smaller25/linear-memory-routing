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
|             |                 | or nn.Module(d_model, n_head, head_dim, mc_cfg);                 |
|             |                 |   forward(x, seg, online, idx[B,T,R], valid) -> weights          |
|             |                 |   (its own scores: selection and mixing are decoupled)           |
| connect     | MC_CONNECT      | fn(q,k,v,g,b,w, chunk, gdn_fn) -> (online_out[B,T,H,V],          |
|             |                 |   memories[B,N,H,K,V])   (T is a multiple of chunk here)         |
| aux_loss    | MC_AUX_LOSS     | fn(rec: RoutingRecord, batch: dict) -> scalar tensor             |

A descriptor may also return ``seg`` as [B,N,m,H,K] (m sub-blocks per segment);
it must then be paired with a router that reduces over m (e.g. ``maxsim``).

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


@register("aux_loss", "span_copy_ce")
def span_copy_ce(rec, batch):
    """Cross-entropy of the router scores toward span-copy gold segments
    (``mslm.data.span_copy``) at every labelled position. The softmax runs over
    ALL eligible past segments before top-k, so unselected segments get a
    gradient too. Needs ``batch['copy_gold']`` [B,T,G] (-1 padded); positions
    without a label are skipped. Returns a zero that keeps the graph when a
    batch has no labels (DDP needs every router parameter in the graph)."""
    zero = rec.scores.masked_fill(~torch.isfinite(rec.scores), 0).sum() * 0   # scores hold -inf
    gold = batch.get("copy_gold")
    if gold is None:
        return zero
    gold = gold.to(rec.scores.device)[:, :rec.scores.shape[1]]
    b, t = torch.nonzero(gold[..., 0] >= 0, as_tuple=True)
    if b.numel() == 0:
        return zero
    logits = rec.scores[b, t].float()                            # [M,N], -inf = not eligible
    logits = logits.masked_fill(~torch.isfinite(logits), -1e4)
    logp = F.log_softmax(logits, dim=-1)
    g = gold[b, t]                                               # [M,G]
    lp = logp.gather(-1, g.clamp(min=0)).masked_fill(g < 0, -torch.inf)
    return -torch.logsumexp(lp, dim=-1).mean()                   # any gold counts


@register("descriptor", "subblock8")
def subblock8_descriptor(keys: torch.Tensor, chunk: int):
    """Mean of normalised keys over 8 equal sub-blocks of each segment
    (seg [B,N,8,H,K]); the online descriptor is the meanpool one. A needle that
    is 2% of a 256-token segment is 1/32 of the segment mean but 1/4 of a
    32-token sub-block mean."""
    m = 8
    if chunk % m:
        raise ValueError(f"subblock8 needs MC_CHUNK divisible by {m}, got {chunk}")
    B, T, H, K = keys.shape
    N = T // chunk
    seg = keys.float().view(B, N, m, chunk // m, H, K).mean(dim=3)
    _, online = meanpool_descriptor(keys, chunk)
    return seg, online


@register("router", "maxsim")
class MaxSimRouter(DotRouter):
    """r_t^i = max_j sum_h <u_t^h, c_{i,j}^h> over the sub-blocks j of segment i."""

    def forward(self, x, seg, online):
        if seg.dim() != 5:
            raise ValueError("router 'maxsim' needs a sub-block descriptor (seg [B,N,m,H,K]), e.g. subblock8")
        B, T, _ = x.shape
        u = self.connector(x).view(B, T, self.n_head, self.head_dim).float()
        scores = torch.einsum("bthk,bnmhk->btnm", u, seg.float()).amax(dim=-1)
        online_score = torch.einsum("bthk,bthk->bt", u, online.float())
        return scores, online_score


@register("gate", "sep_softmax")
class SeparateSoftmaxGate(nn.Module):
    """Mixing weights from their own projection W_g, decoupled from the router.

    The router (trained by an aux loss) only decides WHICH segments are read;
    this gate decides HOW MUCH of each is read: softmax over [online, selected]
    of <W_g x_t, c_i> (sum over heads; sub-block descriptors are averaged).
    W_g learns from the LM loss only. With this gate the router gets no LM
    gradient at all, so it needs MC_AUX_LOSS (checked in MCMixer)."""

    def __init__(self, d_model: int, n_head: int, head_dim: int, mc_cfg=None):
        super().__init__()
        self.n_head, self.head_dim = n_head, head_dim
        self.proj = nn.Linear(d_model, n_head * head_dim, bias=False)

    def forward(self, x, seg, online, idx, valid):
        B, T, _ = x.shape
        if seg.dim() == 5:
            seg = seg.mean(dim=2)
        u = self.proj(x).view(B, T, self.n_head, self.head_dim).float()
        all_scores = torch.einsum("bthk,bnhk->btn", u, seg.float())
        sel = all_scores.gather(-1, idx.clamp(min=0))
        online_score = torch.einsum("bthk,bthk->bt", u, online.float())
        logits = torch.cat([online_score.unsqueeze(-1), sel.masked_fill(~valid, -torch.inf)], dim=-1)
        return torch.softmax(logits, dim=-1)
