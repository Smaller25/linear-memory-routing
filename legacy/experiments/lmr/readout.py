# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Memory-Caching read-out heads: RM / GRM / SSC / MoM / AoM.

Each head combines the *online* segment output ``y_main`` with the per-checkpoint contributions
``y_cached[i]`` -- the outputs of running the scan on the current segment with the write/value path
zeroed and frozen checkpoint ``h_L^(i)`` injected as the initial state. All combination happens at
the recurrent-output level (``[b, l, h, p]``), *before* the output gate / norm.

- **RM**  (Residual Memory): plain sum. Training-free, no parameters.
- **GRM** (Gated RM): per-token, per-checkpoint sigmoid gate
  ``gamma_t^(i) = sigmoid(<u_t, descriptor^(i)>)``, ``u_t = x_t W_u``.
- **SSC** (Sparse Selective Caching): top-k router over checkpoints + load-balance aux loss.
- **MoM** (Mixture-of-Memories read-out): route each checkpoint to one of ``M`` bounded slots, merge
  within slot, then a per-token softmax gate combines the ``M`` slot-outputs (+ optional shared
  slot). Constant ``M`` fan-in regardless of segment count. Switch-style load-balance aux.
- **AoM** (Attention over Memories): softmax attention -- query ``u_t = x_t W_q`` over the
  checkpoint descriptors (keys), weighting the cached contributions (values). Normalised over the
  number of checkpoints.

Every trained head accepts ``low_rank_dim``: it factors the ``hidden -> descriptor_dim`` projection
as ``hidden -> r -> descriptor_dim`` (``r`` ~64-128), cutting the parameter count ~25x at 1.3b. When
``low_rank_dim is None`` the projection is a single ``Linear`` (state-dict keys unchanged, so older
full-rank checkpoints still load).

``descriptor_dim`` is ``num_heads * d_state`` for Mamba2 / ``num_v_heads * head_k_dim`` for GDN --
the dimension of the adapter's state descriptor (see :mod:`lmr.adapters`, :mod:`lmr.state_utils`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stack(y_cached: list[torch.Tensor]) -> torch.Tensor:
    # list of [b, l, h, p] -> [b, i, l, h, p]
    return torch.stack(y_cached, dim=1)


def _make_projector(hidden_size: int, out_dim: int, low_rank_dim: int | None) -> nn.Module:
    """``hidden -> out_dim`` (full ``Linear``) or ``hidden -> r -> out_dim`` (low-rank, ~25x fewer)."""
    if low_rank_dim is None:
        return nn.Linear(hidden_size, out_dim, bias=False)
    return nn.Sequential(
        nn.Linear(hidden_size, low_rank_dim, bias=False),
        nn.Linear(low_rank_dim, out_dim, bias=False),
    )


class ResidualMemory(nn.Module):
    """Training-free residual sum of cached contributions (paper: RM)."""

    is_trainable = False

    def forward(self, y_main, y_cached, x=None, descriptors=None):
        if not y_cached:
            return y_main, None
        return y_main + _stack(y_cached).sum(dim=1), None


class GatedResidualMemory(nn.Module):
    """Input-dependent sigmoid gate over each cached contribution (paper: GRM)."""

    is_trainable = True

    def __init__(self, hidden_size: int, descriptor_dim: int, low_rank_dim: int | None = None):
        super().__init__()
        self.W_u = _make_projector(hidden_size, descriptor_dim, low_rank_dim)

    def forward(self, y_main, y_cached, x, descriptors):
        # x: [b, l, hidden]; descriptors: [b, i, descriptor_dim]
        if not y_cached:
            return y_main, None
        u = self.W_u(x)                                            # [b, l, dd]
        scores = torch.einsum("bld,bid->bli", u, descriptors)     # [b, l, i]
        gamma = torch.sigmoid(scores)                             # [b, l, i]
        contrib = _stack(y_cached)                                # [b, i, l, h, p]
        y = y_main + (gamma.permute(0, 2, 1)[..., None, None] * contrib).sum(dim=1)
        return y, None


def load_balancing_loss(router_probs: torch.Tensor, selection_mask: torch.Tensor) -> torch.Tensor:
    """Switch-style load-balance aux loss over the routed dimension.

    Adapts ``load_balancing_loss_func`` from ``fla/models/mom/modeling_mom.py:41`` to routing over
    *checkpoints* (SSC) or *slots* (MoM) instead of *memory slots* per token.

    - ``router_probs``: ``[..., n]`` softmax router distribution over the ``n`` routed items.
    - ``selection_mask``: ``[..., n]`` 1.0 where item ``n`` is in the (token's) top-k.
    """
    n = router_probs.shape[-1]
    dims = tuple(range(router_probs.dim() - 1))
    tokens_per = selection_mask.float().mean(dim=dims)          # [n]
    prob_per = router_probs.mean(dim=dims)                      # [n]
    return n * torch.sum(tokens_per * prob_per)


class SparseSelectiveCaching(nn.Module):
    """Top-k MoE-style routing over cached checkpoints + load-balance aux (paper: SSC)."""

    is_trainable = True

    def __init__(self, hidden_size: int, descriptor_dim: int, topk: int = 2, aux_scale: float = 1e-2,
                 low_rank_dim: int | None = None):
        super().__init__()
        self.router = _make_projector(hidden_size, descriptor_dim, low_rank_dim)
        self.topk = topk
        self.aux_scale = aux_scale

    def forward(self, y_main, y_cached, x, descriptors):
        if not y_cached:
            return y_main, None
        num_ckpt = len(y_cached)
        u = self.router(x)                                        # [b, l, dd]
        logits = torch.einsum("bld,bid->bli", u, descriptors)     # [b, l, i]
        probs = F.softmax(logits, dim=-1)

        k = min(self.topk, num_ckpt)
        topv, topi = torch.topk(probs, k, dim=-1)                 # [b, l, k]
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)     # renormalise selected

        weights = torch.zeros_like(probs).scatter_(-1, topi, topv)  # [b, l, i]
        selection_mask = torch.zeros_like(probs).scatter_(-1, topi, 1.0)

        contrib = _stack(y_cached)                                # [b, i, l, h, p]
        y = y_main + (weights.permute(0, 2, 1)[..., None, None] * contrib).sum(dim=1)
        aux = self.aux_scale * load_balancing_loss(probs, selection_mask)
        return y, aux


class MoMReadout(nn.Module):
    """Bounded ``M``-slot routing read-out adapted from Mixture-of-Memories (paper: MoM).

    Each cached checkpoint is routed (top-1, by its descriptor) to one of ``num_slots`` slots and
    merged within slot; a per-token softmax gate then combines the ``M`` slot-outputs, plus an
    optional always-on shared slot that sees every checkpoint. The combine fan-in is constant in
    ``M`` regardless of how many segments have been cached. The router contribution is Switch-scaled
    (multiplied by the selected slot's softmax prob) so it receives gradient from the main loss.
    """

    is_trainable = True

    def __init__(self, hidden_size: int, descriptor_dim: int, num_slots: int = 4,
                 shared_slot: bool = True, aux_scale: float = 1e-2, low_rank_dim: int | None = None):
        super().__init__()
        self.num_slots = num_slots
        self.shared_slot = shared_slot
        self.aux_scale = aux_scale
        # checkpoint -> slot assignment from the checkpoint descriptor
        self.slot_router = nn.Linear(descriptor_dim, num_slots, bias=False)
        # per-token gate over the M slots (+ shared slot) from the query
        self.slot_gate = _make_projector(hidden_size, num_slots + (1 if shared_slot else 0), low_rank_dim)

    def forward(self, y_main, y_cached, x, descriptors):
        if not y_cached:
            return y_main, None
        contrib = _stack(y_cached)                                # [b, i, l, h, p]
        slot_logits = self.slot_router(descriptors)               # [b, i, M]
        probs = F.softmax(slot_logits, dim=-1)
        assign = torch.zeros_like(probs).scatter_(
            -1, slot_logits.argmax(dim=-1, keepdim=True), 1.0)     # [b, i, M] one-hot top-1
        gated_assign = assign * probs                             # Switch scale -> router gradient

        slot_out = torch.einsum("bim,bilhp->bmlhp", gated_assign, contrib)  # [b, M, l, h, p]
        gate = F.softmax(self.slot_gate(x), dim=-1)               # [b, l, M(+1)]
        m = self.num_slots
        y = y_main + torch.einsum("blm,bmlhp->blhp", gate[..., :m], slot_out)
        if self.shared_slot:
            shared = contrib.sum(dim=1)                           # [b, l, h, p]
            y = y + gate[..., m][..., None, None] * shared
        aux = self.aux_scale * load_balancing_loss(probs, assign)
        return y, aux


class AttentionOverMemories(nn.Module):
    """Softmax attention over cached checkpoints: query from ``x``, keys = descriptors (paper: AoM)."""

    is_trainable = True

    def __init__(self, hidden_size: int, descriptor_dim: int, low_rank_dim: int | None = None):
        super().__init__()
        self.W_q = _make_projector(hidden_size, descriptor_dim, low_rank_dim)
        self.scale = descriptor_dim ** -0.5

    def forward(self, y_main, y_cached, x, descriptors):
        if not y_cached:
            return y_main, None
        u = self.W_q(x)                                           # [b, l, dd]
        logits = torch.einsum("bld,bid->bli", u, descriptors) * self.scale
        attn = F.softmax(logits, dim=-1)                          # [b, l, i] over checkpoints
        contrib = _stack(y_cached)                                # [b, i, l, h, p]
        y = y_main + torch.einsum("bli,bilhp->blhp", attn, contrib)
        return y, None


def build_readout(kind: str, hidden_size: int, descriptor_dim: int, *, topk: int = 2,
                  aux_scale: float = 1e-2, num_slots: int = 4, shared_slot: bool = True,
                  low_rank_dim: int | None = None) -> nn.Module:
    kind = kind.lower()
    if kind == "rm":
        return ResidualMemory()
    if kind == "grm":
        return GatedResidualMemory(hidden_size, descriptor_dim, low_rank_dim=low_rank_dim)
    if kind == "ssc":
        return SparseSelectiveCaching(hidden_size, descriptor_dim, topk=topk, aux_scale=aux_scale,
                                      low_rank_dim=low_rank_dim)
    if kind == "mom":
        return MoMReadout(hidden_size, descriptor_dim, num_slots=num_slots, shared_slot=shared_slot,
                          aux_scale=aux_scale, low_rank_dim=low_rank_dim)
    if kind == "aom":
        return AttentionOverMemories(hidden_size, descriptor_dim, low_rank_dim=low_rank_dim)
    raise ValueError(f"unknown read-out kind: {kind!r} (expected rm/grm/ssc/mom/aom)")
