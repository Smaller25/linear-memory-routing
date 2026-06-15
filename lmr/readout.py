# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Memory-Caching read-out heads: RM / GRM / SSC.

Each head combines the *online* segment output ``y_main`` with the per-checkpoint
contributions ``y_cached[i]`` -- the outputs of running the scan on the current segment with
input zeroed and frozen checkpoint ``h_L^(i)`` injected as the initial state. All combination
happens at the SSM-output level (``[b, l, h, p]``), *before* the output gate / RMSNorm, which
is exactly the paper's memory read-out ``y_t = M_t(q_t) + sum_{i<s} M_L^(i)(q_t)``.

- **RM**  (Residual Memory): plain sum. Training-free, no parameters.
- **GRM** (Gated RM): per-token, per-checkpoint sigmoid gate
  ``gamma_t^(i) = sigmoid(<u_t, MeanPool(h_L^(i))>)``, ``u_t = x_t W_u``.
- **SSC** (Sparse Selective Caching): top-k router over checkpoints + load-balance aux loss.

``descriptor_dim`` is ``num_heads * d_state`` -- the dimension of ``meanpool_state`` (see
:func:`lmr.state_utils.meanpool_state`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stack(y_cached: list[torch.Tensor]) -> torch.Tensor:
    # list of [b, l, h, p] -> [b, i, l, h, p]
    return torch.stack(y_cached, dim=1)


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

    def __init__(self, hidden_size: int, descriptor_dim: int):
        super().__init__()
        self.W_u = nn.Linear(hidden_size, descriptor_dim, bias=False)

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
    """Switch-style load-balance aux loss over cached checkpoints.

    Adapts ``load_balancing_loss_func`` from ``fla/models/mom/modeling_mom.py:41`` to routing
    over *checkpoints* instead of *memory slots*.

    - ``router_probs``: ``[b, l, i]`` softmax router distribution.
    - ``selection_mask``: ``[b, l, i]`` 1.0 where checkpoint ``i`` is in the token's top-k.
    """
    n = router_probs.shape[-1]
    tokens_per_ckpt = selection_mask.float().mean(dim=(0, 1))    # [i]
    prob_per_ckpt = router_probs.mean(dim=(0, 1))                # [i]
    return n * torch.sum(tokens_per_ckpt * prob_per_ckpt)


class SparseSelectiveCaching(nn.Module):
    """Top-k MoE-style routing over cached checkpoints + load-balance aux (paper: SSC)."""

    is_trainable = True

    def __init__(self, hidden_size: int, descriptor_dim: int, topk: int = 2, aux_scale: float = 1e-2):
        super().__init__()
        self.router = nn.Linear(hidden_size, descriptor_dim, bias=False)
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


def build_readout(kind: str, hidden_size: int, descriptor_dim: int, **kw) -> nn.Module:
    kind = kind.lower()
    if kind == "rm":
        return ResidualMemory()
    if kind == "grm":
        return GatedResidualMemory(hidden_size, descriptor_dim)
    if kind == "ssc":
        return SparseSelectiveCaching(hidden_size, descriptor_dim, **kw)
    raise ValueError(f"unknown read-out kind: {kind!r} (expected rm/grm/ssc)")
