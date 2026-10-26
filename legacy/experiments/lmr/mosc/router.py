# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Mixture of Segment-Cache — the spatial axis of Dynamic-MoSC (Phase-2 component).

When a (dynamic) boundary fires, the segment's summarised recurrent state is dispatched to one of
``M`` parallel **expert memory pools** by a segment-level write-router; at read time a **hard top-k**
router selects over the (pool x cached-segment) bank. Hard top-k is the only read-out our frozen
experiments found to generalise (reports 0008/0009; sweet spot k in [2, 8]).

STATUS: skeleton + interfaces. The full mechanism is the research deliverable (Phase 2); it is
gated on the Phase-0 oracle-chunking kill-test passing first. ``num_pools=1`` reduces this to a
plain hard-top-k read over the temporal cache (== SSC), which is what Phase 0/1 exercise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentCacheRouter(nn.Module):
    """Write-route segment states to ``num_pools`` pools; read via hard top-k over the bank.

    Args:
        d_model: model width.
        num_pools: M parallel expert memories (1 == temporal-only / SSC).
        topk: hard top-k fan-in at read time (k in [2, 8] recommended).
        descriptor_dim: width of the cheap pooled descriptor used for scoring (keeps read O(N*k),
            not O(N*d)).
    """

    def __init__(self, d_model: int, num_pools: int = 4, topk: int = 4, descriptor_dim: int = 64):
        super().__init__()
        self.num_pools = num_pools
        self.topk = topk
        self.write_router = nn.Linear(d_model, num_pools, bias=False)   # segment -> pool logits
        self.query_proj = nn.Linear(d_model, descriptor_dim, bias=False)
        self.key_proj = nn.Linear(d_model, descriptor_dim, bias=False)   # descriptor of a cached seg
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def write(self, segment_summaries: torch.Tensor) -> torch.Tensor:
        """[B, N, d] segment summaries -> [B, N] hard pool assignment (argmax; Switch-style).

        TODO(phase2): top-1 vs top-k_w write; load-balance aux (KEEP aux_scale ~1e-4 — report 0010
        showed a layer-summed Switch aux dominates the loss and kills selectivity).
        """
        return self.write_router(segment_summaries).argmax(-1)

    def read(self, query: torch.Tensor, bank: torch.Tensor, pool_of: torch.Tensor) -> torch.Tensor:
        """Hard top-k read.

        Args:
            query: [B, T, d] per-token query.
            bank:  [B, N, d] cached segment states.
            pool_of: [B, N] pool id per cached segment (from ``write``).
        Returns: [B, T, d] read-out contribution.

        TODO(phase2): restrict the candidate set per query to its routed pool(s) before top-k so the
        parallel axis actually isolates interference (else this is just SSC with extra params).
        """
        q = self.query_proj(query)                         # [B, T, r]
        k = self.key_proj(bank)                             # [B, N, r]
        scores = torch.einsum("btr,bnr->btn", q, k)         # [B, T, N]
        kk = min(self.topk, bank.shape[1])
        topv, topi = scores.topk(kk, dim=-1)                # [B, T, k]
        w = F.softmax(topv, dim=-1)                         # [B, T, k]
        sel = torch.gather(bank[:, None].expand(-1, query.shape[1], -1, -1), 2,
                           topi[..., None].expand(-1, -1, -1, bank.shape[-1]))  # [B, T, k, d]
        out = (w[..., None] * sel).sum(2)                   # [B, T, d]
        return self.out_proj(out)
