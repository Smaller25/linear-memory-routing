# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic-MoSC integrated model = GDN2 backbone + dynamic chunking + segment-cache router.

Forward (teacher-forced):
  1. base hidden + logits from the GDN2 backbone
  2. per-token surprisal from those logits (for ``surprisal`` chunk mode)
  3. segment boundaries (fixed | oracle | surprisal)  -> ``dynamic_chunk``
  4. per-segment summaries -> a cache bank
  5. hard top-k read-out over the bank (optionally routed to M pools) -> ``router``
  6. lm_head(base_hidden + read_out)

STATUS: runnable skeleton for the Phase-0/1 experiments. The segment summary is currently a masked
mean of the base hidden over each segment (a PROXY). Phase-1 TODO: replace with the true GDN2
recurrent state at each boundary via a segmented run (cf. ``lmr/segment_runner.py`` in the frozen
track) so the read-out recovers saturated state, not just re-pooled activations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmr.mosc.backbone import GDN2LM
from lmr.mosc.dynamic_chunk import positions_to_mask, segment_boundaries, token_surprisal
from lmr.mosc.router import SegmentCacheRouter


class DynamicMoSC(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 2,
        head_dim: int = 64,
        num_heads: int = 4,
        chunk_mode: str = "fixed",
        chunk: int = 256,
        num_pools: int = 1,
        topk: int = 4,
        surprisal_min_gap: int = 8,
        use_true_state: bool = False,
    ):
        super().__init__()
        self.backbone = GDN2LM(vocab_size, d_model, n_layers, head_dim, num_heads)
        self.router = SegmentCacheRouter(d_model, num_pools=num_pools, topk=topk)
        self.chunk_mode = chunk_mode
        self.chunk = chunk
        self.surprisal_min_gap = surprisal_min_gap
        self.lm_head = self.backbone.lm_head  # tie read-out head to the backbone's
        # true-state mode: cache each segment's actual GDN2 recurrent state (not a pooled-hidden
        # proxy) and read over it -> tests whether recall comes from the RNN state vs hidden attention.
        self.use_true_state = use_true_state
        self.state_proj = nn.Linear(self.backbone.state_dim, d_model) if use_true_state else None
        # Phase-1: a small head that predicts segment boundaries from the model's own hidden states.
        # Trained (optionally distilled from oracle positions) to recover the per-fact boundaries that
        # made the oracle win — the open question Phase-0 localised.
        self.boundary_predictor = nn.Linear(d_model, 1) if chunk_mode == "learned" else None
        self.boundary_threshold = 0.5  # eval-time sigmoid cutoff for firing a boundary (sweepable)
        self.boundary_loss = None  # set per-forward; consumed by the trainer

    def _segment_summaries(self, hidden: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
        """[B, T, d] hidden + [B, T] boundary mask -> [B, N, d] per-segment masked-mean summaries.

        PROXY (see module docstring). N = max #segments across the batch; short rows are zero-padded
        (their extra all-zero summaries contribute ~0 after the descriptor projection).
        """
        B, T, d = hidden.shape
        seg_id = boundaries.long().cumsum(1) - boundaries.long()  # segment index per token
        N = int(seg_id.max().item()) + 1
        summ = hidden.new_zeros(B, N, d)
        cnt = hidden.new_zeros(B, N, 1)
        summ.scatter_add_(1, seg_id[..., None].expand(-1, -1, d), hidden)
        cnt.scatter_add_(1, seg_id[..., None], torch.ones_like(hidden[..., :1]))
        return summ / cnt.clamp_min(1.0)

    def _seg_bounds(self, oracle_positions, seq_len):
        """(start, end) slices partitioning [0, T], uniform across the batch (oracle/fixed only)."""
        if self.chunk_mode == "oracle":
            ends = sorted(set((oracle_positions[0] + 1).tolist()) | {seq_len})
        else:  # fixed
            ends = sorted(set(range(self.chunk, seq_len, self.chunk)) | {seq_len})
        bounds, prev = [], 0
        for e in ends:
            if e > prev:
                bounds.append((prev, e)); prev = e
        return bounds

    def forward(self, input_ids: torch.Tensor, oracle_positions: torch.Tensor | None = None,
                boundary_distill: float = 0.0):
        B, T = input_ids.shape

        if self.use_true_state:
            # cache the actual recurrent state per segment (oracle/fixed boundaries) and read over it
            bounds = self._seg_bounds(oracle_positions, T)
            h_full, states = self.backbone.run_segmented(input_ids, bounds)   # [B,T,d], [B,N,state_dim]
            self.boundary_loss = h_full.new_zeros(())
            bank = self.state_proj(states)                                    # [B, N, d]
            read = self.router.read(h_full, bank, pool_of=None)
            return self.lm_head(h_full + read)

        base_h = self.backbone(input_ids, return_hidden=True)        # [B, T, d]
        base_logits = self.lm_head(base_h)
        self.boundary_loss = base_h.new_zeros(())

        if self.chunk_mode == "learned":
            blogits = self.boundary_predictor(base_h).squeeze(-1)    # [B, T]
            bnd = (blogits.sigmoid() > self.boundary_threshold).clone()   # hard boundaries at inference
            bnd[:, -1] = True
            if boundary_distill > 0.0 and oracle_positions is not None:
                tgt = positions_to_mask(oracle_positions, T).float()
                pos_w = (tgt.numel() - tgt.sum()) / tgt.sum().clamp_min(1.0)  # boundaries are sparse
                self.boundary_loss = boundary_distill * F.binary_cross_entropy_with_logits(
                    blogits, tgt, pos_weight=pos_w)
        else:
            surp = token_surprisal(base_logits, input_ids) if self.chunk_mode == "surprisal" else None
            bnd = segment_boundaries(
                batch_size=B, seq_len=T, mode=self.chunk_mode, chunk=self.chunk, surprisal=surp,
                min_gap=self.surprisal_min_gap, oracle_positions=oracle_positions, device=input_ids.device,
            )

        self.last_boundaries = bnd                                   # [B, T] for diagnostics
        bank = self._segment_summaries(base_h, bnd)                  # [B, N, d]
        read = self.router.read(base_h, bank, pool_of=None if self.router.num_pools == 1 else
                                self.router.write(bank))             # [B, T, d]
        return self.lm_head(base_h + read)
