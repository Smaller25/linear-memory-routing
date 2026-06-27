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
        budget: float = 0.05,
    ):
        super().__init__()
        self.backbone = GDN2LM(vocab_size, d_model, n_layers, head_dim, num_heads)
        self.router = SegmentCacheRouter(d_model, num_pools=num_pools, topk=topk)
        self.chunk_mode = chunk_mode
        self.chunk = chunk
        self.surprisal_min_gap = surprisal_min_gap
        self.budget = budget                  # unsup: L1 weight on landmark prob (sparsity)
        self.lm_head = self.backbone.lm_head  # tie read-out head to the backbone's
        # true-state mode: cache each segment's actual GDN2 recurrent state (not a pooled-hidden
        # proxy) and read over it -> tests whether recall comes from the RNN state vs hidden attention.
        self.use_true_state = use_true_state
        self.state_proj = nn.Linear(self.backbone.state_dim, d_model) if use_true_state else None
        # boundary head: predicts per-token boundary/landmark logits. `learned` distills it from oracle
        # positions; `unsup` trains it end-to-end (no oracle) via a differentiable landmark-biased read
        # + an L1 sparsity budget — testing whether the boundaries are learnable WITHOUT supervision.
        self.boundary_predictor = nn.Linear(d_model, 1) if chunk_mode in ("learned", "unsup", "unsup_ste") else None
        if chunk_mode == "unsup_ste":
            nn.init.constant_(self.boundary_predictor.bias, 2.0)  # start p~0.88 (avoid empty-cache cold start)
        self.boundary_threshold = 0.5  # eval-time sigmoid cutoff for firing a boundary (sweepable)
        self.boundary_loss = None  # set per-forward; consumed by the trainer

    def _segment_summaries(self, hidden: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
        """[B, T, d] hidden + [B, T] boundary mask -> [B, N, d] per-segment summaries.

        Summary = the hidden at each segment's END (boundary) token — the state right after that
        segment's content. Mean-pooling over the segment dilutes the fact when the segment contains
        filler/blanks (which collapses recall on irregular layouts); the boundary token's hidden does
        not. PROXY for the true recurrent state. N = max #segments; short rows zero-pad.
        """
        B, T, d = hidden.shape
        seg_id = boundaries.long().cumsum(1) - boundaries.long()  # segment index per token
        N = int(seg_id.max().item()) + 1
        # boundary tokens scatter their hidden to bank[seg_id]; non-boundary tokens -> dump slot N.
        # Each segment has exactly one boundary (its end), so each slot gets one clean write.
        tgt = torch.where(boundaries, seg_id, torch.full_like(seg_id, N))
        bank = hidden.new_zeros(B, N + 1, d)
        bank.scatter_(1, tgt[..., None].expand(-1, -1, d), hidden)
        return bank[:, :N]

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

        if self.chunk_mode == "unsup_ste":
            # UNSUPERVISED with a SPARSE hard cache (no full attention): the head's hard boundaries
            # (STE: forward 0/1, backward sigmoid) define a segment-cache; the read is top-k over those
            # ~N compressed summaries only. Each summary is gated by the STE prob so the task gradient
            # reaches the head; an L1 budget prunes boundaries. Train==eval (both hard sparse cache), so
            # the model MUST commit to boundaries — boundary precision/recall is now meaningful.
            logits = self.boundary_predictor(base_h).squeeze(-1)         # [B, T]
            p = logits.sigmoid()
            hard = p > 0.5
            hard_f = hard.float() + (p - p.detach())                    # STE value at each token
            bnd = hard.clone(); bnd[:, -1] = True                        # keep cache non-empty
            self.last_boundaries = bnd
            seg_id = bnd.long().cumsum(1) - bnd.long()
            N = int(seg_id.max().item()) + 1
            tgt = torch.where(bnd, seg_id, torch.full_like(seg_id, N))   # non-boundary -> dump slot
            bank = base_h.new_zeros(B, N + 1, base_h.shape[-1])
            bank.scatter_(1, tgt[..., None].expand(-1, -1, base_h.shape[-1]), base_h)   # boundary-hidden
            gate = base_h.new_zeros(B, N + 1)
            gate.scatter_(1, tgt, hard_f)                               # STE prob per segment
            bank = (bank[:, :N] * gate[:, :N, None])                     # gradient to p via magnitude
            read = self.router.read(base_h, bank, pool_of=None)         # top-k over the SPARSE cache
            self.boundary_loss = self.budget * p.mean()                 # L1 sparsity (no oracle)
            return self.lm_head(base_h + read)

        if self.chunk_mode == "unsup":
            # UNSUPERVISED: no oracle. The head predicts a per-token landmark prob p; the read-out is a
            # causal attention over all tokens biased by log p (so gradient reaches the head), and an
            # L1 budget keeps p sparse. At inference, threshold p for hard boundaries.
            p = self.boundary_predictor(base_h).squeeze(-1).sigmoid()    # [B, T]
            q = self.router.query_proj(base_h)                            # [B, T, r]
            k = self.router.key_proj(base_h)
            scores = torch.einsum("bir,bjr->bij", q, k) / (q.shape[-1] ** 0.5)
            scores = scores + torch.log(p + 1e-6)[:, None, :]             # prefer high-landmark tokens
            causal = torch.triu(torch.ones(T, T, device=base_h.device, dtype=torch.bool), 1)
            scores = scores.masked_fill(causal[None], float("-inf"))
            kk = min(self.router.topk, T)
            topv, topi = scores.topk(kk, dim=-1)                          # [B, T, k]
            w = topv.softmax(-1)
            sel = base_h[:, None].expand(-1, T, -1, -1).gather(
                2, topi[..., None].expand(-1, -1, -1, base_h.shape[-1]))  # [B, T, k, d]
            read = self.router.out_proj((w[..., None] * sel).sum(2))
            self.boundary_loss = self.budget * p.mean()                   # sparsity (no oracle)
            self.last_boundaries = (p > 0.5)
            return self.lm_head(base_h + read)

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
