# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Surprisal-driven dynamic chunking — the temporal axis of Dynamic-MoSC.

Instead of caching the recurrent state at rigid fixed intervals (SSC), place segment boundaries
adaptively so a segment is closer to *per-fact* than *per-256-tokens* — directly attacking report
0010's finding ("MC caches one state per segment, not per fact; keys sharing a segment can't be
disambiguated").

Three boundary modes (all return a boolean ``[B, T]`` mask, True = a boundary *ends* at this token):
  - ``fixed``     : every ``chunk`` tokens (== SSC; the control).
  - ``oracle``    : boundaries at supplied positions (e.g. exactly at each key) — the Phase-0
                    KILL-TEST upper bound. If oracle boundaries don't break the multi-key wall,
                    nothing downstream (surprisal approx, parallel pools) can.
  - ``surprisal`` : boundaries at high-surprisal tokens (NLL peaks), with a min gap so dense regions
                    don't degenerate to per-token (the R1 risk we must measure).

NOTE: ``surprisal`` needs per-token NLL. For a frozen LM use its lm_head; from-scratch, use the
model's own logits (teacher-forced) or a detached proxy. See ``token_surprisal``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_surprisal(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token surprisal -log p(x_t | x_<t) as ``[B, T]`` (shifted; position 0 = 0)."""
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    nll = -logp.gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    return F.pad(nll, (1, 0), value=0.0)


def segment_boundaries(
    *,
    batch_size: int,
    seq_len: int,
    mode: str = "fixed",
    chunk: int = 256,
    surprisal: torch.Tensor | None = None,
    threshold: float | None = None,
    min_gap: int = 8,
    oracle_positions: torch.Tensor | None = None,
    device=None,
) -> torch.Tensor:
    """Return a boolean ``[B, T]`` boundary mask (True at the last token of each segment)."""
    dev = device
    if mode == "fixed":
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=dev)
        mask[:, chunk - 1 :: chunk] = True
        mask[:, -1] = True
        return mask

    if mode == "oracle":
        assert oracle_positions is not None, "oracle mode needs oracle_positions [B, *] of token idx"
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=oracle_positions.device)
        mask.scatter_(1, oracle_positions.clamp_(0, seq_len - 1), True)
        mask[:, -1] = True
        return mask

    if mode == "surprisal":
        assert surprisal is not None, "surprisal mode needs a [B, T] surprisal tensor"
        if threshold is None:  # default: mean + 1 std per sequence
            threshold = (surprisal.mean(1, keepdim=True) + surprisal.std(1, keepdim=True))
        mask = surprisal >= threshold
        # enforce a minimum gap so dense high-surprisal runs don't collapse to per-token (R1)
        mask = _enforce_min_gap(mask, min_gap)
        mask[:, -1] = True
        return mask

    raise ValueError(f"unknown chunk mode: {mode!r}")


def _enforce_min_gap(mask: torch.Tensor, min_gap: int) -> torch.Tensor:
    """Greedily drop boundaries closer than ``min_gap`` to the previous kept one (per row)."""
    if min_gap <= 1:
        return mask
    out = torch.zeros_like(mask)
    for b in range(mask.shape[0]):
        last = -min_gap
        idx = mask[b].nonzero(as_tuple=True)[0]
        for t in idx.tolist():
            if t - last >= min_gap:
                out[b, t] = True
                last = t
    return out


def positions_to_mask(positions: torch.Tensor, seq_len: int) -> torch.Tensor:
    """[B, k] token indices -> boolean ``[B, T]`` mask (used to turn oracle positions into a target
    for the learned boundary predictor)."""
    mask = torch.zeros(positions.shape[0], seq_len, dtype=torch.bool, device=positions.device)
    mask.scatter_(1, positions.clamp(0, seq_len - 1), True)
    return mask


def mqar_oracle_positions(num_kv_pairs: int, batch_size: int, device=None) -> torch.Tensor:
    """Oracle boundaries for Phase-0: one segment per (key,value) fact, in the CONTEXT region.

    ``make_mqar`` lays out the context as the first ``2*num_kv_pairs`` tokens with keys at even
    indices and **values at odd indices** (1, 3, ..., 2k-1). To make each segment ~per-fact, end a
    segment at each value token, so the value's state is cleanly checkpointed. Returns ``[B, k]``.

    NOTE: this is the upper-bound oracle — boundaries at the values being recalled, not at the query
    keys (those live later, in the query region, and contain no values to cache).
    """
    pos = torch.arange(1, 2 * num_kv_pairs, 2, device=device)  # [k] value positions in context
    return pos[None].expand(batch_size, -1).contiguous()
