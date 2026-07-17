# -*- coding: utf-8 -*-
"""Segment sampling and cu_seqlens construction for DynMC training.

Training-time segmentation (plan §2.3):
  - random segment lengths, log-uniform on [64, 1024], quantized to multiples of 64
  - document boundaries force a segment boundary (state reset, plan §2.1)
  - each packed row is a concatenation of documents; segments never cross docs

All boundaries land on multiples of 64 relative to the segment start, matching
the fla chunkwise kernel's 64-token state materialization (plan §1 constraint).
The last segment of a document absorbs the remainder (may be shorter than 64).
"""
from __future__ import annotations

import numpy as np
import torch

CHUNK = 64
SEG_MIN = 64
SEG_MAX = 1024


def sample_seglens(doc_len: int, rng: np.random.Generator,
                   seg_min: int = SEG_MIN, seg_max: int = SEG_MAX) -> list[int]:
    """One document (doc_len tokens) -> list of segment lengths summing to doc_len.

    Lengths ~ log-uniform [seg_min, seg_max], quantized down to multiples of 64.
    The final segment takes the remainder (can be < 64 or > seg_max is impossible).
    """
    out = []
    remaining = doc_len
    while remaining > seg_max:
        raw = float(np.exp(rng.uniform(np.log(seg_min), np.log(seg_max + 1))))
        seg = int(raw // CHUNK) * CHUNK
        seg = max(CHUNK, min(seg, seg_max))
        # avoid leaving a tail shorter than CHUNK when possible
        if remaining - seg < CHUNK:
            seg = remaining
        out.append(seg)
        remaining -= seg
    if remaining > 0:
        out.append(remaining)
    return out


def fixed_seglens(doc_len: int, seg_len: int = 256) -> list[int]:
    """mc-fixed baseline: fixed-length segments, remainder in the last one."""
    out = [seg_len] * (doc_len // seg_len)
    rem = doc_len - seg_len * len(out)
    if rem:
        out.append(rem)
    return out


def build_batch_segments(doc_lens_per_row: list[list[int]],
                         rng: np.random.Generator,
                         mode: str = "random",
                         fixed_len: int = 256):
    """Packed rows -> flattened varlen metadata for the whole (flattened) batch.

    Args:
        doc_lens_per_row: for each row, token counts of the docs packed into it
            (already truncated to the row's ctx length; sum(row) == ctx).
        mode: "random" (log-uniform, plan §2.3) or "fixed" (mc-fixed baseline).

    Returns dict with (torch tensors on CPU):
        cu_seqlens   [S+1] int32 — segment offsets in the flattened [1, B*ctx] layout
        seg_doc_ids  [S]   int64 — global doc id per segment (cross-doc read mask)
        seg_row_ids  [S]   int64 — row index per segment
    """
    cu = [0]
    seg_doc_ids: list[int] = []
    seg_row_ids: list[int] = []
    doc_id = 0
    for row_idx, doc_lens in enumerate(doc_lens_per_row):
        for dl in doc_lens:
            segs = (sample_seglens(dl, rng) if mode == "random"
                    else fixed_seglens(dl, fixed_len))
            for s in segs:
                cu.append(cu[-1] + s)
                seg_doc_ids.append(doc_id)
                seg_row_ids.append(row_idx)
            doc_id += 1
    return {
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32),
        "seg_doc_ids": torch.tensor(seg_doc_ids, dtype=torch.long),
        "seg_row_ids": torch.tensor(seg_row_ids, dtype=torch.long),
    }
