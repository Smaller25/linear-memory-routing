# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Flip-flop language modeling (FFLM) — a state-tracking probe.

Liu et al., "Exposing Attention Glitches with Flip-Flop Language Modeling" (NeurIPS 2023,
arXiv 2306.00946; official data: hf.co/datasets/synthseq/flipflop). A sequence of (instruction,
bit) pairs over {w, r, i} × {0, 1}: ``w b`` WRITEs bit b to a 1-bit register, ``i b`` is an IGNORE
distractor, and after a READ ``r`` the model must output the **most recently written bit**. Long
runs of IGNOREs between a write and its read stress whether the recurrent state *maintains* the
register (state-tracking) — a different axis from MQAR's associative recall.

This is the standard format (not associative recall): keys/values play no role; only the running
1-bit state matters. We supervise only the read-answer positions. Same {input_ids, labels} contract
as ``lmr.tasks.mqar`` (labels ``-100`` except answer positions). NOTE: this probes a capability the
recurrent backbone already has natively — its use here is a *do-no-harm* check (does segment-cache
routing, which resets state per segment, degrade state-tracking?), not a direct memory-caching test.

Token ids: W=0, R=1, I=2 (instructions); B0=3, B1=4 (bits). vocab_size = 5.
"""

from __future__ import annotations

import numpy as np
import torch

IGNORE = -100
W, R, I, B0, B1 = 0, 1, 2, 3, 4
VOCAB = 5


def make_flipflop(
    num_examples: int = 512,
    n_instr: int = 256,
    p_write: float = 0.1,
    p_read: float = 0.1,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """FFLM batch. ``n_instr`` instructions -> token length ``2*n_instr`` (instr, bit pairs).

    ``p_write``/``p_read`` set the write/read rates; the rest are IGNOREs (distractors). Each example
    starts with a WRITE so a register value always exists. Returns next-token ``{input_ids, labels}``
    (``[N, 2*n_instr-1]``); labels are the gold bit only at the position right after each READ.
    """
    rng = np.random.default_rng(seed)
    p_ignore = 1.0 - p_write - p_read
    assert p_ignore >= 0, "p_write + p_read must be <= 1"
    L = 2 * n_instr
    toks = np.empty((num_examples, L), dtype=np.int64)
    labs = np.full((num_examples, L), IGNORE, dtype=np.int64)

    for n in range(num_examples):
        ops = rng.choice([W, R, I], size=n_instr, p=[p_write, p_read, p_ignore])
        ops[0] = W                                            # ensure a value exists before any read
        last = None
        for j, op in enumerate(ops):
            ti = 2 * j
            if op == W:
                b = int(rng.integers(0, 2)); toks[n, ti] = W; toks[n, ti + 1] = B0 + b; last = b
            elif op == I:
                b = int(rng.integers(0, 2)); toks[n, ti] = I; toks[n, ti + 1] = B0 + b
            else:                                             # READ: answer = last written bit
                toks[n, ti] = R; toks[n, ti + 1] = B0 + last
                labs[n, ti] = B0 + last                       # predict the answer right after seeing R

    # label already sits at the READ (predicting) position ti, with target toks[ti+1]; no shift.
    inputs = torch.tensor(toks[:, :-1])
    labels = torch.tensor(labs[:, :-1])
    return {"input_ids": inputs, "labels": labels}


def flipflop_write_positions(input_ids: torch.Tensor) -> list[torch.Tensor]:
    """Oracle boundaries for a Dynamic-MoSC variant: the WRITE token positions (state-change points).

    Per-row variable length, so returns a list of 1-D index tensors (one per example). Not used by
    the do-no-harm vanilla-vs-fixed comparison; provided for a learned/oracle-boundary flip-flop run.
    """
    return [(row == W).nonzero(as_tuple=True)[0] for row in input_ids]
