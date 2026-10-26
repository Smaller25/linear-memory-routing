# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Selective Copying — the standard content-vs-position task (Mamba, Gu & Dao 2023).

`M` data tokens sit at **random positions** among `scatter_len-M` blank/noise tokens; after a marker
the model must reproduce the data tokens **in order**. Because the data is randomly spaced, a fixed
stride cannot align to it — only content-aware selection can — which makes this the standard external
analog of our irregular-MQAR adaptivity probe. Maps onto Dynamic-MoSC directly: the data positions are
the oracle boundaries (the things to cache); the output region reads them back.

Token ids: 0 = BLANK (noise), [1, vocab-1) = data symbols, vocab-1 = MARKER. Returns
``{input_ids, labels, value_pos}`` (same contract as ``lmr.tasks.mqar``); ``value_pos`` ``[N, M]`` =
the data positions = oracle boundaries; labels = the data token only at the output positions.
"""

from __future__ import annotations

import numpy as np
import torch

IGNORE = -100


def make_selective_copying(
    num_examples: int = 512,
    n_data: int = 16,
    vocab_size: int = 64,
    scatter_mult: int = 8,
    scatter_len: int | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """M = ``n_data`` data tokens at random positions in a length-``scatter_len`` (default M*scatter_mult)
    noise region, then MARKER, then the M data tokens in order (teacher-forced). Supervise the output."""
    rng = np.random.default_rng(seed)
    M = n_data
    L = scatter_len if scatter_len is not None else M * scatter_mult
    assert M <= L and vocab_size >= 4
    BLANK, MARKER = 0, vocab_size - 1
    total = L + 1 + M                                  # scatter + marker + output region
    inp = np.zeros((num_examples, total), dtype=np.int64)   # BLANK = 0
    labels = np.full((num_examples, total), IGNORE, dtype=np.int64)
    vpos = np.zeros((num_examples, M), dtype=np.int64)

    for n in range(num_examples):
        data = rng.integers(1, vocab_size - 1, size=M)        # data symbols (repeats allowed)
        pos = np.sort(rng.choice(L, size=M, replace=False))   # random distinct scatter positions
        inp[n, pos] = data
        vpos[n] = pos
        inp[n, L] = MARKER
        inp[n, L + 1:L + 1 + M] = data                        # output region (teacher-forced, in order)
        labels[n, L] = data[0]                                # MARKER -> predict data[0]
        for i in range(M - 1):                                # output_i -> predict data[i+1]
            labels[n, L + 1 + i] = data[i + 1]
    return {"input_ids": torch.tensor(inp), "labels": torch.tensor(labels),
            "value_pos": torch.tensor(vpos)}
