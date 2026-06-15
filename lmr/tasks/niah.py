# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Self-contained Needle-in-a-Haystack / passkey-retrieval generator.

Token-level synthetic long-context recall. A short ``passkey`` (a run of digit tokens) is
hidden at a random depth inside a long run of filler tokens, then queried at the end. The model
must reproduce the passkey, scored as next-token prediction at the answer positions.

Use ``lengths={2048,4096,8192,16384,32768}`` for the length sweep. Returns ``input_ids``
``[N, L]`` and ``labels`` ``[N, L]`` (``-100`` except the answer positions).
"""

from __future__ import annotations

import torch

IGNORE = -100


def make_passkey(
    num_examples: int = 64,
    seq_len: int = 4096,
    vocab_size: int = 8192,
    passkey_len: int = 5,
    depth: float | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Generate passkey-retrieval examples of total length ~``seq_len``.

    ``depth`` in [0, 1] fixes where the needle sits (``None`` = random per example). Special
    marker tokens delimit the needle and the query; the answer (passkey) is appended after the
    query marker and is the only labelled span.
    """
    g = torch.Generator().manual_seed(seed)
    # Reserve a few special tokens at the top of the vocab.
    needle_marker = vocab_size - 1
    query_marker = vocab_size - 2
    filler_hi = vocab_size - 2  # filler/passkey drawn below the markers

    seqs, labels = [], []
    for _ in range(num_examples):
        passkey = torch.randint(1, 10, (passkey_len,), generator=g)  # digit-like tokens
        needle = torch.cat([torch.tensor([needle_marker]), passkey, torch.tensor([needle_marker])])
        query = torch.cat([torch.tensor([query_marker]), passkey])

        filler_len = seq_len - len(needle) - len(query)
        if filler_len < 1:
            raise ValueError(f"seq_len={seq_len} too short for passkey_len={passkey_len}")
        filler = torch.randint(10, filler_hi, (filler_len,), generator=g)

        d = torch.rand(1, generator=g).item() if depth is None else depth
        cut = int(d * filler_len)
        ids = torch.cat([filler[:cut], needle, filler[cut:], query])

        lab = torch.full_like(ids, IGNORE)
        # answer positions: the query marker + passkey[:-1] predict passkey[0:].
        ans_start = len(ids) - passkey_len - 1
        lab[ans_start:ans_start + passkey_len] = passkey
        seqs.append(ids)
        labels.append(lab)

    return {"input_ids": torch.stack(seqs), "labels": torch.stack(labels)}
