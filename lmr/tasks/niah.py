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


# Classic NIAH filler (Mohtashami & Jaggi / Chen et al.): benign repeated sentences so the
# needle is the only salient fact.
_FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    "There and back again. "
)
_PREAMBLE = (
    "There is an important pass key hidden inside a lot of irrelevant text. "
    "Find it and remember it. I will quiz you about the pass key.\n\n"
)
_QUESTION = "\n\nWhat is the pass key? The pass key is"


def make_text_passkey(
    tokenizer,
    num_examples: int = 64,
    seq_len: int = 4096,
    passkey_digits: int = 5,
    depth: float | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Natural-language passkey retrieval — **in-distribution for a text-pretrained LM**.

    Unlike :func:`make_passkey` (random token ids, for from-scratch training), this builds a real
    English NIAH prompt with ``tokenizer`` so a pretrained mamba2/GDN sees familiar text. The needle
    ``"The pass key is N. Remember it."`` is placed at ``depth`` in repeated filler; the prompt ends
    with ``"... The pass key is"`` and the answer span (the digits of ``N``) is the only labelled
    span (teacher-forced next-token scoring, same convention as :func:`make_passkey`).

    Examples are **right-padded** to a common length per batch (mamba is right-pad-invariant — it
    compresses L→R, so trailing pad cannot corrupt earlier real positions; left-pad would). The
    answer-label positions sit on real tokens before the pad, so a single forward scores everyone.
    """
    g = torch.Generator().manual_seed(seed)
    pre_ids = tokenizer(_PREAMBLE, add_special_tokens=False).input_ids
    filler_ids = tokenizer(_FILLER, add_special_tokens=False).input_ids
    q_ids = tokenizer(_QUESTION, add_special_tokens=False).input_ids
    pad_id = tokenizer.eos_token_id or 0

    rows, labels, lens = [], [], []
    for _ in range(num_examples):
        lo, hi = 10 ** (passkey_digits - 1), 10 ** passkey_digits - 1
        key = int(torch.randint(lo, hi + 1, (1,), generator=g).item())
        needle_ids = tokenizer(f" The pass key is {key}. Remember it.", add_special_tokens=False).input_ids
        ans_ids = tokenizer(f" {key}", add_special_tokens=False).input_ids

        # Repeat filler to fill the budget around the needle + fixed prefix/suffix/answer.
        budget = seq_len - len(pre_ids) - len(needle_ids) - len(q_ids) - len(ans_ids)
        if budget < len(filler_ids):
            raise ValueError(f"seq_len={seq_len} too short")
        reps = budget // len(filler_ids)
        hay = filler_ids * reps
        d = torch.rand(1, generator=g).item() if depth is None else depth
        cut = int(d * len(hay))
        ids = pre_ids + hay[:cut] + needle_ids + hay[cut:] + q_ids + ans_ids

        lab = [IGNORE] * len(ids)
        # label position j predicts token j+1; the answer occupies the final len(ans_ids) tokens.
        a0 = len(ids) - len(ans_ids)
        for j in range(len(ans_ids)):
            lab[a0 + j - 1] = ids[a0 + j]
        rows.append(ids)
        labels.append(lab)
        lens.append(len(ids))

    # Right-pad to the max length in the batch (RIGHT-pad for mamba: trailing pad is invariant,
    # left-pad would corrupt the state for every real position).
    maxlen = max(lens)
    ids_t = torch.full((num_examples, maxlen), pad_id, dtype=torch.long)
    lab_t = torch.full((num_examples, maxlen), IGNORE, dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(rows, labels)):
        ids_t[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        lab_t[i, :len(lab)] = torch.tensor(lab, dtype=torch.long)
    return {"input_ids": ids_t, "labels": lab_t}
