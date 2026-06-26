# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Self-contained Multi-Query Associative Recall (MQAR) generator — Zoology-faithful.

Token-level synthetic recall (Arora, Eyuboglu et al., "Zoology"). A context of distinct key→value
pairs is followed by **single query keys** scattered through random filler at **power-law** gaps; the
value is the **label only** (next-token target at the position after the query key) — it is NOT placed
in the input. Non-query slots are random tokens. This is the standard MQAR protocol (the format
small linear models are trained on from scratch); keys/values occupy disjoint vocab halves.

Returns ``input_ids`` ``[N, L]`` and ``labels`` ``[N, L]`` (``-100`` except answer positions).
Re-implements ``zoology.data.multiquery_ar`` standalone (no zoology/wandb deps).
"""

from __future__ import annotations

import numpy as np
import torch

IGNORE = -100


def make_mqar(
    num_examples: int = 512,
    vocab_size: int = 8192,
    num_kv_pairs: int = 64,
    input_seq_len: int | None = None,
    power_a: float = 0.01,
    random_non_queries: bool = True,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Standard MQAR. ``input_seq_len`` defaults to ``4*num_kv_pairs`` (min context+queries)."""
    if input_seq_len is None:
        input_seq_len = max(256, 4 * num_kv_pairs)
    assert input_seq_len % 2 == 0 and vocab_size > input_seq_len
    assert 4 * num_kv_pairs <= input_seq_len, "input_seq_len too short for num_kv_pairs"
    rng = np.random.default_rng(seed)

    context_size = num_kv_pairs * 2
    half = vocab_size // 2
    key_choices = np.arange(1, half)
    value_choices = np.arange(half, vocab_size)

    # unique keys & values per example
    keys = np.stack([rng.choice(key_choices, size=num_kv_pairs, replace=False) for _ in range(num_examples)])
    values = np.stack([rng.choice(value_choices, size=num_kv_pairs, replace=False) for _ in range(num_examples)])
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # power-law gap placement of queries in the post-context region
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()
    gaps = np.stack([rng.choice(space, size=num_kv_pairs, replace=False, p=p) for _ in range(num_examples)])

    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, gaps * 2, values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), IGNORE, dtype=np.int64)
    np.put_along_axis(labels, gaps * 2 + context_size + 1, values=values, axis=1)

    inputs = torch.tensor(examples[:, :-1])
    labels = torch.tensor(labels[:, 1:])
    if random_non_queries:
        rand = torch.randint(vocab_size, size=inputs.shape)
        inputs[inputs == 0] = rand[inputs == 0]
    return {"input_ids": inputs, "labels": labels}


def make_mqar_gapped(
    num_examples: int = 512,
    vocab_size: int = 8192,
    num_kv_pairs: int = 64,
    ctx_filler: int = 128,
    query_filler: int | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Irregular MQAR: the ``(key,value)`` units in the context are separated by **random-length
    filler**, so facts sit at CONTENT-dependent, non-periodic positions. A fixed stride cannot align
    to them — only a content-aware boundary can — so this is the task that actually tests whether the
    boundary head *adapts* (unlike standard MQAR, where facts are at a fixed period and a fixed
    chunk=2 already equals the oracle).

    Context length = ``2*num_kv_pairs + ctx_filler`` (filler split randomly over the k+1 gaps). Returns
    ``{input_ids, labels, value_pos}`` where ``value_pos`` ``[N, k]`` are the (irregular) token
    positions of the values — the oracle boundaries. ``labels`` are the value only at each query
    position (predicted as the token after the query key).
    """
    rng = np.random.default_rng(seed)
    half = vocab_size // 2
    k = num_kv_pairs
    ctx_len = 2 * k + ctx_filler
    Q = query_filler if query_filler is not None else max(4 * k, 2 * k + 2)
    L = ctx_len + Q
    inp = np.zeros((num_examples, L), dtype=np.int64)
    labels = np.full((num_examples, L), IGNORE, dtype=np.int64)
    vpos = np.zeros((num_examples, k), dtype=np.int64)
    key_choices = np.arange(1, half)
    val_choices = np.arange(half, vocab_size)

    for n in range(num_examples):
        keys = rng.choice(key_choices, size=k, replace=False)
        vals = rng.choice(val_choices, size=k, replace=False)
        gaps = rng.multinomial(ctx_filler, np.ones(k + 1) / (k + 1))   # filler split over k+1 gaps
        pos = int(gaps[0])
        for i in range(k):
            inp[n, pos] = keys[i]; inp[n, pos + 1] = vals[i]; vpos[n, i] = pos + 1
            pos += 2 + int(gaps[i + 1])
        # query region: scatter the k keys at distinct positions; supervise value at the key position
        qpos = rng.choice(np.arange(ctx_len, ctx_len + Q - 1), size=k, replace=False)
        for qp, i in zip(qpos, rng.permutation(k)):
            inp[n, qp] = keys[i]; labels[n, qp] = vals[i]

    rand = rng.integers(1, vocab_size, size=inp.shape)
    inp[inp == 0] = rand[inp == 0]                                      # fill filler slots
    return {"input_ids": torch.tensor(inp), "labels": torch.tensor(labels),
            "value_pos": torch.tensor(vpos)}
