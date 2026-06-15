# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Self-contained Multi-Query Associative Recall (MQAR) generator.

Token-level, tokenizer-free synthetic recall task (Zoology-style). Each example lays out a
context of distinct key->value pairs, then a series of query keys; the target is each queried
key's value, scored as next-token prediction at the query-key position.

Keys and values occupy disjoint halves of the vocab so they can't be confused. Returns a dict
of ``input_ids`` ``[N, L]`` and ``labels`` ``[N, L]`` (``-100`` everywhere except the query-key
positions, where the label is the value to predict).
"""

from __future__ import annotations

import torch

IGNORE = -100


def make_mqar(
    num_examples: int = 512,
    vocab_size: int = 8192,
    num_kv_pairs: int = 64,
    num_queries: int | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    if num_queries is None:
        num_queries = num_kv_pairs
    g = torch.Generator().manual_seed(seed)
    half = vocab_size // 2
    key_lo, val_lo = 1, half  # reserve 0 as a separator/pad

    seqs, labels = [], []
    for _ in range(num_examples):
        keys = torch.randperm(half - 1, generator=g)[:num_kv_pairs] + key_lo
        vals = torch.randint(0, half, (num_kv_pairs,), generator=g) + val_lo

        ctx = torch.stack([keys, vals], dim=1).reshape(-1)  # k1 v1 k2 v2 ...

        q_idx = torch.randint(0, num_kv_pairs, (num_queries,), generator=g)
        q_keys, q_vals = keys[q_idx], vals[q_idx]
        # query layout: [query_key, value] pairs; predict value at the key position.
        q_block = torch.stack([q_keys, q_vals], dim=1).reshape(-1)

        ids = torch.cat([ctx, q_block])
        lab = torch.full_like(ids, IGNORE)
        # query-key positions are at offset len(ctx), every other token.
        key_positions = len(ctx) + torch.arange(0, 2 * num_queries, 2)
        lab[key_positions] = q_vals
        seqs.append(ids)
        labels.append(lab)

    return {"input_ids": torch.stack(seqs), "labels": torch.stack(labels)}
