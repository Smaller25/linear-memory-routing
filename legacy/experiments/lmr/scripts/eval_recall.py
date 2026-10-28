# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Arch-neutral synthetic recall evaluation (MQAR + NIAH/passkey length sweep).

A model is supplied as a ``logits_fn(input_ids) -> logits`` callable, so the same scorer works
for a vanilla model forward, the MC segment runner, or the MoM arm. Accuracy is next-token
argmax at the labelled positions (``labels != -100``).
"""

from __future__ import annotations

import torch

from lmr.tasks import make_mqar, make_passkey

IGNORE = -100


@torch.no_grad()
def score(logits_fn, batch, device="cpu", micro_batch=8) -> float:
    ids, labels = batch["input_ids"], batch["labels"]
    correct = total = 0
    for i in range(0, ids.shape[0], micro_batch):
        x = ids[i:i + micro_batch].to(device)
        y = labels[i:i + micro_batch].to(device)
        logits = logits_fn(x)
        pred = logits.argmax(dim=-1)
        mask = y != IGNORE
        correct += (pred[mask] == y[mask]).sum().item()
        total += int(mask.sum().item())
    return correct / max(total, 1)


@torch.no_grad()
def score_hidden(hidden_fn, lm_head, batch, device="cpu", micro_batch=4) -> float:
    """Memory-light scorer: ``hidden_fn(x) -> [b, L, hidden]``, then apply ``lm_head`` ONLY at the
    labelled positions. Avoids materialising full-vocab logits over the whole sequence (which OOMs
    at long context). Equivalent to :func:`score` but feasible at 8k-16k.
    """
    ids, labels = batch["input_ids"], batch["labels"]
    correct = total = 0
    for i in range(0, ids.shape[0], micro_batch):
        x = ids[i:i + micro_batch].to(device)
        y = labels[i:i + micro_batch].to(device)
        hidden = hidden_fn(x)                       # [b, L, H]
        mask = y != IGNORE
        if mask.any():
            sel = lm_head(hidden[mask])             # [n_labelled, vocab]
            correct += (sel.argmax(dim=-1) == y[mask]).sum().item()
            total += int(mask.sum().item())
    return correct / max(total, 1)


def evaluate(logits_fn, device="cpu", lengths=(2048, 4096, 8192, 16384, 32768), seed=0) -> dict:
    """Return a dict of task -> accuracy: MQAR plus passkey at each length."""
    results = {}
    results["mqar"] = score(logits_fn, make_mqar(seed=seed), device)
    for L in lengths:
        results[f"passkey@{L}"] = score(logits_fn, make_passkey(seq_len=L, seed=seed), device)
    return results
