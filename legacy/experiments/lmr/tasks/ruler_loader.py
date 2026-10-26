# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Load NVIDIA RULER prepared jsonl into the teacher-forced {input_ids, labels} format.

RULER (vendored under ``src/ruler``) is the standard *synthetic* long-context benchmark; its
generators build {``input``, ``outputs``, ``answer_prefix``} per (task, length) — real
Paul-Graham-essay haystacks, controlled lengths, standardized task structure (niah_single,
niah_multikey, ...). Generate with ``python scripts/ruler.py prepare --lengths L --tasks ...``.

This adapter scores the *single-answer* tasks (niah_single_*, niah_multikey_*) with the same
memory-light teacher-forced next-token scorer the rest of lmr uses, so vanilla and +SSC are compared
identically through the segment runner. Construction: ``input + answer_prefix + " " + outputs[0]``,
with the answer tokens as the only labelled span (RIGHT-padded for mamba). Multi-answer tasks
(niah_multivalue, vt) need a recall/generation metric and are out of scope for this teacher-forced
loader.
"""

from __future__ import annotations

import json
import os

import torch

IGNORE = -100
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_ruler(task: str, length: int, tokenizer, max_examples: int = 50,
               data_root: str | None = None) -> dict[str, torch.Tensor]:
    """Load ``data/ruler/<length>/<task>/validation.jsonl`` as teacher-forced {input_ids, labels}.

    Scores ``outputs[0]`` (single-answer tasks). Returns right-padded ``[N, T]`` tensors.
    """
    data_root = data_root or os.path.join(REPO_ROOT, "data", "ruler")
    path = os.path.join(data_root, str(length), task, "validation.jsonl")
    pad_id = tokenizer.eos_token_id or 0

    rows, labels, lens = [], [], []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            answer = ex["outputs"][0]
            prefix_ids = tokenizer(ex["input"] + ex.get("answer_prefix", ""),
                                   add_special_tokens=False).input_ids
            ans_ids = tokenizer(" " + answer, add_special_tokens=False).input_ids
            ids = prefix_ids + ans_ids
            lab = [IGNORE] * len(ids)
            a0 = len(ids) - len(ans_ids)
            for j in range(len(ans_ids)):           # label j predicts token j+1 (next-token)
                lab[a0 + j - 1] = ids[a0 + j]
            rows.append(ids); labels.append(lab); lens.append(len(ids))
            if len(rows) >= max_examples:
                break

    maxlen = max(lens)
    ids_t = torch.full((len(rows), maxlen), pad_id, dtype=torch.long)
    lab_t = torch.full((len(rows), maxlen), IGNORE, dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(rows, labels)):
        ids_t[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        lab_t[i, :len(lab)] = torch.tensor(lab, dtype=torch.long)
    return {"input_ids": ids_t, "labels": lab_t}
