# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the mamba2 -> FLA converter.

The CPU test pins the key-rename logic. The end-to-end logit-match needs CUDA + the checkpoint
download and is skipped off-GPU.
"""

import torch

from lmr.converter import convert_state_dict


def test_rename_embedding_and_tie_lm_head():
    src = {
        "backbone.embedding.weight": torch.zeros(8, 4),
        "backbone.layers.0.mixer.in_proj.weight": torch.zeros(2, 4),
        "backbone.norm_f.weight": torch.zeros(4),
    }
    out = convert_state_dict(src)
    assert "backbone.embeddings.weight" in out
    assert "backbone.embedding.weight" not in out
    # untied head -> tied to embeddings
    assert torch.equal(out["lm_head.weight"], out["backbone.embeddings.weight"])
    # mixer/other keys pass through unchanged
    assert "backbone.layers.0.mixer.in_proj.weight" in out
    assert "backbone.norm_f.weight" in out


def test_existing_lm_head_preserved():
    src = {
        "backbone.embedding.weight": torch.zeros(8, 4),
        "lm_head.weight": torch.ones(8, 4),
    }
    out = convert_state_dict(src)
    assert torch.equal(out["lm_head.weight"], torch.ones(8, 4))
