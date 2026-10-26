# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the MoM comparison-arm adapter.

Construction is CPU-safe; the forward pass uses FLA's gated-delta chunk op (Triton/CUDA) and is
skipped off-GPU.
"""

import pytest
import torch

from lmr.mom_adapter import build_mom, forward_with_aux


def test_build_mom_defaults():
    model = build_mom(hidden_size=64, num_hidden_layers=2, vocab_size=64)
    assert model.config.num_memories == 4
    assert model.config.topk == 2
    assert model.config.shared_mem is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MoM forward needs Triton/CUDA")
def test_mom_forward_returns_aux():
    model = build_mom(hidden_size=64, num_hidden_layers=2, vocab_size=64).cuda()
    ids = torch.randint(0, 64, (1, 32), device="cuda")
    logits, aux = forward_with_aux(model, ids)
    assert logits.shape == (1, 32, 64)
    assert aux is not None
