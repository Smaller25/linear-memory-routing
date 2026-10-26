# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the arch-agnostic segment-runner helpers (no model / Triton needed)."""

import torch

from lmr.segment_runner import checkpoint_list, segment_lengths


def test_segment_lengths():
    assert segment_lengths(16, 8) == [8, 8]
    assert segment_lengths(20, 8) == [8, 8, 4]
    assert segment_lengths(5, 8) == [5]


def test_checkpoint_list_flat_is_identity():
    states = [torch.randn(2, 3, 4, 5) for _ in range(5)]
    assert checkpoint_list(states, None) is states
    assert checkpoint_list(states, 1) is states


def test_checkpoint_list_hierarchical_merges_completed_blocks():
    states = [torch.full((1, 2, 2, 2), float(i)) for i in range(5)]
    out = checkpoint_list(states, hierarchical_k=2)
    # two completed blocks of 2 -> coarse sums; one trailing fine state.
    assert len(out) == 3
    assert torch.allclose(out[0], states[0] + states[1])
    assert torch.allclose(out[1], states[2] + states[3])
    assert torch.allclose(out[2], states[4])
