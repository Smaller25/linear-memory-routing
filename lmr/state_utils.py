# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers for inspecting and pooling Mamba2 SSM states used by Memory Caching.

A Mamba2 per-head SSM state has shape ``[b, h, p, n]`` (heads, head_dim, d_state). These
utilities pool it into a routing/gating descriptor (GRM/SSC) and provide a rank diagnostic
for analysing whether cached checkpoints are diverse enough to be worth combining.
"""

from __future__ import annotations

import torch


def meanpool_state(state: torch.Tensor) -> torch.Tensor:
    """Mean-pool an SSM checkpoint ``[b, h, p, n]`` to a per-batch descriptor ``[b, h*n]``.

    Pools over ``head_dim`` (``p``) -- the read-out contracts that axis with ``C`` -- leaving a
    ``(heads, d_state)`` summary flattened to a vector. This is the ``MeanPool(S^(i))`` term in
    the GRM gate ``gamma = sigmoid(<u_t, MeanPool(S^(i))>)`` and the SSC router score.
    """
    if state.dim() != 4:
        raise ValueError(f"expected state of shape [b, h, p, n], got {tuple(state.shape)}")
    b, h, _p, n = state.shape
    return state.mean(dim=2).reshape(b, h * n)


def stack_checkpoints(checkpoints: list[torch.Tensor]) -> torch.Tensor:
    """Stack a list of ``i`` checkpoints ``[b, h, p, n]`` into ``[b, i, h, p, n]``."""
    return torch.stack(checkpoints, dim=1)


def effective_rank(state: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Effective rank (entropy of normalised singular values) of each head's state matrix.

    ``state``: ``[b, h, p, n]`` -> returns ``[b, h]``. A low effective rank means the cached
    memory is near-degenerate (little extra information over the online state).
    """
    s = torch.linalg.svdvals(state.float())  # [b, h, min(p, n)]
    s = s / (s.sum(dim=-1, keepdim=True) + eps)
    entropy = -(s * (s + eps).log()).sum(dim=-1)
    return torch.exp(entropy)


def extract_ssm_states(past_key_values) -> list[torch.Tensor]:
    """Pull per-layer ``recurrent_state`` (the SSM state) out of an FLA ``Cache``.

    Mirrors how :meth:`fla.layers.mamba2.Mamba2Mixer.forward` stores state via
    ``update_layer_cache(..., recurrent_state=ssm_state, ...)``. Returns one tensor per layer
    that carries an SSM state; layers without one are skipped.
    """
    states: list[torch.Tensor] = []
    for layer_state in past_key_values:
        rec = layer_state.get("recurrent_state") if isinstance(layer_state, dict) else None
        if rec is not None:
            states.append(rec)
    return states
