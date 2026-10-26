# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Segment-by-segment Memory-Caching runner for a frozen linear-recurrent LM.

The model is run in segments of ``chunk_size`` tokens. Within a segment every recurrent layer runs
from a *zero* state (the segment is independent); at each segment boundary the layer's final state
is frozen and appended to that layer's checkpoint cache. The read-out (:mod:`lmr.readout`) then
augments each later segment's online output with the contributions of all earlier frozen
checkpoints.

Everything architecture-specific (the per-segment mixer math, state pooling, block structure) lives
behind an :class:`lmr.adapters.Adapter`; ``arch`` selects ``"mamba2"`` or ``"gdn"``. The generic
``run_mixer_with_cache`` is re-exported from :mod:`lmr.adapters.base`.
"""

from __future__ import annotations

import torch

from lmr.adapters import get_adapter, run_mixer_with_cache  # noqa: F401  (re-exported)


def segment_lengths(total: int, chunk_size: int) -> list[int]:
    n_full, rem = divmod(total, chunk_size)
    lens = [chunk_size] * n_full
    if rem:
        lens.append(rem)
    return lens


def checkpoint_list(fine_states: list[torch.Tensor], hierarchical_k: int | None) -> list[torch.Tensor]:
    """Build the checkpoint list a read-out sees from the per-segment final states.

    - ``hierarchical_k=None`` (flat): return ``fine_states`` unchanged.
    - ``hierarchical_k=K``: summarise every completed block of ``K`` consecutive segments into one
      coarse checkpoint (their *sum* -- which, by recurrence linearity, equals the RM-merge of those
      ``K`` states), and keep the states of the current partial block at fine resolution. This bounds
      the checkpoint count to ``~len/K + K`` and adds a long-range scale.
    """
    if not hierarchical_k or hierarchical_k <= 1:
        return fine_states
    n_full = len(fine_states) // hierarchical_k
    coarse = [
        torch.stack(fine_states[i * hierarchical_k:(i + 1) * hierarchical_k], dim=0).sum(dim=0)
        for i in range(n_full)
    ]
    tail = fine_states[n_full * hierarchical_k:]
    return coarse + tail


def bounded_checkpoints(fine_states: list[torch.Tensor], cap: int | None,
                        evict: str = "uniform") -> list[torch.Tensor]:
    """Cap the checkpoints the read-out sees to a constant ``cap`` (bounded memory / O(N·cap) read).

    - ``evict='uniform'``: keep ``cap`` evenly-spaced snapshots (coverage across depth — the needle
      can sit anywhere, so spread retention beats recency).
    - ``evict='recent'``: keep the last ``cap`` (most recent).
    - ``evict='first'``: keep the earliest ``cap``.
    """
    n = len(fine_states)
    if not cap or n <= cap:
        return fine_states
    if evict == "recent":
        return fine_states[-cap:]
    if evict == "first":
        return fine_states[:cap]
    idx = [round(i * (n - 1) / (cap - 1)) for i in range(cap)]   # uniform incl. endpoints
    return [fine_states[i] for i in idx]


def run_segmented_lm(model, input_ids, readouts, chunk_size, backend="naive",
                     return_hidden=False, arch="mamba2", hierarchical_k=None,
                     cache_cap=None, evict="uniform"):
    """Run a frozen linear-recurrent causal LM segment-by-segment with Memory Caching.

    ``readouts``: one read-out head per recurrent layer (e.g. a ``ModuleList``); RM heads are
    shared/param-free, trained heads (GRM/SSC/MoM/AoM) carry per-layer params.
    ``arch``: ``"mamba2"`` or ``"gdn"`` (selects the :class:`lmr.adapters.Adapter`).
    ``hierarchical_k``: if set, present coarse super-segment checkpoints to the read-out (see
    :func:`checkpoint_list`).
    Returns ``(logits:[b, L, vocab], aux_total)``, or with ``return_hidden=True`` the post-final-norm
    hidden states ``[b, L, hidden]`` instead of logits -- applying ``lm_head`` only at the positions
    you need avoids materialising full-vocab logits over every segment, which OOMs at long context.
    """
    adapter = get_adapter(arch)
    blocks = adapter.blocks(model)
    lm_head = adapter.lm_head(model)
    fine: list[list[torch.Tensor]] = [[] for _ in blocks]

    seg_out = []
    aux_total = input_ids.new_zeros((), dtype=torch.float32)
    offset = 0
    for seg_len in segment_lengths(input_ids.shape[1], chunk_size):
        seg_ids = input_ids[:, offset:offset + seg_len]
        offset += seg_len

        hidden = adapter.embed(model, seg_ids)
        new_finals = []
        for li, block in enumerate(blocks):
            cached = checkpoint_list(fine[li], hierarchical_k)
            if cache_cap:
                cached = bounded_checkpoints(cached, cache_cap, evict)
            hidden, final_state, aux = adapter.run_block(block, hidden, cached, readouts[li], backend)
            new_finals.append(final_state.detach())
            if aux is not None:
                aux_total = aux_total + aux

        hidden = adapter.final_norm(model, hidden)
        seg_out.append(hidden if return_hidden else lm_head(hidden))

        for li, fs in enumerate(new_finals):
            fine[li].append(fs)

    return torch.cat(seg_out, dim=1), aux_total
