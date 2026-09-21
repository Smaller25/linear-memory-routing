"""Backbone-independent utilities for Memory Caching segment scans."""

from __future__ import annotations

from collections.abc import Callable

import torch


def scan_segments(
    length: int,
    chunk_size: int,
    scan_fn: Callable[[int, int, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]],
    *,
    checkpoint_mode: str = "independent",
    use_reentrant_checkpoint: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan segments and return online outputs plus one final state per segment.

    ``independent`` implements paper section 3.4's independent compressors;
    ``checkpoint`` caches checkpoints of one continuously updated memory.

    Per-segment activation checkpointing was removed after verifying it stacked
    redundantly with the per-call checkpoint in ``gdn2_ssc_forward`` and the
    per-Block checkpoint in ``lit_gpt/model.py`` — three layers of recompute
    pushed training to ~27x vanilla, contradicting paper Section 5.7's "minimal
    overhead" claim. With batch=8 seq=4096 on H200, peak VRAM (~85 GB) fits in
    143 GB without any activation checkpointing. The ``use_reentrant_checkpoint``
    argument is retained for API stability but currently unused.
    """
    if checkpoint_mode not in {"independent", "checkpoint"}:
        raise ValueError("checkpoint_mode must be 'independent' or 'checkpoint'")
    outputs, states = [], []
    previous = None
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        initial = previous if checkpoint_mode == "checkpoint" else None
        output, state = scan_fn(start, stop, initial)
        outputs.append(output)
        states.append(state)
        previous = state
    return torch.cat(outputs, dim=1), torch.stack(states, dim=1)
