"""Deterministic, resumable sampler over pre-tokenised shards.

The token stream is cut into non-overlapping windows of SEQ_LEN+1 tokens. A
seeded permutation fixes the window order per epoch, and sample number ``n``
(counted over the whole run, all ranks) always maps to the same window, so
resuming only needs the step counter — no loader state to save.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch


class TokenWindows:
    def __init__(self, data_dir: str, split: str, seq_len: int, seed: int):
        with open(os.path.join(data_dir, "meta.json")) as f:
            self.meta = json.load(f)
        files = self.meta[f"{split}_files"]
        self.arrays = [np.memmap(os.path.join(data_dir, n), dtype=np.uint16, mode="r") for n in files]
        self.win = seq_len + 1
        counts = [len(a) // self.win for a in self.arrays]
        self.offsets = np.cumsum([0] + counts)
        self.n = int(self.offsets[-1])
        if self.n == 0:
            raise ValueError(f"{data_dir}: {split} split has fewer than {self.win} tokens")
        self.seed = seed
        self._perm_epoch, self._perm = -1, None

    def _window(self, n: int) -> np.ndarray:
        epoch, i = divmod(n, self.n)
        if epoch != self._perm_epoch:
            self._perm = np.random.default_rng(self.seed + 1000003 * epoch).permutation(self.n)
            self._perm_epoch = epoch
        w = int(self._perm[i])
        f = int(np.searchsorted(self.offsets, w, side="right") - 1)
        start = (w - self.offsets[f]) * self.win
        return np.asarray(self.arrays[f][start:start + self.win], dtype=np.int64)

    def batch(self, sample_ids) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.stack([self._window(int(n)) for n in sample_ids]))
        return x[:, :-1], x[:, 1:]


def sample_ids(step: int, micro: int, accum: int, micro_batch: int, rank: int, world: int) -> list[int]:
    """Global sample numbers for (step, micro-step, rank). Every rank gets distinct samples."""
    per_step = accum * micro_batch * world
    base = step * per_step + (micro * world + rank) * micro_batch
    return list(range(base, base + micro_batch))
