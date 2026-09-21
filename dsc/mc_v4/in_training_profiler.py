"""In-training profiler: samples N consecutive iters and breaks down CUDA time
per component (forward, backward, optimizer.step, DDP all-reduce, dataloader).

Hooks into torch.profiler inside the training loop. Output: JSON written to
log_dir every sample_period_iters (default 200).

Designed to be drop-in: import and call register_hooks(trainer) once.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional

import torch
from torch.profiler import profile, ProfilerActivity, schedule


class InTrainingProfiler:
    """Sample N iters every sample_period_iters, dump per-component breakdown.

    Usage:
        prof = InTrainingProfiler(log_dir="/path/to/logs", sample_period_iters=200, sample_iters=5)
        for it, batch in enumerate(loader):
            prof.begin_iter(it)
            ... forward, backward, optimizer.step ...
            prof.end_iter(it)
    """

    def __init__(
        self,
        log_dir: str,
        sample_period_iters: int = 200,
        sample_iters: int = 5,
        warmup_iters: int = 100,
        rank: int = 0,
    ):
        self.log_dir = log_dir
        self.sample_period = sample_period_iters
        self.sample_iters = sample_iters
        self.warmup = warmup_iters
        self.rank = rank  # only dump artifacts on this rank (avoids 8× duplicates)
        if rank == 0:
            os.makedirs(log_dir, exist_ok=True)

        self._prof: Optional[profile] = None
        self._sample_counter = 0
        self._is_sampling = False
        self._iter_starts: dict = {}
        self._component_times: dict = {}
        # cuda.Event-based manual timing
        self._evt_starts: dict = {}

    def maybe_start_sampling(self, iter_num: int) -> bool:
        """Returns True if entering a sampling window."""
        if iter_num < self.warmup:
            return False
        if (iter_num - self.warmup) % self.sample_period == 0 and not self._is_sampling:
            self._is_sampling = True
            self._sample_counter = 0
            self._prof = profile(
                activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                record_shapes=False,
                with_stack=False,
            )
            self._prof.__enter__()
            return True
        return False

    def maybe_stop_sampling(self, iter_num: int) -> bool:
        """Returns True if just exited a sampling window."""
        if self._is_sampling and self._sample_counter >= self.sample_iters:
            self._prof.__exit__(None, None, None)
            self._dump_report(iter_num)
            self._prof = None
            self._is_sampling = False
            return True
        return False

    def tick(self, iter_num: int):
        """Call at top of each iter. Manages sampling window."""
        self.maybe_start_sampling(iter_num)
        if self._is_sampling:
            self._sample_counter += 1
            self.maybe_stop_sampling(iter_num)

    @contextmanager
    def time_component(self, name: str):
        """Manual cuda.Event timing for a region (forward/backward/etc)."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
            self._component_times.setdefault(name, []).append(ms)

    def _dump_report(self, iter_num: int):
        """Write two artifacts: torch.profiler table + manual timing summary.
        Only writes on rank 0 to avoid 8× duplicate artifacts in FSDP."""
        if self.rank != 0:
            self._component_times.clear()
            return
        ts = int(time.time())
        # 1) Profiler table
        path_prof = os.path.join(self.log_dir, f"profiler_iter{iter_num}_{ts}.txt")
        with open(path_prof, "w") as f:
            f.write(f"=== In-training profiler @ iter {iter_num} ===\n")
            f.write(f"=== {self.sample_iters} iter sample ===\n\n")
            f.write("TOP kernels (CUDA time total):\n")
            f.write(self._prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
            f.write("\n\n")
        # 2) Manual component timing (averages)
        path_manual = os.path.join(self.log_dir, f"components_iter{iter_num}_{ts}.json")
        summary = {}
        for k, v in self._component_times.items():
            summary[k] = {
                "n": len(v),
                "mean_ms": sum(v) / len(v),
                "min_ms": min(v),
                "max_ms": max(v),
            }
        with open(path_manual, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[profiler] wrote {path_prof} + {path_manual}", flush=True)
        self._component_times.clear()
