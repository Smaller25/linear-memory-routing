# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Phase 0: vanilla mamba2-1.3b vs. +MC-RM on the recall suite (training-free).

GPU/local: needs the converted checkpoint and CUDA. Prints a side-by-side accuracy table over
lengths; a measurable RM gain at >=8k with zero training is the Phase-0 success signal.

    python -m lmr.scripts.eval.run_baseline --repo state-spaces/mamba2-1.3b --chunk-size 256
"""

from __future__ import annotations

import argparse

import torch

from lmr.converter import load_fla_mamba2
from lmr.readout import ResidualMemory
from lmr.scripts.eval_recall import evaluate
from lmr.segment_runner import run_segmented_lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="state-spaces/mamba2-1.3b")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = load_fla_mamba2(args.repo, device=args.device, dtype=torch.float32)
    backend = "cuda" if args.device.startswith("cuda") else "naive"
    readouts = [ResidualMemory() for _ in model.backbone.layers]

    def vanilla_fn(ids):
        return model(ids).logits

    def mc_rm_fn(ids):
        logits, _ = run_segmented_lm(model, ids, readouts, args.chunk_size, backend=backend)
        return logits

    vanilla = evaluate(vanilla_fn, device=args.device, lengths=tuple(args.lengths))
    mc_rm = evaluate(mc_rm_fn, device=args.device, lengths=tuple(args.lengths))

    print(f"{'task':>14} | {'vanilla':>8} | {'+MC-RM':>8} | {'delta':>7}")
    print("-" * 48)
    for task in vanilla:
        v, m = vanilla[task], mc_rm[task]
        print(f"{task:>14} | {v:8.3f} | {m:8.3f} | {m - v:+7.3f}")


if __name__ == "__main__":
    main()
