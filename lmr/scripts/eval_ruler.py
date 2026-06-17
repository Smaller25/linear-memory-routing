# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluate vanilla / +RM / +trained-head on NVIDIA RULER single-answer tasks (teacher-forced).

Standardized synthetic long-context recall (real essay haystack) via vendored RULER, scored with
the memory-light teacher-forced scorer so vanilla and the MC heads are compared identically. Prepare
data first:  ``python scripts/ruler.py prepare --lengths 4096 8192 --tasks niah_single_1,niah_multikey_2``

    python -m lmr.scripts.eval_ruler --arch mamba2 --heads ckpt/ssc_heads_2048.pt --variant ssc \
        --topk 2 --tasks niah_single_1 niah_multikey_2 --lengths 4096 8192
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from lmr.adapters import descriptor_dim_for, get_adapter
from lmr.loaders import load_backbone
from lmr.readout import ResidualMemory, build_readout
from lmr.scripts.eval_recall import score_hidden
from lmr.segment_runner import run_segmented_lm
from lmr.tasks.ruler_loader import load_ruler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mamba2", "gdn"], default="mamba2")
    ap.add_argument("--model", "--repo", dest="repo", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--heads", default=None, help="trained head .pt; omit for vanilla/RM only")
    ap.add_argument("--variant", choices=["grm", "ssc", "mom", "aom"], default="ssc")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--num-slots", type=int, default=4)
    ap.add_argument("--low-rank-dim", type=int, default=0, help="match the trained head; 0 = full-rank")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--tasks", nargs="+", default=["niah_single_1", "niah_multikey_2"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    ap.add_argument("--max-examples", type=int, default=50)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    low_rank = None if not args.low_rank_dim else args.low_rank_dim
    dtype = getattr(torch, args.dtype)

    model, tok = load_backbone(args.arch, repo=args.repo, tokenizer=args.tokenizer,
                               device=args.device, dtype=dtype)
    adapter = get_adapter(args.arch)
    lm_head = adapter.lm_head(model)
    rm_heads = [ResidualMemory() for _ in adapter.blocks(model)]
    trained = None
    if args.heads:
        dd = descriptor_dim_for(model, args.arch)
        kw = {"low_rank_dim": low_rank}
        if args.variant == "ssc":
            kw["topk"] = args.topk
        if args.variant == "mom":
            kw["num_slots"] = args.num_slots
        trained = nn.ModuleList([build_readout(args.variant, model.config.hidden_size, dd, **kw)
                                 for _ in adapter.blocks(model)]).to(args.device, dtype=dtype)
        trained.load_state_dict(torch.load(args.heads, map_location=args.device))
        trained.eval()

    def van(x):
        return adapter.vanilla_hidden(model, x)

    def seg(x, hds):
        return run_segmented_lm(model, x, hds, args.chunk_size, backend="cuda",
                                return_hidden=True, arch=args.arch)[0]

    cols = ["vanilla", "+RM"] + (["+" + args.variant] if trained else [])
    print(f"{'task':>16} {'len':>6} | " + " | ".join(f"{c:>8}" for c in cols))
    print("-" * (26 + 11 * len(cols)))
    for task in args.tasks:
        for L in args.lengths:
            batch = load_ruler(task, L, tok, max_examples=args.max_examples)
            vals = [score_hidden(van, lm_head, batch, device=args.device, micro_batch=args.micro_batch),
                    score_hidden(lambda x: seg(x, rm_heads), lm_head, batch, device=args.device, micro_batch=args.micro_batch)]
            if trained:
                vals.append(score_hidden(lambda x: seg(x, trained), lm_head, batch, device=args.device, micro_batch=args.micro_batch))
            print(f"{task:>16} {L:>6} | " + " | ".join(f"{v:8.3f}" for v in vals))


if __name__ == "__main__":
    main()
