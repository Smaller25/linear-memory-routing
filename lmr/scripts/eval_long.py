# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Memory-light long-context eval: vanilla vs +RM vs +trained head on the passkey.

Uses :func:`lmr.scripts.eval_recall.score_hidden` (apply lm_head only at labelled positions) so it
reaches 8k-16k without the full-vocab-logit OOM. Loads trained GRM/SSC/MoM/AoM heads from --heads.
Works for both backbones via ``--arch {mamba2,gdn}``.

    python -m lmr.scripts.eval_long --arch gdn --heads ckpt/ssc_heads.pt --variant ssc \
        --low-rank-dim 64 --lengths 512 2048 4096 8192 --num-examples 32
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
from lmr.tasks import make_text_passkey


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mamba2", "gdn"], default="mamba2")
    ap.add_argument("--model", "--repo", dest="repo", default=None, help="backbone repo; defaults per --arch")
    ap.add_argument("--tokenizer", default=None, help="mamba2 only; gdn ships its own")
    ap.add_argument("--heads", default=None, help="trained head state_dict (.pt); omit for RM-only")
    ap.add_argument("--variant", choices=["grm", "ssc", "mom", "aom"], default="grm")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--num-slots", type=int, default=4)
    ap.add_argument("--low-rank-dim", type=int, default=64, help="must match the trained head; 0 = full-rank")
    ap.add_argument("--hierarchical-k", type=int, default=None)
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 4096, 8192])
    ap.add_argument("--num-examples", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    low_rank_dim = None if not args.low_rank_dim else args.low_rank_dim

    model, tok = load_backbone(args.arch, repo=args.repo, tokenizer=args.tokenizer,
                               device=args.device, dtype=torch.float32)
    backend = "cuda" if args.device.startswith("cuda") else "naive"
    adapter = get_adapter(args.arch)
    lm_head = adapter.lm_head(model)

    rm_heads = [ResidualMemory() for _ in adapter.blocks(model)]
    trained = None
    if args.heads:
        dd = descriptor_dim_for(model, args.arch)
        kw = {"low_rank_dim": low_rank_dim}
        if args.variant == "ssc":
            kw["topk"] = args.topk
        if args.variant == "mom":
            kw["num_slots"] = args.num_slots
        trained = nn.ModuleList([
            build_readout(args.variant, model.config.hidden_size, dd, **kw)
            for _ in adapter.blocks(model)
        ]).to(args.device)
        trained.load_state_dict(torch.load(args.heads, map_location=args.device))
        trained.eval()

    def run(x, hds):
        return run_segmented_lm(model, x, hds, args.chunk_size, backend=backend,
                                return_hidden=True, arch=args.arch,
                                hierarchical_k=args.hierarchical_k)[0]

    def van_hidden(x):
        return adapter.vanilla_hidden(model, x)

    def rm_hidden(x):
        return run(x, rm_heads)

    def tr_hidden(x):
        return run(x, trained)

    cols = ["vanilla", "+RM"] + (["+" + args.variant] if trained else [])
    head = " | ".join(f"{c:>8}" for c in cols)
    print(f"{'length':>8} | {head}")
    print("-" * (10 + 11 * len(cols)))
    for L in args.lengths:
        b = make_text_passkey(tok, num_examples=args.num_examples, seq_len=L, seed=10_000 + L)
        vals = [score_hidden(van_hidden, lm_head, b, device=args.device, micro_batch=args.micro_batch),
                score_hidden(rm_hidden, lm_head, b, device=args.device, micro_batch=args.micro_batch)]
        if trained:
            vals.append(score_hidden(tr_hidden, lm_head, b, device=args.device, micro_batch=args.micro_batch))
        print(f"{L:>8} | " + " | ".join(f"{v:8.3f}" for v in vals))


if __name__ == "__main__":
    main()
