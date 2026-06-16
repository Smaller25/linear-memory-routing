# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Memory-light long-context eval: vanilla vs +RM vs +trained head on the passkey.

Uses :func:`lmr.scripts.eval_recall.score_hidden` (apply lm_head only at labelled positions) so it
reaches 8k-16k without the full-vocab-logit OOM. Loads trained GRM/SSC heads from --heads.

    python -m lmr.scripts.eval_long --heads ckpt/grm_heads_1024.pt --variant grm \
        --lengths 512 2048 4096 8192 --num-examples 32
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from lmr.converter import load_fla_mamba2
from lmr.readout import ResidualMemory, build_readout
from lmr.scripts.eval_recall import score_hidden
from lmr.segment_runner import run_segmented_lm
from lmr.tasks import make_text_passkey

TOKENIZER = "EleutherAI/gpt-neox-20b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="state-spaces/mamba2-1.3b")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--heads", default=None, help="trained head state_dict (.pt); omit for RM-only")
    ap.add_argument("--variant", choices=["grm", "ssc"], default="grm")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 4096, 8192])
    ap.add_argument("--num-examples", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_fla_mamba2(args.repo, device=args.device, dtype=torch.float32)
    backend = "cuda" if args.device.startswith("cuda") else "naive"
    lm_head = model.lm_head

    rm_heads = [ResidualMemory() for _ in model.backbone.layers]
    trained = None
    if args.heads:
        cfg = model.config
        dd = cfg.num_heads * cfg.state_size
        trained = nn.ModuleList([
            build_readout(args.variant, cfg.hidden_size, dd,
                          **({"topk": args.topk} if args.variant == "ssc" else {}))
            for _ in model.backbone.layers
        ]).to(args.device)
        trained.load_state_dict(torch.load(args.heads, map_location=args.device))
        trained.eval()

    def van_hidden(x):
        return model.backbone(x).last_hidden_state

    def rm_hidden(x):
        return run_segmented_lm(model, x, rm_heads, args.chunk_size, backend=backend, return_hidden=True)[0]

    def tr_hidden(x):
        return run_segmented_lm(model, x, trained, args.chunk_size, backend=backend, return_hidden=True)[0]

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
