# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Phase 0 (in-distribution): vanilla mamba2-1.3b vs. +MC-RM on a NATURAL-LANGUAGE passkey.

The synthetic ``make_passkey`` uses random token ids that are OOD for a text-pretrained LM (its
vanilla recall is at the noise floor), so it cannot fairly test training-free MC-RM. This script
uses :func:`lmr.tasks.make_text_passkey` (real English NIAH prompt, gpt-neox tokenizer) so the
pretrained model is in-distribution, then sweeps context length.

    python -m lmr.scripts.eval.eval_text_passkey --repo state-spaces/mamba2-1.3b \
        --chunk-size 256 --lengths 1024 2048 4096 8192
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from lmr.converter import load_fla_mamba2
from lmr.readout import ResidualMemory
from lmr.scripts.eval_recall import score
from lmr.segment_runner import run_segmented_lm

TOKENIZER = "EleutherAI/gpt-neox-20b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="state-spaces/mamba2-1.3b")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--num-examples", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from lmr.tasks import make_text_passkey

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_fla_mamba2(args.repo, device=args.device, dtype=torch.float32)
    backend = "cuda" if args.device.startswith("cuda") else "naive"
    readouts = [ResidualMemory() for _ in model.backbone.layers]

    def vanilla_fn(ids):
        return model(ids).logits

    def mc_rm_fn(ids):
        return run_segmented_lm(model, ids, readouts, args.chunk_size, backend=backend)[0]

    print(f"{'length':>8} | {'vanilla':>8} | {'+MC-RM':>8} | {'delta':>7}")
    print("-" * 42)
    for L in args.lengths:
        batch = make_text_passkey(tok, num_examples=args.num_examples, seq_len=L, seed=0)
        v = score(vanilla_fn, batch, device=args.device)
        m = score(mc_rm_fn, batch, device=args.device)
        print(f"{L:>8} | {v:8.3f} | {m:8.3f} | {m - v:+7.3f}")


if __name__ == "__main__":
    main()
