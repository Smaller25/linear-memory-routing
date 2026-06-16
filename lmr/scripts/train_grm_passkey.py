# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Phase 1: train a GRM/SSC read-out head on the in-distribution passkey (backbone frozen).

Phase-0b showed training-free RM destroys recall on a frozen mamba2 (it sums every cached
checkpoint blindly). This trains ONLY the per-layer read-out head (``W_u`` / router; the 1.3B
backbone is frozen) on the natural-language passkey so the head learns *which* cached segment to
surface for the query. Training uses multi-segment sequences (``train_len > chunk_size``) so the
cache is non-empty and the head actually receives gradient — unlike single-segment MQAR.

After training it evaluates vanilla vs +RM (training-free) vs +trained-head on held-out passkeys
across lengths, so the three sit in one table.

    python -m lmr.scripts.train_grm_passkey --variant grm --train-len 1024 --steps 300 \
        --batch 8 --eval-lengths 512 1024 2048 4096
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from lmr.converter import load_fla_mamba2
from lmr.readout import ResidualMemory, build_readout
from lmr.scripts.eval_recall import score
from lmr.segment_runner import run_segmented_lm
from lmr.tasks import make_text_passkey

TOKENIZER = "EleutherAI/gpt-neox-20b"
IGNORE = -100


def build_heads(model, variant, topk):
    cfg = model.config
    descriptor_dim = cfg.num_heads * cfg.state_size
    return nn.ModuleList([
        build_readout(variant, cfg.hidden_size, descriptor_dim,
                      **({"topk": topk} if variant == "ssc" else {}))
        for _ in model.backbone.layers
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["grm", "ssc"], default="grm")
    ap.add_argument("--repo", default="state-spaces/mamba2-1.3b")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--train-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--eval-lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="heads.pt")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_fla_mamba2(args.repo, device=args.device, dtype=torch.float32)
    model.requires_grad_(False)
    backend = "cuda" if args.device.startswith("cuda") else "naive"

    heads = build_heads(model, args.variant, args.topk).to(args.device)
    opt = torch.optim.AdamW([p for p in heads.parameters() if p.requires_grad], lr=args.lr)
    n_params = sum(p.numel() for p in heads.parameters() if p.requires_grad)
    print(f"[train] variant={args.variant} trainable head params={n_params:,} "
          f"train_len={args.train_len} chunk={args.chunk_size} "
          f"segments={-(-args.train_len // args.chunk_size)}")

    heads.train()
    for step in range(args.steps):
        batch = make_text_passkey(tok, num_examples=args.batch, seq_len=args.train_len, seed=step)
        ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        logits, aux = run_segmented_lm(model, ids, heads, args.chunk_size, backend=backend)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE) + aux
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0 or step == args.steps - 1:
            with torch.no_grad():
                mask = labels != IGNORE
                acc = (logits.argmax(-1)[mask] == labels[mask]).float().mean().item()
            print(f"  step {step:4d} | loss {loss.item():.4f} | aux {float(aux):.4f} | train-acc {acc:.3f}")

    torch.save(heads.state_dict(), args.out)
    print(f"[train] saved heads -> {args.out}")

    # ---- eval: vanilla vs +RM (training-free) vs +trained head, on held-out passkeys ----
    heads.eval()
    rm_heads = [ResidualMemory() for _ in model.backbone.layers]

    def vanilla_fn(x):
        return model(x).logits

    def rm_fn(x):
        return run_segmented_lm(model, x, rm_heads, args.chunk_size, backend=backend)[0]

    def trained_fn(x):
        return run_segmented_lm(model, x, heads, args.chunk_size, backend=backend)[0]

    print(f"\n{'length':>8} | {'vanilla':>8} | {'+RM':>8} | {'+' + args.variant:>8} | {'Δ vs van':>9}")
    print("-" * 56)
    for L in args.eval_lengths:
        b = make_text_passkey(tok, num_examples=64, seq_len=L, seed=10_000 + L)
        v = score(vanilla_fn, b, device=args.device)
        r = score(rm_fn, b, device=args.device)
        t = score(trained_fn, b, device=args.device)
        print(f"{L:>8} | {v:8.3f} | {r:8.3f} | {t:8.3f} | {t - v:+9.3f}")


if __name__ == "__main__":
    main()
