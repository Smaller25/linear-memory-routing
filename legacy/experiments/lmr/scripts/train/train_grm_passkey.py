# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Phase 1+: train a Memory-Caching read-out head (backbone frozen) on the passkey.

Phase-0b showed training-free RM destroys recall on a frozen backbone (it sums every cached
checkpoint blindly). This trains ONLY the per-layer read-out head (the 1.3B backbone is frozen) on
the natural-language passkey so the head learns *which* cached segment to surface for the query.
Training uses multi-segment sequences (``train_len > chunk_size``) so the cache is non-empty and the
head actually receives gradient.

Works for both backbones via ``--arch {mamba2,gdn}`` and any trained head via
``--variant {grm,ssc,mom,aom}``; all trained heads default to low-rank routers (``--low-rank-dim``).

After training it evaluates vanilla vs +RM (training-free) vs +trained-head on held-out passkeys.

    python -m lmr.scripts.train.train_grm_passkey --arch gdn --variant ssc --train-len 1024 --steps 300 \
        --batch 8 --low-rank-dim 64 --eval-lengths 512 1024 2048 4096
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmr.adapters import descriptor_dim_for, get_adapter
from lmr.loaders import load_backbone
from lmr.readout import ResidualMemory, build_readout
from lmr.scripts.eval_recall import score_hidden
from lmr.segment_runner import run_segmented_lm
from lmr.tasks import make_text_multikey, make_text_passkey

IGNORE = -100
TASKS = {"passkey": make_text_passkey, "multikey": make_text_multikey}


def build_heads(model, arch, variant, topk, num_slots, low_rank_dim, aux_scale=1e-2):
    cfg = model.config
    dd = descriptor_dim_for(model, arch)
    kw = {"low_rank_dim": low_rank_dim}
    if variant == "ssc":
        kw["topk"] = topk; kw["aux_scale"] = aux_scale
    if variant == "mom":
        kw["num_slots"] = num_slots; kw["aux_scale"] = aux_scale
    return nn.ModuleList([
        build_readout(variant, cfg.hidden_size, dd, **kw)
        for _ in get_adapter(arch).blocks(model)
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mamba2", "gdn"], default="mamba2")
    ap.add_argument("--variant", choices=["grm", "ssc", "mom", "aom"], default="grm")
    ap.add_argument("--task", choices=["passkey", "multikey"], default="passkey",
                    help="recall task to train (and in-script eval) on")
    ap.add_argument("--model", "--repo", dest="repo", default=None,
                    help="backbone repo; defaults per --arch")
    ap.add_argument("--tokenizer", default=None, help="mamba2 only; gdn ships its own")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--train-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--num-slots", type=int, default=4)
    ap.add_argument("--aux-scale", type=float, default=1e-2, help="SSC/MoM load-balance aux weight; lower for selective tasks (multikey)")
    ap.add_argument("--low-rank-dim", type=int, default=64,
                    help="low-rank router dim; pass 0 for a full-rank router")
    ap.add_argument("--hierarchical-k", type=int, default=None)
    ap.add_argument("--eval-lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="backbone+head dtype; use bfloat16 for large models (e.g. 2.7b)")
    ap.add_argument("--out", default="heads.pt")
    args = ap.parse_args()
    low_rank_dim = None if not args.low_rank_dim else args.low_rank_dim
    dtype = getattr(torch, args.dtype)
    gen = TASKS[args.task]

    model, tok = load_backbone(args.arch, repo=args.repo, tokenizer=args.tokenizer,
                               device=args.device, dtype=dtype)
    model.requires_grad_(False)
    backend = "cuda" if args.device.startswith("cuda") else "naive"
    adapter = get_adapter(args.arch)

    heads = build_heads(model, args.arch, args.variant, args.topk, args.num_slots, low_rank_dim, aux_scale=args.aux_scale)
    heads = heads.to(args.device, dtype=dtype)
    opt = torch.optim.AdamW([p for p in heads.parameters() if p.requires_grad], lr=args.lr)
    n_params = sum(p.numel() for p in heads.parameters() if p.requires_grad)
    print(f"[train] arch={args.arch} variant={args.variant} low_rank_dim={low_rank_dim} "
          f"trainable head params={n_params:,} train_len={args.train_len} chunk={args.chunk_size} "
          f"segments={-(-args.train_len // args.chunk_size)}")

    def run(ids, hds, return_hidden=False):
        return run_segmented_lm(model, ids, hds, args.chunk_size, backend=backend,
                                return_hidden=return_hidden, arch=args.arch,
                                hierarchical_k=args.hierarchical_k)

    heads.train()
    for step in range(args.steps):
        batch = gen(tok, num_examples=args.batch, seq_len=args.train_len, seed=step)
        ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        logits, aux = run(ids, heads)
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
    rm_heads = [ResidualMemory() for _ in adapter.blocks(model)]
    lm_head = adapter.lm_head(model)

    def vanilla_h(x):
        return adapter.vanilla_hidden(model, x)

    def rm_h(x):
        return run(x, rm_heads, return_hidden=True)[0]

    def trained_h(x):
        return run(x, heads, return_hidden=True)[0]

    print(f"\n{'length':>8} | {'vanilla':>8} | {'+RM':>8} | {'+' + args.variant:>8} | {'Δ vs van':>9}")
    print("-" * 56)
    for L in args.eval_lengths:
        b = gen(tok, num_examples=64, seq_len=L, seed=10_000 + L)
        v = score_hidden(vanilla_h, lm_head, b, device=args.device, micro_batch=2)
        r = score_hidden(rm_h, lm_head, b, device=args.device, micro_batch=2)
        t = score_hidden(trained_h, lm_head, b, device=args.device, micro_batch=2)
        print(f"{L:>8} | {v:8.3f} | {r:8.3f} | {t:8.3f} | {t - v:+9.3f}")


if __name__ == "__main__":
    main()
