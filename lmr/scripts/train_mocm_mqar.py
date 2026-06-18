# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""From-scratch MQAR validation of the MoCM parallel-memory axis (M=1 vs M=M).

Trains a tiny 2-layer model whose mixer is :class:`lmr.layers.MoCMMixer` from scratch on token-level
MQAR (our self-contained `make_mqar`, the standard random-token associative-recall task), and reports
recall accuracy across kv-pair counts. ``--num-memories 1`` = single-memory baseline (the ablation);
``--num-memories 4`` = parallel memories (MoM axis). Question: do parallel memories raise the kv-pair
count at which a from-scratch linear model still recalls? (Temporal caching is NOT used here — short
single-segment MQAR; this isolates the parallel axis.)

    python -m lmr.scripts.train_mocm_mqar --num-memories 4 --train-kv 16 32 64 --eval-kv 16 32 64 128
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmr.layers import MoCMMixer
from lmr.tasks.mqar import make_mqar

IGNORE = -100


class TwoLayerMoCM(nn.Module):
    def __init__(self, vocab, d_model, n_layers, num_memories, topk_w, head_dim, shared_mem):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(n_layers)])
        self.mixers = nn.ModuleList([
            MoCMMixer(d_model, num_memories=num_memories, topk_w=topk_w,
                      head_dim=head_dim, shared_mem=shared_mem)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids):
        h = self.embed(ids)
        aux_total = h.new_zeros(())
        for norm, mixer in zip(self.norms, self.mixers):
            out, aux = mixer(norm(h))
            h = h + out
            aux_total = aux_total + aux
        return self.lm_head(self.norm_f(h)), aux_total


@torch.no_grad()
def recall_acc(model, k, vocab, device, n=256):
    batch = make_mqar(num_examples=n, vocab_size=vocab, num_kv_pairs=k, seed=10_000 + k)
    ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
    correct = total = 0
    for i in range(0, n, 64):
        logits, _ = model(ids[i:i + 64])
        m = labels[i:i + 64] != IGNORE
        correct += (logits.argmax(-1)[m] == labels[i:i + 64][m]).sum().item()
        total += int(m.sum())
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-memories", type=int, default=4)
    ap.add_argument("--topk-w", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--no-shared", action="store_true")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--train-kv", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--eval-kv", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = TwoLayerMoCM(args.vocab, args.d_model, args.n_layers, args.num_memories,
                         args.topk_w, args.head_dim, not args.no_shared).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[mocm] M={args.num_memories} topk_w={args.topk_w} d={args.d_model} "
          f"layers={args.n_layers} shared={not args.no_shared} params={n_params:,}")

    model.train()
    g = torch.Generator().manual_seed(0)
    for step in range(args.steps):
        k = args.train_kv[int(torch.randint(0, len(args.train_kv), (1,), generator=g))]
        batch = make_mqar(num_examples=args.batch, vocab_size=args.vocab, num_kv_pairs=k, seed=step)
        ids, labels = batch["input_ids"].to(args.device), batch["labels"].to(args.device)
        logits, aux = model(ids)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE) + aux
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == args.steps - 1:
            print(f"  step {step:5d} | loss {loss.item():.4f} | aux {float(aux):.4f}")

    model.eval()
    print(f"\n{'kv-pairs':>9} | recall-acc")
    print("-" * 24)
    for k in args.eval_kv:
        print(f"{k:>9} | {recall_acc(model, k, args.vocab, args.device):.3f}")


if __name__ == "__main__":
    main()
