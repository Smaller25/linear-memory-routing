# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Flip-flop (FFLM) state-tracking trainer — a do-no-harm check for the segment-cache method.

Trains a from-scratch GDN-2 on flip-flop LM and reports read-accuracy vs sequence length. Compare
``--model gdn2`` (vanilla backbone) against ``--model mosc --chunk-mode fixed`` (segment-cache
read-out): does segmenting the recurrent state (which the read-out does) DEGRADE the backbone's
native state-tracking? Accuracy is measured only at READ-answer positions.

    sbatch scripts/sh_slurm_run.sh python -m lmr.mosc.train_flipflop --model gdn2 \
        --train-len 128 --eval-lens 128 256 512 --steps 4000
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from lmr.mosc.backbone import GDN2LM
from lmr.mosc.mosc_model import DynamicMoSC
from lmr.tasks.flipflop import VOCAB, make_flipflop

IGNORE = -100


def build_model(args):
    if args.model == "gdn2":
        return GDN2LM(VOCAB, d_model=args.d_model, n_layers=args.n_layers,
                      head_dim=args.head_dim, num_heads=args.num_heads)
    return DynamicMoSC(VOCAB, d_model=args.d_model, n_layers=args.n_layers, head_dim=args.head_dim,
                       num_heads=args.num_heads, chunk_mode=args.chunk_mode, chunk=args.chunk,
                       num_pools=args.num_pools, topk=args.topk)


@torch.no_grad()
def read_acc(model, n_instr, device, p_write, p_read, n=256):
    b = make_flipflop(num_examples=n, n_instr=n_instr, p_write=p_write, p_read=p_read, seed=9000 + n_instr)
    ids, labels = b["input_ids"].to(device), b["labels"].to(device)
    correct = total = 0
    for i in range(0, n, 64):
        logits = model(ids[i:i + 64])
        m = labels[i:i + 64] != IGNORE
        if m.any():
            correct += (logits.argmax(-1)[m] == labels[i:i + 64][m]).sum().item()
            total += int(m.sum())
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gdn2", "mosc"], default="gdn2")
    ap.add_argument("--chunk-mode", choices=["fixed", "learned"], default="fixed")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--num-pools", type=int, default=1)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--train-len", type=int, default=128, help="n_instr at train")
    ap.add_argument("--eval-lens", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--p-write", type=float, default=0.1)
    ap.add_argument("--p-read", type=float, default=0.1)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    print(f"model={args.model} chunk={args.chunk_mode if args.model=='mosc' else '-'} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M | device={device}")

    for step in range(1, args.steps + 1):
        b = make_flipflop(num_examples=args.batch, n_instr=args.train_len,
                          p_write=args.p_write, p_read=args.p_read, seed=step)
        ids, labels = b["input_ids"].to(device), b["labels"].to(device)
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1), ignore_index=IGNORE)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == 1:
            print(f"[train] step {step:5d}  loss {loss.item():.4f}")

    print("=== read-accuracy (vs n_instr; state-tracking maintenance) ===")
    model.eval()
    for L in args.eval_lens:
        print(f"  n_instr={L:5d}  acc={read_acc(model, L, device, args.p_write, args.p_read):.3f}")


if __name__ == "__main__":
    main()
