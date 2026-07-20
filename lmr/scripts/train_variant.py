# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Phase 1: train the GRM / SSC read-out heads on synthetic recall (backbone frozen).

Only the per-layer read-out parameters (``W_u`` / router) are trained; the Mamba2 backbone is
frozen, so this is hours on a single A100. SSC adds the load-balance aux loss returned by the
segment runner.

    python -m lmr.scripts.train_variant --variant grm --chunk-size 256 --steps 2000
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmr.converter import load_fla_mamba2
from lmr.tasks import make_mqar
from lmr.readout import build_readout
from lmr.segment_runner import run_segmented_lm

IGNORE = -100


def build_readouts(model, variant, topk):
    cfg = model.config
    descriptor_dim = cfg.num_heads * cfg.state_size
    heads = nn.ModuleList([
        build_readout(variant, cfg.hidden_size, descriptor_dim, **({"topk": topk} if variant == "ssc" else {}))
        for _ in model.backbone.layers
    ])
    return heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["grm", "ssc"], default="grm")
    ap.add_argument("--repo", default="state-spaces/mamba2-1.3b")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="grm_heads.pt")
    args = ap.parse_args()

    model = load_fla_mamba2(args.repo, device=args.device, dtype=torch.float32)
    model.requires_grad_(False)
    backend = "cuda" if args.device.startswith("cuda") else "naive"

    heads = build_readouts(model, args.variant, args.topk).to(args.device)
    opt = torch.optim.AdamW([p for p in heads.parameters() if p.requires_grad], lr=args.lr)

    for step in range(args.steps):
        batch = make_mqar(num_examples=args.chunk_size, seed=step)
        ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)

        logits, aux = run_segmented_lm(model, ids, heads, args.chunk_size, backend=backend)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE)
        loss = loss + aux

        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | aux {float(aux):.4f}")

    torch.save(heads.state_dict(), args.out)
    print(f"saved read-out heads -> {args.out}")


if __name__ == "__main__":
    main()
