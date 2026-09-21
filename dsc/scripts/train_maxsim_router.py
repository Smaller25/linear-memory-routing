#!/usr/bin/env python3
"""Fit a router against sub-block descriptors. Offline, CPU, no forward.

Scoring a segment by its best-matching sub-block instead of by its mean is
the largest single gain measured today: held out, hit@2 0.250 -> 0.306 ->
0.316 -> 0.357 at m = 1 / 2 / 4 / 8, and hit@1 0.128 -> 0.204. m=1 is exactly
the deployed descriptor, so this generalizes it.

The cost is close to nothing. A segment already stores an [H,K,V] state
(262k values here) against a [H,Kd] descriptor of 2048, so eight descriptors
add about 6% to a segment; the extra compute is eight times a routing einsum
that was 0.004% of the forward.

Reads capture_key_identity.py's cache, which stores 8 block means, and
averaging adjacent blocks gives any coarser m from the same capture. Writes
router_L*.pt plus meta.json, because the granularity has to travel with the
checkpoint — a head fitted at m=8 and injected at m=1 would score a formula
it never saw.

Usage:
  python dsc/scripts/train_maxsim_router.py --cache /root/keyid_s44.pt \
      --blocks 8 --layers 0 1 --steps 12000 --out /root/routers_maxsim8
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "dsc")):
    if p not in sys.path:
        sys.path.insert(0, p)
from dsc.mc_baseline.mc_ssc_mlp_router import MLPRouterHead  # noqa: E402


def coarsen(blocks: torch.Tensor, m: int) -> torch.Tensor:
    n, b, h, kd = blocks.shape
    if b % m:
        raise ValueError(f"{b} stored blocks not divisible by m={m}")
    return blocks.view(n, m, b // m, h, kd).mean(dim=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    rows, meta = blob["rows"], blob["meta"]
    D, H, Kd = meta["D"], meta["H"], meta["Kd"]
    stored = meta["blocks"]
    if stored % args.blocks:
        raise SystemExit(f"cache stores {stored} blocks, not divisible by "
                         f"--blocks {args.blocks}")
    cells = sorted({r["cell"] for r in rows})
    print(f"[train] {len(rows)} rows over {len(cells)} cells, stored blocks "
          f"{stored} -> m={args.blocks}", flush=True)

    verdict = {"blocks": args.blocks, "scorer": "dot", "steps": args.steps,
               "cache": args.cache, "n_rows": len(rows), "cells": cells}
    for L in args.layers:
        items = []
        for r in rows:
            elig = r["nseg"] - 1
            if elig <= 0 or r["gold"] >= elig:
                continue
            g = coarsen(torch.tensor(np.asarray(r[f"g{L}"][:elig])).float(),
                        args.blocks)
            if args.blocks == 1:
                # blocks=1 is the deployed layout, [E,H,Kd] with no block
                # axis. The head checks the layout strictly on purpose — that
                # is what caught this — so squeeze here rather than loosening
                # the check.
                g = g[:, 0]
            items.append((torch.tensor(np.asarray(r[f"h{L}"])).float(), g,
                          int(r["gold"])))
        if not items:
            raise SystemExit(f"layer {L}: no usable rows")
        torch.manual_seed(args.seed + L)
        head = MLPRouterHead(D, H, Kd, scorer="dot", blocks=args.blocks)
        head = head.to(args.device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                weight_decay=0.01)
        gen = torch.Generator().manual_seed(args.seed + 1)
        for step in range(args.steps):
            h, g, gold = items[int(torch.randint(len(items), (1,),
                                                 generator=gen))]
            sc = head.scores(h[None, None].to(args.device),
                             g[None].to(args.device))[0, 0]
            loss = F.cross_entropy(sc[None],
                                   torch.tensor([gold], device=args.device))
            opt.zero_grad(); loss.backward(); opt.step()
        # train-set hit only, as a sanity number — never as evidence
        head.eval()
        hit = 0
        with torch.no_grad():
            for h, g, gold in items:
                sc = head.scores(h[None, None].to(args.device),
                                 g[None].to(args.device))[0, 0]
                hit += int(int((sc.argsort(descending=True) == gold)
                               .nonzero()[0, 0]) < 2)
        verdict[f"layer{L}"] = {"train_hit@2": hit / len(items),
                                "n": len(items)}
        torch.save(head.state_dict(), os.path.join(args.out, f"router_L{L}.pt"))
        print(f"[layer {L}] fitted on {len(items)} rows, train hit@2="
              f"{hit / len(items):.3f} (train fit is not evidence)", flush=True)

    json.dump(verdict, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(f"[train] wrote {args.out}/router_L*.pt and meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
