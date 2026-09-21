#!/usr/bin/env python3
"""Learn the document tower. Offline, CPU, no forward.

The score is <u_t, gamma_i>. u_t = f(h_t) is learned; gamma_i = mean_j
L2norm(k_j) is a hand-designed constant. Retrieval normally learns both
towers, and here only the query side ever was — the largest untested
asymmetry in the design.

Three families, all applied ONCE per segment at write time, so none of them
changes what a segment costs to store or what routing costs at read time:

  fixed-mean      today. m=1, no descriptor parameters at all.
  fixed-maxsim    today's best. m blocks, max over them, still no parameters.
  proj            P(gamma), a learned map on the stored descriptor. The
                  symmetric two-tower version of what we already do to h.
  proj-maxsim     the same map, applied per block, max over blocks.
  attn-pool       a learned aggregation over the m blocks instead of max:
                  weights come from each block's own content, so capacity
                  goes where the segment is distinctive rather than to
                  whichever block happens to score highest.
  mix             (1-a)*mean + a*max over blocks, a = sigmoid(theta). The
                  score is linear in the descriptor, so mean-over-blocks IS
                  the m=1 score: a=0 recovers today's deployed descriptor and
                  a=1 recovers maxsim. One scalar, both endpoints
                  representable. That is the point -- max wins when the
                  evidence is one short span and loses when it is spread over
                  the segment, and the router should not have to bet on which
                  regime the query is in.
  mixhead         the same with one a per head (H scalars). Heads need not
                  agree on how local their evidence is.
  mixq            a = sigmoid(w.h_t), one weight vector on the query side.
                  Paired over four initialisations, max beats the mean by
                  0.104 on few_needles (4/4 runs) and loses by 0.039 on
                  many_needles (1/4), so the two regimes want opposite
                  aggregations and a single global a can only split the
                  difference -- which is what the learned one does, landing
                  at 0.56 and matching neither. Both regimes are in this
                  training set, so the constraint is the form of a, not the
                  data. Note also which side this puts the parameters on:
                  every document-side family lost (proj 4/4, attn-pool
                  massively), while the query tower has worked from the
                  start.
  mix@A           a held at a constant A instead of learned. The learned
                  version lands at a = 0.56 in both folds and generalises
                  worse than a = 1, so containing the endpoints does not
                  make it safe -- the endpoints are reachable but CE does
                  not go there. Fixing a separates two different failures:
                  no interior a beats a = 1, or the fit cannot find the one
                  that does.

What is NOT here is the real version: a pooling learned over the per-token
keys. The cache stores block means, so 32 tokens is the finest unit
available. Worth noting that the training cost and the deployment cost part
company there — learning a pooling needs the tokens, applying it does not,
since only gamma is kept.

Two folds by seed, each reported on the seed it did not train on.

--init-seeds repeats every fit from a different initialisation and reports
the spread. This is not optional decoration. a = 0.00 in the fixed-a sweep
and fixed-mean m=1 are the same model -- verified numerically to 1.2e-7 --
and they scored 0.327 and 0.311, because fp error at 1e-7 per step
accumulates over 8000 AdamW steps until the two runs are independent draws
(weights differ by 2.5e-2 after only 500 steps). That accidental duplicate
was the only thing measuring this probe's resolution, and it says a gap
under about 0.02 on 196 items means nothing. Every conclusion drawn from a
single fit per variant has to be read against that.

Usage:
  python dsc/scripts/probe_learned_descriptor.py --cache /root/keyid_cache.pt \
      --layer 0 --blocks 8 --out /root/learned_desc_L0.json
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "dsc")):
    if p not in sys.path:
        sys.path.insert(0, p)


def coarsen(blocks: torch.Tensor, m: int) -> torch.Tensor:
    n, b, h, kd = blocks.shape
    if b % m:
        raise ValueError(f"{b} blocks not divisible by m={m}")
    return blocks.view(n, m, b // m, h, kd).mean(dim=2)


class Scorer(nn.Module):
    """Query tower is always learned; `desc` selects the document tower."""

    def __init__(self, D, H, Kd, desc="fixed", blocks=1, agg="max",
                 fix_a=None):
        super().__init__()
        self.H, self.Kd, self.desc, self.blocks, self.agg = H, Kd, desc, blocks, agg
        self.fix_a = fix_a
        self.q = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, H * Kd))
        self.logit_scale = nn.Parameter(torch.zeros(1))
        if desc == "proj":
            # Per-head linear map on the descriptor: the document-side twin of
            # the query projection, and free at read time because it is
            # applied when the segment is written.
            self.p = nn.Parameter(torch.eye(Kd).repeat(H, 1, 1))
        if agg == "attn":
            self.score_blk = nn.Linear(H * Kd, 1)
        if agg == "mix" and fix_a is None:
            self.mix = nn.Parameter(torch.zeros(1))
        if agg == "mixhead":
            self.mix = nn.Parameter(torch.zeros(H))
        if agg == "mixq":
            self.mix_q = nn.Linear(D, 1)
            nn.init.zeros_(self.mix_q.weight)
            nn.init.zeros_(self.mix_q.bias)

    def forward(self, h, g):                  # h:[D]  g:[Nseg,m,H,Kd]
        u = self.q(h).view(self.H, self.Kd)
        if self.desc == "proj":
            g = torch.einsum("nmhk,hkj->nmhj", g, self.p)
        if self.agg == "mixhead":
            sh = torch.einsum("hk,nmhk->nmh", u, g)
            a = torch.sigmoid(self.mix)
            r = (1 - a) * sh.mean(dim=1) + a * sh.max(dim=1).values
            return r.sum(-1) * self.logit_scale.exp()
        s = torch.einsum("hk,nmhk->nm", u, g) * self.logit_scale.exp()
        if s.shape[1] == 1:
            return s[:, 0]
        if self.agg in ("mix", "mixq"):
            if self.agg == "mixq":
                a = torch.sigmoid(self.mix_q(h))
            else:
                a = (self.fix_a if self.fix_a is not None
                     else torch.sigmoid(self.mix))
            return (1 - a) * s.mean(dim=-1) + a * s.max(dim=-1).values
        if self.agg == "max":
            return s.max(dim=-1).values
        w = torch.softmax(self.score_blk(g.flatten(-2)).squeeze(-1), dim=-1)
        return (s * w).sum(-1)


def hit_at(scores, golds, ks=(1, 2, 4)):
    got = {k: 0 for k in ks}
    n = 0
    for sc, gd in zip(scores, golds):
        r = int((torch.as_tensor(sc).argsort(descending=True) == gd)
                .nonzero()[0, 0])
        for k in ks:
            got[k] += int(r < k)
        n += 1
    return {f"hit@{k}": (got[k] / n if n else None) for k in ks} | {"n": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--variants", default=None,
                    help="comma-separated subset of mean1,maxsim,proj,"
                         "projmaxsim,attn,projattn,mix,mixhead. One variant "
                         "per process lets the sweep run in parallel, which "
                         "matters once every variant is fitted several times")
    ap.add_argument("--init-seeds", type=int, default=1,
                    help="repeat each fit from this many initialisations and "
                         "report mean and spread. 1 gives a single fit, which "
                         "cannot be compared against another single fit "
                         "closer than about 0.02")
    ap.add_argument("--only-fixed-a", action="store_true",
                    help="skip the learned families and sweep a constant a "
                         "over the mixture instead")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    rows, meta = blob["rows"], blob["meta"]
    L, H, Kd, D = args.layer, meta["H"], meta["Kd"], meta["D"]

    items = []
    for r in rows:
        elig = r["nseg"] - 1
        if elig <= 0 or r["gold"] >= elig:
            continue
        items.append({
            "cell": r["cell"], "gold": int(r["gold"]),
            "many": "_n16" in r["cell"] or "_n32" in r["cell"],
            "h": torch.tensor(np.asarray(r[f"h{L}"])).float(),
            "g8": torch.tensor(np.asarray(r[f"g{L}"][:elig])).float()})
    seeds = sorted({it["cell"].split("_")[0] for it in items})
    print(f"[probe] {len(items)} items, layer {L}, seeds {seeds}", flush=True)

    report = {"cache": args.cache, "layer": L, "variants": {}}

    def run(name, desc, m, agg, fix_a=None):
        per_init = []
        for init in range(args.init_seeds):
            flat, mixes = fit_once(desc, m, agg, fix_a, init)
            per_init.append(summarise(flat, mixes))
        row = per_init[0]
        if args.init_seeds > 1:
            row = dict(row)
            for sub in ("all", "few_needles", "many_needles"):
                vals = [r[sub]["hit@2"] for r in per_init]
                row[f"mean_{sub}"] = sum(vals) / len(vals)
                row[f"spread_{sub}"] = max(vals) - min(vals)
                row[f"runs_{sub}"] = vals
        report["variants"][name] = row
        if args.init_seeds > 1:
            print(f"  {name:<22} hit@2 mean={row['mean_all']:.3f} "
                  f"+-{row['spread_all'] / 2:.3f}  "
                  f"few={row['mean_few_needles']:.3f}  "
                  f"many={row['mean_many_needles']:.3f}  "
                  f"(runs {'/'.join(f'{v:.3f}' for v in row['runs_all'])}"
                  f" params={row['desc_params']})", flush=True)
        else:
            print(f"  {name:<22} hit@2 all={row['all']['hit@2']:.3f}  "
                  f"few={row['few_needles']['hit@2']:.3f}  "
                  f"many={row['many_needles']['hit@2']:.3f}  "
                  f"(hit@1={row['all']['hit@1']:.3f} "
                  f"desc_params={row['desc_params']}"
                  + (f" a={'/'.join(f'{x:.2f}' for x in row['mix_a'])}"
                     if "mix_a" in row else "") + ")", flush=True)

    def fit_once(desc, m, agg, fix_a, init):
        flat, mixes = [], []
        for held in seeds:
            tr = [it for it in items if not it["cell"].startswith(held)]
            te = [it for it in items if it["cell"].startswith(held)]
            if not tr or not te:
                continue
            torch.manual_seed(init)
            model = Scorer(D, H, Kd, desc=desc, blocks=m, agg=agg,
                           fix_a=fix_a)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=0.01)
            gen = torch.Generator().manual_seed(1 + init)
            for _ in range(args.steps):
                it = tr[int(torch.randint(len(tr), (1,), generator=gen))]
                g = coarsen(it["g8"], m)
                loss = F.cross_entropy(model(it["h"], g)[None],
                                       torch.tensor([it["gold"]]))
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            if hasattr(model, "mix"):
                mixes.append(torch.sigmoid(model.mix).detach().mean().item())
            elif hasattr(model, "mix_q"):
                with torch.no_grad():
                    mixes.append(float(torch.sigmoid(torch.stack(
                        [model.mix_q(it["h"]) for it in te])).mean()))
            with torch.no_grad():
                for it in te:
                    flat.append((model(it["h"], coarsen(it["g8"], m)),
                                 it["gold"], it["many"]))
        # mix_q lives on the query side, which is the side that has always
        # worked, so it does not belong in a document-tower parameter count.
        desc_params = sum(p.numel() for n, p in model.named_parameters()
                          if n.startswith(("p", "score_blk", "mix"))
                          and not n.startswith("mix_q"))
        return flat, (mixes, desc_params)

    def summarise(flat, mixes_and_params):
        mixes, desc_params = mixes_and_params
        sc = [x[0] for x in flat]; gd = [x[1] for x in flat]
        few = [i for i, x in enumerate(flat) if not x[2]]
        many = [i for i, x in enumerate(flat) if x[2]]
        row = {"all": hit_at(sc, gd),
               "few_needles": hit_at([sc[i] for i in few], [gd[i] for i in few]),
               "many_needles": hit_at([sc[i] for i in many], [gd[i] for i in many]),
               "desc_params": desc_params}
        if mixes:
            row["mix_a"] = mixes
        return row

    if args.only_fixed_a:
        # a = 0.00 and a = 1.00 are the two deployed options, so they double
        # as the identity check on the sweep: they must land on fixed-mean
        # and fixed-maxsim.
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            run(f"mix@a={a:.2f} m={args.blocks}", "fixed", args.blocks,
                "mix", fix_a=a)
    else:
        B = args.blocks
        table = {
            "mean1":      ("fixed-mean m=1",       "fixed", 1, "max"),
            "maxsim":     (f"fixed-maxsim m={B}",  "fixed", B, "max"),
            "proj":       ("proj m=1",             "proj",  1, "max"),
            "projmaxsim": (f"proj-maxsim m={B}",   "proj",  B, "max"),
            "attn":       (f"attn-pool m={B}",     "fixed", B, "attn"),
            "projattn":   (f"proj+attn m={B}",     "proj",  B, "attn"),
            "mix":        (f"mix m={B}",           "fixed", B, "mix"),
            "mixhead":    (f"mixhead m={B}",       "fixed", B, "mixhead"),
            "mixq":       (f"mixq m={B}",          "fixed", B, "mixq"),
        }
        want = (list(table) if not args.variants
                else [v.strip() for v in args.variants.split(",")])
        bad = [v for v in want if v not in table]
        if bad:
            raise SystemExit(f"unknown variants {bad}, pick from {list(table)}")
        for v in want:
            run(*table[v])

    json.dump(report, open(args.out, "w"), indent=1)
    print(f"[probe] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
