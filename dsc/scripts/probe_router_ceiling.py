#!/usr/bin/env python3
"""Why does the routing hit rate stop at 0.20? Three candidates, told apart
offline. CPU only, no backbone forward.

score ~= baseline + hit@2 x (oracle - baseline) with no conversion loss at
k=2, so the whole problem is hit@2, and hit@2 has not moved across a router
architecture change, five loss variants, a training-set change and four
source layers. Something else is binding. It is one of:

  form   the scorer is <u_t, gamma_i> with u L2-normalized per head, a
         strong constraint. Sweep families of increasing freedom at fixed
         data. If they all plateau together, the form is not what binds.
  data   300 training samples for a 3M-parameter head. Fit on 25/50/100% of
         them and look at the val curve. Still rising means more data is the
         lever, and 2250 samples exist across the 45 generated cells.
  descriptor  gamma_i is the mean of 256 L2-normalized keys. It encodes
         needle-ness at AUC 1.00 (09-01 diagnosis), but picking GOLD needs
         the query's particular key to survive that mean. If neither form nor
         data moves val hit@2, the mean pooling is destroying the identity
         and no router reading gamma can recover it.

Deliberately NOT used as evidence: how well a family fits the TRAIN set. An
unconstrained net over [h; gamma_i] can always separate the gold segment for
a memorised h, so a perfect train fit says nothing about information content.
Only the held-out curve is read.

--holdout splits the val rows in two: one half picks the early-stopping
checkpoint, the other is reported and never selected on. Without it every
family early-stops on the same rows it is scored on, which is fair for
COMPARING families but inflates the absolute numbers -- the deployed form
reads 0.232 that way against an honest 0.141.

Usage:
  python dsc/scripts/probe_router_ceiling.py \
      --cache /root/routers_a4A_mixed/cache.pt --layers 0 1 \
      --out /root/ceiling_probe.json
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "dsc")):
    if p not in sys.path:
        sys.path.insert(0, p)


class Cos(nn.Module):
    """The deployed form: per-head L2-normalized u against L2-normalized gamma."""

    def __init__(self, D, H, Kd):
        super().__init__()
        self.H, self.Kd, self.scale = H, Kd, Kd ** -0.5
        self.net = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, H * Kd))

    def forward(self, h, g):                      # h:[D]  g:[N,H,Kd] -> [N]
        u = F.normalize(self.net(h).view(self.H, self.Kd), dim=-1)
        return torch.einsum("hk,nhk->n", u, F.normalize(g, dim=-1)) * self.scale


class HeadWeighted(Cos):
    """Cos plus a learned weight per head — the 09-01 probe's extra freedom."""

    def __init__(self, D, H, Kd):
        super().__init__(D, H, Kd)
        self.w = nn.Parameter(torch.ones(H))

    def forward(self, h, g):
        u = F.normalize(self.net(h).view(self.H, self.Kd), dim=-1)
        per_head = torch.einsum("hk,nhk->nh", u, F.normalize(g, dim=-1))
        return (per_head * self.w).sum(-1) * self.scale


class Dot(nn.Module):
    """Drop the normalization: magnitudes are free to carry information."""

    def __init__(self, D, H, Kd):
        super().__init__()
        self.H, self.Kd = H, Kd
        self.net = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, H * Kd))
        self.logit_scale = nn.Parameter(torch.zeros(1))

    def forward(self, h, g):
        u = self.net(h).view(self.H, self.Kd)
        return torch.einsum("hk,nhk->n", u, g) * self.logit_scale.exp()


class Concat(nn.Module):
    """No structural constraint: a net over [h ; flattened gamma_i]."""

    def __init__(self, D, H, Kd, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D + H * Kd, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, h, g):
        n = g.shape[0]
        x = torch.cat([h.expand(n, -1), g.reshape(n, -1)], dim=-1)
        return self.net(x).squeeze(-1)


class ConcatSmall(Concat):
    """One hidden layer at 128. The cheapest concat that could ship.

    Deployability, not curiosity, decides the width. The scorer runs at every
    position against every eligible segment, so its cost is T x N x MACs.
    At T=8192, ~16 eligible segments and a 370M backbone (~6.1 TFLOP per
    forward): hidden 512 with two hidden layers is ~480 GMAC, about 16% of
    the forward, while hidden 128 with one is ~103 GMAC, about 1.7%.

    Both numbers assume the routing decision is computed ONCE and reused
    across layers. Per-layer scoring multiplies them by 16, which puts even
    the small variant out of reach — layer-shared routing is what makes any
    of this affordable.
    """

    def __init__(self, D, H, Kd, hidden=128):
        nn.Module.__init__(self)
        self.net = nn.Sequential(nn.Linear(D + H * Kd, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))


class LowRank(nn.Module):
    """score_i = <A h, B gamma_i> at rank r, no normalization.

    Same cost as the deployed form (two thin projections plus a dot), so if
    this keeps the gain there is nothing to trade away.
    """

    def __init__(self, D, H, Kd, rank=128):
        super().__init__()
        self.A = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, rank))
        self.B = nn.Linear(H * Kd, rank, bias=False)
        self.logit_scale = nn.Parameter(torch.zeros(1))

    def forward(self, h, g):
        q = self.A(h)
        k = self.B(g.reshape(g.shape[0], -1))
        return (k @ q) * self.logit_scale.exp()


FAMILIES = {"cos": Cos, "headw": HeadWeighted, "dot": Dot, "concat": Concat,
            "concat128": ConcatSmall, "lowrank": LowRank}


def rows_for(split, li):
    out = []
    for d in split:
        elig = d["nseg"] - 1
        if elig <= 0 or d["gold"] >= elig:
            continue
        out.append((torch.tensor(d["h"][li]).float(),
                    torch.tensor(np.asarray(d["g"][li][:elig])).float(),
                    int(d["gold"]), len(d["ndl"]) > 8))
    return out


def hit_at(model, rows, ks=(1, 2, 4)):
    got = {k: 0 for k in ks}
    many = {k: 0 for k in ks}
    n = nm = 0
    with torch.no_grad():
        for h, g, gold, is_many in rows:
            r = int((model(h, g).argsort(descending=True) == gold).nonzero()[0, 0])
            for k in ks:
                got[k] += int(r < k)
                if is_many:
                    many[k] += int(r < k)
            n += 1
            nm += int(is_many)
    return ({f"hit@{k}": got[k] / n for k in ks} | {"n": n},
            {f"hit@{k}": many[k] / nm for k in ks} | {"n": nm} if nm else None)


def split_half(rows, seed=0):
    """Deterministic halves of the held-out rows: one to early-stop on, one to
    report. Shuffled first so the two are not one needle-count each."""
    idx = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed))
    a = [rows[int(i)] for i in idx[: len(rows) // 2]]
    b = [rows[int(i)] for i in idx[len(rows) // 2:]]
    return a, b


def fit(family, D, H, Kd, tr_rows, va_rows, steps, lr, seed, wd=0.01):
    torch.manual_seed(seed)
    model = FAMILIES[family](D, H, Kd)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    gen = torch.Generator().manual_seed(seed + 1)
    best = None
    for step in range(steps):
        i = int(torch.randint(len(tr_rows), (1,), generator=gen))
        h, g, gold, _ = tr_rows[i]
        loss = F.cross_entropy(model(h, g)[None], torch.tensor([gold]))
        opt.zero_grad(); loss.backward(); opt.step()
        # Early stopping on val, so a bigger family is not penalised for
        # overfitting 300 samples when the question is about its ceiling.
        if (step + 1) % max(1, steps // 8) == 0:
            v, _ = hit_at(model, va_rows, ks=(2,))
            if best is None or v["hit@2"] > best[0]:
                best = (v["hit@2"], step + 1,
                        {k: t.detach().clone() for k, t in model.state_dict().items()})
    if best:
        model.load_state_dict(best[2])
    return model, (best[1] if best else steps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0])
    ap.add_argument("--curve-families", nargs="+", default=["cos", "concat"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--holdout", action="store_true",
                    help="early-stop on half the val rows and report the "
                         "other half, so the reported number is not selected "
                         "on (absolute values, not just the ordering)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    meta = blob["meta"]
    D, H, Kd = meta["D"], meta["H"], meta["Kd"]
    print(f"[probe] {args.cache}: train {len(blob['train'])} val "
          f"{len(blob['val'])} D={D} H={H} Kd={Kd} threads={args.threads}",
          flush=True)

    report = {"cache": args.cache, "meta": {"D": D, "H": H, "Kd": Kd},
              "families": {}, "curve": {}}

    report["holdout"] = args.holdout
    for li, L in enumerate(args.layers):
        tr = rows_for(blob["train"], L)
        va_all = rows_for(blob["val"], L)
        if args.holdout:
            va, te = split_half(va_all)
        else:
            va, te = va_all, va_all
        # Native linear connector on the same rows, as the floor.
        conn = (blob.get("connector") or {}).get(L)
        if conn is not None:
            class Native(nn.Module):
                def forward(self, h, g):
                    u = (h @ conn.T.float()).view(H, Kd)
                    return torch.einsum("hk,nhk->n", u, g)
            nat, nat_many = hit_at(Native(), te)
            report["families"].setdefault(f"L{L}", {})["native"] = {
                "val": nat, "val_many_needles": nat_many}
            print(f"[L{L}] native  val hit@2={nat['hit@2']:.3f}", flush=True)

        for fam in args.families:
            t0 = time.time()
            model, at_step = fit(fam, D, H, Kd, tr, va, args.steps, args.lr,
                                 args.seed)
            trh, _ = hit_at(model, tr)
            vah, vah_many = hit_at(model, te)
            nparam = sum(p.numel() for p in model.parameters())
            report["families"].setdefault(f"L{L}", {})[fam] = {
                "params": nparam, "best_step": at_step,
                "train": trh, "val": vah, "val_many_needles": vah_many}
            print(f"[L{L}] {fam:<7} params={nparam / 1e6:.2f}M "
                  f"train hit@2={trh['hit@2']:.3f}  val hit@2={vah['hit@2']:.3f} "
                  f"(hit@1={vah['hit@1']:.3f} hit@4={vah['hit@4']:.3f}) "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    L0 = args.layers[0]
    tr0 = rows_for(blob["train"], L0)
    va0_all = rows_for(blob["val"], L0)
    va0, te0 = split_half(va0_all) if args.holdout else (va0_all, va0_all)
    for fam in args.curve_families:
        for frac in args.fractions:
            k = max(8, int(len(tr0) * frac))
            sub = tr0[:k]
            model, _ = fit(fam, D, H, Kd, sub, va0, args.steps, args.lr,
                           args.seed)
            vah, _ = hit_at(model, te0)
            report["curve"].setdefault(fam, {})[f"{frac:.2f}"] = {
                "n_train": k, "val": vah}
            print(f"[curve L{L0}] {fam:<7} n_train={k:<4} "
                  f"val hit@2={vah['hit@2']:.3f}", flush=True)

    json.dump(report, open(args.out, "w"), indent=1)
    print(f"[probe] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
