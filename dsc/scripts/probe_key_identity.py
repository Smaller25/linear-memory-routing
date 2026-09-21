#!/usr/bin/env python3
"""Can the query's own key find the right needle? Offline, CPU, no forward.

Reads capture_key_identity.py's cache and scores segments several ways, all
reported as hit@k split by needle count, because N=4 and N=16 are different
problems: every lever so far gained on needle-vs-haystack and stalled on
needle-vs-needle.

  deployed     the trained router on the last position against the segment
               mean. The reference line — reproduces the shipped scorer.
  keymatch     <k of the query's key tokens, gamma_i>. NO learned parameters.
               The model already projects keys, and the needle segment
               contains the same string, so if key identity survives the
               descriptor at all this should find it.
  maxsim(m)    keymatch, but a segment scores as its best-matching block of
               m. m=1 is the deployed descriptor; higher m spends m times the
               segment memory to undo the 1/256 dilution of a single key.
  combo        deployed + keymatch, each standardized. Cheap to ship if both
               carry independent signal.
  hq-router    a router fitted on the hidden state at the KEY tokens instead
               of the last position, in case the identity is there but does
               not survive "mentioned in the provided text is".

Fitted variants train on one seed and report on the other, never on the rows
they are scored on.

Usage:
  python dsc/scripts/probe_key_identity.py --cache /root/keyid_cache.pt \
      --router /root/routers_a5_dot --layer 0 --out /root/keyid_probe.json
"""
from __future__ import annotations

import argparse, collections, json, os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "dsc")):
    if p not in sys.path:
        sys.path.insert(0, p)


def coarsen(blocks: torch.Tensor, m: int) -> torch.Tensor:
    """[Nseg,B,H,Kd] -> [Nseg,m,H,Kd] by averaging adjacent blocks. Equal-size
    blocks, so m=1 is exactly the segment mean the deployed descriptor uses."""
    n, b, h, kd = blocks.shape
    if b % m:
        raise ValueError(f"{b} blocks not divisible by m={m}")
    return blocks.view(n, m, b // m, h, kd).mean(dim=2)


def hit_at(scores, golds, ks=(1, 2, 4)):
    got = {k: 0 for k in ks}
    n = 0
    for sc, gd in zip(scores, golds):
        if sc is None or len(sc) == 0 or gd >= len(sc):
            continue
        r = int((torch.as_tensor(sc).argsort(descending=True) == gd)
                .nonzero()[0, 0])
        for k in ks:
            got[k] += int(r < k)
        n += 1
    return {f"hit@{k}": (got[k] / n if n else None) for k in ks} | {"n": n}


def zscore(x: torch.Tensor) -> torch.Tensor:
    s = x.std()
    return (x - x.mean()) / s if float(s) > 0 else x - x.mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--router", default=None, help="dir with router_L*.pt")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--fit-seed", default="seed44", help="unused if absent")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    rows, meta = blob["rows"], blob["meta"]
    L, H, Kd, D = args.layer, meta["H"], meta["Kd"], meta["D"]
    B = meta["blocks"]
    print(f"[probe] {len(rows)} rows, layer {L}, blocks {B}, "
          f"H={H} Kd={Kd} D={D}", flush=True)

    items = []
    for r in rows:
        elig = r["nseg"] - 1
        if elig <= 0 or r["gold"] >= elig:
            continue
        g = torch.tensor(np.asarray(r[f"g{L}"][:elig])).float()   # [E,B,H,Kd]
        items.append({
            "cell": r["cell"], "gold": r["gold"], "many": "_n16" in r["cell"]
            or "_n32" in r["cell"],
            "g": g,
            "h": torch.tensor(np.asarray(r[f"h{L}"])).float(),
            "kq": torch.tensor(np.asarray(r[f"kq{L}"])).float(),   # [H,Kd]
            "hq": torch.tensor(np.asarray(r[f"hq{L}"])).float(),
        })
    if not items:
        raise SystemExit("no usable rows")
    seeds = sorted({it["cell"].split("_")[0] for it in items})
    print(f"[probe] seeds present: {seeds}", flush=True)

    report = {"cache": args.cache, "layer": L, "variants": {}}

    def record(name, scorer, subset=None):
        use = subset if subset is not None else items
        sc = [scorer(it) for it in use]
        gd = [it["gold"] for it in use]
        few = [i for i, it in enumerate(use) if not it["many"]]
        many = [i for i, it in enumerate(use) if it["many"]]
        row = {"all": hit_at(sc, gd),
               "few_needles": hit_at([sc[i] for i in few], [gd[i] for i in few]),
               "many_needles": hit_at([sc[i] for i in many], [gd[i] for i in many])}
        report["variants"][name] = row
        f2 = row["few_needles"]["hit@2"]
        m2 = row["many_needles"]["hit@2"]
        print(f"  {name:<22} hit@2 all={row['all']['hit@2']:.3f}  "
              f"few={f2 if f2 is None else round(f2, 3)}  "
              f"many={m2 if m2 is None else round(m2, 3)}  "
              f"(hit@1={row['all']['hit@1']:.3f} n={row['all']['n']})",
              flush=True)
        return row

    # keymatch, and the max-sim curve over descriptor memory
    def keymatch(m):
        def f(it):
            gm = coarsen(it["g"], m)                       # [E,m,H,Kd]
            s = torch.einsum("hk,emhk->em", it["kq"], gm)   # [E,m]
            return s.max(dim=-1).values
        return f

    print("[probe] training-free")
    for m in [1, 2, 4, B]:
        if B % m == 0:
            record(f"keymatch maxsim m={m}", keymatch(m))

    # the deployed scorer, for the reference line
    dep = None
    if args.router:
        from dsc.mc_baseline.mc_ssc_mlp_router import MLPRouterHead
        sd = torch.load(os.path.join(args.router, f"router_L{L}.pt"),
                        map_location="cpu")
        head = MLPRouterHead(D, H, Kd,
                             scorer="dot" if "logit_scale" in sd else "cos")
        head.load_state_dict(sd, strict=True)
        head.eval()

        def dep_scorer(it):
            gm = coarsen(it["g"], 1)[:, 0]                 # [E,H,Kd]
            with torch.no_grad():
                return head.scores(it["h"][None, None], gm[None])[0, 0]
        dep = dep_scorer
        print("[probe] deployed reference")
        record("deployed router", dep_scorer)

        print("[probe] combination")
        km1 = keymatch(1)

        def combo(it):
            return zscore(dep(it)) + zscore(km1(it))
        record("deployed + keymatch", combo)

    # Is the key's identity present at the KEY tokens but lost by the last
    # position? Fit the same head on hq instead of h. Two folds by seed, each
    # reported on the seed it did not see.
    from dsc.mc_baseline.mc_ssc_mlp_router import MLPRouterHead

    def fit_on(field):
        folds = collections.defaultdict(list)
        for held in seeds:
            tr = [it for it in items if not it["cell"].startswith(held)]
            te = [it for it in items if it["cell"].startswith(held)]
            if not tr or not te:
                continue
            torch.manual_seed(0)
            head = MLPRouterHead(D, H, Kd, scorer="dot")
            opt = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                    weight_decay=0.01)
            gen = torch.Generator().manual_seed(1)
            for _ in range(args.steps):
                it = tr[int(torch.randint(len(tr), (1,), generator=gen))]
                gm = coarsen(it["g"], 1)[:, 0]
                sc = head.scores(it[field][None, None], gm[None])[0, 0]
                loss = F.cross_entropy(sc[None],
                                       torch.tensor([it["gold"]]))
                opt.zero_grad(); loss.backward(); opt.step()
            head.eval()
            with torch.no_grad():
                for it in te:
                    gm = coarsen(it["g"], 1)[:, 0]
                    folds[it["cell"]].append(
                        (head.scores(it[field][None, None], gm[None])[0, 0],
                         it["gold"], it["many"]))
        flat = [v for vs in folds.values() for v in vs]
        return flat

    print("[probe] fitted, two folds by seed (reported on the held-out seed)")
    for field, name in (("h", "fitted on h (last pos)"),
                        ("hq", "fitted on hq (key toks)")):
        flat = fit_on(field)
        if not flat:
            continue
        sc = [x[0] for x in flat]; gd = [x[1] for x in flat]
        few = [i for i, x in enumerate(flat) if not x[2]]
        many = [i for i, x in enumerate(flat) if x[2]]
        row = {"all": hit_at(sc, gd),
               "few_needles": hit_at([sc[i] for i in few], [gd[i] for i in few]),
               "many_needles": hit_at([sc[i] for i in many], [gd[i] for i in many])}
        report["variants"][name] = row
        f2, m2 = row["few_needles"]["hit@2"], row["many_needles"]["hit@2"]
        print(f"  {name:<22} hit@2 all={row['all']['hit@2']:.3f}  "
              f"few={f2 if f2 is None else round(f2, 3)}  "
              f"many={m2 if m2 is None else round(m2, 3)}  "
              f"(hit@1={row['all']['hit@1']:.3f} n={row['all']['n']})",
              flush=True)

    # Sub-segment descriptors were only tested above through keymatch, which
    # sits at chance, so that could not detect a max-sim effect either way.
    # Fit the router against max-sim over m blocks instead: same data, same
    # folds, only the descriptor granularity changes. m=1 is the deployed
    # descriptor, so this is the memory/accuracy curve for a scorer that
    # actually works.
    def fit_maxsim(m):
        out = []
        for held in seeds:
            tr = [it for it in items if not it["cell"].startswith(held)]
            te = [it for it in items if it["cell"].startswith(held)]
            if not tr or not te:
                continue
            torch.manual_seed(0)
            head = MLPRouterHead(D, H, Kd, scorer="dot")
            opt = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                    weight_decay=0.01)
            gen = torch.Generator().manual_seed(1)

            def score(it, hd):
                gm = coarsen(it["g"], m)                    # [E,m,H,Kd]
                u = hd.u(it["h"][None, None], normalize=False)[0, 0]
                return torch.einsum("hk,emhk->em", u, gm).max(dim=-1).values \
                    * hd.logit_scale.exp()

            for _ in range(args.steps):
                it = tr[int(torch.randint(len(tr), (1,), generator=gen))]
                loss = F.cross_entropy(score(it, head)[None],
                                       torch.tensor([it["gold"]]))
                opt.zero_grad(); loss.backward(); opt.step()
            head.eval()
            with torch.no_grad():
                for it in te:
                    out.append((score(it, head), it["gold"], it["many"]))
        return out

    print("[probe] fitted router x descriptor granularity (max-sim over m)")
    for m in [1, 2, 4, B]:
        if B % m:
            continue
        flat = fit_maxsim(m)
        if not flat:
            continue
        sc = [x[0] for x in flat]; gd = [x[1] for x in flat]
        few = [i for i, x in enumerate(flat) if not x[2]]
        many = [i for i, x in enumerate(flat) if x[2]]
        row = {"all": hit_at(sc, gd),
               "few_needles": hit_at([sc[i] for i in few], [gd[i] for i in few]),
               "many_needles": hit_at([sc[i] for i in many], [gd[i] for i in many])}
        name = f"fitted maxsim m={m}"
        report["variants"][name] = row
        f2, m2 = row["few_needles"]["hit@2"], row["many_needles"]["hit@2"]
        print(f"  {name:<22} hit@2 all={row['all']['hit@2']:.3f}  "
              f"few={f2 if f2 is None else round(f2, 3)}  "
              f"many={m2 if m2 is None else round(m2, 3)}  "
              f"(hit@1={row['all']['hit@1']:.3f} n={row['all']['n']})",
              flush=True)

    json.dump(report, open(args.out, "w"), indent=1)
    print(f"[probe] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
