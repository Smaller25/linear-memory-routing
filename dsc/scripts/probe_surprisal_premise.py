#!/usr/bin/env python3
"""Is the answer span actually high-surprisal? The gate before fitting anything.

Weighting a descriptor by surprisal only helps if the tokens a diverse-key
query must match are the surprising ones. That is an assumption about the
data, not about the model, and it is answerable from a capture with no router,
no fitting and no GPU.

Three numbers per cell, all computed from `surprisal` and `key_span` which
`capture_key_identity.py` now stores:

  ratio      mean surprisal of the gold segment's answer tokens, over the
             mean of every other token in that segment. 1.0 means weighting
             changes nothing; the intervention needs this comfortably above 1.
  share@tau  the answer tokens' share of total weight inside their own
             32-token block at each tau. At tau=0 this is just their token
             share, which is the quantity the whole dilution argument is
             about.
  rank       where the answer tokens sit in the segment's surprisal ordering
             (0.0 = most surprising token in the segment).

A ratio near 1 kills the idea for this benchmark before any GPU is spent. A
large ratio is necessary but not sufficient: it could also mean the benchmark
is easy in a way natural text is not, which is what --by-format is for.

Usage:
  python dsc/scripts/probe_surprisal_premise.py --cache /root/keyid_cache.pt
"""
from __future__ import annotations

import argparse, json, os, sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "dsc")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--floor", type=float, default=1e-2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    rows, meta = blob["rows"], blob["meta"]
    chunk, blocks = meta["chunk"], meta["blocks"]
    per = chunk // blocks
    if "surprisal" not in rows[0]:
        raise SystemExit(
            "this capture has no per-token surprisal — recapture with the "
            "current capture_key_identity.py")

    cells = defaultdict(list)
    for r in rows:
        sur = np.asarray(r["surprisal"], dtype=np.float64)
        if "needle_span" not in r:
            raise SystemExit(
                "this capture stores only key_span (the question's copy of "
                "the key). Recapture: the premise is about the needle's "
                "answer tokens, and the question's copy is trivially "
                "predictable, which would make the check unfalsifiable.")
        lo, hi = r["needle_span"]
        gold = r["gold"]
        seg_lo, seg_hi = gold * chunk, min((gold + 1) * chunk, len(sur))
        if not (seg_lo <= lo < seg_hi):
            # token_position_answer and the tokenizer disagreed about where
            # the answer lands; skip rather than measure the wrong span.
            continue
        cells[r["cell"]].append((sur, lo, hi, seg_lo, seg_hi))

    if not cells:
        raise SystemExit(
            "no sample had its answer span inside the gold segment — the "
            "span locator and token_position_answer disagree. Fix that "
            "before reading any number here.")

    report = {"cache": args.cache, "cells": {}}
    print(f"{'cell':<26} {'n':>4} {'ratio':>7} {'rank':>6} " +
          " ".join(f"share@{t:g}".rjust(10) for t in args.taus))
    for cell in sorted(cells):
        rs, rk, shares = [], [], defaultdict(list)
        for sur, lo, hi, seg_lo, seg_hi in cells[cell]:
            seg = sur[seg_lo:seg_hi]
            ans = sur[lo:hi]
            other = np.concatenate([sur[seg_lo:lo], sur[hi:seg_hi]])
            if not len(ans) or not len(other):
                continue
            rs.append(ans.mean() / max(other.mean(), 1e-9))
            rk.append(float((seg > ans.mean()).mean()))
            blo = ((lo - seg_lo) // per) * per + seg_lo
            blk = sur[blo:blo + per]
            a_in = sur[max(lo, blo):min(hi, blo + per)]
            for t in args.taus:
                wb = np.clip(blk, args.floor, None) ** t
                wa = np.clip(a_in, args.floor, None) ** t
                shares[t].append(wa.sum() / max(wb.sum(), 1e-9))
        if not rs:
            continue
        row = {"n": len(rs), "ratio": float(np.mean(rs)),
               "rank": float(np.mean(rk)),
               "share": {str(t): float(np.mean(shares[t])) for t in args.taus}}
        report["cells"][cell] = row
        print(f"{cell:<26} {row['n']:>4} {row['ratio']:>7.2f} "
              f"{row['rank']:>6.3f} " +
              " ".join(f"{row['share'][str(t)]:>10.3f}" for t in args.taus))

    allr = [c["ratio"] for c in report["cells"].values()]
    print(f"\npooled surprisal ratio {np.mean(allr):.2f} "
          f"(1.0 = weighting cannot help)")
    for t in args.taus:
        v = np.mean([c["share"][str(t)] for c in report["cells"].values()])
        print(f"  tau={t:<4g} answer tokens hold {v:.3f} of their block's weight")
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"[probe] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
