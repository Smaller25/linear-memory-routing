#!/usr/bin/env python3
"""score = P(gold selected) x P(correct | selected) + P(correct | not selected).

Measuring only the product hides which factor moved, and that hid the most
important result of the campaign for a day. Routing hit went 20.5% -> 32.5%
(p = 0.0079) while the score stayed flat, because conversion fell about as
much as selection rose:

            N     P(sel)   P(ok|sel)   score
  prev      4      0.300      0.533     16.0
  a5dot     4      0.440      0.295     20.0
  prev     16      0.080      0.750     14.0
  a5dot    16      0.210      0.095      7.0
  oracle    4      1.000      0.67      67.0
  oracle   16      1.000      0.69      69.0

Conversion sits 2-7x below the oracle's, so selection at 1.0 with today's
read would still only reach 9.5 at N=16. That is what redirected the work
from the router to the gate weight.

Pairing works because measure_hit.py and the eval both index the first
--max-examples rows of the same generated file, so (cell, sample_index) is a
common key. Selection comes from measure_hit's shared_hit; correctness is
recomputed from pred/ref with the same rule the eval scores by (every ref
substring present, case-insensitive).

Cells matching --exclude-cells are dropped, which is how a router's own
TRAINING seed is kept out: hit on training cells runs 0.640 / 0.360 against
0.440 / 0.210 held out, and pooling them flatters the arm.

Usage:
  python dsc/scripts/decompose_score.py \
      --arm /root/a6_headline/a5dot-3seed --hits /root/a5_scorer_data/hit_a5_dot.jsonl \
      --label a5dot --exclude-cells seed44
"""
from __future__ import annotations

import argparse, collections, glob, json, os, re


def correctness(arm_dir: str) -> dict:
    out = {}
    files = sorted(glob.glob(os.path.join(arm_dir, "per_sample", "*.jsonl")))
    if not files:
        raise SystemExit(f"no per_sample/*.jsonl under {arm_dir}")
    for f in files:
        cell = os.path.splitext(os.path.basename(f))[0]
        for line in open(f):
            r = json.loads(line)
            refs = r["ref"] if isinstance(r["ref"], list) else [r["ref"]]
            pred = (r.get("pred") or "").lower()
            out[(cell, r["sample_index"])] = all(
                str(x).lower() in pred for x in refs)
    return out


def selection(path: str) -> dict:
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[(r["cell"], r["sample_index"])] = bool(r["shared_hit"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="eval arm dir (correctness)")
    ap.add_argument("--hits", required=True, help="measure_hit.py JSONL")
    ap.add_argument("--label", default=None)
    ap.add_argument("--exclude-cells", default=None, metavar="REGEX",
                    help="drop matching cells, e.g. the arm's training seed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    label = args.label or os.path.basename(args.arm.rstrip("/"))

    ok, sel = correctness(args.arm), selection(args.hits)
    keys = sorted(set(ok) & set(sel))
    if not keys:
        raise SystemExit("the arm and the hit file share no (cell, index) keys")
    drop = re.compile(args.exclude_cells) if args.exclude_cells else None
    if drop:
        before = len({c for c, _ in keys})
        keys = [k for k in keys if not drop.search(k[0])]
        after = len({c for c, _ in keys})
        print(f"[filter] {before - after} cells dropped by "
              f"{args.exclude_cells!r}", flush=True)

    by_n = collections.defaultdict(list)
    for k in keys:
        m = re.search(r"_n(\d+)$", k[0])
        by_n[int(m.group(1)) if m else -1].append(k)

    rows = []
    print(f"\n{label}")
    print(f"{'N':>4}{'n':>6}{'P(sel)':>9}{'P(ok|sel)':>11}{'P(ok|not)':>11}"
          f"{'score':>8}{'check':>8}")
    for n in sorted(by_n):
        ks = by_n[n]
        s = [k for k in ks if sel[k]]
        ns = [k for k in ks if not sel[k]]
        p_sel = len(s) / len(ks)
        p_ok_s = (sum(ok[k] for k in s) / len(s)) if s else None
        p_ok_n = (sum(ok[k] for k in ns) / len(ns)) if ns else None
        score = 100 * sum(ok[k] for k in ks) / len(ks)
        # Reassembling the factors must return the score, or the two files
        # are not describing the same items.
        chk = 100 * (p_sel * (p_ok_s or 0) + (1 - p_sel) * (p_ok_n or 0))
        rows.append({"needles": n, "n": len(ks), "p_selected": p_sel,
                     "p_correct_given_selected": p_ok_s,
                     "p_correct_given_not": p_ok_n, "score": score,
                     "n_selected": len(s)})
        print(f"{n:>4}{len(ks):>6}{p_sel:>9.3f}"
              + (f"{p_ok_s:>11.3f}" if p_ok_s is not None else f"{'-':>11}")
              + (f"{p_ok_n:>11.3f}" if p_ok_n is not None else f"{'-':>11}")
              + f"{score:>8.1f}{chk:>8.1f}")
        if abs(chk - score) > 0.05:
            raise SystemExit(
                f"factors do not reassemble ({chk:.2f} vs {score:.2f}) — the "
                "hit file and the arm are not indexing the same items")
    print("\n  P(ok|sel) is the read: of the items where gold entered the "
          "top-k, how many answered.\n  Compare it against the oracle arm, "
          "whose score IS its conversion since P(sel)=1.")
    if args.out:
        json.dump({"label": label, "arm": args.arm, "hits": args.hits,
                   "excluded": args.exclude_cells, "rows": rows},
                  open(args.out, "w"), indent=1)
        print(f"[decompose] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
