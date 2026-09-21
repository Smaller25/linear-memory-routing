#!/usr/bin/env python3
"""Paired item-level comparison of two arms. Use this instead of eyeballing
the two marginal scores.

Every arm answers the SAME prompts, so the comparison is paired. Comparing
marginal scores with independent-proportion errors throws the pairing away
and inflates the uncertainty: the layer-shared routing arm at 18.0 against
the baseline at 4.0 reads as 1.7 standard errors that way, and as 8 gained /
1 lost / p = 0.039 when paired. It cuts the other way too — a 10.0 against
18.0 that looked like a halved conversion was 2 gained / 6 lost, p = 0.29.

Only discordant items carry information, so the exact two-sided sign test
(equivalently McNemar) is the whole analysis. That also means conditioning on
a ceiling arm does NOT add power: dropping items the oracle cannot solve
leaves the discordant counts unchanged. What a ceiling arm gives is a
meaningful denominator ("8 of the 27 achievable items"), which belongs in the
write-up, not in the test.

The detection floor is printed because it is the number to check BEFORE
spending GPU. p < 0.05 needs 6 one-directional discordant items, so at n=50 a
real effect has to be worth about 12 points to be visible at all. An arm whose
predicted effect sits under the floor cannot pay for itself.

Usage:
  # score (recomputed string_match) between two eval arm dirs
  python dsc/scripts/compare_arms.py --a /root/a1_stage2/mc-top2-base \
      --b /root/a3_sources/bc-L0-v3 --ceiling /root/a2_broadcast/oracle-top2

  # routing hit rate between two measure_hit.py outputs
  python dsc/scripts/compare_arms.py --metric hit \
      --a /root/hit_base.jsonl --b /root/hit_shared.jsonl
"""
from __future__ import annotations

import argparse, glob, json, math, os
from math import comb


def sign_test_p(a: int, b: int) -> float:
    """Exact two-sided sign test on discordant pairs."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def detection_floor(n: int, need: int = 6) -> float:
    """Best-case floor: the effect needed if EVERY discordant item goes one way.

    Optimistic on purpose, as a planning number. Real discordance splits both
    ways, which costs power — see required_n for what the observed split
    actually implies.
    """
    return need / n if n else float("nan")


def required_n(n: int, a_only: int, b_only: int) -> int | None:
    """Items needed for this observed effect and discordance to reach p<0.05.

    McNemar's z is (b - a) / sqrt(b + a). Discordant pairs and the imbalance
    both scale with n, so z scales with sqrt(n) and the shortfall is
    n * (1.96 / z)^2. This is the number to look at when a comparison lands
    in the 0.05-0.3 range: it says whether one more seed settles it or ten do.
    """
    d = a_only + b_only
    if d == 0 or a_only == b_only:
        return None
    z = abs(b_only - a_only) / math.sqrt(d)
    if z >= 1.96:
        return n
    return int(math.ceil(n * (1.96 / z) ** 2))


def string_match(rec: dict) -> bool:
    """RULER string_match as the eval applies it: every ref substring present."""
    refs = rec["ref"]
    refs = refs if isinstance(refs, list) else [refs]
    pred = (rec.get("pred") or "").lower()
    return all(str(r).lower() in pred for r in refs)


def load_arm(path: str, metric: str, hit_field: str) -> dict:
    """-> {(cell, sample_index): bool}. Accepts an eval arm dir or a JSONL."""
    files = []
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "per_sample", "*.jsonl")))
        if not files:
            files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    elif path.endswith(".jsonl"):
        files = [path]
    if not files:
        raise SystemExit(f"no per-sample JSONL under {path}")
    out = {}
    for f in files:
        cell = os.path.splitext(os.path.basename(f))[0]
        for line in open(f):
            r = json.loads(line)
            key = (r.get("cell", cell), r["sample_index"])
            if metric == "score":
                out[key] = string_match(r)
            else:
                v = r.get(hit_field)
                if v is None:
                    raise SystemExit(
                        f"{f} has no field {hit_field!r} — is this a "
                        "measure_hit.py output?")
                out[key] = bool(v) if isinstance(v, bool) else v > 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference arm (dir or jsonl)")
    ap.add_argument("--b", required=True, help="arm under test")
    ap.add_argument("--ceiling", default=None,
                    help="optional ceiling arm (e.g. oracle) — reports the "
                         "achievable denominator, does not affect the test")
    ap.add_argument("--metric", default="score", choices=["score", "hit"])
    ap.add_argument("--hit-field", default="shared_hit",
                    help="field to read for --metric hit")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--only-cells", default=None, metavar="REGEX",
                    help="keep only cells matching this regex. Use it to drop "
                         "an arm's TRAINING seed — those rows favour it, and "
                         "pooled significance can rest on them")
    ap.add_argument("--out", default=None, help="write the table as JSON")
    args = ap.parse_args()

    la = args.label_a or os.path.basename(args.a.rstrip("/"))
    lb = args.label_b or os.path.basename(args.b.rstrip("/"))
    da = load_arm(args.a, args.metric, args.hit_field)
    db = load_arm(args.b, args.metric, args.hit_field)
    dc = load_arm(args.ceiling, args.metric, args.hit_field) if args.ceiling else None

    shared = sorted(set(da) & set(db))
    if args.only_cells:
        import re
        keep = re.compile(args.only_cells)
        dropped = sorted({c for c, _ in shared if not keep.search(c)})
        shared = [k for k in shared if keep.search(k[0])]
        if dropped:
            print(f"[filter] dropped cells: {', '.join(dropped)}", flush=True)
    if not shared:
        raise SystemExit("the two arms share no (cell, sample_index) keys")
    missing = (len(da) - len(shared), len(db) - len(shared))
    if any(missing):
        print(f"[warn] {missing[0]} items only in A, {missing[1]} only in B — "
              "comparing the intersection", flush=True)

    cells = sorted({c for c, _ in shared})
    rows, totals = [], [0, 0, 0, 0]
    for cell in cells + (["ALL"] if len(cells) > 1 else []):
        keys = shared if cell == "ALL" else [k for k in shared if k[0] == cell]
        a_only = sum(1 for k in keys if da[k] and not db[k])
        b_only = sum(1 for k in keys if db[k] and not da[k])
        both = sum(1 for k in keys if da[k] and db[k])
        none = sum(1 for k in keys if not da[k] and not db[k])
        n = len(keys)
        row = {"cell": cell, "n": n,
               "a_rate": (both + a_only) / n, "b_rate": (both + b_only) / n,
               "a_only": a_only, "b_only": b_only, "both": both, "none": none,
               "discordant": a_only + b_only,
               "p": sign_test_p(a_only, b_only),
               "detection_floor": detection_floor(n),
               "required_n": required_n(n, a_only, b_only)}
        if dc:
            ck = [k for k in keys if k in dc]
            solvable = sum(1 for k in ck if dc[k])
            row["ceiling_solvable"] = solvable
            row["b_of_solvable"] = (
                sum(1 for k in ck if db[k] and dc[k]) / solvable
                if solvable else None)
        rows.append(row)
        if cell != "ALL":
            for i, v in enumerate((a_only, b_only, both, none)):
                totals[i] += v

    print(f"\n{la}  ->  {lb}     metric={args.metric}"
          + (f" ({args.hit_field})" if args.metric == "hit" else ""))
    hdr = ("cell", "n", "A", "B", "A-only", "B-only", "both", "none", "p",
           "need n")
    w = (26, 5, 7, 7, 7, 7, 6, 6, 8, 8)
    print("".join(h.rjust(x) for h, x in zip(hdr, w)))
    for r in rows:
        print(r["cell"].rjust(26)
              + str(r["n"]).rjust(5)
              + f"{100 * r['a_rate']:.1f}".rjust(7)
              + f"{100 * r['b_rate']:.1f}".rjust(7)
              + str(r["a_only"]).rjust(7) + str(r["b_only"]).rjust(7)
              + str(r["both"]).rjust(6) + str(r["none"]).rjust(6)
              + f"{r['p']:.4f}".rjust(8)
              + (str(r["required_n"]) if r["required_n"] else "-").rjust(8))
    if dc:
        print()
        for r in rows:
            if r.get("ceiling_solvable"):
                print(f"  {r['cell']}: ceiling solves "
                      f"{r['ceiling_solvable']}/{r['n']}; {lb} got "
                      f"{r['b_of_solvable']:.1%} of those")
    print("\n  need n = items required for THIS observed effect and discordance "
          "to reach p<0.05; equals n when\n  already significant, '-' when the "
          "split is even. Before running an arm, the planning floor is "
          "6/n\n  one-directional (12pp at n=50, 6pp at n=100) — but real "
          "discordance goes both ways, so\n  treat that as a best case. Cheaper "
          "than raising n: test the routing hit, not the score.")
    if args.out:
        json.dump({"a": la, "b": lb, "metric": args.metric, "rows": rows},
                  open(args.out, "w"), indent=1)
        print(f"\n[compare] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
