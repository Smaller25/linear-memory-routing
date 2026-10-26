"""Paired, item-level comparison of two eval results (optionally several seeds each).

    python -m mslm.eval.compare --a runs/vanilla_s{0,1,2}/eval/X --b runs/mc_s{0,1,2}/eval/X \
        [--route-a none --route-b learned]

Items are matched on (length, task, index) — the eval set is fixed by
EVAL_DATA_SEED, so every model saw the same items. With several seeds per
side the per-item score is averaged over seeds first. Reports per (length,
task): mean A, mean B, mean diff, items B>A / B<A, exact sign-test p-value,
and a 95% paired-bootstrap CI of the diff.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict


def load(dirs: list[str], route: str | None):
    per = defaultdict(list)
    for d in dirs:
        path = d if d.endswith(".jsonl") else os.path.join(d, "items.jsonl")
        for line in open(path):
            r = json.loads(line)
            if route and r["route"] != route:
                continue
            per[(r["length"], r["task"], r["index"])].append(r["score"])
    return {k: sum(v) / len(v) for k, v in per.items()}


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def bootstrap_ci(diffs, n_boot=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def compare(a: dict, b: dict):
    groups = defaultdict(list)
    for key in sorted(set(a) & set(b)):
        groups[key[:2]].append((a[key], b[key]))
    rows = []
    for (length, task), pairs in sorted(groups.items()):
        diffs = [y - x for x, y in pairs]
        wins = sum(d > 0 for d in diffs)
        losses = sum(d < 0 for d in diffs)
        lo, hi = bootstrap_ci(diffs)
        rows.append({"length": length, "task": task, "n": len(pairs),
                     "a": 100 * sum(x for x, _ in pairs) / len(pairs),
                     "b": 100 * sum(y for _, y in pairs) / len(pairs),
                     "diff": 100 * sum(diffs) / len(diffs), "b_wins": wins, "b_losses": losses,
                     "sign_p": sign_test(wins, losses), "ci95": (100 * lo, 100 * hi)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--route-a", default=None)
    ap.add_argument("--route-b", default=None)
    args = ap.parse_args()
    rows = compare(load(args.a, args.route_a), load(args.b, args.route_b))
    print(f"{'len':>6} {'task':<18} {'n':>4} {'A':>6} {'B':>6} {'B-A':>6} {'W/L':>7} {'p':>6}  95%CI")
    for r in rows:
        print(f"{r['length']:>6} {r['task']:<18} {r['n']:>4} {r['a']:6.1f} {r['b']:6.1f} {r['diff']:+6.1f} "
              f"{r['b_wins']:>3}/{r['b_losses']:<3} {r['sign_p']:6.3f}  [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}]")


if __name__ == "__main__":
    main()
