# -*- coding: utf-8 -*-
"""Per-source length-upsampled epoch plan (Fu et al., arXiv:2402.10171 §5.3).

Recipe (paper-exact):
  - source token shares stay at the ORIGINAL mixture (we use the empirical
    shares of the chunk1 pool, which reproduce SlimPajama's published
    67/15/4.5/4.5/4.5/2.5/2.0 within noise; both are logged)
  - within each source, documents longer than 4K tokens are upsampled from
    their natural ~30% token share to LONG_TARGET=70%
  - sampling preserves the within-bucket length distribution: shuffle bucket,
    take docs until the bucket's token budget is met; if a bucket is smaller
    than its budget it is repeated in full (true upsampling), reshuffled per
    pass

Output: plan_upsample_15b.npy — uint64 doc ids in final (shuffled) order,
consumed sequentially by data.PackedDocIterator. Plus a stats json.

Run on the VESSL CPU node after tokenization:
    python build_upsample_plan.py --data /root/smaller/dynmc/data/tokenized \
        --target-tokens 15e9 --out plan_upsample_15b.npy
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

LONG_THRESHOLD = 4096
LONG_TARGET = 0.70
SOURCE_NAMES = ["CommonCrawl", "C4", "Github", "Book", "ArXiv", "Wikipedia", "StackExchange"]
# tokenize_slimpajama.py SOURCE_IDS 순서: CC=0, C4=1, Github=2, Book=3, ArXiv=4, Wiki=5, SE=6
ID2NAME = {0: "CommonCrawl", 1: "C4", 2: "Github", 3: "Book", 4: "ArXiv",
           5: "Wikipedia", 6: "StackExchange"}


def load_index(data_dir: str):
    lens, srcs = [], []
    for p in sorted(glob.glob(os.path.join(data_dir, "doclens-*.npy"))):
        lens.append(np.load(p))
        srcs.append(np.load(p.replace("doclens", "sources")))
    doc_len = np.concatenate(lens).astype(np.int64)
    doc_src = np.concatenate(srcs)
    return doc_len, doc_src


def sample_bucket(ids: np.ndarray, doc_len: np.ndarray, budget: int,
                  rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle-and-take until budget; repeat the (reshuffled) bucket if needed."""
    out, got = [], 0
    if len(ids) == 0 or budget <= 0:
        return out
    while got < budget:
        perm = rng.permutation(ids)
        csum = np.cumsum(doc_len[perm])
        if got + csum[-1] <= budget:      # bucket 전체 < 잔여 budget → 통째로 (진짜 upsample)
            out.append(perm)
            got += int(csum[-1])
        else:
            cut = int(np.searchsorted(csum, budget - got)) + 1
            out.append(perm[:cut])
            got += int(csum[cut - 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target-tokens", type=float, default=15e9)
    ap.add_argument("--out", default="plan_upsample_15b.npy")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    doc_len, doc_src = load_index(a.data)
    total_pool = int(doc_len.sum())
    target = int(a.target_tokens)
    rng = np.random.default_rng(a.seed)
    print(f"pool: {len(doc_len)/1e6:.1f}M docs, {total_pool/1e9:.2f}B tokens; target {target/1e9:.1f}B")

    stats = {"pool_tokens": total_pool, "target_tokens": target,
             "long_threshold": LONG_THRESHOLD, "long_target": LONG_TARGET, "sources": {}}
    pieces = []
    for sid, name in ID2NAME.items():
        mask = doc_src == sid
        ids = np.nonzero(mask)[0]
        toks = int(doc_len[ids].sum())
        share = toks / total_pool
        budget = int(round(target * share))
        long_ids = ids[doc_len[ids] > LONG_THRESHOLD]
        short_ids = ids[doc_len[ids] <= LONG_THRESHOLD]
        nat_long = int(doc_len[long_ids].sum()) / max(toks, 1)
        b_long = int(round(budget * LONG_TARGET))
        b_short = budget - b_long
        pieces += sample_bucket(long_ids, doc_len, b_long, rng)
        pieces += sample_bucket(short_ids, doc_len, b_short, rng)
        stats["sources"][name] = dict(pool_tokens=toks, share=round(share, 4),
                                      natural_long_frac=round(nat_long, 4),
                                      budget=budget, budget_long=b_long)
        print(f"  {name:14s} share {share:6.2%} natural-long {nat_long:6.2%} "
              f"budget {budget/1e9:.3f}B (long {b_long/1e9:.3f}B)")

    unknown = int(doc_len[doc_src == 255].sum())
    if unknown:
        print(f"  WARNING: {unknown/1e9:.3f}B tokens with unknown source (excluded)")

    plan = np.concatenate(pieces).astype(np.uint64)
    rng.shuffle(plan)
    achieved = int(doc_len[plan.astype(np.int64)].sum())
    n_repeat = len(plan) - len(np.unique(plan))
    stats.update(plan_docs=len(plan), plan_tokens=achieved, repeated_docs=int(n_repeat))
    print(f"plan: {len(plan)/1e6:.2f}M docs, {achieved/1e9:.2f}B tokens, "
          f"repeats {n_repeat/1e6:.2f}M")

    out = os.path.join(a.data, a.out)
    np.save(out, plan)
    with open(out.replace(".npy", "_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"UPSAMPLE_PLAN_COMPLETE -> {out}")


if __name__ == "__main__":
    main()
