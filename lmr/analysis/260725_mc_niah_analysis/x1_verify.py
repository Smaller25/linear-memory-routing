"""Task 1 (mc-niah 0025 rev2): X1 dump verification.

CPU-only, no torch/model load needed — just reads the npz+meta sidecars the
GPU dump (x1_dump.py) wrote and cross-checks them against results/e1_routing.json.

Two checks:
  1. schema: keys present (u, q, csub_raw, c_full, stock_scores + meta),
     shapes consistent, q not missing/all-zero (rev2 retrieval-query addition).
  2. numeric: recompute per-layer hit@2 straight from stock_scores + meta
     (same rule as routing_stats.analyze_sample: eligible iff gold_seg <
     cur_seg; hit iff gold_seg among the top-TOPK scores, future/current
     already masked to -inf inside stock_scores) and diff against
     results/e1_routing.json results[model][dataset].per_layer[*].hit_at_2.

Known caveat (not a dump bug, see report): the underlying niah_single_1 /
niah_multikey_1 jsonl files contain a handful of duplicate "index" values
that refer to genuinely different input text. x1_dump.py names files by
sample_id, so the second row silently overwrites the first on disk -> the
dump has fewer unique samples (e.g. 47/50 for niah_single_1) than E1's
in-memory aggregate (which counts all 50 rows). This script reports
n_dump vs e1's n_total/n_eligible per dataset so the gap is visible.

Usage: /data2/sohyung/conda-envs/sh_infocap/bin/python x1_verify.py
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
DUMP_ROOT = os.path.join(MC_OUT, "x1_dump")
E1_PATH = os.path.join(HERE, "results", "e1_routing.json")

MODELS = ["mc-5B", "mc-30B"]
DATASETS = ["niah_single_1", "niah_multikey_1", "paired_S_multi", "paired_D_multi"]
TOPK = 2
TOL_PP = 3.0  # percentage points

REQUIRED_KEYS = {"u", "q", "csub_raw", "c_full", "stock_scores"}
REQUIRED_META = {"gold_seg", "cur_seg", "n_seg", "n_tok", "eligible", "key_segs",
                  "sample_id"}


def check_schema(model, dsname, n_spot=2):
    """Spot-check n_spot npz+meta pairs for key presence, shapes, q sanity."""
    d = os.path.join(DUMP_ROOT, model, dsname)
    files = sorted(glob.glob(os.path.join(d, "*.npz")))[:n_spot]
    problems = []
    for fp in files:
        z = np.load(fp)
        keys = set(z.files)
        missing = REQUIRED_KEYS - keys
        if missing:
            problems.append(f"{fp}: missing npz keys {missing}")
            continue
        u, q = z["u"], z["q"]
        if q.shape != u.shape:
            problems.append(f"{fp}: q.shape {q.shape} != u.shape {u.shape}")
        if np.allclose(q, 0):
            problems.append(f"{fp}: q is all-zero (rev1 dump? missing rev2 retrieval query)")
        qn = np.linalg.norm(q.astype(np.float32), axis=-1)
        if not (0.99 < qn.mean() < 1.01):
            problems.append(f"{fp}: q L2-norm mean={qn.mean():.4f}, expected ~1.0")
        meta_fp = fp.replace(".npz", ".meta.json")
        if not os.path.exists(meta_fp):
            problems.append(f"{fp}: no meta.json sidecar")
        else:
            meta = json.load(open(meta_fp))
            missing_m = REQUIRED_META - set(meta.keys())
            if missing_m:
                problems.append(f"{meta_fp}: missing meta keys {missing_m}")
    return files, problems


def load_dump_hits(model, dsname):
    """Recompute per-layer hit@2 (eligible samples only) from stock_scores+meta.

    Returns (per_layer_hit_rate: list[float or None], n_dump, n_eligible_dump).
    """
    d = os.path.join(DUMP_ROOT, model, dsname)
    npz_files = sorted(glob.glob(os.path.join(d, "*.npz")))
    n_dump = len(npz_files)
    L = None
    hit_sum = None
    hit_n = None
    n_eligible = 0
    for fp in npz_files:
        meta = json.load(open(fp.replace(".npz", ".meta.json")))
        if not meta["eligible"]:
            continue
        n_eligible += 1
        z = np.load(fp)
        scores = z["stock_scores"]  # [L,N] fp32, future/cur already -inf
        if L is None:
            L = scores.shape[0]
            hit_sum = [0] * L
            hit_n = [0] * L
        gold_seg = meta["gold_seg"]
        for layer in range(L):
            order = np.argsort(-scores[layer])  # descending
            hit = gold_seg in order[:TOPK].tolist()
            hit_sum[layer] += int(hit)
            hit_n[layer] += 1
    if L is None:
        return [], n_dump, 0
    rates = [hit_sum[i] / hit_n[i] if hit_n[i] else None for i in range(L)]
    return rates, n_dump, n_eligible


def main():
    e1 = json.load(open(E1_PATH))
    all_ok = True
    max_dev_pp = 0.0
    max_dev_loc = None
    lines = []

    for model in MODELS:
        for dsname in DATASETS:
            files, problems = check_schema(model, dsname)
            if problems:
                all_ok = False
                for p in problems:
                    lines.append(f"[SCHEMA-FAIL] {p}")
            dump_rates, n_dump, n_eligible_dump = load_dump_hits(model, dsname)

            e1_agg = e1["results"].get(model, {}).get(dsname)
            if e1_agg is None:
                lines.append(f"[MISSING] no e1_routing.json entry for {model}/{dsname}")
                all_ok = False
                continue
            e1_per_layer = e1_agg["per_layer"]
            best_layer = e1_agg["best_layer"]

            lines.append(f"\n=== {model}/{dsname} ===  "
                         f"dump: n_files={n_dump} n_eligible={n_eligible_dump}  "
                         f"e1: n_total={e1_agg['n_total']} n_eligible={e1_agg['n_eligible']}")
            lines.append(f"{'layer':>5}  {'e1_hit@2':>9}  {'dump_hit@2':>10}  {'dev(pp)':>8}")
            for i, pl in enumerate(e1_per_layer):
                e1_rate = pl["hit_at_2"]
                dump_rate = dump_rates[i] if i < len(dump_rates) else None
                if e1_rate is None or dump_rate is None:
                    dev_str = "n/a"
                else:
                    dev = (dump_rate - e1_rate) * 100
                    dev_str = f"{dev:+.1f}"
                    if abs(dev) > max_dev_pp:
                        max_dev_pp = abs(dev)
                        max_dev_loc = f"{model}/{dsname} layer{i}"
                    if abs(dev) > TOL_PP:
                        all_ok = False
                marker = " " + ("layer=best" if i == best_layer else "")
                e1_s = f"{e1_rate:.3f}" if e1_rate is not None else "None"
                dump_s = f"{dump_rate:.3f}" if dump_rate is not None else "None"
                lines.append(f"{i:>5}  {e1_s:>9}  {dump_s:>10}  {dev_str:>8}{marker}")

            if best_layer is not None:
                e1_best_hit = e1_agg["best_layer_hit_at_2"]
                dump_best_hit = dump_rates[best_layer] if best_layer < len(dump_rates) else None
                if e1_best_hit is not None and dump_best_hit is not None:
                    dev = (dump_best_hit - e1_best_hit) * 100
                    ok = abs(dev) <= TOL_PP
                    all_ok = all_ok and ok
                    lines.append(f"  best_layer={best_layer}: e1={e1_best_hit:.3f} "
                                 f"dump={dump_best_hit:.3f} dev={dev:+.1f}pp "
                                 f"[{'OK' if ok else 'FAIL'}]")

    print("\n".join(lines))
    print(f"\n=== SUMMARY === max_deviation={max_dev_pp:.1f}pp at {max_dev_loc}  "
          f"tolerance=±{TOL_PP}pp  overall={'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
