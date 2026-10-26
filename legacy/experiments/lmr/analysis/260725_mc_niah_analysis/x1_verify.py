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

Fixed filename-collision bug (2026-07-27): the underlying niah_single_1 /
niah_multikey_1 jsonl files contain a handful of duplicate "index" values
that refer to genuinely different input text (vendored RULER's
src/ruler/gen/synthetic/niah.py:~281 shadows the loop `index` with a char
offset). x1_dump.py used to name files by that (non-unique) sample_id/
index, so later rows silently overwrote earlier ones on disk (dump had
only 47/50, 49/50 unique samples). x1_dump.py now keys filenames by the
enumeration position `ri` (always unique) and keeps sample_id inside
meta.json only. This script still prints n_dump vs e1's n_total/n_eligible
so a regression would be visible again.

Small-N note: paired_S_multi / paired_D_multi have only 16 samples, so a
single bf16 A100-vs-rtx6000 hit/miss flip swings hit@2 by 100/16=6.25pp —
larger than the flat TOL_PP=3.0 bar. tol_for() relaxes the bar to an
explicit "<=1.5 sample flips" allowance for N<=SMALL_N_THRESHOLD so the
overall verdict isn't spuriously FAIL on ordinary small-N noise.

Review fixes (2026-07-27):
  1. best_layer check now verifies argmax IDENTITY, not just the value at
     e1's index: it takes the dump's own max hit@2 across layers and asks
     whether e1's best_layer is within that dataset's tolerance of that
     max (i.e. a member of the tie set), so a case where the dump's true
     best layer moved elsewhere no longer passes silently.
  2. SUMMARY now reports the worst VIOLATION (deviation minus that row's
     own applied tolerance, only when positive) instead of the largest
     raw deviation — the raw-max used to point at an innocent passing
     small-N row while the actual over-tolerance row went unmentioned.
     overall PASS/FAIL is derived purely from per-row violations (dev >
     that row's own tol), same as it always was; only the human-readable
     summary line was misleading before.

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
TOL_PP = 3.0  # percentage points, flat bar for the ~50-sample datasets
SMALL_N_THRESHOLD = 16  # paired_S_multi / paired_D_multi have only 16 samples

REQUIRED_KEYS = {"u", "q", "csub_raw", "c_full", "stock_scores"}
REQUIRED_META = {"gold_seg", "cur_seg", "n_seg", "n_tok", "eligible", "key_segs",
                  "sample_id"}
# paired_S_multi / paired_D_multi rows additionally carry pair_id + condition
# (set by routing_stats._rows_for for kind="b" datasets); the plain niah_*
# datasets don't have these fields at all, so they're required conditionally
# rather than folded into the flat REQUIRED_META set.
REQUIRED_META_PAIRED = {"pair_id", "condition"}


def tol_for(n_eligible_dump):
    """Deviation tolerance (pp) for a given eligible-sample count.

    At N<=SMALL_N_THRESHOLD a single bf16 A100-vs-rtx6000 hit/miss flip
    already moves hit@2 by 100/N pp (e.g. 6.25pp at N=16) — well past the
    flat TOL_PP bar. Judge those by an explicit "<=1.5 sample flips"
    allowance (the 1.5 = 1 flip + 50% slack for rounding) instead, so the
    verdict doesn't go spuriously FAIL on ordinary small-N noise. Larger
    datasets keep the flat bar.
    """
    if n_eligible_dump and n_eligible_dump <= SMALL_N_THRESHOLD:
        return (1.5 / n_eligible_dump) * 100
    return TOL_PP


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
            required = REQUIRED_META | (REQUIRED_META_PAIRED if dsname.startswith("paired_") else set())
            missing_m = required - set(meta.keys())
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


def _argmax_tied(rates, tol_pp):
    """Index of the max non-None rate, plus the set of indices within
    tol_pp (percentage points) of that max — i.e. the "tie set" a noisy
    re-measurement could plausibly have picked instead. Returns
    (argmax_idx, dump_max, tied_set); argmax_idx/dump_max are None if every
    rate is None."""
    scored = [(i, r) for i, r in enumerate(rates) if r is not None]
    if not scored:
        return None, None, set()
    dump_max = max(r for _, r in scored)
    argmax_idx = next(i for i, r in scored if r == dump_max)
    tied = {i for i, r in scored if (dump_max - r) * 100 <= tol_pp}
    return argmax_idx, dump_max, tied


def main():
    e1 = json.load(open(E1_PATH))
    all_ok = True
    worst_violation_pp = 0.0   # dev - tol, only tracked when positive (an actual violation)
    worst_violation_loc = None
    worst_violation_tol = None
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
            tol = tol_for(n_eligible_dump)

            lines.append(f"\n=== {model}/{dsname} ===  "
                         f"dump: n_files={n_dump} n_eligible={n_eligible_dump}  "
                         f"e1: n_total={e1_agg['n_total']} n_eligible={e1_agg['n_eligible']}  "
                         f"applied_tol=±{tol:.1f}pp")
            lines.append(f"{'layer':>5}  {'e1_hit@2':>9}  {'dump_hit@2':>10}  {'dev(pp)':>8}")
            for i, pl in enumerate(e1_per_layer):
                e1_rate = pl["hit_at_2"]
                dump_rate = dump_rates[i] if i < len(dump_rates) else None
                if e1_rate is None or dump_rate is None:
                    dev_str = "n/a"
                else:
                    dev = (dump_rate - e1_rate) * 100
                    dev_str = f"{dev:+.1f}"
                    violation = abs(dev) - tol
                    if violation > worst_violation_pp:
                        worst_violation_pp = violation
                        worst_violation_loc = f"{model}/{dsname} layer{i}"
                        worst_violation_tol = tol
                    if violation > 0:
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
                    ok = abs(dev) <= tol
                    all_ok = all_ok and ok
                    lines.append(f"  best_layer VALUE: e1_best_layer={best_layer} "
                                 f"e1={e1_best_hit:.3f} dump={dump_best_hit:.3f} "
                                 f"dev={dev:+.1f}pp tol=±{tol:.1f}pp [{'OK' if ok else 'FAIL'}]")
                    violation = abs(dev) - tol
                    if violation > worst_violation_pp:
                        worst_violation_pp = violation
                        worst_violation_loc = f"{model}/{dsname} best_layer_value"
                        worst_violation_tol = tol

                # Identity check: does the dump's OWN argmax land on (or
                # within tol of) e1's best_layer? This is independent of
                # the value check above — a layer's hit@2 at e1's best
                # index can match within tol while a *different* layer is
                # now the dump's true argmax (e.g. if two layers were
                # close and noise flipped which one is highest). Catch
                # that by checking whether e1's best_layer is inside the
                # dump's own tie set around its max.
                argmax_idx, dump_max, tied = _argmax_tied(dump_rates, tol)
                if argmax_idx is not None:
                    identity_ok = best_layer in tied
                    all_ok = all_ok and identity_ok
                    lines.append(f"  best_layer IDENTITY: e1_best_layer={best_layer} "
                                 f"dump_argmax={argmax_idx} (dump_max={dump_max:.3f}) "
                                 f"tied_within_tol={sorted(tied)} "
                                 f"[{'OK' if identity_ok else 'FAIL'}]")
                    if not identity_ok:
                        dump_at_e1_best = dump_rates[best_layer] if best_layer < len(dump_rates) and dump_rates[best_layer] is not None else None
                        gap = (dump_max - dump_at_e1_best) * 100 if dump_at_e1_best is not None else float("inf")
                        violation = gap - tol
                        if violation > worst_violation_pp:
                            worst_violation_pp = violation
                            worst_violation_loc = f"{model}/{dsname} best_layer_identity"
                            worst_violation_tol = tol

    print("\n".join(lines))
    if worst_violation_loc is None:
        print(f"\n=== SUMMARY === no per-row violations (all deviations within their "
              f"row's own applied tolerance)  overall={'PASS' if all_ok else 'FAIL'}")
    else:
        print(f"\n=== SUMMARY === worst violation: {worst_violation_pp:.1f}pp over its "
              f"applied tolerance (±{worst_violation_tol:.1f}pp) at {worst_violation_loc}  "
              f"overall={'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
