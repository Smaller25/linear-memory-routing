"""X2 corrections (adversarial review, Task 3 round 2): the reviewer's
stronger null hypothesis + structural top-2 ceiling. Both are properties of
the DATA/tokenization only (model-independent — no forward pass, no GPU),
so they're computed once per (task, length) and are identical across
mc-5B/mc-30B.

## needle_null_hit2 ("perfect needle detector, zero key discrimination")

The plain chance baseline (2/cur_seg, already in x2_probe.py) assumes the
router picks 2 segments uniformly at random from ALL past segments. That's
too weak a null: a real "needle detector" (spec's H_outlier/H_template,
either of which implies (M)) would first narrow the candidate set down to
segments that actually contain SOME needle (any key), and only then fail to
discriminate which one is the queried key. The reviewer's null formalizes
that stronger detector: top-2 chosen uniformly at random from the *eligible
needle-bearing segments* (M of them), not from all cur_seg past segments.
For a queried-and-eligible needle whose own segment is one of those M
candidates, P(hit@2) = min(1, 2/M) (same combinatorics as the plain chance
formula, with n_past replaced by M — this is <= plain chance only when
M < cur_seg, which is the common case since most past segments don't
contain any needle at all).

## structural_ceiling (max achievable macro hit@2 given top-2 slots)

If G (>2) distinct queried-and-eligible needle segments exist, no single
top-2 selection can cover all of them — routing scores are computed ONCE
per sample (one query position, shared across every needle checked; see
x2_probe.analyze_sample_multi), so this is a genuine structural cap, not a
per-needle-independent one. If duplicate needles land in the same segment
(plausible here — see note below — since cur_seg is often only 4-7 at
length=2048, a mini birthday-paradox regime for 4 needles), the best
achievable is "cover the segments with the most needles first": sort
segments by needle-count descending, take the top 2, sum their counts,
divide by n_e (eligible queried needle count). If G<=2, both fit in top-2
trivially -> ceiling 1.0 regardless of counts.

## ceiling-normalized skill (item 3i)

skill = (macro_hit2 - chance_hit2) / (structural_ceiling - chance_hit2)
i.e. where the observed value sits between the plain-chance floor and the
best-achievable-given-top-2 ceiling for that task's own data. Falls back to
None when ceiling == chance (degenerate, shouldn't happen in practice since
ceiling >= chance always by construction: the null both draws from a
superset of segments, and the ceiling numerator is an upper bound on hits).
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as mcdata

CHUNK = mcdata.CHUNK
MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")


def structural_stats_for_sample(cur_seg, needles, gold_needles):
    """Pure function (no tokenizer/model): given the already-computed
    cur_seg, the FULL needle list (any key) and the gold (queried) needle
    list for one sample, returns {"n_e", "M", "needle_null_hit2",
    "structural_ceiling"} or None if the sample has 0 eligible queried
    needles (matches x2_probe's eligibility exclusion — same convention,
    excluded from every aggregate, not silently zeroed)."""
    gold_eligible = [n for n in gold_needles if n["seg"] < cur_seg]
    n_e = len(gold_eligible)
    if n_e == 0:
        return None

    all_eligible_segs = sorted({n["seg"] for n in needles if n["seg"] < cur_seg})
    M = len(all_eligible_segs)
    needle_null_hit2 = min(1.0, 2.0 / M) if M > 0 else None

    seg_counts = {}
    for n in gold_eligible:
        seg_counts[n["seg"]] = seg_counts.get(n["seg"], 0) + 1
    G = len(seg_counts)
    if G <= 2:
        ceiling = 1.0
    else:
        top2_sum = sum(sorted(seg_counts.values(), reverse=True)[:2])
        ceiling = top2_sum / n_e

    return {"n_e": n_e, "M": M, "G": G, "needle_null_hit2": needle_null_hit2,
            "structural_ceiling": ceiling}


def _rows_for(task, length):
    path = os.path.join(MC_OUT, "data", str(length), task, "validation.jsonl")
    return [json.loads(l) for l in open(path)]


def compute_dataset_structural(task, length, tok, chunk=CHUNK):
    """Tokenizes every row (no model/GPU) to get cur_seg the same way
    x2_probe.analyze_sample_multi does (t=T-1, cur_seg=t//chunk), calls
    annotate(), and aggregates structural_stats_for_sample over all
    eligible samples (mean — these are per-sample scalars, not per-needle,
    so no two-level macro distinction is needed here)."""
    rows = _rows_for(task, length)
    null_vals, ceil_vals = [], []
    n_eligible = 0
    for r in rows:
        ann = mcdata.annotate(r["input"], tok)
        ids = tok(r["input"], add_special_tokens=False).input_ids
        T = len(ids)
        t = T - 1
        cur_seg = t // chunk
        stats = structural_stats_for_sample(cur_seg, ann["needles"], ann["gold_needles"])
        if stats is None:
            continue
        n_eligible += 1
        if stats["needle_null_hit2"] is not None:
            null_vals.append(stats["needle_null_hit2"])
        ceil_vals.append(stats["structural_ceiling"])

    needle_null_hit2 = sum(null_vals) / len(null_vals) if null_vals else None
    structural_ceiling = sum(ceil_vals) / len(ceil_vals) if ceil_vals else None
    return {
        "task": task, "length": length,
        "n_total": len(rows), "n_eligible_samples": n_eligible,
        "needle_null_hit2": needle_null_hit2,
        "needle_null_hit2_formula": (
            "mean_over_eligible_samples(min(1, 2/M)), M = count of eligible "
            "(seg < cur_seg) segments containing ANY needle (any key) -- "
            "the 'perfect needle detector, zero key discrimination' null"),
        "structural_ceiling": structural_ceiling,
        "structural_ceiling_formula": (
            "mean_over_eligible_samples(1.0 if G<=2 else "
            "top2_needle_counts_sum/n_e), G = count of distinct eligible "
            "GOLD (queried) segments, n_e = count of eligible queried "
            "needle instances -- max achievable macro hit@2 given routing "
            "scores are computed once per sample (single shared top-2, not "
            "independent per needle) and possible needle-segment collisions "
            "(plausible here: cur_seg is often only 4-7 at length=2048, a "
            "mini birthday-paradox regime for 4 needles)"),
    }


def main():
    import argparse
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "x2_structural.json"))
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(mcdata.TOKENIZER)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import x2_probe as xp
    out = {}
    for task, length in xp.all_combos():
        agg = compute_dataset_structural(task, length, tok)
        out.setdefault(task, {})[str(length)] = agg
        print(f"[x2_null] {task}@{length}: needle_null_hit2={agg['needle_null_hit2']} "
              f"structural_ceiling={agg['structural_ceiling']} "
              f"n_eligible={agg['n_eligible_samples']}/{agg['n_total']}")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[x2_null] wrote {a.out}")


if __name__ == "__main__":
    main()
