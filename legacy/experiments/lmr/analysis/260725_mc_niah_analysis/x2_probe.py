"""X2 (Task 3): multi-query/multivalue per-key routing hit@2 — the Tier 1
test of claim (M) "router is a needle detector, not a key discriminator"
(spec §3, docs/superpowers/specs/2026-07-27-0025-router-diagnosis-spec.md).

Reuses the E1 machinery verbatim (routing_stats.capture_hidden +
routing_scores_at — same score formula, same answer-position-only forward
pass, 0 extra forward passes) but scores against MULTIPLE gold needles per
sample instead of one:

  - niah_multiquery : num_needle_k clamped to 4 (== num_needle_q), all 4
    keys queried -> 4 distinct-key needles, all gold.
  - niah_multivalue : 1 key, 4 value-needles (same key, 4 sentences) -> 4
    same-key needles, all gold.
  - niah_multiquery_q2 (2048 only): needle-count sweep, num_needle_q=2.
  - niah_multikey_1  : matched control — same needle *count* in-context (4
    keys) but only 1 queried -> 1 gold needle (this is what E1 already
    reports as "niah_multikey_1"; recomputed here at all 3 lengths with the
    exact same per-needle machinery so the X2 table is apples-to-apples).

Metrics (spec §3 + §7, NOT final-answer accuracy — routing-level only):
  - macro hit@2: per sample, mean over ELIGIBLE queried needles of
    "needle's seg in top-2(routing scores)"; then mean over samples
    (two-level macro, so samples with more queried needles don't dominate).
  - auxiliary macro hit@k, k := number of queried needles in that sample
    (inference-time-only: same scores, just a wider top-k slice). k is
    clipped to cur_seg (can't select more distinct past segments than
    exist) when forming the hit set, but the chance formula below uses the
    unclipped k (matches spec: chance = k/n_past, capped at 1.0 when
    k >= n_past).
  - eligibility: a queried needle is eligible iff its segment < cur_seg
    (t=T-1's own segment). Ineligible needles are excluded from the
    numerator and counted (n_ineligible_needles) — never silently dropped.
  - chance (re-derived per spec §7, multi-gold case):
      chance_hit2(sample)  = min(1, 2 / cur_seg)   [same for every needle
                              in that sample since cur_seg is sample-level]
      chance_hitk(sample)  = min(1, k / cur_seg)
    dataset chance = mean over eligible samples of the per-sample value.

Resumable: results/x2_multiquery.json is loaded first; any (model, task,
length) combo already present with n_total == len(rows) on disk is skipped
entirely (no model forward passes wasted). Saved after every (task, length)
combo finishes, so a VESSL container restart mid-run loses at most one
in-flight combo.
"""
import argparse, json, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_mc, data as mcdata
from routing_stats import capture_hidden, routing_scores_at

MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CHUNK, TOPK = load_mc.CHUNK, load_mc.TOPK

LENGTHS = (2048, 4096, 8192)
CORE_TASKS = ("niah_multiquery", "niah_multivalue", "niah_multikey_1")
# needle-count sweep (q=2) + niah_single_2 (essay-haystack, 1 needle/1 query —
# the clean single-needle control; niah_single_1 in E1 is noise-haystack, a
# confound for comparing against the essay-haystack X2 tasks — see adversarial
# review / task-3-report.md "single_2" section), both 2048-only.
EXTRA_TASKS_BY_LENGTH = {2048: ("niah_multiquery_q2", "niah_single_2")}


def _dataset_path(task, length):
    return os.path.join(MC_OUT, "data", str(length), task, "validation.jsonl")


def _rows_for(task, length):
    path = _dataset_path(task, length)
    return [json.loads(l) for l in open(path)]


def all_combos():
    combos = []
    for length in LENGTHS:
        for task in CORE_TASKS:
            combos.append((task, length))
        for task in EXTRA_TASKS_BY_LENGTH.get(length, ()):
            combos.append((task, length))
    return combos


@torch.no_grad()
def analyze_sample_multi(model, tok, input_text, layers):
    """Multi-gold analog of routing_stats.analyze_sample. Returns per-sample
    dict with per_layer:[{layer, hit2:[bool per eligible needle],
    hitk:[bool per eligible needle]}] plus eligibility/k bookkeeping."""
    ann = mcdata.annotate(input_text, tok)
    ids = torch.tensor([tok(input_text, add_special_tokens=False).input_ids],
                       device="cuda")
    T = ids.shape[1]
    t = T - 1
    cur_seg = t // CHUNK
    gold_needles = ann["gold_needles"]
    n_queried = len(gold_needles)
    eligible_needles = [n for n in gold_needles if n["seg"] < cur_seg]
    n_eligible = len(eligible_needles)
    n_ineligible = n_queried - n_eligible
    k = n_queried  # spec §3: "k를 needle 수에 맞춰 올린 조건"
    topk_eff = min(k, cur_seg) if cur_seg > 0 else 0

    hiddens = capture_hidden(model, ids)
    per_layer = []
    for i, attn in layers:
        s = routing_scores_at(attn, hiddens[i], t)
        order = torch.argsort(s, descending=True).tolist()
        top2 = set(order[:TOPK])
        topk_set = set(order[:topk_eff]) if topk_eff > 0 else set()
        hit2 = [nd["seg"] in top2 for nd in eligible_needles]
        hitk = [nd["seg"] in topk_set for nd in eligible_needles]
        per_layer.append({"layer": i, "hit2": hit2, "hitk": hitk})
    return {"cur_seg": cur_seg, "n_seg": ann["n_seg"], "n_queried": n_queried,
            "n_eligible_needles": n_eligible, "n_ineligible_needles": n_ineligible,
            "k": k, "topk_eff": topk_eff, "per_layer": per_layer}


def aggregate_dataset(task, length, n_rows, n_layers, sample_results):
    """Pure (no GPU/model) aggregation: sample_results is the list of
    per-sample `res` dicts as returned by analyze_sample_multi (or an
    equivalent synthetic construction — see tests/lmr/test_mc_niah_x2.py).
    Kept separate from GPU sample generation so the macro/chance math is
    unit-testable on CPU, mirroring the x1_dump.py (GPU) / x1_analyze.py
    (CPU) split used for X1."""
    layer_hit2 = {i: [] for i in range(n_layers)}   # per-sample macro hit2, one layer
    layer_hitk = {i: [] for i in range(n_layers)}
    chance2_list, chancek_list = [], []
    per_sample_out = []
    n_eligible_samples = 0

    for fallback_idx, res in enumerate(sample_results):
        # sample_idx must be the TRUE row index within the dataset file, not
        # the position within `sample_results` — run_dataset skips failed
        # samples before this list is built, so enumerate() position and
        # true row index diverge as soon as any sample upstream fails
        # (review-flagged misalignment). run_dataset stamps res["row_idx"]
        # with the true index; fall back to enumerate() only for callers
        # (e.g. synthetic tests) that construct `res` dicts without it.
        sample_idx = res.get("row_idx", fallback_idx)
        rec = {"sample_idx": sample_idx, "cur_seg": res["cur_seg"], "n_seg": res["n_seg"],
              "n_queried": res["n_queried"], "n_eligible_needles": res["n_eligible_needles"],
              "n_ineligible_needles": res["n_ineligible_needles"], "k": res["k"]}
        if res["n_eligible_needles"] > 0 and res["cur_seg"] > 0:
            n_eligible_samples += 1
            chance2_list.append(min(1.0, 2.0 / res["cur_seg"]))
            chancek_list.append(min(1.0, res["k"] / res["cur_seg"]))
            layer_recs = []
            for pl in res["per_layer"]:
                m2 = sum(pl["hit2"]) / len(pl["hit2"])
                mk = sum(pl["hitk"]) / len(pl["hitk"])
                layer_hit2[pl["layer"]].append(m2)
                layer_hitk[pl["layer"]].append(mk)
                layer_recs.append({"layer": pl["layer"], "macro_hit2": m2, "macro_hitk": mk})
            rec["per_layer"] = layer_recs
        else:
            rec["per_layer"] = None
        per_sample_out.append(rec)

    per_layer_agg = []
    best_layer, best_val = None, -1.0
    for i in range(n_layers):
        vals2, valsk = layer_hit2[i], layer_hitk[i]
        m2 = sum(vals2) / len(vals2) if vals2 else None
        mk = sum(valsk) / len(valsk) if valsk else None
        per_layer_agg.append({"layer": i, "macro_hit2": m2, "macro_hitk": mk,
                              "n_samples": len(vals2)})
        if m2 is not None and m2 > best_val:
            best_val, best_layer = m2, i
    chance_hit2 = sum(chance2_list) / len(chance2_list) if chance2_list else None
    chance_hitk = sum(chancek_list) / len(chancek_list) if chancek_list else None

    out = {
        "task": task, "length": length,
        "n_total": n_rows, "n_eligible_samples": n_eligible_samples,
        "n_ineligible_samples": n_rows - n_eligible_samples,
        "per_layer": per_layer_agg,
        "best_layer": best_layer,
        "best_layer_macro_hit2": best_val if best_layer is not None else None,
        "best_layer_macro_hitk": (per_layer_agg[best_layer]["macro_hitk"]
                                  if best_layer is not None else None),
        "chance_hit2": chance_hit2,
        "chance_hit2_formula": "mean_over_eligible_samples(min(1, 2/cur_seg))",
        "chance_hitk": chance_hitk,
        "chance_hitk_formula": "mean_over_eligible_samples(min(1, k/cur_seg)), k=n_queried_needles",
    }
    return out, per_sample_out


def run_dataset(model, tok, layers, task, length):
    rows = _rows_for(task, length)
    n_layers = len(layers)
    sample_results = []
    n_failed = 0
    for ri, r in enumerate(rows):
        try:
            res = analyze_sample_multi(model, tok, r["input"], layers)
        except Exception as e:
            n_failed += 1
            print(f"[x2][warn] {task}/{length} sample {ri} failed: {e}", flush=True)
            continue
        res["row_idx"] = ri
        sample_results.append(res)
        print(f"[x2] {task}/{length} {ri+1}/{len(rows)} "
              f"n_eligible={res['n_eligible_needles']}/{res['n_queried']} "
              f"cur_seg={res['cur_seg']}", flush=True)
    out, per_sample_out = aggregate_dataset(task, length, len(rows), n_layers, sample_results)
    out["n_failed"] = n_failed
    return out, per_sample_out


def _load_existing(path):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            return None
    return None


def _already_done(existing, model, task, length):
    if not existing:
        return False
    try:
        entry = existing["results"][model][task][str(length)]
    except (KeyError, TypeError):
        return False
    expected_n = len(_rows_for(task, length))
    return entry.get("n_total") == expected_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mc-5B", "mc-30B"])
    ap.add_argument("--force", action="store_true", help="recompute even if already present")
    a = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    os.makedirs(os.path.join(MC_OUT, "results"), exist_ok=True)

    out_paths = [os.path.join(MC_OUT, "results", "x2_multiquery.json"),
                os.path.join(RES, "x2_multiquery.json")]
    per_sample_paths = [os.path.join(MC_OUT, "results", "x2_per_sample.json"),
                        os.path.join(RES, "x2_per_sample.json")]

    results = {}
    for p in out_paths:
        prev = _load_existing(p)
        if prev:
            for m, by_task in prev.get("results", {}).items():
                results.setdefault(m, {})
                for t, by_len in by_task.items():
                    results[m].setdefault(t, {})
                    results[m][t].update(by_len)

    per_sample_all = {}
    for p in per_sample_paths:
        if os.path.exists(p):
            try:
                per_sample_all = json.load(open(p))
                break
            except Exception:
                pass

    combos = all_combos()
    todo = [(t, l) for (t, l) in combos
            if a.force or not _already_done({"results": results}, a.model, t, l)]
    print(f"[x2] {a.model}: {len(todo)}/{len(combos)} (task,length) combos to run "
          f"(skip existing unless --force)", flush=True)

    if not todo:
        print(f"[x2] {a.model}: nothing to do, all combos already present", flush=True)
        return

    tok = load_mc.load_tokenizer()
    model = load_mc.load_model(a.model)
    layers = load_mc.mc_layers(model)
    print(f"[x2] {a.model}: n_layers={len(layers)}", flush=True)

    results.setdefault(a.model, {})
    per_sample_all.setdefault(a.model, {})

    for task, length in todo:
        print(f"[x2] === {a.model} / {task} @ {length} ===", flush=True)
        agg, per_sample = run_dataset(model, tok, layers, task, length)
        results[a.model].setdefault(task, {})[str(length)] = agg
        per_sample_all[a.model].setdefault(task, {})[str(length)] = per_sample

        out = {"meta": {"topk": TOPK, "chunk": CHUNK, "lengths": list(LENGTHS),
                        "core_tasks": list(CORE_TASKS),
                        "extra_tasks_by_length": EXTRA_TASKS_BY_LENGTH,
                        "models": sorted(results.keys()),
                        "best_layer_tiebreak": (
                            "strict '>' scan over layers 0..n_layers-1 in ascending "
                            "order against macro_hit2 (best_val initialized -1.0) -> "
                            "on a tie the LOWEST-index layer wins (first strict max, "
                            "later equal values do not replace it). Identical algorithm "
                            "to E1's routing_stats.run_model best-layer selection (same "
                            "strict '>' / ascending-index scan there too) -- verified by "
                            "diffing E1's and X2's independently-recomputed per-layer "
                            "niah_multikey_1/mc-30B hit@2 arrays layer-by-layer: 14/16 "
                            "layers are bit-for-bit identical between the two runs, and "
                            "the other 2 (layers 4 and 6) differ by EXACTLY 1/47 samples "
                            "each (a single sample's hit/miss flipped between the two "
                            "independent bf16 forward passes -- at the declared ~2.3pp "
                            "bf16 noise floor, 1/47=2.13pp). That single flip was enough "
                            "to raise layer 4 from 0.787 (below layer 14's 0.809, so E1 "
                            "picked layer 14) up to a tie with layer 14 at 0.809 in X2's "
                            "run -- and since 4<14, the identical first-strict-max "
                            "tie-break then picked layer 4 instead. IMPORTANT: this is "
                            "NOT a tie-break algorithm discrepancy (the algorithm agrees "
                            "in both runs, as the layer-by-layer diff shows) -- it is the "
                            "tie-break correctly and deterministically resolving a NEW "
                            "near-tie that bf16 noise created between two runs of "
                            "nominally the same computation. Practical caution: with "
                            "n=47-50 per cell, best_layer INDICES are noise-unstable "
                            "(this one flipped 14->4 on a single-sample perturbation) and "
                            "should not be over-interpreted as a fixed per-model/per-task "
                            "identity -- report the associated hit@2/skill VALUE, which is "
                            "far more stable than which specific layer index attains it.")},
               "results": results}
        for p in out_paths:
            json.dump(out, open(p, "w"), indent=2)
        for p in per_sample_paths:
            json.dump(per_sample_all, open(p, "w"))
        print(f"[x2] {a.model}/{task}@{length}: best_layer={agg['best_layer']} "
              f"macro_hit2={agg['best_layer_macro_hit2']} "
              f"chance_hit2={agg['chance_hit2']} "
              f"n_eligible={agg['n_eligible_samples']}/{agg['n_total']} "
              f"-> saved", flush=True)

    del model
    torch.cuda.empty_cache()
    print(f"[x2] {a.model}: ALL DONE", flush=True)


if __name__ == "__main__":
    main()
