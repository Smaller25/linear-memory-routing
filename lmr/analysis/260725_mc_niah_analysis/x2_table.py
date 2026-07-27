"""X2 (Task 3) output: results/x2_table.md (task x length x model) + optional
layer-curve figure @2048 + results/x2_verdict.json. CPU-only, reads:
  - results/x2_multiquery.json  (x2_probe.py, VESSL)
  - results/e1_routing.json     (niah_single_1 reference — NOISE-haystack,
                                  see corrections note below)
  - results/x2_structural.json  (x2_null.py — needle_null_hit2/ceiling,
                                  model-independent)
  - results/x2_per_sample.json  (optional, not committed — per-sample
                                  cur_seg/k/per-layer hitk, used only to
                                  compute the cur_seg>k-restricted hit@k
                                  and saturation-fraction columns; those
                                  columns are omitted gracefully if absent)
No hardcoded numbers — every value in the table/figure traces back to a
JSON file (spec §7: "모든 수치는 JSON에 먼저 쓰고 보고서는 JSON만 참조").

--- Adversarial-review corrections (round 2) baked into this version ---
1. niah_single_1 is labeled NOISE-haystack in every row it appears in — it
   is NOT directly comparable to the essay-haystack X2 tasks. niah_single_2
   (essay-haystack, 1 needle/1 query, added this round) is the clean
   single-needle control.
2. needle_null_hit2 ("perfect needle detector, zero key discrimination")
   and structural_ceiling (max achievable macro hit@2 given one shared
   top-2 per sample) columns added, plus ceiling-normalized skill =
   (obs - chance) / (ceiling - chance).
3. hit@k is also reported restricted to the cur_seg > k subset (a large
   fraction of rows are structurally saturated at cur_seg <= k, where
   hit@k is trivially ~1 by construction — disclosed via frac_saturated).
4. The "monotonic decline" framing (multikey k=1 -> q2 k=2 -> multiquery
   k=4) is dropped — it isn't monotone in the raw numbers, and the q=2
   sweep changes total needle COUNT IN CONTEXT (2, not 4) alongside k, so
   it isn't a clean k-only manipulation; noted in the table preamble.
5. "clears the noise floor Nx" language is replaced with sampling-SE
   framing (see se_hit2 column: sqrt(p(1-p)/n)) plus an explicit note that
   best-layer selection (picking the max over 16 layers) is upward-biased.
6. The multiquery-vs-multikey ordering at 8K is reported as observed
   per-model, not asserted as a universal reversal.
"""
import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# display label overrides — item (iv): haystack type must be visible at the
# point of comparison, not just in a footnote.
TASK_LABELS = {
    "niah_single_1 (E1 ref)": "niah_single_1 (E1 ref, NOISE-haystack — not directly comparable)",
    "niah_single_2": "niah_single_2 (essay-haystack, single-needle control)",
    "niah_multiquery_q2": "niah_multiquery_q2 (q=2 sweep — ALSO only 2 needles in context, not 4)",
}


def _label(task):
    return TASK_LABELS.get(task, task)


def _fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def _fmt_ratio(x):
    return "-" if x is None else f"{x:.2f}x"


def _fmt_pp(x):
    return "-" if x is None else f"{x * 100:.1f}pp"


def load(res_path):
    return json.load(open(res_path))


def _ratio(obs, chance):
    if obs is None or chance is None or chance == 0:
        return None
    return obs / chance


def _se(p, n):
    """Sampling standard error of a proportion, sqrt(p(1-p)/n). Item (v):
    replaces the earlier "clears the noise floor Nx" framing. This is an
    approximation (treats the two-level macro average as a simple
    proportion at the reported n_eligible), adequate for the directional
    "is this gap plausibly noise" read the numbers are used for here."""
    if p is None or n is None or n <= 0:
        return None
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def _restricted_hitk(per_sample_task_len, best_layer, k):
    """Item (ii): hit@k restricted to samples where cur_seg > k (i.e. NOT
    structurally saturated — where selecting all cur_seg past segments
    already satisfies "top-k" trivially, so hit@k=1 by construction
    regardless of routing quality). Also recomputes chance_hitk over the
    SAME restricted subset (k/cur_seg, no min(1,...) cap needed since
    cur_seg>k there) — comparing the restricted observed value against the
    *unrestricted* chance would be an apples-to-oranges mismatch, since
    chance is uniformly lower in the restricted subset too. Returns
    (macro_hitk_restricted, chance_hitk_restricted, frac_saturated,
    n_restricted, n_total_eligible), all-None if per_sample_task_len is
    falsy (per-sample file absent)."""
    if not per_sample_task_len:
        return None, None, None, None, None
    restricted_vals, restricted_chance = [], []
    n_saturated = 0
    for rec in per_sample_task_len:
        if rec.get("per_layer") is None:
            continue
        if rec["cur_seg"] <= rec["k"]:
            n_saturated += 1
            continue
        restricted_chance.append(rec["k"] / rec["cur_seg"])
        layer_rec = next((pl for pl in rec["per_layer"] if pl["layer"] == best_layer), None)
        if layer_rec is not None:
            restricted_vals.append(layer_rec["macro_hitk"])
    n_eligible = sum(1 for r in per_sample_task_len if r.get("per_layer") is not None)
    frac_saturated = n_saturated / n_eligible if n_eligible else None
    macro_hitk_restricted = (sum(restricted_vals) / len(restricted_vals)
                             if restricted_vals else None)
    chance_hitk_restricted = (sum(restricted_chance) / len(restricted_chance)
                              if restricted_chance else None)
    return (macro_hitk_restricted, chance_hitk_restricted, frac_saturated,
            len(restricted_vals), n_eligible)


def build_rows(x2, e1, structural, per_sample):
    """One row per (task, length, model). Adds niah_single_1 (E1 ref,
    NOISE-haystack) from e1_routing.json, 2048 only."""
    rows = []
    models = x2["meta"]["models"]
    for model in models:
        by_task = x2["results"].get(model, {})
        for task, by_len in by_task.items():
            for length_s, agg in by_len.items():
                length = int(length_s)
                struct = (structural or {}).get(task, {}).get(length_s)
                ps = (per_sample or {}).get(model, {}).get(task, {}).get(length_s)
                # k is constant per dataset (n_queried) — read it off any
                # per-sample record rather than threading it separately.
                k_val = ps[0]["k"] if ps else None
                hitk_restricted, chance_hitk_restricted, frac_sat, n_restr, n_hitk_eligible = (
                    _restricted_hitk(ps, agg["best_layer"], k_val) if (ps and k_val is not None)
                    else (None, None, None, None, None))
                rows.append({
                    "task": task, "length": length, "model": model,
                    "best_layer": agg["best_layer"],
                    "macro_hit2": agg["best_layer_macro_hit2"],
                    "chance_hit2": agg["chance_hit2"],
                    "needle_null_hit2": struct["needle_null_hit2"] if struct else None,
                    "structural_ceiling": struct["structural_ceiling"] if struct else None,
                    "macro_hitk": agg["best_layer_macro_hitk"],
                    "chance_hitk": agg["chance_hitk"],
                    "macro_hitk_restricted": hitk_restricted,
                    "chance_hitk_restricted": chance_hitk_restricted,
                    "frac_saturated_hitk": frac_sat,
                    "n_eligible": agg["n_eligible_samples"], "n_total": agg["n_total"],
                })
    if e1:
        for model in models:
            ds = e1.get("results", {}).get(model, {}).get("niah_single_1")
            if ds:
                rows.append({
                    "task": "niah_single_1 (E1 ref)", "length": 2048, "model": model,
                    "best_layer": ds["best_layer"],
                    "macro_hit2": ds["best_layer_hit_at_2"], "chance_hit2": ds["chance_hit2"],
                    "needle_null_hit2": None, "structural_ceiling": None,
                    "macro_hitk": None, "chance_hitk": None,
                    "macro_hitk_restricted": None, "chance_hitk_restricted": None,
                    "frac_saturated_hitk": None,
                    "n_eligible": ds["n_eligible"], "n_total": ds["n_total"],
                })
    return rows


def make_table_md(rows):
    order_task = {"niah_single_1 (E1 ref)": 0, "niah_single_2": 1, "niah_multiquery": 2,
                 "niah_multiquery_q2": 3, "niah_multivalue": 4, "niah_multikey_1": 5}
    rows_sorted = sorted(rows, key=lambda r: (r["length"], order_task.get(r["task"], 99),
                                              r["model"]))
    lines = [
        "# X2 results: multi-query/multivalue per-key routing hit@2",
        "",
        "**Corrections round 2 (adversarial review)**: niah_single_1 is NOISE-haystack "
        "(not directly comparable to the essay-haystack tasks below) — niah_single_2 "
        "(essay-haystack, single-needle) is the clean control. The \"monotonic decline\" "
        "claim across the needle-count sweep is dropped (not monotone in the raw "
        "numbers; the q=2 sweep also changes total needle count IN CONTEXT, not just "
        "queried count, so it isn't a clean k-only manipulation). \"Clears the noise "
        "floor Nx\" language is replaced by sampling SE (`se_hit2` = sqrt(p(1-p)/n) at "
        "each row's own n_eligible); best-layer selection (max over 16 layers) is "
        "upward-biased, so treat se_hit2 as a lower bound on the true uncertainty.",
        "",
        "## Table 1 — hit@2, chance, needle-null, structural ceiling, skill",
        "",
        "`needle_null_hit2` = reviewer's stronger null: top-2 uniform over "
        "*needle-bearing* eligible segments (any key), not all past segments — "
        "the \"perfect needle detector, zero key discrimination\" baseline. "
        "`ceiling` = max achievable macro hit@2 given ONE shared top-2 per sample "
        "(routing score is computed once per sample, not independently per needle) "
        "— can exceed 2/(#queried needles) when needles happen to collide into the "
        "same segment (plausible here: cur_seg is often only 4-7 at length=2048). "
        "`skill` = (macro_hit2 - chance_hit2) / (ceiling - chance_hit2) — 0 means "
        "no better than chance, 1 means saturating the structural ceiling. "
        "`se_hit2` = sqrt(p(1-p)/n_eligible), a rough sampling SE for macro_hit2.",
        "",
        "| task | length | model | layer | hit@2 | chance | needle-null | ceiling | "
        "skill | hit2/chance | se_hit2 | n (elig/tot) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows_sorted:
        r2 = _ratio(r["macro_hit2"], r["chance_hit2"])
        skill = None
        if r["structural_ceiling"] is not None and r["macro_hit2"] is not None:
            denom = r["structural_ceiling"] - r["chance_hit2"]
            skill = (r["macro_hit2"] - r["chance_hit2"]) / denom if denom > 1e-9 else None
        se = _se(r["macro_hit2"], r["n_eligible"])
        lines.append(
            f"| {_label(r['task'])} | {r['length']} | {r['model']} | {r['best_layer']} | "
            f"{_fmt(r['macro_hit2'])} | {_fmt(r['chance_hit2'])} | "
            f"{_fmt(r['needle_null_hit2'])} | {_fmt(r['structural_ceiling'])} | "
            f"{_fmt(skill, 2)} | {_fmt_ratio(r2)} | {_fmt_pp(se)} | "
            f"{r['n_eligible']}/{r['n_total']} |"
        )
    lines += [
        "",
        "## Table 2 — hit@k detail (k = number of queried needles) and saturation",
        "",
        "`hit@k (all)` uses every eligible sample at that row's best layer (same as "
        "Table 1's companion column in the previous round). `hit@k (cur_seg>k only)` "
        "restricts to samples where cur_seg > k — i.e. NOT structurally saturated "
        "(when cur_seg <= k, selecting the top-k literally selects every available "
        "past segment, so hit@k is trivially forced high regardless of routing "
        "quality). `frac_saturated` discloses what fraction of eligible samples were "
        "in that trivial regime — item (ii) of the corrections: a large fraction of "
        "multiquery@2048 rows are saturated (cur_seg=4=k), so the unrestricted hit@k "
        "column overstates the routing signal at k.",
        "",
        "| task | length | model | hit@k (all) | chance hit@k (all) | "
        "hit@k (cur_seg>k only) | chance hit@k (cur_seg>k only) | frac_saturated |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows_sorted:
        if r["macro_hitk"] is None:
            continue
        # k isn't stored directly on the row (constant per task: 4 for
        # multiquery/multivalue, 2 for q2, 1 for multikey_1/single_2/
        # single_1) so it's shown via the task label, not a numeric column.
        lines.append(
            f"| {_label(r['task'])} | {r['length']} | {r['model']} | "
            f"{_fmt(r['macro_hitk'])} | {_fmt(r['chance_hitk'])} | "
            f"{_fmt(r['macro_hitk_restricted'])} | {_fmt(r['chance_hitk_restricted'])} | "
            f"{_fmt(r['frac_saturated_hitk'], 2)} |"
        )
    return "\n".join(lines) + "\n"


def make_figure(x2, out_paths, length=2048):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = x2["meta"]["models"]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, models):
        by_task = x2["results"].get(model, {})
        for task, by_len in by_task.items():
            agg = by_len.get(str(length))
            if not agg:
                continue
            layers = [pl["layer"] for pl in agg["per_layer"]]
            hits = [pl["macro_hit2"] if pl["macro_hit2"] is not None else float("nan")
                    for pl in agg["per_layer"]]
            ax.plot(layers, hits, marker="o", label=task)
        ax.set_title(f"{model} @ {length}")
        ax.set_xlabel("layer")
        ax.set_ylabel("macro hit@2")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=150)
    plt.close(fig)


def make_verdict_json(rows):
    """Derived comparison numbers used for the (M) call, at the spec's
    primary comparison length (2048), corrected per adversarial review
    round 2: adds needle_null_hit2/structural_ceiling/skill alongside the
    plain chance ratio, and includes niah_single_2 (essay control)."""
    by_key = {(r["task"], r["length"], r["model"]): r for r in rows}
    out = {"length": 2048, "models": sorted({r["model"] for r in rows}), "rows": {}}
    for model in out["models"]:
        entry = {}
        for task in ("niah_single_1 (E1 ref)", "niah_single_2", "niah_multiquery",
                     "niah_multiquery_q2", "niah_multivalue", "niah_multikey_1"):
            r = by_key.get((task, 2048, model))
            if not r:
                continue
            skill = None
            if r["structural_ceiling"] is not None and r["macro_hit2"] is not None:
                denom = r["structural_ceiling"] - r["chance_hit2"]
                skill = (r["macro_hit2"] - r["chance_hit2"]) / denom if denom > 1e-9 else None
            entry[task] = {
                "macro_hit2": r["macro_hit2"], "chance_hit2": r["chance_hit2"],
                "hit2_over_chance": _ratio(r["macro_hit2"], r["chance_hit2"]),
                "needle_null_hit2": r["needle_null_hit2"],
                "structural_ceiling": r["structural_ceiling"],
                "ceiling_normalized_skill": skill,
                "se_hit2": _se(r["macro_hit2"], r["n_eligible"]),
                "macro_hitk": r["macro_hitk"], "chance_hitk": r["chance_hitk"],
                "hitk_over_chance": _ratio(r["macro_hitk"], r["chance_hitk"]),
                "macro_hitk_restricted_cur_seg_gt_k": r["macro_hitk_restricted"],
                "chance_hitk_restricted_cur_seg_gt_k": r["chance_hitk_restricted"],
                "hitk_restricted_over_chance": _ratio(r["macro_hitk_restricted"],
                                                      r["chance_hitk_restricted"]),
                "frac_saturated_hitk": r["frac_saturated_hitk"],
                "n_eligible": r["n_eligible"], "n_total": r["n_total"],
            }
        out["rows"][model] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x2-json", default=os.path.join(RES, "x2_multiquery.json"))
    ap.add_argument("--e1-json", default=os.path.join(RES, "e1_routing.json"))
    ap.add_argument("--structural-json", default=os.path.join(RES, "x2_structural.json"))
    ap.add_argument("--per-sample-json", default=os.path.join(RES, "x2_per_sample.json"))
    ap.add_argument("--out-md", default=os.path.join(RES, "x2_table.md"))
    ap.add_argument("--out-fig", default=os.path.join(RES, "x2_layers_2048.png"))
    ap.add_argument("--out-verdict-json", default=os.path.join(RES, "x2_verdict.json"))
    a = ap.parse_args()

    x2 = load(a.x2_json)
    e1 = load(a.e1_json) if os.path.exists(a.e1_json) else None
    structural = load(a.structural_json) if os.path.exists(a.structural_json) else None
    per_sample = load(a.per_sample_json) if os.path.exists(a.per_sample_json) else None
    if per_sample is None:
        print("[x2_table][warn] per-sample JSON not found — hit@k restricted/"
              "frac_saturated columns will be empty ('-')")

    rows = build_rows(x2, e1, structural, per_sample)
    md = make_table_md(rows)
    with open(a.out_md, "w") as f:
        f.write(md)
    print(f"[x2_table] wrote {a.out_md}")

    make_figure(x2, [a.out_fig])
    print(f"[x2_table] wrote {a.out_fig}")

    verdict = make_verdict_json(rows)
    with open(a.out_verdict_json, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"[x2_table] wrote {a.out_verdict_json}")


if __name__ == "__main__":
    main()
