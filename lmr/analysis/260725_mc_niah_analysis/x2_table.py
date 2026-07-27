"""X2 (Task 3) output: results/x2_table.md (task x length x model) + optional
layer-curve figure @2048. CPU-only, reads results/x2_multiquery.json (written
by x2_probe.py on VESSL) and results/e1_routing.json (for the niah_single_1
best-layer hit@2 reference, per the brief: "single reference = existing
e1_routing.json niah_single_1 best hit@2"). No hardcoded numbers — every
value in the table/figure traces back to the JSON files (spec §7: "모든
수치는 JSON에 먼저 쓰고 보고서는 JSON만 참조").
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def _fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def _fmt_ratio(x):
    return "-" if x is None else f"{x:.2f}x"


def load(res_path):
    return json.load(open(res_path))


def build_rows(x2, e1):
    """One row per (task, length, model). Adds a synthetic 'niah_single_1'
    row (2048 only) pulled straight from e1_routing.json as the (M) single
    reference, so the verdict comparison lives in the same table."""
    rows = []
    models = x2["meta"]["models"]
    for model in models:
        by_task = x2["results"].get(model, {})
        for task, by_len in by_task.items():
            for length_s, agg in by_len.items():
                rows.append({
                    "task": task, "length": int(length_s), "model": model,
                    "macro_hit2": agg["best_layer_macro_hit2"],
                    "macro_hitk": agg["best_layer_macro_hitk"],
                    "best_layer": agg["best_layer"],
                    "chance_hit2": agg["chance_hit2"],
                    "chance_hitk": agg["chance_hitk"],
                    "n_eligible": agg["n_eligible_samples"], "n_total": agg["n_total"],
                })
    # single reference (spec: "single reference = existing e1_routing.json
    # niah_single_1 best hit@2"), 2048 only — e1 didn't run at 4096/8192.
    if e1:
        for model in models:
            ds = e1.get("results", {}).get(model, {}).get("niah_single_1")
            if ds:
                rows.append({
                    "task": "niah_single_1 (E1 ref)", "length": 2048, "model": model,
                    "macro_hit2": ds["best_layer_hit_at_2"], "macro_hitk": None,
                    "best_layer": ds["best_layer"],
                    "chance_hit2": ds["chance_hit2"], "chance_hitk": None,
                    "n_eligible": ds["n_eligible"], "n_total": ds["n_total"],
                })
    return rows


def _ratio(obs, chance):
    if obs is None or chance is None or chance == 0:
        return None
    return obs / chance


def make_table_md(rows):
    order_task = {"niah_single_1 (E1 ref)": 0, "niah_multiquery": 1,
                 "niah_multiquery_q2": 2, "niah_multivalue": 3, "niah_multikey_1": 4}
    rows_sorted = sorted(rows, key=lambda r: (r["length"], order_task.get(r["task"], 99),
                                              r["model"]))
    lines = [
        "# X2 results: multi-query/multivalue per-key routing hit@2",
        "",
        "task × length × model. `macro_hit2`/`macro_hitk` are best-layer values "
        "(layer chosen by highest macro_hit2 for that task/length/model); "
        "`k` for hit@k = number of queried needles in that task "
        "(4 for multiquery/multivalue, 2 for multiquery_q2, 1 for multikey_1/single). "
        "chance columns are re-derived per §7 (multi-gold: mean over eligible "
        "samples of min(1, m/cur_seg), m=2 or k). "
        "**`hit2/chance` is the key comparison column** — task chance levels differ "
        "substantially (more needle sentences -> smaller haystack budget -> fewer "
        "past segments -> higher chance), so raw macro_hit2 is not directly "
        "comparable across tasks; the ratio to that task's own chance is.",
        "",
        "| task | length | model | best layer | macro hit@2 | chance hit@2 | "
        "hit2/chance | macro hit@k | chance hit@k | hitk/chance | n (eligible/total) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows_sorted:
        r2 = _ratio(r["macro_hit2"], r["chance_hit2"])
        rk = _ratio(r["macro_hitk"], r["chance_hitk"])
        lines.append(
            f"| {r['task']} | {r['length']} | {r['model']} | {r['best_layer']} | "
            f"{_fmt(r['macro_hit2'])} | {_fmt(r['chance_hit2'])} | {_fmt_ratio(r2)} | "
            f"{_fmt(r['macro_hitk'])} | {_fmt(r['chance_hitk'])} | {_fmt_ratio(rk)} | "
            f"{r['n_eligible']}/{r['n_total']} |"
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
    """Derived ratio-to-chance numbers used for the (M) call, at the spec's
    primary comparison length (2048). Raw macro_hit2 is NOT directly
    comparable across tasks here — task chance levels differ by up to 1.7x
    (more needle sentences -> smaller haystack budget -> fewer past segments
    -> higher chance) — so the ratio to each task's own chance is the
    apples-to-apples number the verdict is actually based on. Written to its
    own JSON (not folded into x2_multiquery.json, which is the raw
    resumable-merge probe output) so this derived comparison is also
    JSON-first per spec §7, not just narrated in a report."""
    by_key = {(r["task"], r["length"], r["model"]): r for r in rows}
    out = {"length": 2048, "models": sorted({r["model"] for r in rows}), "rows": {}}
    for model in out["models"]:
        entry = {}
        for task in ("niah_single_1 (E1 ref)", "niah_multiquery", "niah_multiquery_q2",
                     "niah_multivalue", "niah_multikey_1"):
            r = by_key.get((task, 2048, model))
            if not r:
                continue
            entry[task] = {
                "macro_hit2": r["macro_hit2"], "chance_hit2": r["chance_hit2"],
                "hit2_over_chance": _ratio(r["macro_hit2"], r["chance_hit2"]),
                "macro_hitk": r["macro_hitk"], "chance_hitk": r["chance_hitk"],
                "hitk_over_chance": _ratio(r["macro_hitk"], r["chance_hitk"]),
            }
        out["rows"][model] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x2-json", default=os.path.join(RES, "x2_multiquery.json"))
    ap.add_argument("--e1-json", default=os.path.join(RES, "e1_routing.json"))
    ap.add_argument("--out-md", default=os.path.join(RES, "x2_table.md"))
    ap.add_argument("--out-fig", default=os.path.join(RES, "x2_layers_2048.png"))
    ap.add_argument("--out-verdict-json", default=os.path.join(RES, "x2_verdict.json"))
    a = ap.parse_args()

    x2 = load(a.x2_json)
    e1 = load(a.e1_json) if os.path.exists(a.e1_json) else None

    rows = build_rows(x2, e1)
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
