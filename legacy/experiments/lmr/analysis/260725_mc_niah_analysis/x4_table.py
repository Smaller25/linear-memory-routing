"""Build results/x4_table.md from results/x4_random_routing.json.

Spec §5/§7: stock / recent / random / vanilla-anchor rows x length columns,
chance_upper + observed/chance ratio columns. All numbers read from the JSON
(no hardcoding) except the vanilla-5B anchor (external reference, from the
collaborator 4-way document cited in the rev1 plan Task 4 preamble; there is
no vanilla-30B anchor, footnoted below).
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768)

# External reference only (different model, different protocol -- see note
# below). Values are EM% at {1K,2K,4K,8K,16K,32K}, from the collaborator
# 4-way document (rev1 plan Task 4 preamble). No vanilla-30B anchor exists.
VANILLA_5B_ANCHOR = {
    "niah_single_1": [100, 90, 54, 16, 4, 0],
    "niah_multikey_1": [18, 20, 12, 4, 2, 2],
}


def _fmt(x, pct=True):
    if x is None:
        return "-"
    return f"{x*100:.1f}" if pct else f"{x:.3f}"


def _fmt_ratio(obs, chance):
    if obs is None or chance is None or chance == 0:
        return "-"
    return f"{obs/chance:.2f}x"


def build_section(model, task, cell_by_length):
    lines = []
    lines.append(f"### {model} / {task}")
    lines.append("")
    header = "| row |" + "".join(f" {l} |" for l in LENGTHS)
    sep = "|---|" + "".join(" ---: |" for _ in LENGTHS)
    lines.append(header)
    lines.append(sep)

    def row(name, get):
        vals = [get(cell_by_length.get(str(l), {})) for l in LENGTHS]
        lines.append(f"| {name} |" + "".join(f" {_fmt(v)} |" for v in vals))

    row("stock (EM%)", lambda c: c.get("stock"))
    row("recent (EM%)", lambda c: c.get("recent"))
    row("random mean (EM%)", lambda c: (c.get("random") or {}).get("mean"))

    # per-seed spread (min-max across the 5 random seeds), context for noise
    per_seed_vals = []
    for l in LENGTHS:
        c = cell_by_length.get(str(l), {})
        seeds = (c.get("random") or {}).get("per_seed")
        if seeds:
            per_seed_vals.append(f"{min(seeds)*100:.1f}-{max(seeds)*100:.1f}")
        else:
            per_seed_vals.append("-")
    lines.append("| random min-max (EM%) |" + "".join(f" {v} |" for v in per_seed_vals))

    if model == "mc-5B" and task in VANILLA_5B_ANCHOR:
        anchor = VANILLA_5B_ANCHOR[task]
        lines.append("| vanilla-5B anchor (EM%, external ref.) |" +
                    "".join(f" {v} |" for v in anchor))

    row("chance_upper (EM%)", lambda c: c.get("chance_upper"))
    lines.append("| stock/chance_upper ratio |" +
                "".join(f" {_fmt_ratio(cell_by_length.get(str(l), {}).get('stock'), cell_by_length.get(str(l), {}).get('chance_upper'))} |"
                        for l in LENGTHS))

    for l in LENGTHS:
        c = cell_by_length.get(str(l), {})
        if "meta" in c:
            lines.append(f"\n_note: length={l} sample count reduced to "
                         f"{c['meta']['num_samples']} ({c['meta']['reason']}, "
                         f"probe={c['meta'].get('probe_seconds_per_sample')}s/sample)_")

    lines.append("")
    return "\n".join(lines)


def main():
    path_repo = os.path.join(RES, "x4_random_routing.json")
    path_mcout = os.path.join(MC_OUT, "results", "x4_random_routing.json")
    path = path_repo if os.path.exists(path_repo) else path_mcout
    obj = json.load(open(path))
    results = obj.get("results", {})
    meta = obj.get("meta", {})

    out = []
    out.append("# X4 — random-routing MC ablation")
    out.append("")
    out.append(f"n_gen={meta.get('n_gen')} chunk={meta.get('chunk')} topk={meta.get('topk')} "
               f"injection_score_mode={meta.get('injection_score_mode')!r}")
    out.append("")
    out.append("chance_upper = 2/(N-1) * 0.6, N = ctx/256 (spec §5/§7).")
    out.append("")
    out.append("vanilla-5B anchor rows are an EXTERNAL REFERENCE from a different model "
               "(vanilla GDN-2, no memory caching) measured under a DIFFERENT protocol "
               "(collaborator 4-way document) -- not directly comparable score-for-score "
               "to the stock/recent/random rows above them (own data + own scorer, this "
               "experiment). They contextualize the *qualitative* long-context collapse "
               "pattern only. No vanilla-30B anchor exists.")
    out.append("")

    for model in sorted(results.keys()):
        for task in sorted(results[model].keys()):
            out.append(build_section(model, task, results[model][task]))

    md = "\n".join(out)
    for p in (os.path.join(RES, "x4_table.md"),):
        open(p, "w").write(md)
        print(f"[x4_table] wrote {p}")


if __name__ == "__main__":
    main()
