"""X4 grid driver: (model x task x length x mode[+seed]) grid, spec §5.

Cell = one (model, task, length, mode[, seed]) combination. Cell-level
skip-existing resumability: a per-cell raw-results file
($MC_OUT/results/x4_raw.json) records every finished cell keyed by a flat
string; on restart (container recycle etc.) already-recorded cells are
skipped and only missing ones run. This is the "container restart safe"
design the brief requires — NOT sample-level resumability (a half-finished
cell is simply re-run from scratch; cells are the atomic unit, matching
rev1 plan Task 4's "JSON에 셀 단위 기록, 재제출 안전").

Usage (on VESSL, one job per (model,task) — 4 jobs total, per the brief's
chunking plan):
    PY x4_run.py --model mc-5B --task niah_single_1
    PY x4_run.py --model mc-5B --task niah_multikey_1
    PY x4_run.py --model mc-30B --task niah_single_1
    PY x4_run.py --model mc-30B --task niah_multikey_1

Each invocation processes all 6 lengths ascending, all 7 mode-cells
(stock, recent, random x5 seeds) per length, merging into the shared raw
JSON (safe to run multiple (model,task) jobs concurrently — see
_load_raw/_save_raw's read-then-update-then-write-back-merged pattern,
though in practice each job only ever writes cells for its OWN
(model,task), so concurrent writers never touch the same top-level key).
"""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_mc
import x4_gen

MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

LENGTHS_DEFAULT = (1024, 2048, 4096, 8192, 16384, 32768)
SEEDS_DEFAULT = x4_gen.SEEDS_DEFAULT
N_GEN_DEFAULT = x4_gen.N_GEN_DEFAULT
NUM_SAMPLES_DEFAULT = 50
REDUCED_NUM_SAMPLES = 25
# Budget guard (task brief): "잔여 예산 초과가 예상되면 32K/16K만 25샘플로 감축".
# A length's 7 mode-cells are timed via a small probe BEFORE any of them run
# (see _decide_num_samples) so the sample count is decided once, uniformly,
# for that whole length -- never split 50/25 across modes of the same
# length (spec §7: bf16 noise aside, comparisons must hold n constant).
PROBE_N = 2
BUDGET_SECONDS_PER_LENGTH = 3 * 3600  # 3h for all 7 modes at a given length


def _raw_path():
    return os.path.join(MC_OUT, "results", "x4_raw.json")


def _load_raw():
    p = _raw_path()
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return {}


def _save_raw(raw):
    # merge with whatever is on disk right now (another concurrent (model,
    # task) job may have written cells since we last loaded) so a slow job
    # never clobbers a fast one's progress.
    p = _raw_path()
    on_disk = _load_raw()
    on_disk.update(raw)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(on_disk, open(p, "w"))
    return on_disk


def _cell_key(model, task, length, mode, seed=None):
    if mode == "random":
        return f"{model}|{task}|{length}|random|seed{seed}"
    return f"{model}|{task}|{length}|{mode}"


def _rows_for(task, length, num_samples):
    path = os.path.join(MC_OUT, "data", str(length), task, "validation.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[:num_samples]


def _run_rows(model, tok, engine, rows, n_gen):
    out_rows = []
    for r in rows:
        ids_list = tok(r["input"], add_special_tokens=False).input_ids
        gen = x4_gen.incremental_generate(model, engine, ids_list, n_gen=n_gen)
        pred = tok.decode(gen)
        correct = all(o.lower() in pred.lower() for o in r["outputs"])
        out_rows.append({"index": r.get("index", r.get("pair_id")), "pred": pred,
                         "correct": correct})
    n = len(out_rows)
    score = sum(1 for r in out_rows if r["correct"]) / n if n else 0.0
    return {"score": score, "n": n, "rows": out_rows}


def _decide_num_samples(model, tok, engine, task, length, n_gen):
    """Probe timing for a length BEFORE committing to a sample count for all
    7 of its mode-cells (kept uniform across modes -- see module docstring).
    Only probes lengths >= 16384 (shorter lengths are cheap by construction:
    generation cost is dominated by per-step current-segment replay, capped
    at `chunk_size` tokens, not by total context length)."""
    if length < 16384:
        return NUM_SAMPLES_DEFAULT, None
    probe_rows = _rows_for(task, length, PROBE_N)
    x4_gen.set_mode(engine.overrides, "stock", seed=0)
    t0 = time.time()
    for r in probe_rows:
        ids_list = tok(r["input"], add_special_tokens=False).input_ids
        x4_gen.incremental_generate(model, engine, ids_list, n_gen=n_gen)
    elapsed = time.time() - t0
    per_sample = elapsed / max(1, len(probe_rows))
    projected = per_sample * 7 * NUM_SAMPLES_DEFAULT
    print(f"[x4][probe] length={length}: {per_sample:.2f}s/sample "
          f"(probe n={len(probe_rows)}) -> projected {projected/3600:.2f}h for "
          f"7 modes x {NUM_SAMPLES_DEFAULT} samples (budget {BUDGET_SECONDS_PER_LENGTH/3600:.1f}h)",
          flush=True)
    if projected > BUDGET_SECONDS_PER_LENGTH:
        print(f"[x4][probe][WARN] length={length}: projected time exceeds budget -- "
              f"reducing to {REDUCED_NUM_SAMPLES} samples for this length", flush=True)
        return REDUCED_NUM_SAMPLES, per_sample
    return NUM_SAMPLES_DEFAULT, per_sample


def run_grid(model_kind, task, lengths=LENGTHS_DEFAULT, seeds=SEEDS_DEFAULT,
            n_gen=N_GEN_DEFAULT):
    tok = load_mc.load_tokenizer()
    model = load_mc.load_model(model_kind)
    overrides = x4_gen.patch_override(model, mode="stock", seed=0)
    engine = x4_gen.IncrementalEngine(model, overrides)
    print(f"[x4] {model_kind}/{task}: n_layers={len(overrides)}", flush=True)

    meta_notes = {}
    for length in lengths:
        num_samples, probe_seconds = _decide_num_samples(model, tok, engine, task, length, n_gen)
        if num_samples != NUM_SAMPLES_DEFAULT:
            meta_notes[str(length)] = {"num_samples": num_samples,
                                       "probe_seconds_per_sample": probe_seconds,
                                       "reason": "projected time exceeded budget"}
        rows = _rows_for(task, length, num_samples)
        t_length0 = time.time()

        modes = [("stock", None), ("recent", None)] + [("random", s) for s in seeds]
        for mode, seed in modes:
            key = _cell_key(model_kind, task, length, mode, seed)
            raw = _load_raw()
            if key in raw and raw[key].get("n") == len(rows):
                print(f"[x4][skip] {key} already done (n={raw[key]['n']})", flush=True)
                continue
            x4_gen.set_mode(overrides, mode, seed=(seed if seed is not None else 0))
            t0 = time.time()
            cell = _run_rows(model, tok, engine, rows, n_gen)
            dt = time.time() - t0
            cell["wall_seconds"] = dt
            cell["seed"] = seed
            raw = _load_raw()
            raw[key] = cell
            _save_raw(raw)
            print(f"[x4] {key}: score={cell['score']:.3f} n={cell['n']} "
                  f"({dt:.1f}s, {dt/max(1,len(rows)):.2f}s/sample)", flush=True)
        print(f"[x4][length-done] {model_kind}/{task}/{length}: "
              f"{time.time()-t_length0:.1f}s total for 7 modes", flush=True)

    del model
    torch.cuda.empty_cache()
    return meta_notes


def aggregate(model_kind, task, lengths=LENGTHS_DEFAULT, seeds=SEEDS_DEFAULT,
             meta_notes=None):
    """Read x4_raw.json and build the aggregate results[model][task][length]
    schema the brief specifies, plus chance_upper. Written by main() into
    results/x4_random_routing.json (merged across all (model,task) jobs that
    have run so far)."""
    raw = _load_raw()
    out = {}
    for length in lengths:
        n_seg = length // x4_gen.CHUNK_DEFAULT
        cell = {"chance_upper": x4_gen.chance_upper(n_seg)}
        stock_key = _cell_key(model_kind, task, length, "stock")
        recent_key = _cell_key(model_kind, task, length, "recent")
        if stock_key in raw:
            cell["stock"] = raw[stock_key]["score"]
            cell["n"] = raw[stock_key]["n"]
        if recent_key in raw:
            cell["recent"] = raw[recent_key]["score"]
        per_seed = []
        for s in seeds:
            k = _cell_key(model_kind, task, length, "random", s)
            if k in raw:
                per_seed.append(raw[k]["score"])
        if per_seed:
            cell["random"] = {"mean": sum(per_seed) / len(per_seed), "per_seed": per_seed}
        if meta_notes and str(length) in meta_notes:
            cell["meta"] = meta_notes[str(length)]
        out[str(length)] = cell
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mc-5B", "mc-30B"])
    ap.add_argument("--task", required=True, choices=["niah_single_1", "niah_multikey_1"])
    ap.add_argument("--lengths", type=str, default=",".join(str(l) for l in LENGTHS_DEFAULT))
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS_DEFAULT))
    ap.add_argument("--n-gen", type=int, default=N_GEN_DEFAULT)
    a = ap.parse_args()

    lengths = tuple(int(x) for x in a.lengths.split(",") if x.strip())
    seeds = tuple(int(x) for x in a.seeds.split(",") if x.strip())

    meta_notes = run_grid(a.model, a.task, lengths=lengths, seeds=seeds, n_gen=a.n_gen)
    agg = aggregate(a.model, a.task, lengths=lengths, seeds=seeds, meta_notes=meta_notes)

    os.makedirs(RES, exist_ok=True)
    os.makedirs(os.path.join(MC_OUT, "results"), exist_ok=True)
    out_repo = os.path.join(RES, "x4_random_routing.json")
    out_mcout = os.path.join(MC_OUT, "results", "x4_random_routing.json")

    merged = {"meta": {"n_gen": a.n_gen, "chunk": x4_gen.CHUNK_DEFAULT,
                       "topk": x4_gen.TOPK_DEFAULT,
                       "injection_score_mode": "max(selected_real_scores_max, online)",
                       "num_samples_default": NUM_SAMPLES_DEFAULT,
                       "seeds": list(seeds), "lengths": list(lengths)},
             "results": {}}
    for p in (out_repo, out_mcout):
        if os.path.exists(p):
            try:
                prev = json.load(open(p))
                merged["results"].update(prev.get("results", {}))
            except Exception:
                pass
    merged["results"].setdefault(a.model, {})[a.task] = agg

    for p in (out_repo, out_mcout):
        json.dump(merged, open(p, "w"), indent=2)
        print(f"[x4] wrote {p}", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
