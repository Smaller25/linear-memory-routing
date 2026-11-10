#!/usr/bin/env python3
"""Summarise finished or running runs for docs/EXPERIMENTS.md and the experiment queue (docs/QUEUE.md).
Reads only train.jsonl, wandb_sync.log and eval/*/summary.json; standard library only, so it runs on
the login node.

    python3 scripts/harvest.py RUNS_ROOT mc_desc_sb16_50m_ctx8192_s0 mc_desc_mma_50m_ctx8192_s0 ...
    python3 scripts/harvest.py RUNS_ROOT 'mc_desc_*'            # glob over run folder names
    python3 scripts/harvest.py --traj RUNS_ROOT mc_desc_mma_50m_ctx8192_s0   # + every eval point

One line per run: state (end / running step / missing), last eval step, val_loss, ruler_hit at the last
eval and averaged over the last 3, copy_hit, random baselines, and whether wandb_sync reported "Synced".
"""

import argparse
import fnmatch
import json
import os


def read_run(d):
    ev, last, end, start = {}, None, None, None
    try:
        for line in open(os.path.join(d, "train.jsonl")):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "ruler_hit" in r or "val_loss" in r:
                ev[r["step"]] = r
            elif "loss" in r:
                last = r
            elif r.get("event") == "start":
                start = r
            elif r.get("event") == "end":
                end = r
    except FileNotFoundError:
        return None
    return [ev[k] for k in sorted(ev)], last, start, end


def wandb_state(d):
    try:
        txt = open(os.path.join(d, "wandb_sync.log")).read()
    except FileNotFoundError:
        return "no-log"
    return "synced" if "Synced" in txt else ("error" if "rror" in txt else "uploading")


def evals(d):
    out = []
    root = os.path.join(d, "eval")
    for ck in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        try:
            res = json.load(open(os.path.join(root, ck, "summary.json")))["results"]
        except (FileNotFoundError, ValueError, KeyError):
            continue
        for r in res:
            out.append(f"{ck}/{r['route']}/{r['task']}/{r['length']}={r['string_match_all']:.1f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", action="store_true", help="also print step:val/ruler/copy for every eval")
    ap.add_argument("root")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    names = sorted(os.listdir(a.root))
    picked = []
    for pat in a.runs:
        hits = [n for n in names if fnmatch.fnmatch(n, pat)]
        picked += hits or [pat]
    for n in dict.fromkeys(picked):
        d = os.path.join(a.root, n)
        got = read_run(d)
        if got is None:
            print(f"{n:40s} missing")
            continue
        ev, last, start, end = got
        state = "end" if end and start and end["step"] == start.get("stop_step", start.get("total_steps")) \
            else f"step {last['step'] if last else 0}"
        line = f"{n:40s} {state:10s} wandb={wandb_state(d)}"
        if ev:
            f, l3 = ev[-1], ev[-3:]
            avg = lambda k: sum(r.get(k, 0) for r in l3) / len(l3)
            line += (f" eval@{f['step']} val={f['val_loss']:.4f}")
            if "ruler_hit" in f:
                line += (f" ruler={f['ruler_hit']:.4f} ruler3={avg('ruler_hit'):.4f}"
                         f" copy={f.get('copy_hit', float('nan')):.4f}"
                         f" rrand={f.get('ruler_random_hit', float('nan')):.4f}"
                         f" crand={f.get('copy_random_hit', float('nan')):.4f}")
        if last:
            line += f" aux={last.get('aux_loss', 0):.3f} tok/s={last.get('tok_per_s', 0):.0f}"
        print(line)
        if a.traj and ev:
            print("   traj " + " ".join(f"{r['step']}:{r['val_loss']:.3f}/{r.get('ruler_hit', 0):.3f}/"
                                       f"{r.get('copy_hit', 0):.3f}" for r in ev))
            if "ruler_hit_by_layer" in ev[-1]:
                print("   ruler_L " + " ".join(f"{x:.2f}" for x in ev[-1]["ruler_hit_by_layer"]))
        for e in evals(d):
            print("   eval " + e)


if __name__ == "__main__":
    main()
