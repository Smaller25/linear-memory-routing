#!/usr/bin/env python3
"""Generate the diverse-key NIAH benchmark data (needles x lengths x seeds).

Sweeps the vendored RULER generator (linear-memory-routing/src/ruler/gen/
prepare.py) over the ``niah_diversekey_{haystack}_{N}`` tasks added to
synthetic.yaml. All generation constants follow the pinned experiment
protocol:

    * tokenizer  : TinyLlama/TinyLlama_v1.1, tokenizer_type=hf  (CONSTANT)
    * template   : model_template_type=base                      (CONSTANT)
    * num_samples: 50 per cell                                   (CONSTANT)
    * seeds      : >= 3 generation seeds, one directory each
                   (RULER seed variance was measured up to 26.6pp — a single
                   seed table is not admissible)

Output layout (one directory per generation seed, so no cell ever
overwrites another):

    <data_root>/seed{S}/{length}/niah_diversekey_{hay}_{N}/validation.jsonl

A ``gen_manifest.json`` at <data_root> records every parameter of the sweep.

Usage (defaults reproduce the experiment grid):
    python dsc/scripts/gen_diverse_key_niah.py \
        --data-root linear-memory-routing/data/ruler_diverse_key

Smoke test (tiny, CPU-only, ~1 min):
    python dsc/scripts/gen_diverse_key_niah.py \
        --data-root /tmp/dk_smoke --lengths 2048 --needles 1 4 \
        --seeds 42 --num-samples 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The RULER generator used to live in a nested linear-memory-routing checkout
# inside long-gdn. In this repository it is top-level, at src/ruler/gen.
GEN_DIR = os.path.join(REPO, "src", "ruler", "gen")
PREPARE = os.path.join(GEN_DIR, "prepare.py")

TOKENIZER = "TinyLlama/TinyLlama_v1.1"  # CONSTANT: chunk boundaries + gold labels depend on it


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--needles", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                    help="generation seeds; >= 3 required for the main table")
    ap.add_argument("--haystack", choices=["essay", "noise"], default="essay",
                    help="essay = niah_multikey lineage (default); noise = niah_single_1 lineage")
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    if len(args.seeds) < 3 and args.num_samples >= 50:
        print("[warn] fewer than 3 generation seeds — NOT main-table-ready "
              "(protocol requires mean±sd over >= 3 seeds)", flush=True)

    os.makedirs(args.data_root, exist_ok=True)
    manifest = {
        "generator": "src/ruler/gen/prepare.py",
        "tasks": [f"niah_diversekey_{args.haystack}_{n}" for n in args.needles],
        "tokenizer": args.tokenizer,
        "tokenizer_type": "hf",
        "model_template_type": "base",
        "num_samples": args.num_samples,
        "lengths": args.lengths,
        "needles": args.needles,
        "seeds": args.seeds,
        "haystack": args.haystack,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cells": [],
    }

    n_cells = len(args.seeds) * len(args.lengths) * len(args.needles)
    done = 0
    failures = []
    for seed in args.seeds:
        for length in args.lengths:
            save_dir = os.path.join(args.data_root, f"seed{seed}", str(length))
            for needles in args.needles:
                task = f"niah_diversekey_{args.haystack}_{needles}"
                out_file = os.path.join(save_dir, task, "validation.jsonl")
                done += 1
                cell = {"seed": seed, "length": length, "needles": needles,
                        "task": task, "path": os.path.relpath(out_file, args.data_root)}
                if args.skip_existing and os.path.exists(out_file):
                    with open(out_file) as f:
                        n_lines = sum(1 for _ in f)
                    if n_lines == args.num_samples:
                        print(f"[{done}/{n_cells}] skip (exists, {n_lines} samples): {out_file}",
                              flush=True)
                        cell["status"] = "existing"
                        manifest["cells"].append(cell)
                        continue
                cmd = [
                    sys.executable, PREPARE,
                    "--save_dir", save_dir,
                    "--benchmark", "synthetic",
                    "--task", task,
                    "--tokenizer_path", args.tokenizer,
                    "--tokenizer_type", "hf",
                    "--max_seq_length", str(length),
                    "--model_template_type", "base",
                    "--num_samples", str(args.num_samples),
                    "--random_seed", str(seed),
                ]
                print(f"[{done}/{n_cells}] gen seed={seed} len={length} N={needles}",
                      flush=True)
                # prepare.py shells out to bare `python`; make sure it
                # resolves to THIS interpreter (wonderwords/nltk live there).
                env = dict(os.environ)
                env["PATH"] = (os.path.dirname(sys.executable) + os.pathsep
                               + env.get("PATH", ""))
                proc = subprocess.run(cmd, cwd=GEN_DIR, env=env)
                ok = proc.returncode == 0 and os.path.exists(out_file)
                cell["status"] = "generated" if ok else "FAILED"
                manifest["cells"].append(cell)
                if not ok:
                    failures.append(cell)
                    print(f"[error] generation failed: {cell}", flush=True)

    manifest_path = os.path.join(args.data_root, "gen_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[saved] {manifest_path}", flush=True)
    if failures:
        print(f"[FAIL] {len(failures)} cells failed", flush=True)
        return 1
    print(f"[done] {n_cells} cells", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
