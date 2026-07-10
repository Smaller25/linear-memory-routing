# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Query-conditioned routing probe — WHY does SSC work on single but fail on multi-key?

The pooled-descriptor AUC probe (diag_descriptor_probe) measures query-INDEPENDENT needle-vs-filler
separability, which cannot explain the multi-key gap (the multi difficulty is query-DEPENDENT:
"which of the K key-chunks does THIS query want"). This probe measures exactly that, from the trained
SSC router's own scores (`logits = router(x_query) . descriptor_chunk`):

  top1_all       : is the queried key's chunk the router's #1 among ALL chunks? (needle-vs-everything)
  top1_amongkeys : ... among ONLY the K key-bearing chunks? (the which-key test, chance = 1/K)

Single-needle has K=1 key-chunk, so `top1_amongkeys` is trivially 1.0 — that IS the explanation of why
single works (no which-key problem). Multi-key has K>1: if top1_amongkeys ≈ 1/K, the router cannot pick
the queried key given the query (H2 confirmed = the gap); if high, routing works and the failure is
downstream extraction. Scored at the answer position (last answer-prefix token), per layer.

Run: env FLA_CONV_BACKEND=triton python -m lmr.analysis.routing_probe --arch mamba2 \
       --model state-spaces/mamba2-370m --heads ckpt/ssc_370m.pt --low-rank-dim 64 --chunk-size 256 \
       --tasks niah_single_1 niah_multikey_2 --lengths 2048 4096
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from lmr.adapters import descriptor_dim_for, get_adapter
from lmr.loaders import load_backbone
from lmr.readout import SparseSelectiveCaching, build_readout
from lmr.segment_runner import run_segmented_lm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def key_chunks_and_query(input_text, tokenizer, chunk_size, single):
    """Return (set of chunk indices holding a key-value, queried chunk index). Uses the RULER text."""
    q = re.search(r"What is the special magic number(?:s)? for ([\w-]+)", input_text)
    if not q:
        return set(), -1
    queried = q.group(1)
    # every "... for <kw> is[: ] <number>" in the body marks a key-value; map its first mention to a chunk
    chunk_for = {}
    for kw in set(re.findall(r"for ([\w-]+) is[: ]+", input_text)):
        p = input_text.find(kw)
        if p < 0:
            continue
        n_tok = len(tokenizer(input_text[:p], add_special_tokens=False).input_ids)
        chunk_for[kw] = n_tok // chunk_size
    if single:  # keep only the queried key's chunk as the single needle
        chunk_for = {queried: chunk_for[queried]} if queried in chunk_for else {}
    if queried not in chunk_for:
        return set(), -1
    return set(chunk_for.values()), chunk_for[queried]


def patch_capture_logits(trained, store):
    def make(mod, li):
        orig = mod.forward
        def new(y_main, y_cached, x, descriptors):
            if y_cached and descriptors is not None and descriptors.shape[1] > 0:
                u = mod.router(x)                                        # [b,l,dd]
                logits = torch.einsum("bld,bid->bli", u, descriptors)    # [b,l,i]
                store.append((li, logits.detach().float().cpu()))
            return orig(y_main, y_cached, x, descriptors)
        return new
    i = 0
    for h in trained:
        if isinstance(h, SparseSelectiveCaching):
            h.forward = make(h, i)
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mamba2")
    ap.add_argument("--model", default="state-spaces/mamba2-370m")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--low-rank-dim", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--tasks", nargs="+", default=["niah_single_1", "niah_multikey_2"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument("--max-examples", type=int, default=50)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="ckpt/routing_probe.json")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    model, tok = load_backbone(args.arch, repo=args.model, tokenizer=args.tokenizer,
                              device=args.device, dtype=dtype)
    adapter = get_adapter(args.arch)
    n_layers = len(list(adapter.blocks(model)))
    dd = descriptor_dim_for(model, args.arch)
    trained = nn.ModuleList([build_readout("ssc", model.config.hidden_size, dd,
                                           topk=args.topk, low_rank_dim=args.low_rank_dim)
                             for _ in range(n_layers)]).to(args.device, dtype=dtype)
    trained.load_state_dict(torch.load(args.heads, map_location=args.device))
    trained.eval()
    store = []
    patch_capture_logits(trained, store)

    def seg(x):
        return run_segmented_lm(model, x, trained, args.chunk_size, backend="cuda",
                                return_hidden=True, arch=args.arch, hierarchical_k=None,
                                cache_cap=None)[0]

    print(f"[routing-probe] {args.arch} {args.model} chunk={args.chunk_size} heads={args.heads}", flush=True)
    results = {}
    for task in args.tasks:
        single = task.startswith("niah_single")
        for L in args.lengths:
            path = os.path.join(REPO_ROOT, "data", "ruler", str(L), task, "validation.jsonl")
            if not os.path.exists(path):
                continue
            rows = [json.loads(l) for l in open(path)][:args.max_examples]
            # per-layer running tallies
            all_hit = defaultdict(int); key_hit = defaultdict(int); nseen = defaultdict(int)
            Ks = []; nused = 0
            for ex in rows:
                kc, qchunk = key_chunks_and_query(ex["input"], tok, args.chunk_size, single)
                if qchunk < 0 or not kc:
                    continue
                prefix = tok(ex["input"] + ex.get("answer_prefix", ""), add_special_tokens=False).input_ids
                qpos = len(prefix) - 1                                   # answer position (routes the answer)
                ids = torch.tensor([prefix], dtype=torch.long, device=args.device)
                store.clear()
                with torch.no_grad():
                    seg(ids)
                # collect per-layer logits at the answer position
                per_layer = defaultdict(list)
                for li, lg in store:
                    if lg.shape[1] > qpos:
                        per_layer[li].append(lg[0, qpos])               # [num_ckpt]
                nused += 1; Ks.append(len(kc))
                key_idx = sorted(kc)
                for li, lst in per_layer.items():
                    s = lst[-1]                                          # [num_ckpt] scores over chunks
                    if s.shape[0] <= qchunk:
                        continue
                    nseen[li] += 1
                    if int(s.argmax()) == qchunk:                        # top-1 among ALL chunks
                        all_hit[li] += 1
                    ks = torch.tensor([s[c] for c in key_idx])          # restrict to key-chunks
                    if key_idx[int(ks.argmax())] == qchunk:              # top-1 among key-chunks
                        key_hit[li] += 1
            if nused == 0:
                continue
            meanK = float(np.mean(Ks))
            layers = sorted(nseen)
            top1_all = {li: all_hit[li] / nseen[li] for li in layers}
            top1_key = {li: key_hit[li] / nseen[li] for li in layers}
            best_all = max(top1_all.values()); best_key = max(top1_key.values())
            print(f"\n=== {task}@{L}  used={nused}  mean #key-chunks K={meanK:.1f}  (chance among-keys={1/meanK:.3f}) ===")
            print(f"  top1 among ALL chunks : mean {np.mean(list(top1_all.values())):.3f}  best-layer {best_all:.3f}")
            print(f"  top1 among KEY chunks : mean {np.mean(list(top1_key.values())):.3f}  best-layer {best_key:.3f}   <- the which-key test")
            results[f"{task}@{L}"] = {"n": nused, "meanK": meanK,
                                      "top1_all": top1_all, "top1_amongkeys": top1_key}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n[routing-probe] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
