#!/usr/bin/env python3
"""Post-condition gate for the broadcast arm. Run BEFORE reading its score.

Broadcast means one claim that is directly observable: every MC layer routes
on the SAME segment indices. A bus that silently failed over to per-layer
routing would still produce a complete run with a plausible score, so the
claim is checked against the artifact rather than the flag being set.

Checks, on a handful of held-out samples:

  A. identity — every broadcast layer's route_indices equals the source
     layer's, at every position. Layers before the source are excluded by
     construction: they cannot read the bus and run their own router.
  B. all-or-nothing gold — because the picks are shared, gold is either in
     every layer's top-k or in none. The per-sample gold-hit fraction over
     layers must be exactly 0.0 or 1.0, never in between.
  C. hit rate — reports the fraction of samples where gold made the shared
     top-k. This is the arm's ceiling: if the read delivered perfectly, the
     score could not exceed this times the oracle score.

Usage:
  python dsc/scripts/verify_broadcast_routing.py \
      --ckpt /root/dk_local/ckpts/ssc30b/checkpoint-30B-model-ckpt.pth \
      --router-dir /root/mlp_router_v2 --data-root /root/dk_data_full \
      --source mlp --topk 2 --cells 8192:16 8192:4 --seed 43 --max-samples 20
"""
from __future__ import annotations

import argparse, json, os, sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "dsc")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--router-dir", default=None)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--config-name", default="mc_370M")
    ap.add_argument("--tokenizer", default="TinyLlama/TinyLlama_v1.1")
    ap.add_argument("--source", default="mlp", choices=["mlp", "native"])
    ap.add_argument("--source-layer", type=int, default=0)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--cells", nargs="+", default=["8192:16", "8192:4"])
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from lit_gpt.config import Config
    from lit_gpt.model import GPT
    cfg = Config.from_name(args.config_name, mc_topk=args.topk)
    chunk = getattr(cfg, "mc_chunk_size", 256)
    model = GPT(cfg).to("cuda").to(torch.bfloat16).eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"ckpt/config mismatch: {len(missing)} missing, "
                           f"{len(unexpected)} unexpected")

    from dsc.mc_baseline.mc_ssc_mlp_router import enable_broadcast_routing
    layers_on = enable_broadcast_routing(model, args.source_layer,
                                         args.source, args.router_dir)
    print(f"[gate] broadcast on {len(layers_on)} layers, source=L"
          f"{args.source_layer}/{args.source}, topk={args.topk}", flush=True)

    mc_layers = [m for m in model.modules()
                 if m.__class__.__name__ == "MemoryCachingGDN2Layer"]
    for lyr in mc_layers:
        def _fwd(hidden_states, attention_mask=None, past_key_values=None,
                 use_cache=False, output_attentions=False, _lyr=lyr, **kw):
            out, res = _lyr.forward_with_diagnostics(hidden_states)
            _lyr.last_route_indices = res.route_indices.detach()
            return out, None, past_key_values
        lyr.forward = _fwd

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    samples = []
    for cell in args.cells:
        ctx, ndl = (int(x) for x in cell.split(":"))
        path = os.path.join(args.data_root, f"seed{args.seed}", str(ctx),
                            f"niah_diversekey_essay_{ndl}", "validation.jsonl")
        samples += [json.loads(l) for l in open(path)][:args.max_samples]
    print(f"[gate] {len(samples)} samples", flush=True)

    failures, hits, n = [], 0, 0
    for i, s in enumerate(samples):
        prompt = s["input"] + s.get("answer_prefix", "")
        ids = tok(prompt, return_tensors="pt",
                  add_special_tokens=False).input_ids.to("cuda")
        for lyr in mc_layers:
            lyr.last_route_indices = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(ids)
        src = args.source_layer
        bcast = mc_layers[src:]
        ref = bcast[0].last_route_indices
        if ref is None:
            failures.append(f"sample {i}: source L{src} recorded no routing")
            continue
        for li, lyr in enumerate(bcast[1:], start=src + 1):
            got = lyr.last_route_indices
            if got is None:
                failures.append(f"sample {i} L{li}: no routing recorded")
            elif got.shape != ref.shape or not torch.equal(got, ref):
                bad = int((got != ref).sum()) if got.shape == ref.shape else -1
                failures.append(
                    f"sample {i} L{li}: A route_indices differ from the "
                    f"source L{src} ({bad} positions)")
        gold = s["token_position_answer"] // chunk
        last = ids.shape[1] - 1
        per_layer = [bool((lyr.last_route_indices[0, last] == gold).any())
                     for lyr in bcast
                     if lyr.last_route_indices is not None]
        frac = sum(per_layer) / len(per_layer)
        if frac not in (0.0, 1.0):
            failures.append(
                f"sample {i}: B gold-hit fraction {frac:.3f} is neither 0 nor "
                "1 — the layers did not share a decision")
        hits += int(frac == 1.0); n += 1
        if (i + 1) % 10 == 0:
            print(f"[gate] {i + 1}/{len(samples)} hit_rate={hits / n:.3f}",
                  flush=True)

    report = {"source": args.source, "source_layer": args.source_layer,
              "topk": args.topk, "n": n,
              "shared_topk_gold_hit_rate": (hits / n) if n else None,
              "cells": args.cells, "seed": args.seed,
              "verdict": "PASS" if not failures else "FAIL",
              "failures": failures[:20]}
    out = (args.out or f"/root/bcast_gate_{args.source}_L"
           f"{args.source_layer}_k{args.topk}.json")
    json.dump(report, open(out, "w"), indent=1)
    print(json.dumps({k: report[k] for k in
                      ("verdict", "n", "shared_topk_gold_hit_rate")}, indent=1))
    if failures:
        print("FAILURES:"); [print(" ", f) for f in failures[:20]]
    print(f"[gate] wrote {out}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
