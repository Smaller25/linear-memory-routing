#!/usr/bin/env python3
"""Is gold SELECTED but not READ? Measure the gate weight it actually gets.

The broadcast arm put gold inside the shared top-8 of all 16 layers 57.5% of
the time and scored 10, while the oracle put gold at top-1 of all 16 layers
and scored 82. Selection cannot be the whole story. The candidate mechanism
is the gate: selection decides which segments enter softmax(gate_logits), but
the LOGITS are each layer's native score, which is at chance for gold. Gold
can be selected and still receive a weight indistinguishable from the seven
haystack segments beside it.

This reads the weights off SSCOutput at the query position, per layer:
  online_weight        how much of the read stays on the current segment
  gold_route_weight    gold's share, when gold is selected at all
  max_route_weight     the largest share any selected segment got
A gold share far below max, in an arm where gold is reliably selected, is the
dilution mechanism observed rather than argued.

Usage:
  python dsc/scripts/measure_gold_gate_weight.py --ckpt <ckpt> \
      --data-root /root/dk_data_full --arm bcast-mlp --topk 8 \
      --router-dir /root/mlp_router_v2 --cells 8192:16 --seed 42 --max-samples 25
  python dsc/scripts/measure_gold_gate_weight.py --ckpt <ckpt> \
      --data-root /root/dk_data_full --arm oracle --topk 2 ...
"""
from __future__ import annotations

import argparse, json, os, statistics, sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "dsc")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--arm", required=True,
                    choices=["base", "bcast-mlp", "bcast-native", "oracle"])
    ap.add_argument("--gate-mode", default="native",
                    choices=["native", "order", "top1", "boost"])
    ap.add_argument("--gate-margin", type=float, default=1.0)
    ap.add_argument("--gate-scope", default="all",
                    choices=["all", "last_segment"])
    ap.add_argument("--source-layer", type=int, default=0)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--router-dir", default=None)
    ap.add_argument("--config-name", default="mc_370M")
    ap.add_argument("--tokenizer", default="TinyLlama/TinyLlama_v1.1")
    ap.add_argument("--cells", nargs="+", default=["8192:16", "8192:4"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=25)
    ap.add_argument("--out", required=True)
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

    oracle_aggs = []
    if args.arm == "oracle":
        from dsc.mc_baseline.mc_ssc_oracle import enable_oracle_routing
        oracle_aggs = enable_oracle_routing(model)
        if not oracle_aggs:
            raise RuntimeError("oracle attached to 0 aggregators")
    elif args.arm.startswith("bcast"):
        from dsc.mc_baseline.mc_ssc_mlp_router import enable_broadcast_routing
        src = "mlp" if args.arm == "bcast-mlp" else "native"
        on = enable_broadcast_routing(model, args.source_layer, src,
                                      args.router_dir, "cuda",
                                      args.gate_mode, args.gate_margin,
                                      gate_scope=args.gate_scope)
        if len(on) < 2:
            raise RuntimeError(f"broadcast attached to {len(on)} layers")
    print(f"[measure] arm={args.arm} topk={args.topk} "
          f"gate={args.gate_mode} margin={args.gate_margin}", flush=True)

    mc_layers = [m for m in model.modules()
                 if m.__class__.__name__ == "MemoryCachingGDN2Layer"]
    for lyr in mc_layers:
        def _fwd(hidden_states, attention_mask=None, past_key_values=None,
                 use_cache=False, output_attentions=False, _lyr=lyr, **kw):
            out, res = _lyr.forward_with_diagnostics(hidden_states)
            _lyr.diag = res
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

    gold_w, max_w, online_w, sel_rate, top1_rate = [], [], [], [], []
    # The metric that should predict the read: given gold was selected, did it
    # end up carrying the most weight of the selected segments? Under the
    # native gate it usually does not (gold 0.123, the other pick 0.242), and
    # that is what the gate modes are meant to fix. Prompt-only, so this
    # screens a gate change without paying for generation.
    gold_argmax = []
    for i, s in enumerate(samples):
        prompt = s["input"] + s.get("answer_prefix", "")
        ids = tok(prompt, return_tensors="pt",
                  add_special_tokens=False).input_ids.to("cuda")
        gold = s["token_position_answer"] // chunk
        if oracle_aggs:
            gt = torch.tensor([gold], dtype=torch.long)
            for agg in oracle_aggs:
                agg.oracle_gold = gt
        for lyr in mc_layers:
            lyr.diag = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(ids)
        last = ids.shape[1] - 1
        for lyr in mc_layers:
            r = lyr.diag
            if r is None:
                continue
            idx = r.route_indices[0, last]
            w = r.route_weights[0, last].float()
            online_w.append(float(r.online_weight[0, last, 0]))
            max_w.append(float(w.max()) if w.numel() else 0.0)
            hit = (idx == gold)
            sel_rate.append(float(hit.any()))
            top1_rate.append(float(idx.numel() > 0 and int(idx[0]) == gold))
            if bool(hit.any()):
                pos = int(hit.nonzero()[0, 0])
                gold_w.append(float(w[pos]))
                gold_argmax.append(float(int(w.argmax()) == pos))
        if (i + 1) % 10 == 0:
            print(f"[measure] {i + 1}/{len(samples)}", flush=True)

    def stat(xs):
        return {"mean": statistics.fmean(xs), "median": statistics.median(xs),
                "n": len(xs)} if xs else None

    rep = {"arm": args.arm, "topk": args.topk, "gate_mode": args.gate_mode,
           "gate_margin": args.gate_margin,
           "gold_is_argmax_given_selected": (
               statistics.fmean(gold_argmax) if gold_argmax else None),
           "cells": args.cells,
           "seed": args.seed, "samples": len(samples),
           "layer_obs": len(online_w),
           "gold_selected_rate": statistics.fmean(sel_rate) if sel_rate else None,
           "gold_top1_rate": statistics.fmean(top1_rate) if top1_rate else None,
           "gold_route_weight": stat(gold_w),
           "max_route_weight": stat(max_w),
           "online_weight": stat(online_w)}
    json.dump(rep, open(args.out, "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print(f"[measure] wrote {args.out}")


if __name__ == "__main__":
    main()
