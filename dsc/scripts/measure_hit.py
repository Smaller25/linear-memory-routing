#!/usr/bin/env python3
"""Routing hit rate, batched, prompt-only. The cheap screen before a score arm.

The score is a thresholded readout of two factors, score ~= hit x conversion,
and it is the product that is noisy. The hit factor -- did the gold segment
enter the top-k at the query position -- is a per-item binary that needs ONE
forward pass, no generation. That is where the cost asymmetry lives:

  score arm   ~9 s/item   (up to 48 decode steps per item)
  hit, bs=1   ~3 s/item
  hit, bs=8   under 1 s/item

So hit can be measured at 5x the item count for a fraction of the cost, which
drops the paired detection floor from about 12 points at n=50 to about 2 at
n=250. Screen every variant here; spend generation only once the hit gain,
times (oracle - baseline), clears the floor of the score arm you can afford.

Batching is right-padded, and that is safe for SELECTION specifically.
Eligibility at position t admits only segments strictly before t's own
segment, so for a row of length L the eligible segments all end at or before
L-1 and their descriptors contain no pad tokens. Gate WEIGHTS are a different
story -- pads land in the current segment and perturb the online score -- so
this script deliberately reports selection only. For weights use
measure_gold_gate_weight.py at bs=1.

Writes one JSONL row per item so compare_arms.py can pair on it:
  python dsc/scripts/compare_arms.py --metric hit --a hit_base.jsonl \
      --b hit_shared.jsonl

Usage:
  python dsc/scripts/measure_hit.py --ckpt <ckpt> --data-root /root/dk_data_full \
      --arm shared --router-dir /root/routers_v3_gvn03 --source-layer 0 \
      --topk 2 --cells 8192:4 8192:16 --seeds 42 43 --max-samples 50 \
      --batch-size 8 --out /root/hit_shared.jsonl
"""
from __future__ import annotations

import argparse, json, os, statistics, sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "dsc")):
    if p not in sys.path:
        sys.path.insert(0, p)


def build_model(args):
    from lit_gpt.config import Config
    from lit_gpt.model import GPT
    cfg = Config.from_name(args.config_name, mc_topk=args.topk)
    model = GPT(cfg).to(args.device).to(torch.bfloat16).eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"ckpt/config mismatch: {len(missing)} missing (e.g. "
            f"{missing[:3]}), {len(unexpected)} unexpected — a partially "
            "loaded model still produces a plausible hit rate")
    return model, getattr(cfg, "mc_chunk_size", 256)


def attach_arm(model, args):
    """Returns a short description of what was attached, or raises."""
    if args.arm == "native":
        return "native linear router, per layer"
    if args.arm == "select":
        from dsc.mc_baseline.mc_ssc_mlp_router import enable_mlp_router
        on = enable_mlp_router(model, args.router_dir, None, "select",
                               args.device)
        if not on:
            raise RuntimeError("--arm select attached to 0 layers")
        return f"MLP router in select mode on layers {on}"
    from dsc.mc_baseline.mc_ssc_mlp_router import enable_broadcast_routing
    on = enable_broadcast_routing(model, args.source_layer, args.shared_source,
                                  args.router_dir, args.device)
    if len(on) < 2:
        raise RuntimeError(f"--arm shared attached to {len(on)} layers")
    return (f"layer-shared routing, source L{args.source_layer}/"
            f"{args.shared_source}, {len(on)} layers reuse it")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="shared",
                    choices=["native", "select", "shared"])
    ap.add_argument("--router-dir", default=None)
    ap.add_argument("--source-layer", type=int, default=0)
    ap.add_argument("--shared-source", default="mlp",
                    choices=["mlp", "native"])
    ap.add_argument("--config-name", default="mc_370M")
    ap.add_argument("--tokenizer", default="TinyLlama/TinyLlama_v1.1")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--cells", nargs="+", default=["8192:4", "8192:16"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pad-id", type=int, default=0)
    args = ap.parse_args()
    if args.arm != "native" and not args.router_dir:
        ap.error(f"--arm {args.arm} needs --router-dir")

    model, chunk = build_model(args)
    print(f"[hit] {attach_arm(model, args)}", flush=True)

    mc_layers = [m for m in model.modules()
                 if m.__class__.__name__ == "MemoryCachingGDN2Layer"]
    if not mc_layers:
        raise RuntimeError("found 0 MemoryCachingGDN2Layer")
    for lyr in mc_layers:
        def _fwd(hidden_states, attention_mask=None, past_key_values=None,
                 use_cache=False, output_attentions=False, _lyr=lyr, **kw):
            out, res = _lyr.forward_with_diagnostics(hidden_states)
            _lyr.last_route_indices = res.route_indices.detach()
            return out, None, past_key_values
        lyr.forward = _fwd

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    items = []
    for seed in args.seeds:
        for cell in args.cells:
            ctx, ndl = (int(x) for x in cell.split(":"))
            path = os.path.join(args.data_root, f"seed{seed}", str(ctx),
                                f"niah_diversekey_essay_{ndl}",
                                "validation.jsonl")
            if not os.path.exists(path):
                print(f"[skip] {path}", flush=True)
                continue
            for i, line in enumerate(open(path)):
                if i >= args.max_samples:
                    break
                s = json.loads(line)
                items.append((f"seed{seed}_len{ctx}_n{ndl}", i, s))
    if not items:
        raise RuntimeError("no items found")
    print(f"[hit] {len(items)} items, batch {args.batch_size}", flush=True)

    fh = open(args.out, "w")
    n_done = 0
    for start in range(0, len(items), args.batch_size):
        chunk_items = items[start:start + args.batch_size]
        ids_list, lens = [], []
        for _, _, s in chunk_items:
            prompt = s["input"] + s.get("answer_prefix", "")
            t = tok(prompt, return_tensors="pt",
                    add_special_tokens=False).input_ids[0]
            ids_list.append(t); lens.append(len(t))
        maxlen = max(lens)
        batch = torch.full((len(ids_list), maxlen), args.pad_id,
                           dtype=torch.long)
        for b, t in enumerate(ids_list):
            batch[b, :len(t)] = t
        batch = batch.to(args.device)
        for lyr in mc_layers:
            lyr.last_route_indices = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(batch)

        for b, (cell, idx, s) in enumerate(chunk_items):
            last = lens[b] - 1
            gold = s["token_position_answer"] // chunk
            n_seg = (lens[b] + chunk - 1) // chunk
            per_layer, picks = [], []
            for lyr in mc_layers:
                ri = lyr.last_route_indices
                if ri is None or last >= ri.shape[1]:
                    continue
                sel = ri[b, last]
                per_layer.append(bool((sel == gold).any().item()))
                picks.append(tuple(int(x) for x in sel.tolist()))
            if not per_layer:
                continue
            agree = len(set(picks)) == 1
            rec = {"cell": cell, "sample_index": idx, "gold_segment": gold,
                   "n_segments": n_seg, "eligible": max(0, n_seg - 1),
                   "layer_hit_frac": sum(per_layer) / len(per_layer),
                   "layers_agree": agree,
                   "shared_hit": all(per_layer) if agree else None,
                   "any_layer_hit": any(per_layer),
                   "n_layers": len(per_layer)}
            if rec["shared_hit"] is None:
                # Not a shared-routing arm: there is no single decision, so
                # fall back to the per-layer fraction and say so.
                rec["shared_hit"] = sum(per_layer) / len(per_layer) > 0.5
                rec["shared_hit_is_majority"] = True
            fh.write(json.dumps(rec) + "\n")
            n_done += 1
        if (start // args.batch_size) % 5 == 0:
            print(f"[hit] {n_done}/{len(items)}", flush=True)
    fh.close()

    rows = [json.loads(l) for l in open(args.out)]
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)
    print(f"\n{'cell':<26}{'n':>5}{'shared_hit':>12}{'layer_frac':>12}"
          f"{'agree':>8}{'floor':>8}")
    for cell in sorted(by_cell) + (["ALL"] if len(by_cell) > 1 else []):
        rs = rows if cell == "ALL" else by_cell[cell]
        print(f"{cell:<26}{len(rs):>5}"
              f"{statistics.fmean(1.0 if r['shared_hit'] else 0.0 for r in rs):>12.3f}"
              f"{statistics.fmean(r['layer_hit_frac'] for r in rs):>12.3f}"
              f"{statistics.fmean(1.0 if r['layers_agree'] else 0.0 for r in rs):>8.2f}"
              f"{6 / len(rs):>8.3f}")
    print("\n  floor = paired detection floor at this n (6 discordant items "
          "for p<0.05). Selection only; use\n  measure_gold_gate_weight.py "
          "for gate weights, which right padding does perturb.")
    print(f"[hit] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
