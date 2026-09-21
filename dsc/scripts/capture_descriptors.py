#!/usr/bin/env python3
"""Capture routing descriptors + queries for the descriptor-quality diagnosis.

For each NIAH sample, one prompt-only forward; per selected MC layer we stash
  u        = W_u h_t at the FINAL prompt position           [H, K]
  gamma    = per-segment descriptors (mean of L2-norm keys) [Nseg, H, K]
and label every segment: gold (holds the queried needle), needle (holds any
needle), or haystack. Saved fp16 to one npz per cell.

Question this feeds: is the ReLU/top-k failure a missing signal (descriptors
carry nothing), a wasted signal (linear classifier on per-head cos sims beats
the router's uniform head-sum), or extrapolation (signal at 2K, gone at 8K)?

Usage:
    python dsc/scripts/capture_descriptors.py \
        --ckpt <ssc30b.pth> --data-root /root/dk_data_full \
        --cells 2048:16 8192:16 8192:4 --layers 0 5 10 15 \
        --max-samples 50 --out /root/descr_capture
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "dsc")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dsc.mc_baseline.mc_ssc import segment_key_sums  # noqa: E402
from dsc.mc_gdn2.ssc import GDN2SSC  # noqa: E402

NEEDLE_PHRASE = "special magic numbers for "


def attach_capture(model) -> list:
    """Instance-shadow each GDN2SSC forward to stash (u, summaries), then run
    the original math untouched (same wrap-only pattern as the eval hooks)."""
    captured = []
    for module in model.modules():
        if module.__class__.__name__ != "GDN2SSC":
            continue

        def make_fwd(agg, orig):
            def fwd(hidden_states, queries, keys, online_output, memories):
                u = agg.connector(hidden_states).view(
                    *hidden_states.shape[:2], agg.num_heads, agg.head_qk_dim)
                agg.cap_u_last = u[:, -1].detach().float().cpu()
                agg.cap_summaries = segment_key_sums(
                    keys, agg.chunk_size).detach().float().cpu()
                return orig(hidden_states, queries, keys, online_output, memories)
            return fwd

        module.forward = make_fwd(module, module.forward)
        captured.append(module)
    return captured


def needle_segments(tok, text: str, chunk: int) -> list[int]:
    """Segments containing ANY needle statement (offset-mapped)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    segs = set()
    start = 0
    while True:
        i = text.find(NEEDLE_PHRASE, start)
        if i < 0:
            break
        for t, (a, b) in enumerate(offsets):
            if a <= i < b:
                segs.add(t // chunk)
                break
        start = i + 1
    return sorted(segs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-name", default="mc_370M")
    ap.add_argument("--config-overrides", nargs="*", default=["mc_topk=2"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--cells", nargs="+", default=["2048:16", "8192:16", "8192:4"],
                    help="ctx:needles pairs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 5, 10, 15])
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from lit_gpt.config import Config
    from lit_gpt.model import GPT
    overrides = {}
    for pair in args.config_overrides:
        k, v = pair.split("=", 1)
        overrides[k] = int(v)
    cfg = Config.from_name(args.config_name, **overrides)
    chunk = getattr(cfg, "mc_chunk_size", 256)
    model = GPT(cfg).to("cuda").to(torch.bfloat16)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing[:3], unexpected[:3])
    model.eval()
    aggs = attach_capture(model)
    print(f"[capture] hooked {len(aggs)} SSC aggregators; layers kept: {args.layers}")
    assert aggs, "no aggregators hooked"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")

    os.makedirs(args.out, exist_ok=True)
    for cell in args.cells:
        ctx, ndl = (int(x) for x in cell.split(":"))
        path = os.path.join(args.data_root, f"seed{args.seed}", str(ctx),
                            f"niah_diversekey_essay_{ndl}", "validation.jsonl")
        samples = [json.loads(l) for l in open(path)][: args.max_samples]
        U, G, GOLD, NDLSEG, NSEG = [], [], [], [], []
        for s in samples:
            prompt = s["input"] + s.get("answer_prefix", "")
            ids = tok(prompt, return_tensors="pt",
                      add_special_tokens=False).input_ids.to("cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(ids)
            U.append(np.stack([aggs[l].cap_u_last[0].numpy()
                               for l in args.layers]))          # [L,H,K]
            G.append(np.stack([aggs[l].cap_summaries[0].numpy()
                               for l in args.layers]))          # [L,Nseg,H,K]
            GOLD.append(s["token_position_answer"] // chunk)
            nsegs = needle_segments(tok, s["input"], chunk)
            NDLSEG.append(nsegs)
            NSEG.append(G[-1].shape[1])
        out = os.path.join(args.out, f"cap_{ctx}_n{ndl}.npz")
        # prompts vary in length -> Nseg varies per sample: zero-pad the
        # segment axis to the cell max (nseg[] carries the real count).
        max_seg = max(g.shape[1] for g in G)
        G = [np.pad(g, ((0, 0), (0, max_seg - g.shape[1]), (0, 0), (0, 0)))
             for g in G]
        np.savez_compressed(
            out,
            u=np.stack(U).astype(np.float16),
            gamma=np.stack(G).astype(np.float16),
            gold=np.array(GOLD),
            nseg=np.array(NSEG),
            needle_segs=np.array([json.dumps(x) for x in NDLSEG]),
            layers=np.array(args.layers),
        )
        print(f"[saved] {out}  u{np.stack(U).shape} gamma{np.stack(G).shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
