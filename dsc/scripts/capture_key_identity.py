#!/usr/bin/env python3
"""Capture what a router would need to tell one needle from another.

N=16 is where the work is stuck: 6.0 against an oracle 71.3, and every lever
so far moved needle-vs-haystack rather than needle-vs-needle. The dot scorer
gained +0.164 where needles are few and +0.037 where they are many, which is
the signature of a feature that says "this segment holds a distinctive
concentrated phrase" — true of all sixteen needles at once.

Picking the right needle needs the query's particular key. Two things are
missing from the current setup, and both are captured here in one pass.

THE QUERY'S KEY, IN KEY SPACE. answer_prefix spells it out: "The special
magic number for telling-chatter mentioned in the provided text is". The
router reads only the last position's hidden state, so the key's identity has
to survive a few tokens of "mentioned in the provided text is" and a learned
projection. But the model already computes a key projection, and the needle
segment contains the same string, so the query key's own k vectors can be
matched against the segment descriptors with NO learned parameters. That is
the cheapest thing that could work, so it is measured first.

SUB-SEGMENT DESCRIPTORS. gamma_i is the mean of 256 L2-normalized keys, which
dilutes any single key by 1/256. Splitting a segment into m blocks and
scoring by the best-matching block (max-sim, as dense retrievers do over
tokens) recovers identity at m times the memory. m=1 IS the deployed
descriptor, so storing m=8 and averaging adjacent blocks offline gives the
whole curve — how much identity does each doubling of segment memory buy —
from a single capture.

Layers 0 and 1 only: that is where routing signal lives (gold-AUC 0.742 /
0.706 against 0.51-0.55 everywhere else) and where the shared decision is
computed. All sixteen layers at m=8 would be ~16 MB per sample.

Usage:
  python dsc/scripts/capture_key_identity.py --ckpt <ckpt> \
      --data-root /root/dk_data_full --cells 8192:4 8192:16 --seeds 42 43 \
      --max-samples 50 --blocks 8 --layers 0 1 --out /root/keyid_cache.pt
"""
from __future__ import annotations

import argparse, json, os, re, sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "dsc")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dsc.mc_baseline.mc_ssc_mlp_router import (  # noqa: E402
    block_weighted_sums, surprisal_weights, token_surprisal)

KEY_RE = re.compile(r"magic number(?:s)? for (.+?) mentioned", re.I)


def query_key(sample: dict) -> str | None:
    m = KEY_RE.search(sample.get("answer_prefix") or "")
    return m.group(1).strip() if m else None


def answer_token_span(tok, prompt: str, answer, hint: int | None = None
                     ) -> tuple[int, int] | None:
    """Token span of the answer VALUE inside the prompt.

    `key_token_span` deliberately takes the LAST occurrence of the key, which
    is the question's copy in the final segment. The surprisal premise is
    about the needle instead: the answer value sitting in the gold segment.
    Those are different spans and conflating them makes the premise
    unfalsifiable, since the question's copy is trivially predictable.

    `hint` is `token_position_answer`; the occurrence nearest to it is taken,
    because a magic number can appear both in the needle and in the answer
    prefix of some templates.
    """
    text = answer[0] if isinstance(answer, (list, tuple)) else answer
    if not text:
        return None
    best = None
    start = 0
    while True:
        c = prompt.find(str(text), start)
        if c < 0:
            break
        start = c + 1
        pre = len(tok(prompt[:c], add_special_tokens=False).input_ids)
        span = (pre, pre + len(tok(str(text), add_special_tokens=False).input_ids))
        if hint is None:
            return span
        if best is None or abs(span[0] - hint) < abs(best[0] - hint):
            best = span
    return best


def key_token_span(tok, prompt: str, key: str) -> tuple[int, int] | None:
    """Token span of the LAST occurrence of `key` — the one in the question,
    not the one buried in the haystack."""
    ch = prompt.rfind(key)
    if ch < 0:
        return None
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    lo, hi = None, None
    for t, (a, b) in enumerate(enc["offset_mapping"]):
        if b > ch and a < ch + len(key):
            lo = t if lo is None else lo
            hi = t + 1
    return (lo, hi) if lo is not None else None


def block_means(keys: torch.Tensor, chunk: int, blocks: int) -> torch.Tensor:
    """[B,T,H,Kd] -> [B,Nseg,blocks,H,Kd]. Block b of a segment is the mean of
    its slice of the chunk; averaging adjacent blocks offline recovers any
    coarser m, and m=1 reproduces segment_key_sums exactly."""
    b, t, h, kd = keys.shape
    nseg = (t + chunk - 1) // chunk
    pad = nseg * chunk - t
    if pad:
        keys = torch.cat([keys, keys.new_zeros(b, pad, h, kd)], dim=1)
    per = chunk // blocks
    if chunk % blocks:
        raise ValueError(f"chunk {chunk} not divisible by blocks {blocks}")
    return keys.view(b, nseg, blocks, per, h, kd).mean(dim=3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config-name", default="mc_370M")
    ap.add_argument("--tokenizer", default="TinyLlama/TinyLlama_v1.1")
    ap.add_argument("--cells", nargs="+", default=["8192:4", "8192:16"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0],
                    help="surprisal exponents to store descriptors for. Each "
                         "one costs another copy of the block sums, so keep "
                         "--layers short when sweeping many.")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()

    from lit_gpt.config import Config
    from lit_gpt.model import GPT
    cfg = Config.from_name(args.config_name, mc_topk=2)
    chunk = getattr(cfg, "mc_chunk_size", 256)
    model = GPT(cfg).to("cuda").to(torch.bfloat16).eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"ckpt/config mismatch: {len(missing)} missing, "
                           f"{len(unexpected)} unexpected")

    # Same wrap-only discipline as every other capture here: shadow
    # GDN2SSC.forward, stash the tensors it was handed, then run the original
    # math so the residual stream is the real one. `keys` is already
    # routing_keys = L2norm(k), which is what the deployed router scores.
    aggs = []
    for m in model.modules():
        if m.__class__.__name__ != "GDN2SSC":
            continue

        def mk(agg, orig):
            def fwd(hidden_states, queries, keys, online_output, memories):
                agg.cap_h = hidden_states.detach()
                agg.cap_k = keys.detach()
                return orig(hidden_states, queries, keys, online_output,
                            memories)
            return fwd

        m.forward = mk(m, m.forward)
        aggs.append(m)
    if not aggs:
        raise RuntimeError("found 0 GDN2SSC aggregators")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    rows, skipped, no_needle = [], 0, 0
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
                key = query_key(s)
                prompt = s["input"] + s.get("answer_prefix", "")
                span = key_token_span(tok, prompt, key) if key else None
                if span is None:
                    skipped += 1
                    continue
                ids = tok(prompt, return_tensors="pt",
                          add_special_tokens=False).input_ids.to("cuda")
                for a in aggs:
                    a.cap_h = a.cap_k = None
                with torch.no_grad(), torch.autocast("cuda",
                                                     dtype=torch.bfloat16):
                    out = model(ids)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                sur = token_surprisal(logits.float(), ids)[0]  # [T], nats
                lo, hi = span
                rec = {"cell": f"seed{seed}_len{ctx}_n{ndl}",
                       "sample_index": i, "key": key,
                       "gold": s["token_position_answer"] // chunk,
                       "nseg": (ids.shape[1] + chunk - 1) // chunk,
                       "key_span": [lo, hi]}
                nspan = answer_token_span(
                    tok, prompt, s.get("outputs"),
                    s.get("token_position_answer"))
                if nspan is None:
                    no_needle += 1
                    continue
                rec["needle_span"] = list(nspan)
                rec["surprisal"] = sur.cpu().numpy().astype(np.float16)
                for li, L in enumerate(args.layers):
                    a = aggs[L]
                    blk = block_means(a.cap_k.float(), chunk, args.blocks)
                    rec[f"g{L}"] = blk[0].cpu().numpy().astype(np.float16)
                    # Store the block-level SUMS rather than the means, one
                    # pair per tau. Sums compose: adjacent blocks merge by
                    # adding numerator and denominator, so a single capture
                    # serves both the tau sweep and the m sweep. Means would
                    # only serve the tau sweep.
                    for ti, tau in enumerate(args.taus):
                        w = surprisal_weights(sur[None], tau)
                        num, den = block_weighted_sums(
                            a.cap_k.float(), chunk, args.blocks, w)
                        rec[f"gs{L}_t{ti}"] = num[0].cpu().numpy(
                        ).astype(np.float16)
                        rec[f"ws{L}_t{ti}"] = den[0].cpu().numpy(
                        ).astype(np.float32)
                    rec[f"h{L}"] = a.cap_h[0, -1].float().cpu().numpy(
                    ).astype(np.float16)
                    # The query key's own k vectors, averaged over its tokens:
                    # no learned parameters, same space as the descriptors.
                    rec[f"kq{L}"] = a.cap_k[0, lo:hi].float().mean(
                        dim=0).cpu().numpy().astype(np.float16)
                    rec[f"hq{L}"] = a.cap_h[0, lo:hi].float().mean(
                        dim=0).cpu().numpy().astype(np.float16)
                rows.append(rec)
            print(f"[cap] seed{seed} {cell}: {len(rows)} rows so far",
                  flush=True)

    if skipped:
        print(f"[warn] {skipped} samples had no locatable query key", flush=True)
    if no_needle:
        print(f"[warn] {no_needle} samples had no locatable answer span",
              flush=True)
    if not rows:
        raise RuntimeError("captured nothing")
    meta = {"D": cfg.n_embd, "H": aggs[0].num_heads, "Kd": aggs[0].head_qk_dim,
            "chunk": chunk, "blocks": args.blocks, "layers": args.layers,
            "taus": list(args.taus)}
    torch.save({"rows": rows, "meta": meta}, args.out)
    print(f"[capture] {len(rows)} rows, meta {meta}")
    print(f"[capture] wrote {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
