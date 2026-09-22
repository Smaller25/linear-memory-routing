#!/usr/bin/env python3
"""Train a per-segment soft prompt to make segments routable. Track 2 of 0027.

Only `n_prefix + n_suffix` embedding vectors are trained; the backbone stays
frozen and never receives a gradient (asserted, not assumed). The objective is
contrastive over segments: the router's own score for the gold segment against
every other eligible segment, cross-entropy on the gold index. That is the
same target `train_mlp_router.py` optimises, so a gain here is attributable to
the prompt and not to a different loss.

Two guards are not optional, both paid for by earlier failures in this project.

**A perplexity regression check.** This is the first intervention here that
touches the prefill rather than the routing score. The uniform gate margin also
touched every prefill position, looked oracle-like at the query position, and
still collapsed the score to 0.0 at N=16 because the residual stream was
corrupted (0027 §5). Routing hit alone would not have caught that. So the
script measures held-out token perplexity with the prompt attached and refuses
to report a routing number if it has drifted past --ppl-tolerance.

**A p=0 control in the same run.** A prefix of zero vectors is still a token
and does not recover the unmodified model, so `p=0` -- no insertion at all --
is the only exact baseline. Measuring it here rather than quoting an earlier
number keeps the comparison inside one protocol.

Usage:
  python dsc/scripts/train_soft_prompt.py --ckpt CKPT --config-name mc_370M \
      --data-root DATA --n-prefix 8 --steps 2000 --out /root/sp_p8
"""
from __future__ import annotations

import argparse, json, os, sys

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "dsc")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dsc.mc_baseline.mc_ssc import segment_key_sums  # noqa: E402
from dsc.mc_baseline.mc_ssc_soft_prompt import (  # noqa: E402
    attach_soft_prompt, expand_ids, plan_expansion, prompt_state_dict,
    trainable_report)


def capture_router_inputs(model):
    """Shadow every GDN2SSC so one forward yields (h, keys) per layer.

    Calls the original forward and returns its result. A shadow that skipped
    the wrapped call once deleted sixteen attention sublayers here and
    produced a plausible AUC that had to be withdrawn, so calling `orig` is
    the only acceptable shape for a capture hook in this repository.
    """
    aggs = {}
    for i, m in enumerate(
            [m for m in model.modules() if m.__class__.__name__ == "GDN2SSC"]):
        def mk(agg, orig):
            def fwd(hidden_states, queries, keys, online_output, memories):
                agg.cap_h, agg.cap_k = hidden_states, keys
                return orig(hidden_states, queries, keys, online_output,
                            memories)
            return fwd
        m.forward = mk(m, m.forward)
        aggs[i] = m
    if not aggs:
        raise RuntimeError("no GDN2SSC layers found — wrong config?")
    return aggs


def route_logits(layer, chunk: int, n_elig: int) -> torch.Tensor:
    """[n_elig] deployed routing scores at the last position.

    Uses the layer's own frozen connector and the same descriptor the model
    routes with, so the only thing the gradient can change is what the prompt
    made the keys become.
    """
    h, k = layer.cap_h, layer.cap_k
    H, Kd = k.shape[-2], k.shape[-1]
    u = layer.connector(h[:, -1]).view(1, H, Kd).float()
    g = segment_key_sums(k, chunk)[:, :n_elig].float()
    return torch.einsum("bhk,bnhk->bn", u, g)[0]


def token_ppl(model, ids, pm, sp) -> float:
    """Held-out token perplexity, skipping targets a prompt slot displaced.

    The prompt makes the LM objective ill-defined exactly at segment
    boundaries: the position that used to predict the next token now predicts
    a trained vector with no token id. Those positions are excluded rather
    than scored against an arbitrary target, and the exclusion is the same
    with and without the prompt so the two numbers stay comparable.
    """
    with torch.no_grad():
        if pm is None:
            sp.set_plan(None) if sp is not None else None
            logits = model(ids)
            pred, tgt = logits[0, :-1], ids[0, 1:]
        else:
            sp.set_plan(pm)
            logits = model(expand_ids(ids, pm))
            src = pm.new_of_old
            keep = (src[1:] - src[:-1]) == 1
            pred = logits[0, src[:-1][keep]]
            tgt = ids[0, 1:][keep]
        return float(torch.exp(F.cross_entropy(pred.float(), tgt)))


def load_rows(data_root, seed, cells, max_samples, tok, chunk):
    rows = []
    for cell in cells:
        ctx, ndl = (int(x) for x in cell.split(":"))
        path = os.path.join(data_root, f"seed{seed}", str(ctx),
                            f"niah_diversekey_essay_{ndl}", "validation.jsonl")
        if not os.path.exists(path):
            print(f"[skip] {path}", flush=True)
            continue
        for i, line in enumerate(open(path)):
            if i >= max_samples:
                break
            s = json.loads(line)
            prompt = s["input"] + s.get("answer_prefix", "")
            ids = tok(prompt, return_tensors="pt",
                      add_special_tokens=False).input_ids
            gold = s["token_position_answer"] // chunk
            nseg = (ids.shape[1] + chunk - 1) // chunk
            if gold >= nseg - 1:
                continue          # answer is in the current segment: unroutable
            rows.append({"ids": ids, "gold": gold, "nseg": nseg,
                         "cell": f"seed{seed}_len{ctx}_n{ndl}"})
    if not rows:
        raise RuntimeError(f"no usable rows under {data_root}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-name", default="mc_370M")
    ap.add_argument("--config-overrides", default="mc_topk=2")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tokenizer", default="TinyLlama/TinyLlama_v1.1")
    ap.add_argument("--train-cells", nargs="+",
                    default=["2048:8", "4096:8", "8192:8", "8192:16"])
    ap.add_argument("--train-seed", type=int, default=45,
                    help="must be a seed the evaluation never uses")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--n-prefix", type=int, default=8)
    ap.add_argument("--n-suffix", type=int, default=0)
    ap.add_argument("--warm-start", default=".",
                    help="text whose token embeddings seed the prompt. Random "
                         "init costs the first thousand steps; a punctuation "
                         "or separator token is a cheap, in-distribution "
                         "starting point")
    ap.add_argument("--layers", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ppl-tolerance", type=float, default=0.05,
                    help="fractional held-out ppl increase that fails the run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[sp] training {args.n_prefix}+{args.n_suffix} vectors and "
          f"nothing else", flush=True)

    from lit_gpt.config import Config
    from lit_gpt.model import GPT
    from transformers import AutoTokenizer
    over = dict(kv.split("=") for kv in args.config_overrides.split(",") if kv)
    cfg = Config.from_name(args.config_name,
                           **{k: int(v) for k, v in over.items()})
    chunk = getattr(cfg, "mc_chunk_size", 256)
    model = GPT(cfg).to(args.device).to(torch.bfloat16).eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"ckpt/config mismatch: {len(missing)} missing, "
                           f"{len(unexpected)} unexpected")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    ws = tok(args.warm_start, add_special_tokens=False).input_ids or None
    sp = attach_soft_prompt(model, args.n_prefix, args.n_suffix, ws)
    print("[sp] " + trainable_report(sp), flush=True)
    n_back = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_prompt = sum(v.numel() for v in (sp.prefix, sp.suffix) if v is not None)
    if n_back != n_prompt:
        raise RuntimeError(
            f"{n_back} trainable parameters but the prompt is only "
            f"{n_prompt} — the backbone is not frozen")

    aggs = capture_router_inputs(model)
    src = args.layers[0]
    if src not in aggs:
        raise RuntimeError(f"source layer {src} outside 0..{len(aggs) - 1}")
    rows = load_rows(args.data_root, args.train_seed, args.train_cells,
                     args.max_samples, tok, chunk)
    print(f"[sp] {len(rows)} training rows, source layer L{src}", flush=True)

    # Perplexity before anything is trained, on a row held out of training.
    held = rows.pop()
    ppl0 = token_ppl(model, held["ids"].to(args.device), None, sp)

    opt = torch.optim.AdamW([v for v in (sp.prefix, sp.suffix)
                             if v is not None], lr=args.lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(0)
    hits = losses = seen = 0
    for step in range(args.steps):
        r = rows[int(torch.randint(len(rows), (1,), generator=gen))]
        ids = r["ids"].to(args.device)
        pm = plan_expansion(ids.shape[1], chunk, args.n_prefix,
                            args.n_suffix, device=args.device)
        sp.set_plan(pm)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(expand_ids(ids, pm))
        # gold is unchanged by construction: the layout keeps
        # position // chunk_out equal to the original position // chunk_in
        n_elig = r["nseg"] - 1
        logits = route_logits(aggs[src], pm.chunk_out, n_elig)
        loss = F.cross_entropy(logits[None],
                               torch.tensor([r["gold"]], device=args.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if sp.wte.weight.grad is not None:
            raise RuntimeError("the embedding table received a gradient — "
                               "the backbone is not frozen")
        opt.step()
        losses += float(loss)
        hits += int(logits.argmax().item() == r["gold"])
        seen += 1
        if (step + 1) % 200 == 0:
            print(f"[sp] step {step + 1:>5}  loss {losses / seen:.4f}  "
                  f"train top1 {hits / seen:.3f}", flush=True)
            hits = losses = seen = 0

    # The gate. Routing hit alone would not have caught the gate-margin
    # failure, which looked oracle-like at the query position and still
    # corrupted every prefill position.
    ppl1 = token_ppl(model, held["ids"].to(args.device),
                     plan_expansion(held["ids"].shape[1], chunk,
                                    args.n_prefix, args.n_suffix,
                                    device=args.device), sp)
    drift = (ppl1 - ppl0) / ppl0
    verdict = {"n_prefix": args.n_prefix, "n_suffix": args.n_suffix,
               "warm_start": args.warm_start, "chunk_in": chunk,
               "chunk_out": chunk + args.n_prefix + args.n_suffix,
               "train_seed": args.train_seed, "steps": args.steps,
               "source_layer": src, "n_rows": len(rows),
               "ppl_before": ppl0, "ppl_after": ppl1, "ppl_drift": drift,
               "trainable": sum(v.numel() for v in (sp.prefix, sp.suffix)
                                if v is not None)}
    torch.save(prompt_state_dict(sp), os.path.join(args.out, "prompt.pt"))
    json.dump(verdict, open(os.path.join(args.out, "verdict.json"), "w"),
              indent=1)
    print(f"[sp] ppl {ppl0:.3f} -> {ppl1:.3f} ({drift:+.1%})", flush=True)
    if drift > args.ppl_tolerance:
        raise SystemExit(
            f"[sp] FAILED the perplexity gate: {drift:+.1%} exceeds "
            f"{args.ppl_tolerance:+.1%}. A routing number from this prompt "
            "would be measuring a corrupted prefill, which is how the gate "
            "margin passed its own read check and still scored 0.0 at N=16.")
    print(f"[sp] wrote {args.out}/prompt.pt and verdict.json", flush=True)
    return 0
