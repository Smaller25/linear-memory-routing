#!/usr/bin/env python3
"""Wiring gate for a-1 stage 2. Run BEFORE reading any diverse-key score.

An injection that loads the wrong file onto the wrong layer, or that scores a
descriptor the router was not trained on, still produces a complete run with
a plausible number. The first a-1 capture failed exactly that way: its shadow
returned zeros from the layer forward, which deletes every attention
sublayer, and it pooled raw projected keys instead of the normalized routing
keys — and nothing downstream complained.

Three checks on the LIVE injected model, over held-out samples:

  A. formula equivalence — the in-model routing score equals an offline
     recompute from the tensors that forward received (catches an einsum,
     normalization, or head-layout mismatch between train and inject).
  B. AUC reproduction — the injected router's gold-AUC matches the value
     train_mlp_router.py wrote for that layer (catches per-layer file
     misalignment and any train/inject descriptor mismatch).
  C. native control — the untouched linear connector, scored on the same
     in-model tensors, sits at its known chance level (catches a capture or
     forward that made the task easier than the real one).

Usage:
  python dsc/scripts/verify_mlp_router_injection.py \
      --ckpt /root/dk_local/ckpts/ssc30b/checkpoint-30B-model-ckpt.pth \
      --router-dir /root/mlp_router_v2 --data-root /root/dk_data_full \
      --cells 8192:16 8192:4 --seed 43 --max-samples 400

--cells must match the val cells train_mlp_router.py used, or check B
compares AUCs measured on different held-out sets and can fail for a reason
that has nothing to do with the wiring.
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

AUC_TOL = 0.03
# Relative, not absolute: what matters is the score noise against the spread
# the ranking has to resolve. An absolute bound is uncalibrated — the same
# 8e-3 is fatal for segments 1e-3 apart and irrelevant for segments 1 apart.
FORMULA_REL_TOL = 0.02
CONTROL_RANGE = (0.35, 0.60)


def gold_auc(scores, gold_idx):
    from sklearn.metrics import roc_auc_score
    y, x = [], []
    for sc, gd in zip(scores, gold_idx):
        for j, v in enumerate(sc):
            y.append(1 if j == gd else 0); x.append(float(v))
    return float(roc_auc_score(y, x)) if 0 < sum(y) < len(y) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--router-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--config-name", default="mc_370M")
    ap.add_argument("--config-overrides", nargs="*", default=["mc_topk=2"])
    ap.add_argument("--tokenizer", default="TinyLlama/TinyLlama_v1.1")
    ap.add_argument("--cells", nargs="+", default=["8192:16", "8192:4"],
                    help="held-out cells; must match train_mlp_router.py's "
                         "--val-cells so check B compares like with like")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from lit_gpt.config import Config
    from lit_gpt.model import GPT
    ov = {k: int(v) for k, v in (p.split("=", 1) for p in args.config_overrides)}
    cfg = Config.from_name(args.config_name, **ov)
    chunk = getattr(cfg, "mc_chunk_size", 256)
    model = GPT(cfg).to("cuda").to(torch.bfloat16).eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"ckpt/config mismatch: {len(missing)} missing, "
                           f"{len(unexpected)} unexpected")

    from dsc.mc_baseline.mc_ssc_mlp_router import enable_mlp_router
    # observe: score and log, but leave selection and gating on the native
    # path. Checks A-C are about the wiring, and the routers were fitted on
    # activations from a natively routed model — letting the MLP steer here
    # would shift every downstream layer's input and make check B fail for a
    # reason that is not a wiring defect.
    injected = enable_mlp_router(model, args.router_dir, None, "observe")
    print(f"[gate] injected layers {injected} (mode=observe)", flush=True)

    mc_layers = [m for m in model.modules()
                 if m.__class__.__name__ == "MemoryCachingGDN2Layer"]
    aggs = [lyr.ssc for lyr in mc_layers]
    for agg in aggs:
        if getattr(agg, "mlp_router", None) is not None:
            agg.log_mlp_scores = True

    # Wrap the (already injected) instance forward to stash the tensors it
    # receives, then call it. Instance attributes win over the class, so this
    # sits on top of the injected forward without replacing it.
    for agg in aggs:
        def mk(a, orig):
            def fwd(hidden_states, queries, keys, online_output, memories):
                a.cap_h = hidden_states[:, -1].detach().float().cpu()
                a.cap_g = segment_key_sums(
                    keys, a.chunk_size).detach().float().cpu()
                return orig(hidden_states, queries, keys, online_output, memories)
            return fwd
        agg.forward = mk(agg, agg.forward)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    samples = []
    for cell in args.cells:
        ctx, ndl = (int(x) for x in cell.split(":"))
        path = os.path.join(args.data_root, f"seed{args.seed}", str(ctx),
                            f"niah_diversekey_essay_{ndl}", "validation.jsonl")
        got = [json.loads(l) for l in open(path)][:args.max_samples]
        samples += got
        print(f"[gate] {len(got)} samples from {path}", flush=True)
    if not samples:
        raise RuntimeError("no held-out samples found")

    per_layer = {L: {"mlp": [], "native": [], "gold": [], "dmax": 0.0,
                     "spread": []} for L in injected}
    for n, s in enumerate(samples):
        prompt = s["input"] + s.get("answer_prefix", "")
        ids = tok(prompt, return_tensors="pt",
                  add_special_tokens=False).input_ids.to("cuda")
        for a in aggs:
            a.cap_h = a.cap_g = None
            a.last_mlp_scores = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(ids)
        gold = s["token_position_answer"] // chunk
        for L in injected:
            agg = aggs[L]
            nseg = agg.cap_g.shape[1]
            elig = nseg - 1
            if elig <= 0 or gold >= elig:
                continue
            h = agg.cap_h.cuda()
            g = agg.cap_g[:, :elig].cuda()
            head = agg.mlp_router
            with torch.no_grad():
                off = head.scores(h[:, None], g)[0, 0].cpu()
                u_nat = agg.connector(
                    h.to(agg.connector.weight.dtype)
                ).view(agg.num_heads, agg.head_qk_dim).float()
                nat = torch.einsum("hk,nhk->n", u_nat, g[0].float()).cpu()
            live = agg.last_mlp_scores[0, -1, :elig]
            per_layer[L]["dmax"] = max(per_layer[L]["dmax"],
                                       float((off - live).abs().max()))
            per_layer[L]["spread"].append(float(live.max() - live.min()))
            per_layer[L]["mlp"].append(live.tolist())
            per_layer[L]["native"].append(nat.tolist())
            per_layer[L]["gold"].append(gold)
        if (n + 1) % 20 == 0:
            print(f"[gate] {n + 1}/{len(samples)}", flush=True)

    verdict_path = os.path.join(args.router_dir, "verdict.json")
    trained = json.load(open(verdict_path)) if os.path.exists(verdict_path) else {}

    report, failures = {}, []
    for L in injected:
        d = per_layer[L]
        if not d["mlp"]:
            failures.append(f"L{L}: no usable samples"); continue
        a_mlp = gold_auc(d["mlp"], d["gold"])
        a_nat = gold_auc(d["native"], d["gold"])
        ref = (trained.get(f"layer{L}") or {}).get("gold_auc")
        spread = sum(d["spread"]) / len(d["spread"])
        rel = d["dmax"] / spread if spread > 0 else float("inf")
        row = {"in_model_mlp_auc": a_mlp, "in_model_native_auc": a_nat,
               "trained_auc": ref, "formula_max_abs_diff": d["dmax"],
               "mean_score_spread": spread, "formula_rel_diff": rel,
               "n": len(d["mlp"])}
        if rel > FORMULA_REL_TOL:
            failures.append(f"L{L}: A formula mismatch, max|off-live|="
                            f"{d['dmax']:.2e} = {rel:.1%} of the {spread:.3f} "
                            f"score spread (> {FORMULA_REL_TOL:.0%})")
        if ref is not None and a_mlp is not None and abs(a_mlp - ref) > AUC_TOL:
            failures.append(f"L{L}: B AUC {a_mlp:.3f} != trained {ref:.3f} "
                            f"(tol {AUC_TOL})")
        if a_nat is not None and not CONTROL_RANGE[0] <= a_nat <= CONTROL_RANGE[1]:
            failures.append(f"L{L}: C native control {a_nat:.3f} outside "
                            f"{CONTROL_RANGE} — the in-model task is not the "
                            "one the baseline was measured on")
        report[f"layer{L}"] = row
        print(f"[L{L}] mlp={a_mlp} native={a_nat} trained={ref} "
              f"dmax={d['dmax']:.2e} ({rel:.2%} of spread {spread:.3f})",
              flush=True)

    # Informational second pass. Under select the MLP actually steers, so
    # every layer past the first sees hidden states the router was not fitted
    # on. That drift is the thing to know if the score comes out flat: it
    # separates "routing did not improve in-model" from "routing improved but
    # the top-k threshold still was not crossed". Not gated on.
    if not failures:
        for L in injected:
            aggs[L].mlp_router_mode = "select"
        shifted = {L: {"mlp": [], "gold": []} for L in injected}
        for s_i, smp in enumerate(samples):
            prompt = smp["input"] + smp.get("answer_prefix", "")
            ids = tok(prompt, return_tensors="pt",
                      add_special_tokens=False).input_ids.to("cuda")
            for a in aggs:
                a.last_mlp_scores = None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(ids)
            gd = smp["token_position_answer"] // chunk
            for L in injected:
                sc = aggs[L].last_mlp_scores
                elig = sc.shape[-1] - 1
                if elig <= 0 or gd >= elig:
                    continue
                shifted[L]["mlp"].append(sc[0, -1, :elig].tolist())
                shifted[L]["gold"].append(gd)
        for L in injected:
            if shifted[L]["mlp"]:
                a_sh = gold_auc(shifted[L]["mlp"], shifted[L]["gold"])
                report[f"layer{L}"]["select_mode_auc"] = a_sh
                print(f"[L{L}] select-mode in-model AUC = {a_sh}", flush=True)

    report["mode"] = "observe (+ select second pass)"
    report["cells"] = args.cells
    report["seed"] = args.seed
    report["verdict"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures
    out = args.out or os.path.join(args.router_dir, "injection_gate.json")
    json.dump(report, open(out, "w"), indent=1)
    print(json.dumps({"verdict": report["verdict"], "failures": failures},
                     indent=1))
    print(f"[gate] wrote {out}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
