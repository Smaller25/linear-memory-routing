#!/usr/bin/env python3
"""a-1: train an MLP router on FROZEN backbone activations (retrieval CE).

Backbone is frozen, so per-layer query-position hidden states h and segment
descriptors gamma are FIXED. Only the router u = f(h) is learned. So we
capture (h, gamma, gold) ONCE with one backbone forward per sample, then fit
the router offline on the cached tensors — no repeated GPU forward.

Capture correctness (v2, 2026-09-07). The first version shadowed the LAYER
forward and returned zeros to skip the SSC read. Block.forward is x = x + h,
so that silently deleted all 16 GDN-2 attention sublayers: every layer past 0
was captured from a model that does not exist, and it also pooled the RAW
projected k instead of the L2-normalized routing_keys the router actually
scores (gdn2_ssc_forward normalizes before pooling; segment_key_sums does
not). Both made the reported gold-AUC unusable. v2 shadows GDN2SSC.forward
instead, stashes exactly the tensors that forward receives, and calls the
original math — the residual stream stays intact and the descriptor is the
deployed one. Cost: the real read runs (one plain eval forward per sample).

Router per layer (independent), replacing the linear connector W_u:
    u_h = L2norm( MLP(h) )            # MLP: Linear(D, D) - GELU - Linear(D, H*Kd)
    score_i = < u , L2norm(gamma_i) > summed over heads     # cosine, scaled
    loss = CE( softmax(scores over eligible segs), gold )   # + needle aux
Success signal: gold-AUC of the learned score (linear router baseline ~0.46).

Stage 2 (separate) injects the trained MLP into the model for a real
diverse-key eval; this script writes router weights + the AUC verdict.

Usage:
  python dsc/scripts/train_mlp_router.py --ckpt <ssc30b.pth> \
      --data-root /root/dk_data_full --train-cells 2048:8 4096:8 8192:8 \
      --val-cells 8192:16 --max-samples 400 --out /root/mlp_router_v2

--layers defaults to all 16 MC layers: the capture forward is shared, so
extra layers cost cache size and per-layer fit time, not GPU forwards. All 16
matter because routing is per-layer independent — injecting a subset leaves
the rest on the chance-level linear router and makes a flat result unreadable.
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "dsc")):
    if p not in sys.path:
        sys.path.insert(0, p)
from dsc.mc_baseline.mc_ssc import segment_key_sums  # noqa: E402

NEEDLE = "special magic numbers for "


def needle_segs(tok, text, chunk):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]; segs = set(); start = 0
    while (i := text.find(NEEDLE, start)) >= 0:
        for t, (a, b) in enumerate(off):
            if a <= i < b:
                segs.add(t // chunk); break
        start = i + 1
    return segs


def capture(model, aggs, tok, data_root, cells, seed, layers, max_s, chunk):
    """Return list of per-sample dicts: h[L,D], gamma[L,Nseg,H,Kd], gold, ndl."""
    out = []
    for cell in cells:
        ctx, ndl = (int(x) for x in cell.split(":"))
        path = os.path.join(data_root, f"seed{seed}", str(ctx),
                            f"niah_diversekey_essay_{ndl}", "validation.jsonl")
        if not os.path.exists(path):
            print(f"[skip] no {path}"); continue
        samples = [json.loads(l) for l in open(path)][:max_s]
        for s in samples:
            prompt = s["input"] + s.get("answer_prefix", "")
            ids = tok(prompt, return_tensors="pt",
                      add_special_tokens=False).input_ids.to("cuda")
            for a in aggs:
                a.cap_h = a.cap_g = None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(ids)
            missing = [l for l in layers if aggs[l].cap_h is None]
            if missing:
                raise RuntimeError(
                    f"capture hook did not fire for layers {missing} — the "
                    "shadow is not on the forward that actually runs")
            h = np.stack([aggs[l].cap_h[0].numpy() for l in layers])       # [L,D]
            g = np.stack([aggs[l].cap_g[0].numpy() for l in layers])       # [L,Nseg,H,Kd]
            out.append(dict(h=h.astype(np.float16), g=g.astype(np.float16),
                            gold=s["token_position_answer"] // chunk,
                            ndl=needle_segs(tok, s["input"], chunk),
                            nseg=g.shape[1]))
        print(f"[cap] {cell}: {len(samples)} samples")
    return out


class MLPRouter(nn.Module):
    """u = f(h) scored against the segment descriptors.

    ``scorer="cos"`` is the deployed form: L2-normalize u and gamma per head,
    then a scaled dot. ``scorer="dot"`` drops both normalizations and learns
    the temperature instead.

    The normalization was a training-stability choice — segment_key_sums
    mean-pools so the descriptor magnitude stays near 1 and softmax does not
    collapse at init — and it costs routing information. gamma's magnitude
    says how concentrated a segment's keys are (256 similar keys average to
    something large, scattered keys to something small), which bears directly
    on whether a segment holds a distinctive phrase. Held-out probe, val
    hit@2: L0 0.220 -> 0.280, L1 0.240 -> 0.300, and L0 hit@1 0.040 -> 0.180,
    at identical cost (`probe_router_ceiling.py`).
    """

    def __init__(self, D, H, Kd, hidden=None, scorer="cos"):
        super().__init__()
        if scorer not in ("cos", "dot"):
            raise ValueError(f"scorer must be 'cos' or 'dot', got {scorer!r}")
        hidden = hidden or D
        self.H, self.Kd, self.scorer = H, Kd, scorer
        self.net = nn.Sequential(nn.Linear(D, hidden), nn.GELU(),
                                 nn.Linear(hidden, H * Kd))
        self.scale = Kd ** -0.5
        if scorer == "dot":
            self.logit_scale = nn.Parameter(torch.zeros(1))

    def scores(self, h, gamma):
        # h:[B,D] gamma:[B,Nseg,H,Kd] -> scores:[B,Nseg]
        u = self.net(h).view(-1, self.H, self.Kd)
        if self.scorer == "dot":
            return torch.einsum("bhk,bnhk->bn", u, gamma.float()) \
                * self.logit_scale.exp()
        u = F.normalize(u, dim=-1)
        gam = F.normalize(gamma.float(), dim=-1)
        return torch.einsum("bhk,bnhk->bn", u, gam) * self.scale


def linear_router_auc(rows_h, rows_g, gold, W, H, Kd):
    """gold-AUC of the NATIVE linear router on the captured tensors.

    Positive control on the capture itself, not a result. The native router
    is known to score at chance on this task (~0.46 measured in-model), so a
    correct capture must reproduce chance here. The first capture version
    returned zeros from the layer forward, which deleted every attention
    sublayer and turned the descriptor into a near-lexical needle detector —
    that inflates BOTH the linear and the MLP score, so this number is what
    catches it before a single GPU-hour goes into the wrong cache.
    """
    scs, gidx, emask = [], [], []
    for h, g, gd in zip(rows_h, rows_g, gold):
        elig = g.shape[0] - 1
        if elig <= 0 or gd >= elig:
            continue
        u = (torch.as_tensor(h).float() @ W.T.float()).view(H, Kd)
        sc = torch.einsum("hk,nhk->n", u, torch.as_tensor(g[:elig]).float())
        scs.append(sc.tolist()); gidx.append(gd); emask.append([True] * elig)
    return auc(scs, gidx, emask)


def hit_at_k(scores, gold_idx, ks=(1, 2, 4, 8)):
    """Fraction of samples where gold lands in the top k.

    This, not AUC, is what the score follows. Measured on the broadcast arm:
    score ~= baseline + hit@2 * (oracle - baseline), with no conversion loss
    at k=2 (N=4: predicted 14.0 routing gain, observed 14.0). AUC tolerates
    reorderings among the leaders; top-2 membership does not, so optimizing
    AUC is optimizing the wrong thing once the picks are broadcast.
    """
    out = {k: 0 for k in ks}
    n = 0
    for sc, gd in zip(scores, gold_idx):
        if gd < 0 or len(sc) == 0 or gd >= len(sc):
            continue  # gold sits in the excluded current segment
        t = torch.as_tensor(sc)
        rank = int((t.argsort(descending=True) == gd).nonzero()[0, 0])
        for k in ks:
            out[k] += int(rank < k)
        n += 1
    return {f"hit@{k}": (out[k] / n if n else None) for k in ks} | {"n": n}


def auc(scores, gold_idx, eligible_mask):
    from sklearn.metrics import roc_auc_score
    y, x = [], []
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            if eligible_mask[i][j]:
                y.append(1 if j == gold_idx[i] else 0); x.append(scores[i][j])
    return float(roc_auc_score(y, x)) if 0 < sum(y) < len(y) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-name", default="mc_370M")
    ap.add_argument("--config-overrides", nargs="*", default=["mc_topk=2"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--train-cells", nargs="+", default=["2048:8", "4096:8", "8192:8"])
    ap.add_argument("--val-cells", nargs="+", default=["8192:16"])
    ap.add_argument("--train-seed", type=int, default=42)
    ap.add_argument("--val-seed", type=int, default=43)
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(16)),
                    help="MC-layer indices to fit a router for (default: all "
                         "16 — a partial injection cannot be read)")
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--needle-aux", type=float, default=0.0,
                    help="weight on 'raise all needle segments'. Default 0: "
                         "it opposes gold selection at high needle counts")
    ap.add_argument("--gold-vs-needle", type=float, default=0.0,
                    help="weight on CE restricted to needle segments — gold "
                         "against the other needles, the hard part at N=16")
    ap.add_argument("--scorer", default="cos", choices=["cos", "dot"],
                    help="cos: the deployed L2-normalized form. dot: drop "
                         "both normalizations, learn the temperature")
    ap.add_argument("--fit-seed", type=int, default=0,
                    help="router init seed, held fixed across loss variants")
    ap.add_argument("--control-only", action="store_true",
                    help="report the native linear router's gold-AUC for "
                         "every layer and stop, without fitting and without "
                         "raising. The 0.35-0.60 band was calibrated on the "
                         "370M; when a different model trips it, the first "
                         "question is whether the capture broke or the model "
                         "genuinely routes differently, and that is settled "
                         "by running this same path on the 370M.")
    ap.add_argument("--from-cache", default=None, metavar="CACHE_PT",
                    help="reuse a previous run's cache.pt and skip the capture "
                         "entirely (no backbone forward). Loss and metric "
                         "variants are then seconds apart, not an hour")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.from_cache:
        blob = torch.load(args.from_cache, map_location="cpu",
                          weights_only=False)
        tr, va, meta = blob["train"], blob["val"], blob["meta"]
        D, H, Kd = meta["D"], meta["H"], meta["Kd"]
        chunk = meta["chunk"]
        if sorted(args.layers) != sorted(meta["layers"]):
            raise RuntimeError(
                f"--layers {args.layers} not all in the cache "
                f"{meta['layers']} — the cache fixes which layers exist")
        conn = blob.get("connector")
        print(f"[cache] {args.from_cache}: train {len(tr)} val {len(va)}, "
              f"layers {meta['layers']}, "
              f"connector weights {'present' if conn else 'ABSENT'}",
              flush=True)
        fit_routers(args, tr, va, D, H, Kd, conn)
        return

    from lit_gpt.config import Config
    from lit_gpt.model import GPT
    ov = {k: int(v) for k, v in (p.split("=", 1) for p in args.config_overrides)}
    cfg = Config.from_name(args.config_name, **ov)
    chunk = getattr(cfg, "mc_chunk_size", 256)
    model = GPT(cfg).to("cuda").to(torch.bfloat16).eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(sd, strict=False)
    for p in model.parameters():
        p.requires_grad_(False)

    # Shadow GDN2SSC.forward and then run the ORIGINAL math. Its arguments are
    # by construction the router's real inputs: hidden_states is what the
    # connector consumes, and keys is routing_keys = L2norm(k), already
    # normalized by gdn2_ssc_forward before pooling (segment_key_sums only
    # mean-pools, it does not normalize). Calling orig() keeps the residual
    # stream intact, which shadowing the layer and returning zeros did not:
    # Block.forward is x = x + h, so a zero return deletes the attention
    # sublayer for every downstream layer. Same wrap-only pattern as
    # capture_descriptors.py and the eval hooks; mc_ssc.py stays untouched.
    aggs = []
    for m in model.modules():
        if m.__class__.__name__ == "GDN2SSC":
            def mk(agg, orig):
                def fwd(hidden_states, queries, keys, online_output, memories):
                    agg.cap_h = hidden_states[:, -1].detach().float().cpu()
                    agg.cap_g = segment_key_sums(
                        keys, agg.chunk_size).detach().float().cpu()
                    return orig(hidden_states, queries, keys, online_output,
                                memories)
                return fwd
            m.forward = mk(m, m.forward)
            aggs.append(m)
    if not aggs:
        raise RuntimeError("found 0 GDN2SSC aggregators to capture from")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
    D = cfg.n_embd
    H = aggs[0].num_heads; Kd = aggs[0].head_qk_dim

    print("[capture] train ...")
    tr = capture(model, aggs, tok, args.data_root, args.train_cells,
                 args.train_seed, args.layers, args.max_samples, chunk)
    print("[capture] val ...")
    va = capture(model, aggs, tok, args.data_root, args.val_cells,
                 args.val_seed, args.layers, args.max_samples, chunk)

    conn = {L: aggs[L].connector.weight.detach().cpu() for L in args.layers}
    torch.save({"train": tr, "val": va, "connector": conn,
                "meta": dict(D=D, H=H, Kd=Kd, layers=args.layers, chunk=chunk)},
               os.path.join(args.out, "cache.pt"))
    fit_routers(args, tr, va, D, H, Kd, conn)


def fit_routers(args, tr, va, D, H, Kd, conn):
    """Fit one router per layer on cached tensors and write the verdict."""
    # Needle count per sample is not stored, but the needle-segment set is,
    # and hit@2 collapses from 0.28 to 0.04 between N=4 and N=16 — reporting
    # one pooled number would hide the only axis that matters.
    if args.control_only:
        print("[control] native linear router on the captured tensors")
        out = {}
        for li, L in enumerate(args.layers):
            if conn is None or L not in conn:
                print(f"  layer {L:<3} no connector captured")
                continue
            a = linear_router_auc(
                [d["h"][li] for d in va], [d["g"][li] for d in va],
                [d["gold"] for d in va], conn[L], H, Kd)
            out[L] = a
            flag = "" if 0.35 <= a <= 0.60 else "   <-- outside the 370M band"
            print(f"  layer {L:<3} gold-AUC {a:.3f}{flag}", flush=True)
        if out:
            lo, hi = min(out.values()), max(out.values())
            print(f"[control] {len(out)} layers, range {lo:.3f}-{hi:.3f}, "
                  f"mean {sum(out.values())/len(out):.3f}")
        os.makedirs(args.out, exist_ok=True)
        json.dump({"control_only": True, "auc": out},
                  open(os.path.join(args.out, "control.json"), "w"), indent=1)
        return

    def bucket(d):
        return "many_needles" if len(d["ndl"]) > 8 else "few_needles"
    buckets = sorted({bucket(d) for d in va})

    # ---- capture control, before any MLP number is read ----
    # A broken capture is not layer-selective. The withdrawn zeros bug
    # deleted 16 attention sublayers and pushed every layer to 0.71-0.79, so
    # a majority test still kills it. A per-layer test does not survive
    # contact with a second model: mc_50M puts L0/L1/L2 at 0.330/0.419/0.352
    # while its other 13 layers sit at 0.54-0.59, and mc_370M reproduces
    # 0.507-0.554 on all 16 through this same code. Below 0.5 is not absent
    # signal, it is inverted signal -- 0.33 carries as much as 0.67 -- so the
    # early layers of the small model are informative and its hand-designed
    # descriptor reads them backwards. Failing the run there would have
    # thrown away the finding.
    controls = {}
    for li, L in enumerate(args.layers):
        if conn is not None and L in conn:
            controls[L] = linear_router_auc(
                [d["h"][li] for d in va], [d["g"][li] for d in va],
                [d["gold"] for d in va], conn[L], H, Kd)
    if controls:
        bad = {L: a for L, a in controls.items() if not 0.35 <= a <= 0.60}
        if bad:
            print("[control] outside the 0.35-0.60 band: "
                  + ", ".join(f"L{L} {a:.3f}" for L, a in sorted(bad.items())),
                  flush=True)
        if len(bad) > max(1, len(controls) // 4):
            raise RuntimeError(
                f"capture control FAILED: {len(bad)} of {len(controls)} "
                f"layers put the native linear router outside 0.35-0.60 "
                f"({min(controls.values()):.3f}-{max(controls.values()):.3f}). "
                "A capture that misses the deployed routing path distorts "
                "every layer, so this is the capture, not the model. Fix it "
                "before reading any MLP number.")
        print(f"[control] PASS: {len(controls) - len(bad)}/{len(controls)} "
              f"layers in band, mean "
              f"{sum(controls.values())/len(controls):.3f}", flush=True)

    # ---- train one MLP router per layer ----
    verdict = {"capture_version": 2}
    for li, L in enumerate(args.layers):
        base_auc = controls.get(L)
        # Same init for every loss variant, or a loss comparison also
        # compares two random initialisations.
        torch.manual_seed(args.fit_seed + L)
        router = MLPRouter(D, H, Kd, scorer=args.scorer).to("cuda")
        opt = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=0.01)
        rows = [(torch.tensor(d["h"][li]).float().cuda(),
                 torch.tensor(np.asarray(d["g"][li])).float().cuda(),
                 d["gold"], d["ndl"], d["nseg"]) for d in tr]
        gen = torch.Generator().manual_seed(0)
        for step in range(args.steps):
            k = int(torch.randint(len(rows), (1,), generator=gen))
            h, g, gold, ndl, nseg = rows[k]
            elig = nseg - 1
            if gold >= elig:  # gold in the (excluded) current segment
                continue
            sc = router.scores(h[None], g[None, :elig])[0]      # [elig]
            loss = F.cross_entropy(sc[None], torch.tensor([gold]).cuda())
            if args.needle_aux > 0 and ndl:
                # Raises EVERY needle segment. At N=16 more than half the
                # eligible segments are needles, so this term fights the main
                # CE, which has to single one of them out. Kept behind a flag
                # only to reproduce the earlier runs.
                tgt = torch.zeros(elig, device="cuda")
                for j in ndl:
                    if j < elig:
                        tgt[j] = 1.0
                if tgt.sum() > 0:
                    loss = loss + args.needle_aux * F.binary_cross_entropy_with_logits(
                        sc, tgt)
            if args.gold_vs_needle > 0 and ndl:
                # The discrimination that is actually hard. Descriptors already
                # encode needle-ness at AUC 1.00 (09-01 diagnosis), so telling
                # needles from haystack is free; telling GOLD from the other
                # needles is the whole task at high N. Restrict the CE
                # denominator to needle segments to spend capacity there.
                cand = sorted({j for j in ndl if j < elig} | {gold})
                if len(cand) > 1 and gold in cand:
                    loss = loss + args.gold_vs_needle * F.cross_entropy(
                        sc[cand][None],
                        torch.tensor([cand.index(gold)]).cuda())
            opt.zero_grad(); loss.backward(); opt.step()
        # ---- eval gold-AUC on val ----
        router.eval()
        with torch.no_grad():
            allsc, gidx, emask = [], [], []
            for d in va:
                elig = d["nseg"] - 1
                if elig <= 0 or d["gold"] >= elig:
                    allsc.append([]); gidx.append(-1); emask.append([])
                    continue
                h = torch.tensor(d["h"][li]).float().cuda()
                g = torch.tensor(np.asarray(d["g"][li])).float().cuda()
                sc = router.scores(h[None], g[None, :elig])[0].cpu().numpy()
                allsc.append(list(sc)); gidx.append(d["gold"])
                emask.append([True] * elig)
            a = auc(allsc, gidx, emask)
        row = {"gold_auc": a, "linear_control_auc": base_auc}
        row.update(hit_at_k(allsc, gidx))
        for b in buckets:
            sel = [i for i, d in enumerate(va) if bucket(d) == b
                   and i < len(allsc)]
            if sel:
                row[b] = hit_at_k([allsc[i] for i in sel],
                                  [gidx[i] for i in sel])
        verdict[f"layer{L}"] = row
        torch.save(router.state_dict(), os.path.join(args.out, f"router_L{L}.pt"))
        print(f"[layer {L}] AUC={a} " + " ".join(
            f"{k}={row[k]:.3f}" for k in ("hit@1", "hit@2", "hit@4", "hit@8")
            if row.get(k) is not None) + " | " + " ".join(
            f"{b}:hit@2={row[b]['hit@2']:.3f}(n={row[b]['n']})"
            for b in buckets if b in row), flush=True)

    verdict["baseline_linear_router_auc"] = 0.46
    verdict["scorer"] = args.scorer
    verdict["loss"] = {"needle_aux": args.needle_aux,
                       "gold_vs_needle": args.gold_vs_needle,
                       "steps": args.steps, "lr": args.lr}
    json.dump(verdict, open(os.path.join(args.out, "verdict.json"), "w"), indent=1)
    print(json.dumps(verdict, indent=1))




if __name__ == "__main__":
    main()
