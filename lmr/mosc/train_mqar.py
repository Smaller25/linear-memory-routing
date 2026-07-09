# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""From-scratch MQAR trainer for the Dynamic-MoSC track (GDN2 backbone).

Models:
  --model gdn2  : the vanilla GDN2 backbone (baseline / Phase-0 ``vanilla``).
  --model mosc  : Dynamic-MoSC (--chunk-mode fixed|oracle|surprisal, --num-pools, --topk).

Phase-0 kill-test (segment-level routing under the BEST case): compare ``vanilla`` vs
``mosc --chunk-mode oracle`` on multi-key MQAR. If oracle boundaries don't beat vanilla, segment-
level routing is dead (pivot to consolidation). Run via Slurm:

    sbatch scripts/sh_slurm_run.sh python -m lmr.mosc.train_mqar \
        --model mosc --chunk-mode oracle --train-kv 16 32 --eval-kv 16 32 64 --steps 3000

GOTCHA: MQAR has a delayed phase transition (~2000 steps) — loss sits at random (~ln(vocab/2)) then
drops sharply. Run >= 3000 steps; earlier <=1500-step runs look "stuck" but aren't (report README).
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from lmr.mosc.backbone import GDN2LM
from lmr.mosc.dynamic_chunk import mqar_oracle_positions, token_surprisal
from lmr.mosc.mosc_model import DynamicMoSC
from lmr.tasks.mqar import make_mqar, make_mqar_gapped
from lmr.tasks.selective_copying import make_selective_copying

IGNORE = -100


def build_model(args, vocab):
    if args.model == "gdn2":
        return GDN2LM(vocab, d_model=args.d_model, n_layers=args.n_layers,
                      head_dim=args.head_dim, num_heads=args.num_heads)
    return DynamicMoSC(vocab, d_model=args.d_model, n_layers=args.n_layers,
                       head_dim=args.head_dim, num_heads=args.num_heads,
                       chunk_mode=args.chunk_mode, chunk=args.chunk,
                       num_pools=args.num_pools, topk=args.topk,
                       use_true_state=args.true_state, budget=args.budget,
                       density_signal=args.density_signal, target_rate=args.target_rate,
                       density_fire=args.density_fire,
                       cache_mode=args.cache_mode, cache_budget=args.cache_budget)


def gen_batch(ctx_filler, n, vocab, k, seq_len, seed, task="mqar", scatter_mult=8):
    """Return (input_ids, labels, oracle_pos).

    task=selcopy -> Selective Copying (the standard content-vs-position task): k data tokens at random
    positions among k*scatter_mult noise tokens, reproduced in order; oracle_pos = the data positions.
    task=mqar: ctx_filler>0 -> IRREGULAR MQAR (facts at random positions; oracle_pos = per-row value
    positions); ctx_filler==0 -> standard MQAR (oracle_pos None, from the fixed period at use)."""
    if task == "selcopy":
        b = make_selective_copying(num_examples=n, n_data=k, vocab_size=vocab,
                                   scatter_mult=scatter_mult, seed=seed)
        return b["input_ids"], b["labels"], b["value_pos"]
    if ctx_filler > 0:
        b = make_mqar_gapped(num_examples=n, vocab_size=vocab, num_kv_pairs=k,
                             ctx_filler=ctx_filler, seed=seed)
        return b["input_ids"], b["labels"], b["value_pos"]
    b = make_mqar(num_examples=n, vocab_size=vocab, num_kv_pairs=k, input_seq_len=seq_len, seed=seed)
    return b["input_ids"], b["labels"], None


def build_salience_gate(mode, ids, vpos, model, eps=0.1, scale=2.0):
    """Per-token retention gate [B,T] in [eps,1] for the constant-memory salience-gated backbone.
    oracle: 1.0 at value (needle) positions, eps at filler — the upper bound (needs --ctx-filler>0).
    surprisal: monotone in DETACHED token surprisal (frozen signal -> no co-training drift)."""
    if mode == "none":
        return None
    B, T = ids.shape
    if mode == "oracle":
        if vpos is None:
            raise ValueError("oracle salience needs value positions — use --ctx-filler > 0")
        gate = torch.full((B, T), eps, device=ids.device)
        gate.scatter_(1, vpos, 1.0)
        return gate
    if mode == "surprisal":
        with torch.no_grad():
            s = token_surprisal(model(ids), ids)                 # ungated, detached signal [B,T]
            s = (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-5)
            return eps + (1 - eps) * torch.sigmoid(scale * s)
    raise ValueError(mode)


def run_model(model, ids, k, is_mosc, distill=0.0, oracle_pos=None, salience_gate=None):
    if not is_mosc:
        return model(ids, salience_gate=salience_gate)

    def _oracle():  # explicit per-row value positions (irregular) or the fixed-period fallback
        return oracle_pos if oracle_pos is not None else mqar_oracle_positions(k, ids.shape[0], device=ids.device)

    if model.chunk_mode == "oracle":
        return model(ids, oracle_positions=_oracle())
    if model.chunk_mode in ("learned", "unsup_ste"):
        # distill from oracle when distill>0 (learned: always; unsup_ste: warm-up phase only). At
        # eval (distill=0) the head segments from its own predictions.
        return model(ids, oracle_positions=_oracle() if distill > 0 else None, boundary_distill=distill)
    return model(ids)


@torch.no_grad()
def recall_acc(model, k, vocab, device, is_mosc, seq_len=None, ctx_filler=0, n=256,
               task="mqar", scatter_mult=8, salience="none", salience_eps=0.1, salience_scale=2.0):
    ids, labels, vpos = gen_batch(ctx_filler, n, vocab, k, seq_len, 10_000 + k, task, scatter_mult)
    ids, labels = ids.to(device), labels.to(device)
    vpos = vpos.to(device) if vpos is not None else None
    correct = total = 0
    for i in range(0, n, 64):
        op = vpos[i:i + 64] if vpos is not None else None
        sg = build_salience_gate(salience, ids[i:i + 64], op, model, salience_eps, salience_scale)
        logits = run_model(model, ids[i:i + 64], k, is_mosc, oracle_pos=op, salience_gate=sg)
        pred = logits.argmax(-1)
        m = labels[i:i + 64] != IGNORE
        correct += (pred[m] == labels[i:i + 64][m]).sum().item()
        total += m.sum().item()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mqar", "selcopy"], default="mqar",
                    help="selcopy = Selective Copying (standard content-vs-position task); train-kv/"
                         "eval-kv are then #data tokens to copy, scattered among --scatter-mult*k noise")
    ap.add_argument("--scatter-mult", type=int, default=8,
                    help="selcopy: noise region length = scatter-mult * (#data tokens)")
    ap.add_argument("--salience", choices=["none", "oracle", "surprisal"], default="none",
                    help="constant-memory salience-gated retention (--model gdn2): gate each token's "
                         "mixer input so filler writes weakly to the fixed state. oracle=needle "
                         "positions (upper bound, needs --ctx-filler>0); surprisal=detached signal")
    ap.add_argument("--salience-eps", type=float, default=0.1, help="gate floor for non-salient tokens")
    ap.add_argument("--salience-scale", type=float, default=2.0, help="surprisal gate sigmoid steepness")
    ap.add_argument("--model", choices=["gdn2", "mosc"], default="gdn2")
    ap.add_argument("--chunk-mode", choices=["fixed", "oracle", "surprisal", "learned", "unsup", "unsup_ste", "density"], default="fixed")
    ap.add_argument("--density-signal", choices=["surprisal", "entropy", "cosdist"], default="surprisal",
                    help="chunk-mode=density: intrinsic info-density signal to segment on (no oracle)")
    ap.add_argument("--target-rate", type=float, default=0.1,
                    help="chunk-mode=density: target boundary firing rate (cache budget); the rate loss "
                         "anchors mean(p) to this two-sided so it cannot collapse to 0")
    ap.add_argument("--density-fire", choices=["quantile", "threshold"], default="quantile",
                    help="chunk-mode=density: quantile=fire top target-rate fraction per row (robust); "
                         "threshold=fixed 0.5 cutoff (fires nothing if p is diffuse below 0.5)")
    ap.add_argument("--cache-mode", choices=["full", "capped", "hier"], default="full",
                    help="AXIS-2 bounded memory when #segments > cache-budget: full=keep all (ceiling); "
                         "capped=keep recent B, drop older (0011 baseline); hier=recent fine + older "
                         "compressed via learned merge (bounded, lossy-not-dropped)")
    ap.add_argument("--cache-budget", type=int, default=0,
                    help="AXIS-2: max segments retained (0 = unlimited). The recall-vs-budget curve of "
                         "capped vs hier is the axis-2 experiment (cf. 0011 ∝B/N degradation)")
    ap.add_argument("--boundary-distill", type=float, default=1.0,
                    help="weight on the oracle-boundary distillation loss (chunk-mode=learned)")
    ap.add_argument("--warmup-steps", type=int, default=-1,
                    help="unsup_ste warm-start: distill the boundary head for the first N steps, then "
                         "drop the oracle (task-loss only). -1 = distill always on (learned mode).")
    ap.add_argument("--budget", type=float, default=0.05,
                    help="chunk-mode=unsup: L1 sparsity weight on the landmark prob (no oracle)")
    ap.add_argument("--eval-thresholds", type=float, nargs="*", default=[0.5, 0.3, 0.2, 0.1, 0.05],
                    help="learned mode: re-eval the trained model at these boundary thresholds")
    ap.add_argument("--dump-boundaries", default=None,
                    help="learned mode: save predicted segment lengths per eval-kv to this .npz (for viz)")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--true-state", action="store_true",
                    help="cache the TRUE GDN2 recurrent state per segment (not pooled-hidden proxy)")
    ap.add_argument("--num-pools", type=int, default=1)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--train-kv", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--eval-kv", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--ctx-filler", type=int, default=0,
                    help=">0 -> IRREGULAR MQAR: insert this many random filler tokens in the context so "
                         "facts sit at non-periodic positions (tests adaptive vs fixed-stride boundaries)")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)  # MQAR needs the high lr to hit the transition
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_mosc = args.model == "mosc"
    model = build_model(args, args.vocab).to(device).train()
    # match the validated MoCM MQAR recipe (lmr/scripts/train_mocm_mqar.py): high lr + wd + grad clip
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    print(f"model={args.model} chunk={args.chunk_mode} pools={args.num_pools} lr={args.lr} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M | device={device}")

    for step in range(1, args.steps + 1):
        k = args.train_kv[step % len(args.train_kv)]
        ids, labels, vpos = gen_batch(args.ctx_filler, args.batch, args.vocab, k, args.seq_len, step,
                                      args.task, args.scatter_mult)
        ids, labels = ids.to(device), labels.to(device)
        vpos = vpos.to(device) if vpos is not None else None
        distill = args.boundary_distill if (args.warmup_steps < 0 or step <= args.warmup_steps) else 0.0
        sgate = build_salience_gate(args.salience, ids, vpos, model, args.salience_eps, args.salience_scale)
        logits = run_model(model, ids, k, is_mosc, distill=distill, oracle_pos=vpos, salience_gate=sgate)
        loss = F.cross_entropy(logits.reshape(-1, args.vocab), labels.reshape(-1), ignore_index=IGNORE)
        if getattr(model, "boundary_loss", None) is not None:
            loss = loss + model.boundary_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == 1:
            print(f"[train] step {step:5d}  loss {loss.item():.4f}")

    print("=== recall accuracy (vs #kv pairs) ===")
    model.eval()
    for k in args.eval_kv:
        print(f"  kv={k:4d}  acc={recall_acc(model, k, args.vocab, device, is_mosc, args.seq_len, args.ctx_filler, task=args.task, scatter_mult=args.scatter_mult, salience=args.salience, salience_eps=args.salience_eps, salience_scale=args.salience_scale):.3f}")

    # learned mode: sweep the eval-time boundary threshold (no retrain). Diagnosis says the head
    # UNDER-FIRES at 0.5 (precision ~1.0, recall low); a lower cutoff should fire more boundaries and
    # recover recall. Lists recall across kv at each threshold.
    if is_mosc and model.chunk_mode == "learned" and args.eval_thresholds:
        print("=== boundary-threshold sweep (recall-acc across kv) ===")
        for thr in args.eval_thresholds:
            model.boundary_threshold = thr
            accs = [recall_acc(model, k, args.vocab, device, is_mosc, args.seq_len, args.ctx_filler, task=args.task, scatter_mult=args.scatter_mult) for k in args.eval_kv]
            print(f"  thr={thr:.2f}  " + "  ".join(f"kv{k}={a:.2f}" for k, a in zip(args.eval_kv, accs)))
        model.boundary_threshold = 0.5

    # boundary-quality diagnostic for the learned predictor: how many boundaries does it fire at
    # eval, and how well do they match the oracle (per-fact) positions?
    if is_mosc and model.chunk_mode in ("learned", "unsup", "unsup_ste", "density"):
        from lmr.mosc.dynamic_chunk import positions_to_mask
        print("=== learned-boundary quality (predicted vs oracle) ===")
        for k in args.eval_kv:
            ids, _, vpos = gen_batch(args.ctx_filler, 64, args.vocab, k, args.seq_len, 20_000 + k, args.task, args.scatter_mult)
            ids = ids.to(device); vpos = vpos.to(device) if vpos is not None else None
            with torch.no_grad():
                model(ids)
            pred = model.last_boundaries.clone(); pred[:, -1] = False  # ignore the forced last
            op = vpos if vpos is not None else mqar_oracle_positions(k, ids.shape[0], device)
            tgt = positions_to_mask(op, ids.shape[1])
            tp = (pred & tgt).sum().item()
            prec = tp / max(pred.sum().item(), 1)
            rec = tp / max(tgt.sum().item(), 1)
            # THRESHOLD-FREE: take the top-k positions by p (k = #facts) and overlap with the true
            # facts — answers "does the head rank facts highest?" even if p never crosses 0.5.
            tf = float("nan")
            if getattr(model, "last_p", None) is not None:
                p = model.last_p.clone()
                p[:, -1] = -1.0                                   # exclude the forced-last boundary
                topi = p.topk(min(k, p.shape[1] - 1), dim=1).indices
                tfmask = positions_to_mask(topi, ids.shape[1])
                tf = ((tfmask & tgt).sum().item()) / max(tgt.sum().item(), 1)   # top-k recall == prec
            print(f"  kv={k:4d}  pred/seq={pred.float().sum(1).mean():.1f} (oracle={k})  "
                  f"precision={prec:.2f} recall={rec:.2f}  | top-k(p) overlap={tf:.2f}")

        # dump predicted segment lengths (gaps between consecutive boundaries) for visualization
        if args.dump_boundaries:
            import numpy as np
            out = {}
            for k in args.eval_kv:
                ids, _, _ = gen_batch(args.ctx_filler, 64, args.vocab, k, args.seq_len, 30_000 + k, args.task, args.scatter_mult)
                ids = ids.to(device)
                with torch.no_grad():
                    model(ids)
                bnd = model.last_boundaries                       # [B, T] bool (last token forced True)
                T = bnd.shape[1]
                seglens = []
                for row in bnd:
                    idx = row.nonzero(as_tuple=True)[0].tolist()   # boundary end positions
                    prev = -1
                    for e in idx:
                        seglens.append(e - prev); prev = e
                out[f"kv{k}_seglens"] = np.array(seglens, dtype=np.int64)
                out[f"kv{k}_T"] = np.array([T])
            np.savez(args.dump_boundaries, **out)
            print(f"[dump] segment lengths saved -> {args.dump_boundaries}")


if __name__ == "__main__":
    main()
