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
from lmr.mosc.dynamic_chunk import mqar_oracle_positions
from lmr.mosc.mosc_model import DynamicMoSC
from lmr.tasks.mqar import make_mqar

IGNORE = -100


def build_model(args, vocab):
    if args.model == "gdn2":
        return GDN2LM(vocab, d_model=args.d_model, n_layers=args.n_layers,
                      head_dim=args.head_dim, num_heads=args.num_heads)
    return DynamicMoSC(vocab, d_model=args.d_model, n_layers=args.n_layers,
                       head_dim=args.head_dim, num_heads=args.num_heads,
                       chunk_mode=args.chunk_mode, chunk=args.chunk,
                       num_pools=args.num_pools, topk=args.topk)


def run_model(model, ids, k, is_mosc, distill=0.0):
    if not is_mosc:
        return model(ids)
    if model.chunk_mode == "oracle":
        oracle = mqar_oracle_positions(k, ids.shape[0], device=ids.device)
        return model(ids, oracle_positions=oracle)
    if model.chunk_mode == "learned":
        # distill the boundary head from oracle positions during TRAINING only; at eval (distill=0)
        # the model must segment from its own predictions — no oracle.
        oracle = mqar_oracle_positions(k, ids.shape[0], device=ids.device) if distill > 0 else None
        return model(ids, oracle_positions=oracle, boundary_distill=distill)
    return model(ids)


@torch.no_grad()
def recall_acc(model, k, vocab, device, is_mosc, seq_len=None, n=256):
    batch = make_mqar(num_examples=n, vocab_size=vocab, num_kv_pairs=k,
                      input_seq_len=seq_len, seed=10_000 + k)
    ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
    correct = total = 0
    for i in range(0, n, 64):
        logits = run_model(model, ids[i:i + 64], k, is_mosc)
        pred = logits.argmax(-1)
        m = labels[i:i + 64] != IGNORE
        correct += (pred[m] == labels[i:i + 64][m]).sum().item()
        total += m.sum().item()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gdn2", "mosc"], default="gdn2")
    ap.add_argument("--chunk-mode", choices=["fixed", "oracle", "surprisal", "learned"], default="fixed")
    ap.add_argument("--boundary-distill", type=float, default=1.0,
                    help="weight on the oracle-boundary distillation loss (chunk-mode=learned)")
    ap.add_argument("--eval-thresholds", type=float, nargs="*", default=[0.5, 0.3, 0.2, 0.1, 0.05],
                    help="learned mode: re-eval the trained model at these boundary thresholds")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--num-pools", type=int, default=1)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--train-kv", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--eval-kv", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--seq-len", type=int, default=None)
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
        batch = make_mqar(num_examples=args.batch, vocab_size=args.vocab, num_kv_pairs=k,
                          input_seq_len=args.seq_len, seed=step)
        ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
        logits = run_model(model, ids, k, is_mosc, distill=args.boundary_distill)
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
        print(f"  kv={k:4d}  acc={recall_acc(model, k, args.vocab, device, is_mosc, args.seq_len):.3f}")

    # learned mode: sweep the eval-time boundary threshold (no retrain). Diagnosis says the head
    # UNDER-FIRES at 0.5 (precision ~1.0, recall low); a lower cutoff should fire more boundaries and
    # recover recall. Lists recall across kv at each threshold.
    if is_mosc and model.chunk_mode == "learned" and args.eval_thresholds:
        print("=== boundary-threshold sweep (recall-acc across kv) ===")
        for thr in args.eval_thresholds:
            model.boundary_threshold = thr
            accs = [recall_acc(model, k, args.vocab, device, is_mosc, args.seq_len) for k in args.eval_kv]
            print(f"  thr={thr:.2f}  " + "  ".join(f"kv{k}={a:.2f}" for k, a in zip(args.eval_kv, accs)))
        model.boundary_threshold = 0.5

    # boundary-quality diagnostic for the learned predictor: how many boundaries does it fire at
    # eval, and how well do they match the oracle (per-fact) positions?
    if is_mosc and model.chunk_mode == "learned":
        from lmr.mosc.dynamic_chunk import positions_to_mask
        print("=== learned-boundary quality (predicted vs oracle) ===")
        for k in args.eval_kv:
            b = make_mqar(num_examples=64, vocab_size=args.vocab, num_kv_pairs=k,
                          input_seq_len=args.seq_len, seed=20_000 + k)
            ids = b["input_ids"].to(device)
            with torch.no_grad():
                model(ids)
            pred = model.last_boundaries.clone(); pred[:, -1] = False  # ignore the forced last
            tgt = positions_to_mask(mqar_oracle_positions(k, ids.shape[0], device), ids.shape[1])
            tp = (pred & tgt).sum().item()
            prec = tp / max(pred.sum().item(), 1)
            rec = tp / max(tgt.sum().item(), 1)
            print(f"  kv={k:4d}  pred/seq={pred.float().sum(1).mean():.1f} (oracle={k})  "
                  f"precision={prec:.2f} recall={rec:.2f}")


if __name__ == "__main__":
    main()
