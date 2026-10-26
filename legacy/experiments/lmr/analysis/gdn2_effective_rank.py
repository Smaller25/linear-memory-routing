# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Axis-3 spike — port the SSM effective-rank / state-saturation measurement to GDN-2.

Mirrors `Smaller25/SSM_Rank_Analysis` (Mamba-2): the effective rank of the recurrent state rises with
context length and **plateaus at a per-head saturation point T\***. There the state is `(nheads,
headdim, d_state)`; here GDN-2's recurrent state is `(nheads, head_k, head_v)` per layer, captured via
``GDN2LM.run_segmented`` (fla Cache state-threading). For each context length T we read the threaded
state and compute, per head, the effective rank of its `[head_k, head_v]` matrix.

effective_rank(M) = exp(Shannon entropy of the normalized singular values) — identical to the repo.
T* = first T at which the (head-averaged) rank reaches `--sat-ratio` of its max.

This is a PORT/INFRA spike: it validates the measurement on GDN-2's state object and prints the
rank-vs-T curve + T* + per-head spread (head heterogeneity, the repo's Type A/B/C). Run on a trained
checkpoint later for the scientific rank≈#facts claim; here the model may be random-init.

    sbatch scripts/sh_slurm_run.sh python -m lmr.analysis.gdn2_effective_rank \
        --task mqar --kv 64 --seq-len 512 --t-step 16
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from lmr.mosc.backbone import GDN2LM
from lmr.tasks.mqar import make_mqar


def effective_rank(matrix: torch.Tensor, eps: float = 1e-9) -> float:
    """matrix: [K, V] -> exp(entropy of normalized singular values). Identical to SSM_Rank_Analysis."""
    s = torch.linalg.svdvals(matrix.float())
    s = s / (s.sum() + eps)
    entropy = -(s * (s + eps).log()).sum()
    return float(entropy.exp())


def find_saturation_point(traj: np.ndarray, ratio: float = 0.95) -> int:
    """First index where the trajectory reaches `ratio` of its max (the repo's T* rule)."""
    thr = ratio * float(np.max(traj))
    for i, r in enumerate(traj):
        if r >= thr:
            return i
    return len(traj) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mqar", "random"], default="mqar")
    ap.add_argument("--kv", type=int, default=64, help="mqar: #kv pairs (facts)")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--t-step", type=int, default=16, help="measure the state every t-step tokens")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--sat-ratio", type=float, default=0.95)
    ap.add_argument("--ckpt", default=None, help="optional GDN2LM state_dict to load (trained model)")
    ap.add_argument("--out", default="ckpt/gdn2_effrank.npz")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GDN2LM(args.vocab, d_model=args.d_model, n_layers=args.n_layers,
                   head_dim=args.head_dim, num_heads=args.num_heads).to(device).eval()
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"loaded checkpoint {args.ckpt}")

    # input
    if args.task == "mqar":
        b = make_mqar(num_examples=args.batch, vocab_size=args.vocab, num_kv_pairs=args.kv,
                      input_seq_len=args.seq_len, seed=0)
        ids = b["input_ids"][:, :args.seq_len].to(device)
    else:
        ids = torch.randint(0, args.vocab, (args.batch, args.seq_len), device=device)
    T = ids.shape[1]

    # cumulative segment bounds at each measured T -> run_segmented threads the state and captures it
    # at every cut, so states[:, i] is the recurrent state after seeing tokens [0, T_RANGE[i]).
    T_RANGE = list(range(args.t_step, T + 1, args.t_step))
    if T_RANGE[-1] != T:
        T_RANGE.append(T)
    bounds, prev = [], 0
    for t in T_RANGE:
        bounds.append((prev, t)); prev = t

    with torch.no_grad():
        _, states = model.run_segmented(ids, bounds)            # states: [B, N, H*K*V]
    H, K = args.num_heads, args.head_dim
    V = int(states.shape[-1] // (H * K))
    states = states.reshape(states.shape[0], states.shape[1], H, K, V)   # [B, N, H, K, V]
    print(f"model={'ckpt' if args.ckpt else 'random-init'} task={args.task} kv={args.kv} "
          f"H={H} K={K} V={V} | device={device} | measuring {len(T_RANGE)} context lengths")

    # per-head effective rank at each T (averaged over batch)
    B, N = states.shape[0], states.shape[1]
    per_head = np.zeros((N, H))                                 # [T-index, head]
    for i in range(N):
        for h in range(H):
            rks = [effective_rank(states[bi, i, h]) for bi in range(B)]
            per_head[i, h] = float(np.mean(rks))
    rank_traj = per_head.mean(axis=1)                           # head-averaged rank(T)

    tstar_idx = find_saturation_point(rank_traj, args.sat_ratio)
    print("\n=== effective rank vs context length (last layer, head-avg) ===")
    for i, t in enumerate(T_RANGE):
        bar = "#" * int(round(rank_traj[i]))
        print(f"  T={t:4d}  rank={rank_traj[i]:6.2f}  spread[{per_head[i].min():.1f},{per_head[i].max():.1f}]  {bar}")
    print(f"\nT* (rank >= {args.sat_ratio:.0%} of max={rank_traj.max():.2f}) = T_RANGE[{tstar_idx}] = {T_RANGE[tstar_idx]} tokens")
    print(f"max effective rank {rank_traj.max():.2f} / capacity min(K,V)={min(K, V)} "
          f"({100*rank_traj.max()/min(K, V):.0f}% of capacity)")
    # head heterogeneity: per-head T* (the repo's Type A/B/C signal)
    head_tstar = [T_RANGE[find_saturation_point(per_head[:, h], args.sat_ratio)] for h in range(H)]
    print(f"per-head T*: {head_tstar}  (spread => heterogeneous heads, cf. repo Type A/B/C)")

    np.savez(args.out, T_RANGE=np.array(T_RANGE), rank_traj=rank_traj, per_head=per_head,
             tstar=T_RANGE[tstar_idx], head_tstar=np.array(head_tstar), capacity=min(K, V))
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
