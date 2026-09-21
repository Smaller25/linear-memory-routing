"""Test v5 kernel produces gradients matching v2 reference, tighter than v3/v4.

v5 = v3c (fwd → TF32) + v3 (bwd_q → TF32, bwd_w → TF32) + v4 (bwd_mem → TF32).

Acceptance:
  - fwd: should be tighter than v3 (TF32 > bf16, ~0.1% rel max vs ~1%)
  - grad_q, grad_w: same — TF32 path tighter than v3
  - grad_mem: same as v4 (already TF32)
  - All should be ≤ 0.05% rel max (vs v3's ~1% and v4's mixed ~1%)

If v5 grads <0.1% rel across all four, the 1B-token training should match v2 PPL/RULER.
"""
from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from dsc.mc_baseline.cached_memory_read import _SSCGatherRead as V2
from dsc.mc_v3 import _SSCGatherReadV3c as V3C
from dsc.mc_v4.cached_memory_read_v4 import _SSCGatherReadV4 as V4
from dsc.mc_v5.cached_memory_read_v5 import _SSCGatherReadV5 as V5


def _make_and_compare(B, T, H, K, V, N, R, scale, normalize, label):
    device = "cuda"
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    m = torch.randn(B, N, H, K, V, device=device, dtype=torch.float32)
    idx = torch.randint(0, N, (B, T, R), device=device, dtype=torch.long)
    w = torch.randn(B, T, R, device=device, dtype=torch.bfloat16)

    if N >= 4:
        for t in range(T):
            for b in range(B):
                idx[b, t, 0] = (t + b) % N
                idx[b, t, 1] = (t * 2 + b + 1) % N

    def fwd_bwd(kernel_cls):
        q_ = q.clone().requires_grad_(True)
        m_ = m.clone().requires_grad_(True)
        w_ = w.clone().requires_grad_(True)
        out = kernel_cls.apply(q_, m_, idx, w_, scale, normalize)
        gq, gm, gw = torch.autograd.grad(out.sum(), [q_, m_, w_], retain_graph=False)
        return out.float(), gq.float(), gm.float(), gw.float()

    out_v2, gq_v2, gm_v2, gw_v2 = fwd_bwd(V2)
    out_v3, gq_v3, gm_v3, gw_v3 = fwd_bwd(V3C)
    out_v4, gq_v4, gm_v4, gw_v4 = fwd_bwd(V4)
    out_v5, gq_v5, gm_v5, gw_v5 = fwd_bwd(V5)

    def rel(a, b):
        d = (a - b).abs().max().item()
        r = d / (b.abs().max().item() + 1e-9)
        return d, r

    fwd_d3, fwd_r3 = rel(out_v3, out_v2)
    fwd_d4, fwd_r4 = rel(out_v4, out_v2)
    fwd_d5, fwd_r5 = rel(out_v5, out_v2)
    gq_d3, gq_r3 = rel(gq_v3, gq_v2)
    gq_d4, gq_r4 = rel(gq_v4, gq_v2)
    gq_d5, gq_r5 = rel(gq_v5, gq_v2)
    gw_d3, gw_r3 = rel(gw_v3, gw_v2)
    gw_d4, gw_r4 = rel(gw_v4, gw_v2)
    gw_d5, gw_r5 = rel(gw_v5, gw_v2)
    gm_d3, gm_r3 = rel(gm_v3, gm_v2)
    gm_d4, gm_r4 = rel(gm_v4, gm_v2)
    gm_d5, gm_r5 = rel(gm_v5, gm_v2)

    print(f"\n=== {label} shapes B={B} T={T} H={H} K={K} V={V} N={N} R={R} ===")
    print(f"  fwd           v3 (bf16):  rel={fwd_r3*100:.3f}%")
    print(f"  fwd           v4 (mixed): rel={fwd_r4*100:.3f}%")
    print(f"  fwd           v5 (TF32):  rel={fwd_r5*100:.3f}%  ← should be << v3")
    print(f"  grad_q        v3:         rel={gq_r3*100:.3f}%   v4: {gq_r4*100:.3f}%   v5: {gq_r5*100:.3f}%")
    print(f"  grad_w        v3:         rel={gw_r3*100:.3f}%   v4: {gw_r4*100:.3f}%   v5: {gw_r5*100:.3f}%")
    print(f"  grad_mem      v3.2 (bf):  rel={gm_r3*100:.3f}%")
    print(f"  grad_mem      v4 (TF32):  rel={gm_r4*100:.3f}%")
    print(f"  grad_mem      v5 (TF32):  rel={gm_r5*100:.3f}%")

    v5_better_fwd = fwd_r5 < fwd_r3
    v5_better_gq = gq_r5 < gq_r3
    v5_better_gw = gw_r5 < gw_r3
    v5_all_tight = max(fwd_r5, gq_r5, gw_r5, gm_r5) < 0.001  # all ≤ 0.1%

    print(f"  v5 fwd tighter than v3:   {'YES' if v5_better_fwd else 'NO'}  ({fwd_r3/max(fwd_r5,1e-9):.1f}x)")
    print(f"  v5 gq  tighter than v3:   {'YES' if v5_better_gq else 'NO'}  ({gq_r3/max(gq_r5,1e-9):.1f}x)")
    print(f"  v5 gw  tighter than v3:   {'YES' if v5_better_gw else 'NO'}  ({gw_r3/max(gw_r5,1e-9):.1f}x)")
    print(f"  v5 all ≤ 0.1% rel:        {'YES' if v5_all_tight else 'NO'}  (target)")
    return v5_better_fwd and v5_better_gq and v5_better_gw and v5_all_tight


def smoke():
    B, T, H, K, V, N, R = 2, 16, 4, 32, 32, 4, 2
    return _make_and_compare(B, T, H, K, V, N, R, 0.5, True, "smoke")


def training_shapes():
    B, T, H, K, V, N, R = 4, 256, 8, 128, 128, 16, 2
    return _make_and_compare(B, T, H, K, V, N, R, 1.0, True, "training")


if __name__ == "__main__":
    smoke_ok = smoke()
    train_ok = training_shapes()
    sys.exit(0 if smoke_ok and train_ok else 1)
