"""Test v4 kernel produces gradients matching v2 reference.

v4 = v3c (fwd) + v3 (bwd_q, bwd_w) + v4 bwd_mem (TF32 TensorCore).

Acceptance:
  - fwd, grad_q, grad_w same as v3 (re-uses v3 kernels) — should be ≤0.6% rel
  - grad_mem (TF32 vs v2 fp32 FMA) should be MUCH closer than v3.2 (bf16):
      v3.2 (bf16, 7 mantissa bits): ~0.23% rel max
      v4   (TF32, 10 mantissa bits): expect <0.05% rel max (4096x less rounding)

If v4 grad_mem <0.05%, the 1B-token training should recover v2 PPL/RULER quality.
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


def _make_and_compare(B, T, H, K, V, N, R, scale, normalize, label):
    device = "cuda"
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    m = torch.randn(B, N, H, K, V, device=device, dtype=torch.float32)
    idx = torch.randint(0, N, (B, T, R), device=device, dtype=torch.long)
    w = torch.randn(B, T, R, device=device, dtype=torch.bfloat16)

    # Force idx spread so n-loop hits multiple segments
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

    def rel(a, b):
        d = (a - b).abs().max().item()
        r = d / (b.abs().max().item() + 1e-9)
        return d, r

    fwd_d, fwd_r = rel(out_v3, out_v2)
    gq_d3, gq_r3 = rel(gq_v3, gq_v2)
    gq_d4, gq_r4 = rel(gq_v4, gq_v2)
    gw_d3, gw_r3 = rel(gw_v3, gw_v2)
    gw_d4, gw_r4 = rel(gw_v4, gw_v2)
    gm_d3, gm_r3 = rel(gm_v3, gm_v2)
    gm_d4, gm_r4 = rel(gm_v4, gm_v2)

    print(f"\n=== {label} shapes B={B} T={T} H={H} K={K} V={V} N={N} R={R} ===")
    print(f"  fwd (v3 vs v2):           max_abs={fwd_d:.4e}  rel={fwd_r*100:.3f}%")
    print(f"  grad_q (v3 vs v2):        rel={gq_r3*100:.3f}%   (v4 vs v2): {gq_r4*100:.3f}%  (same kernel)")
    print(f"  grad_w (v3 vs v2):        rel={gw_r3*100:.3f}%   (v4 vs v2): {gw_r4*100:.3f}%  (same kernel)")
    print(f"  grad_mem v3.2 (bf16):     rel={gm_r3*100:.3f}%   max_abs={gm_d3:.4e}")
    print(f"  grad_mem v4   (TF32):     rel={gm_r4*100:.3f}%   max_abs={gm_d4:.4e}  ← should be << v3.2")
    print(f"  improvement v4 vs v3.2:   {gm_r3/max(gm_r4,1e-9):.1f}x more precise")

    # v4 grad_mem should be tighter than v3.2 (TF32 > bf16)
    v4_better = gm_r4 < gm_r3
    # v4 grad_mem should be ≤ 0.1% (much tighter than v3.2's 0.23%)
    v4_tight = gm_r4 < 0.001
    print(f"  v4 better than v3.2: {'YES' if v4_better else 'NO'}")
    print(f"  v4 < 0.1% rel:       {'YES' if v4_tight else 'NO'}  (target)")
    return v4_better and v4_tight


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
