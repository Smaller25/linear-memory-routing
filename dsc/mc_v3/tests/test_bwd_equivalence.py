"""Test v3 bwd_qw kernel produces gradients matching v2.

Compares grad_q, grad_w, grad_mem from:
  - v2 (original implementation)
  - v3c (new: segment-conditional fwd + v3 bwd_qw + v3.2 bwd_mem bf16 TensorCore)

Acceptance: max_abs on grad_q, grad_w, grad_mem < 5% of |grad|_max.
v3.2 bwd_mem casts operands to bf16 for TensorCore → ~2-3% rounding expected.
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


def smoke():
    """Tiny shape for fast iteration."""
    B, T, H, K, V, N, R = 2, 16, 4, 32, 32, 4, 2
    scale = 0.5
    normalize = True
    return _make_and_compare(B, T, H, K, V, N, R, scale, normalize, "smoke")


def training_shapes():
    """Training shapes (smaller H to keep memory reasonable on shared GPU)."""
    B, T, H, K, V, N, R = 4, 256, 8, 128, 128, 16, 2
    scale = 1.0
    normalize = True
    return _make_and_compare(B, T, H, K, V, N, R, scale, normalize, "training")


def _make_and_compare(B, T, H, K, V, N, R, scale, normalize, label):
    device = "cuda"
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    m = torch.randn(B, N, H, K, V, device=device, dtype=torch.float32)
    idx = torch.randint(0, N, (B, T, R), device=device, dtype=torch.long)
    w = torch.randn(B, T, R, device=device, dtype=torch.bfloat16)

    # Make idx deterministic-but-spread (force some matches across segments)
    # so the n-loop in v3 doesn't always hit idx=0
    if N >= 4:
        for t in range(T):
            for b in range(B):
                idx[b, t, 0] = (t + b) % N
                idx[b, t, 1] = (t * 2 + b + 1) % N

    # v2 forward+backward
    q_v2 = q.clone().requires_grad_(True)
    m_v2 = m.clone().requires_grad_(True)
    w_v2 = w.clone().requires_grad_(True)
    out_v2 = V2.apply(q_v2, m_v2, idx, w_v2, scale, normalize)
    gq_v2, gm_v2, gw_v2 = torch.autograd.grad(
        out_v2.sum(), [q_v2, m_v2, w_v2], retain_graph=False)

    # v3c forward+backward
    q_v3 = q.clone().requires_grad_(True)
    m_v3 = m.clone().requires_grad_(True)
    w_v3 = w.clone().requires_grad_(True)
    out_v3 = V3C.apply(q_v3, m_v3, idx, w_v3, scale, normalize)
    gq_v3, gm_v3, gw_v3 = torch.autograd.grad(
        out_v3.sum(), [q_v3, m_v3, w_v3], retain_graph=False)

    # Compare forward output
    fwd_diff = (out_v2.float() - out_v3.float()).abs()
    fwd_max = fwd_diff.max().item()
    fwd_rel = fwd_max / (out_v2.abs().float().max().item() + 1e-9)

    # Compare grad_q
    gq_diff = (gq_v2.float() - gq_v3.float()).abs()
    gq_max = gq_diff.max().item()
    gq_rel = gq_max / (gq_v2.abs().float().max().item() + 1e-9)

    # Compare grad_w
    gw_diff = (gw_v2.float() - gw_v3.float()).abs()
    gw_max = gw_diff.max().item()
    gw_rel = gw_max / (gw_v2.abs().float().max().item() + 1e-9)

    # Compare grad_mem (should be ~bit-exact since same kernel)
    gm_diff = (gm_v2.float() - gm_v3.float()).abs()
    gm_max = gm_diff.max().item()
    gm_rel = gm_max / (gm_v2.abs().float().max().item() + 1e-9)

    print(f"\n=== {label} shapes B={B} T={T} H={H} K={K} V={V} N={N} R={R} ===")
    print(f"  fwd   max_abs={fwd_max:.4e}  rel_max={fwd_rel*100:.3f}%")
    print(f"  grad_q   max_abs={gq_max:.4e}  rel_max={gq_rel*100:.3f}%")
    print(f"  grad_w   max_abs={gw_max:.4e}  rel_max={gw_rel*100:.3f}%")
    print(f"  grad_mem max_abs={gm_max:.4e}  rel_max={gm_rel*100:.3f}%  (v3.2 bf16 TensorCore, expect <3%)")

    # Acceptance: <5% rel for grad_q/grad_w/grad_mem
    ok = (fwd_rel < 0.05) and (gq_rel < 0.05) and (gw_rel < 0.05) and (gm_rel < 0.05)
    print(f"  PASS" if ok else f"  FAIL")
    return ok


if __name__ == "__main__":
    smoke_ok = smoke()
    train_ok = training_shapes()
    sys.exit(0 if smoke_ok and train_ok else 1)
