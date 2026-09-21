"""Estimate MC v3 iter time from kernel-level benchmarks.

Measures fwd, bwd_qw, bwd_mem under same GPU contention (so ratios are
realistic for the actual training scenario where v3 replaces v2).
"""
from __future__ import annotations

import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def bench_fn(fn, warmup=3, iters=20):
    for _ in range(warmup):
        out = fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2], out


def main():
    B, T, H, K, V, N, R = 8, 4096, 16, 128, 128, 16, 2
    scale = 1.0
    normalize = True
    device = "cuda"
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    m = torch.randn(B, N, H, K, V, device=device, dtype=torch.float32)
    idx = torch.randint(0, N, (B, T, R), device=device, dtype=torch.long)
    w = torch.randn(B, T, R, device=device, dtype=torch.bfloat16)

    from dsc.mc_baseline.cached_memory_read import _SSCGatherRead as V2
    from dsc.mc_v3 import _SSCGatherReadV3c as V3C

    # v2 fwd+bwd via autograd
    q_v2 = q.clone().requires_grad_(True)
    m_v2 = m.clone().requires_grad_(True)
    w_v2 = w.clone().requires_grad_(True)

    def call_v2():
        q_v2.grad = None; m_v2.grad = None; w_v2.grad = None
        out = V2.apply(q_v2, m_v2, idx, w_v2, scale, normalize)
        out.sum().backward()
        return out
    v2_ms, _ = bench_fn(call_v2)

    # v3c fwd+bwd via autograd
    q_v3 = q.clone().requires_grad_(True)
    m_v3 = m.clone().requires_grad_(True)
    w_v3 = w.clone().requires_grad_(True)

    def call_v3():
        q_v3.grad = None; m_v3.grad = None; w_v3.grad = None
        out = V3C.apply(q_v3, m_v3, idx, w_v3, scale, normalize)
        out.sum().backward()
        return out
    v3_ms, _ = bench_fn(call_v3)

    print(f"\n=== SSC fwd+bwd per-call (training shapes B=8 T=4K H=16) ===")
    print(f"  v2 (original)           : {v2_ms:.2f} ms/call")
    print(f"  v3c (split bwd + v3c fwd): {v3_ms:.2f} ms/call  ({v2_ms/v3_ms:.2f}x speedup)")
    print()
    # 16 layers, fwd runs 3x per iter (fwd + checkpoint recompute + grad-ckpt-recompute)
    # bwd runs 1x per iter per layer
    v2_iter_ssc = v2_ms * 16 * 2  # rough: fwd + bwd per layer (assume fwd~bwd)
    v3_iter_ssc = v3_ms * 16 * 2
    print(f"  estimate per-iter SSC overhead:")
    print(f"    v2 : {v2_iter_ssc:.0f} ms (16 layers x [fwd+bwd] x ~x2 ckpt overhead)")
    print(f"    v3 : {v3_iter_ssc:.0f} ms")
    print()
    vanilla_baseline = 393  # ms/iter
    print(f"  vanilla baseline: {vanilla_baseline} ms/iter")
    print(f"  MC v2 (actual measured): 1295 ms/iter = {1295/vanilla_baseline:.2f}x vanilla")
    # v3 savings: scale v2 SSC overhead by v3/v2 ratio
    ssc_v2_actual = 1295 - vanilla_baseline  # 902ms
    ssc_v3_actual = ssc_v2_actual * (v3_ms / v2_ms)
    mc_v3_iter = vanilla_baseline + ssc_v3_actual
    print(f"  MC v3 (projected): {mc_v3_iter:.0f} ms/iter = {mc_v3_iter/vanilla_baseline:.2f}x vanilla")


if __name__ == "__main__":
    main()
