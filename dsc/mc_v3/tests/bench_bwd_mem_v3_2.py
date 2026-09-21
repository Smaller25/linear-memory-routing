"""Benchmark v2 vs v3.2 bwd_mem kernel (bf16 TensorCore dot).

v3.2 = identical to v2 grid (B, H, N), but casts q_scaled and match_wg
to bf16 before tl.dot to enable TensorCore (v2 uses fp32 with allow_tf32=False
which forces CUDA-core FMA).
"""
from __future__ import annotations

import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from dsc.mc_baseline.cached_memory_read import _ssc_gather_read_bwd_mem_noatomic_kernel as V2_MEM
from dsc.mc_v3.cached_memory_read_v3 import _ssc_gather_read_bwd_mem_kernel_v3_2 as V32_MEM


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
    return times[len(times) // 2]


def main():
    B, T, H, K, V, N, R = 8, 4096, 16, 128, 128, 16, 2
    scale, normalize = 1.0, True
    device = "cuda"
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    idx = torch.randint(0, N, (B, T, R), device=device, dtype=torch.long)
    w = torch.randn(B, T, R, device=device, dtype=torch.bfloat16)
    go = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16).float()
    grad_mem = torch.zeros(B, N, H, K, V, device=device, dtype=torch.float32)
    print(f"shapes: B={B} T={T} H={H} K={K} V={V} N={N} R={R}")

    # v2 baseline (current best: BLOCK_T=32 nw=8 ns=3 = 16.85ms in perf doc)
    def call_v2_base():
        grad_mem.zero_()
        V2_MEM[(B, H, N)](
            go, q, idx, w, grad_mem, scale, T, N,
            H=H, K=K, V=V, R=R,
            stride_gob=go.stride(0), stride_got=go.stride(1), stride_goh=go.stride(2),
            stride_qb=q.stride(0), stride_qt=q.stride(1), stride_qh=q.stride(2),
            stride_ib=idx.stride(0), stride_it=idx.stride(1),
            stride_wb=w.stride(0), stride_wt=w.stride(1),
            stride_gmb=grad_mem.stride(0), stride_gmn=grad_mem.stride(1), stride_gmh=grad_mem.stride(2),
            normalize_queries=normalize, BLOCK_T=32, num_warps=8, num_stages=3,
        )
        return grad_mem

    v2_ms = bench_fn(call_v2_base)
    print(f"\n  v2 BASELINE (BLOCK_T=32 nw=8 ns=3): {v2_ms:.2f} ms")

    # v3.2 sweep
    print(f"\n  v3.2 (bf16 TensorCore) sweep:")
    for bt in (16, 32, 64, 128):
        for nw in (4, 8, 16):
            for ns in (1, 2, 3, 4):
                try:
                    def call_v32():
                        grad_mem.zero_()
                        V32_MEM[(B, H, N)](
                            go, q, idx, w, grad_mem, scale, T, N,
                            H=H, K=K, V=V, R=R,
                            stride_gob=go.stride(0), stride_got=go.stride(1), stride_goh=go.stride(2),
                            stride_qb=q.stride(0), stride_qt=q.stride(1), stride_qh=q.stride(2),
                            stride_ib=idx.stride(0), stride_it=idx.stride(1),
                            stride_wb=w.stride(0), stride_wt=w.stride(1),
                            stride_gmb=grad_mem.stride(0), stride_gmn=grad_mem.stride(1), stride_gmh=grad_mem.stride(2),
                            normalize_queries=normalize, BLOCK_T=bt,
                            num_warps=nw, num_stages=ns,
                        )
                        return grad_mem
                    ms = bench_fn(call_v32, warmup=2, iters=10)
                    speedup = v2_ms / ms
                    star = " ***" if ms < v2_ms * 0.4 else (" **" if ms < v2_ms * 0.6 else (" *" if ms < v2_ms else ""))
                    print(f"    BLOCK_T={bt:3d} nw={nw:2d} ns={ns}: {ms:6.2f} ms  ({speedup:.2f}x v2){star}")
                except Exception as e:
                    pass


if __name__ == "__main__":
    main()
