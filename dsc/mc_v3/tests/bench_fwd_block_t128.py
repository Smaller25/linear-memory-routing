"""Quick targeted sweep: just BLOCK_T=128/256/512 for v3c fwd.

Smaller sweep than bench_fwd.py so it finishes faster under contention.
"""
from __future__ import annotations

import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from dsc.mc_v3.cached_memory_read_v3 import _ssc_gather_read_fwd_kernel_v3c as V3CFWD


def bench_fn(fn, warmup=3, iters=15):
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
    scale = 1.0
    normalize = True
    device = "cuda"
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    m = torch.randn(B, N, H, K, V, device=device, dtype=torch.float32)
    idx = torch.randint(0, N, (B, T, R), device=device, dtype=torch.long)
    w = torch.randn(B, T, R, device=device, dtype=torch.bfloat16)
    out = torch.empty((B, T, H, V), device=device, dtype=torch.float32)

    print(f"shapes: B={B} T={T} H={H} K={K} V={V} N={N} R={R}", flush=True)
    baseline = 2.48  # BLOCK_T=64 num_warps=8 num_stages=2 (from full sweep)
    print(f"baseline: BLOCK_T=64 nw=8 ns=2 = {baseline} ms (from prior sweep)", flush=True)
    print(flush=True)

    for bt in (128, 256):
        for nw in (4, 8, 16):
            for ns in (1, 2, 3):
                try:
                    grid = (B, H, T // bt)
                    def call_v3c():
                        V3CFWD[grid](
                            q, m, idx, w, out, scale, T, N,
                            H=H, K=K, V=V, R=R,
                            stride_qb=q.stride(0), stride_qt=q.stride(1), stride_qh=q.stride(2),
                            stride_mb=m.stride(0), stride_mn=m.stride(1), stride_mh=m.stride(2),
                            stride_ib=idx.stride(0), stride_it=idx.stride(1),
                            stride_wb=w.stride(0), stride_wt=w.stride(1),
                            stride_ob=out.stride(0), stride_ot=out.stride(1), stride_oh=out.stride(2),
                            normalize_queries=normalize, BLOCK_T=bt,
                            num_warps=nw, num_stages=ns,
                        )
                        return out
                    ms = bench_fn(call_v3c, warmup=2, iters=8)
                    speedup = baseline / ms
                    star = " ***" if ms < baseline * 0.7 else (" **" if ms < baseline else "")
                    print(f"  BLOCK_T={bt:3d} nw={nw:2d} ns={ns}: {ms:6.2f} ms  ({speedup:.2f}x baseline){star}", flush=True)
                except Exception as e:
                    print(f"  BLOCK_T={bt:3d} nw={nw:2d} ns={ns}: FAILED ({type(e).__name__}: {str(e)[:80]})", flush=True)


if __name__ == "__main__":
    main()
