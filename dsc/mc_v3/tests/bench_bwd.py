"""Quick per-kernel benchmark of v2 fwd / bwd_qw / bwd_mem at training shapes.

Purpose: confirm which backward kernel is the bottleneck before writing v3.

Run on a single GPU briefly during vanilla training (each call is ~50ms,
total benchmark ~3s of GPU time). Use a small CUDA_VISIBLE_DEVICES subset.

Usage:
    CUDA_VISIBLE_DEVICES=0 python dsc/mc_v3/tests/bench_bwd.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from dsc.mc_baseline.cached_memory_read import (
    _ssc_gather_read_fwd_kernel,
    _ssc_gather_read_bwd_qw_kernel,
    _ssc_gather_read_bwd_mem_noatomic_kernel,
    _SSCGatherRead,
)


def bench_fn(fn, warmup=3, iters=20):
    """Median of `iters` runs after `warmup` warmups."""
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
    # Training shapes: B=8 T=4096 H=16 K=V=128 N=16 R=2
    B, T, H, K, V, N, R = 8, 4096, 16, 128, 128, 16, 2
    scale = 1.0
    normalize = True

    device = "cuda"
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    m = torch.randn(B, N, H, K, V, device=device, dtype=torch.float32)
    idx = torch.randint(0, N, (B, T, R), device=device, dtype=torch.long)
    w = torch.randn(B, T, R, device=device, dtype=torch.bfloat16)

    print(f"shapes: B={B} T={T} H={H} K={K} V={V} N={N} R={R}")
    print(f"device: {torch.cuda.get_device_name(0)}")

    # --- Forward direct kernel call ---
    out = torch.empty((B, T, H, V), device=device, dtype=torch.float32)
    BLOCK_T_FWD = 1  # v2 default
    grid_fwd = (B, H, T // BLOCK_T_FWD)

    def call_fwd():
        _ssc_gather_read_fwd_kernel[grid_fwd](
            q, m, idx, w, out,
            scale, T, N,
            H=H, K=K, V=V, R=R,
            stride_qb=q.stride(0), stride_qt=q.stride(1), stride_qh=q.stride(2),
            stride_mb=m.stride(0), stride_mn=m.stride(1), stride_mh=m.stride(2),
            stride_ib=idx.stride(0), stride_it=idx.stride(1),
            stride_wb=w.stride(0), stride_wt=w.stride(1),
            stride_ob=out.stride(0), stride_ot=out.stride(1), stride_oh=out.stride(2),
            normalize_queries=normalize,
            BLOCK_T=BLOCK_T_FWD,
            num_warps=4, num_stages=2,
        )
        return out

    fwd_ms, _ = bench_fn(call_fwd)
    print(f"\nFWD kernel (BLOCK_T={BLOCK_T_FWD}): {fwd_ms:.3f} ms/call")

    # --- Backward via autograd (calls bwd_qw + bwd_mem) ---
    q_g = q.clone().requires_grad_(True)
    m_g = m.clone().requires_grad_(True)
    w_g = w.clone().requires_grad_(True)

    def call_bwd():
        q_g.grad = None
        m_g.grad = None
        w_g.grad = None
        out = _SSCGatherRead.apply(q_g, m_g, idx, w_g, scale, normalize)
        loss = out.sum()
        loss.backward()
        return out

    bwd_ms, _ = bench_fn(call_bwd)
    print(f"BWD total (autograd, calls bwd_qw + bwd_mem): {bwd_ms:.3f} ms/call")

    # --- bwd_qw kernel directly ---
    go = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16).float()
    grad_q = torch.zeros_like(q)
    grad_w = torch.zeros_like(w)
    BLOCK_T_QW = 1
    grid_qw = (B, H, T // BLOCK_T_QW)

    def call_bwd_qw():
        grad_w.zero_()
        _ssc_gather_read_bwd_qw_kernel[grid_qw](
            go, q, m, idx, w,
            grad_q, grad_w,
            scale, T, N,
            H=H, K=K, V=V, R=R,
            stride_gob=go.stride(0), stride_got=go.stride(1), stride_goh=go.stride(2),
            stride_qb=q.stride(0), stride_qt=q.stride(1), stride_qh=q.stride(2),
            stride_mb=m.stride(0), stride_mn=m.stride(1), stride_mh=m.stride(2),
            stride_ib=idx.stride(0), stride_it=idx.stride(1),
            stride_wb=w.stride(0), stride_wt=w.stride(1),
            stride_gqb=grad_q.stride(0), stride_gqt=grad_q.stride(1), stride_gqh=grad_q.stride(2),
            stride_gwb=grad_w.stride(0), stride_gwt=grad_w.stride(1),
            normalize_queries=normalize,
            BLOCK_T=BLOCK_T_QW,
            num_warps=4, num_stages=2,
        )
        return grad_q

    bwd_qw_ms, _ = bench_fn(call_bwd_qw)
    print(f"BWD_QW kernel (BLOCK_T={BLOCK_T_QW}): {bwd_qw_ms:.3f} ms/call")

    # --- bwd_mem kernel directly ---
    grad_mem = torch.zeros_like(m)
    BLOCK_T_MEM = 32
    grid_mem = (B, H, N)

    def call_bwd_mem():
        grad_mem.zero_()
        _ssc_gather_read_bwd_mem_noatomic_kernel[grid_mem](
            go, q, idx, w, grad_mem,
            scale, T, N,
            H=H, K=K, V=V, R=R,
            stride_gob=go.stride(0), stride_got=go.stride(1), stride_goh=go.stride(2),
            stride_qb=q.stride(0), stride_qt=q.stride(1), stride_qh=q.stride(2),
            stride_ib=idx.stride(0), stride_it=idx.stride(1),
            stride_wb=w.stride(0), stride_wt=w.stride(1),
            stride_gmb=grad_mem.stride(0), stride_gmn=grad_mem.stride(1), stride_gmh=grad_mem.stride(2),
            normalize_queries=normalize,
            BLOCK_T=BLOCK_T_MEM,
            num_warps=8, num_stages=3,
        )
        return grad_mem

    bwd_mem_ms, _ = bench_fn(call_bwd_mem)
    print(f"BWD_MEM kernel (BLOCK_T={BLOCK_T_MEM}): {bwd_mem_ms:.3f} ms/call")

    # --- v3 bwd_qw kernel directly ---
    from dsc.mc_v3.cached_memory_read_v3 import (
        _ssc_gather_read_bwd_q_grad_kernel_v3 as KernA,
        _ssc_gather_read_bwd_w_grad_kernel_v3 as KernB,
    )
    BLOCK_T_Q_V3 = 16
    grid_q_v3 = (B, H, T // BLOCK_T_Q_V3)
    BLOCK_T_W_V3 = 8
    grid_w_v3 = (B, T // BLOCK_T_W_V3)

    def call_bwd_qw_v3():
        _KernA = KernA
        _KernB = KernB
        _KernA[grid_q_v3](
            go, q, m, idx, w,
            grad_q,
            scale, T, N,
            H=H, K=K, V=V, R=R,
            stride_gob=go.stride(0), stride_got=go.stride(1), stride_goh=go.stride(2),
            stride_qb=q.stride(0), stride_qt=q.stride(1), stride_qh=q.stride(2),
            stride_mb=m.stride(0), stride_mn=m.stride(1), stride_mh=m.stride(2),
            stride_ib=idx.stride(0), stride_it=idx.stride(1),
            stride_wb=w.stride(0), stride_wt=w.stride(1),
            stride_gqb=grad_q.stride(0), stride_gqt=grad_q.stride(1), stride_gqh=grad_q.stride(2),
            normalize_queries=normalize,
            BLOCK_T=BLOCK_T_Q_V3,
            num_warps=8, num_stages=2,
        )
        _KernB[grid_w_v3](
            go, q, m, idx, w,
            grad_w,
            scale, T, N,
            H=H, K=K, V=V, R=R,
            stride_gob=go.stride(0), stride_got=go.stride(1), stride_goh=go.stride(2),
            stride_qb=q.stride(0), stride_qt=q.stride(1), stride_qh=q.stride(2),
            stride_mb=m.stride(0), stride_mn=m.stride(1), stride_mh=m.stride(2),
            stride_ib=idx.stride(0), stride_it=idx.stride(1),
            stride_wb=w.stride(0), stride_wt=w.stride(1),
            stride_gwb=grad_w.stride(0), stride_gwt=grad_w.stride(1),
            normalize_queries=normalize,
            BLOCK_T=BLOCK_T_W_V3,
            num_warps=4, num_stages=1,
        )
        return grad_q

    bwd_qw_v3_ms, _ = bench_fn(call_bwd_qw_v3)
    print(f"BWD_QW_v3 split (BLOCK_T_Q={BLOCK_T_Q_V3}, BLOCK_T_W={BLOCK_T_W_V3}): {bwd_qw_v3_ms:.3f} ms/call  (v2: {bwd_qw_ms:.3f} ms)")

    # Try BLOCK_T_Q=32 sweep
    for bt_q in (16, 32, 64, 128):
        for bt_w in (8, 16, 32, 64):
            try:
                grid_q_sweep = (B, H, T // bt_q)
                grid_w_sweep = (B, T // bt_w)
                kern_stages_q = 2 if bt_q <= 16 else 1
                kern_stages_w = 1

                def call_sweep():
                    KernA[grid_q_sweep](
                        go, q, m, idx, w,
                        grad_q,
                        scale, T, N,
                        H=H, K=K, V=V, R=R,
                        stride_gob=go.stride(0), stride_got=go.stride(1), stride_goh=go.stride(2),
                        stride_qb=q.stride(0), stride_qt=q.stride(1), stride_qh=q.stride(2),
                        stride_mb=m.stride(0), stride_mn=m.stride(1), stride_mh=m.stride(2),
                        stride_ib=idx.stride(0), stride_it=idx.stride(1),
                        stride_wb=w.stride(0), stride_wt=w.stride(1),
                        stride_gqb=grad_q.stride(0), stride_gqt=grad_q.stride(1), stride_gqh=grad_q.stride(2),
                        normalize_queries=normalize,
                        BLOCK_T=bt_q,
                        num_warps=8, num_stages=kern_stages_q,
                    )
                    KernB[grid_w_sweep](
                        go, q, m, idx, w,
                        grad_w,
                        scale, T, N,
                        H=H, K=K, V=V, R=R,
                        stride_gob=go.stride(0), stride_got=go.stride(1), stride_goh=go.stride(2),
                        stride_qb=q.stride(0), stride_qt=q.stride(1), stride_qh=q.stride(2),
                        stride_mb=m.stride(0), stride_mn=m.stride(1), stride_mh=m.stride(2),
                        stride_ib=idx.stride(0), stride_it=idx.stride(1),
                        stride_wb=w.stride(0), stride_wt=w.stride(1),
                        stride_gwb=grad_w.stride(0), stride_gwt=grad_w.stride(1),
                        normalize_queries=normalize,
                        BLOCK_T=bt_w,
                        num_warps=4, num_stages=kern_stages_w,
                    )
                    return grad_q
                ms, _ = bench_fn(call_sweep, warmup=2, iters=10)
                print(f"  sweep BLOCK_T_Q={bt_q:3d} BLOCK_T_W={bt_w:3d}: {ms:.2f} ms")
            except Exception as e:
                print(f"  sweep BLOCK_T_Q={bt_q:3d} BLOCK_T_W={bt_w:3d}: FAILED ({type(e).__name__}: {str(e)[:50]})")

    # --- v3c full backward via autograd ---
    from dsc.mc_v3 import _SSCGatherReadV3c as V3C
    q_v3 = q.clone().requires_grad_(True)
    m_v3 = m.clone().requires_grad_(True)
    w_v3 = w.clone().requires_grad_(True)

    def call_bwd_v3c():
        q_v3.grad = None
        m_v3.grad = None
        w_v3.grad = None
        out = V3C.apply(q_v3, m_v3, idx, w_v3, scale, normalize)
        loss = out.sum()
        loss.backward()
        return out

    bwd_v3c_ms, _ = bench_fn(call_bwd_v3c)
    print(f"BWD_v3c total (autograd, v3 bwd_qw + v2 bwd_mem): {bwd_v3c_ms:.3f} ms/call  (v2 total: {bwd_ms:.3f} ms)")

    # --- Summary ---
    print(f"\n=== summary ===")
    print(f"  fwd (v2)       : {fwd_ms:6.2f} ms/call  x16 layers x3 (ckpt) = {fwd_ms*16*3:7.1f} ms/iter")
    print(f"  bwd_qw (v2)    : {bwd_qw_ms:6.2f} ms/call  x16 layers        = {bwd_qw_ms*16:7.1f} ms/iter")
    print(f"  bwd_qw (v3)    : {bwd_qw_v3_ms:6.2f} ms/call  x16 layers        = {bwd_qw_v3_ms*16:7.1f} ms/iter  ({bwd_qw_ms/bwd_qw_v3_ms:.2f}x speedup)")
    print(f"  bwd_mem (v2)   : {bwd_mem_ms:6.2f} ms/call  x16 layers        = {bwd_mem_ms*16:7.1f} ms/iter")
    total_ssc_v2 = fwd_ms * 16 * 3 + (bwd_qw_ms + bwd_mem_ms) * 16
    total_ssc_v3 = fwd_ms * 16 * 3 + (bwd_qw_v3_ms + bwd_mem_ms) * 16
    print(f"  total SSC v2 : {total_ssc_v2:.1f} ms/iter  -> MC iter ≈ {393 + total_ssc_v2:.0f} ms")
    print(f"  total SSC v3 : {total_ssc_v3:.1f} ms/iter  -> MC iter ≈ {393 + total_ssc_v3:.0f} ms")


if __name__ == "__main__":
    main()
