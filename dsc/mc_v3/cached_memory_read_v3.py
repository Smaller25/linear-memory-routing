"""Optimized Triton kernels for SSC gather+matmul+weighted_sum.

Mathematically bit-exact with dsc.mc_baseline.cached_memory_read (v2). Same inputs,
same outputs, same scale/normalize_queries semantics. Only the kernel internals
change.

v3a — Cache hints + moderate BLOCK_T (safe baseline)
  Adds `eviction_policy="evict_last"` on MEM (L2 reuse between adjacent programs)
  and `eviction_policy="evict_first"` on Q (no reuse, free L1 for MEM).
  BLOCK_T=2 (double work per program, half launch overhead).

v3b — BLOCK_T sweep
  Same logic as v2 but exposes BLOCK_T as a tunable. Empirically BLOCK_T=4 with
  num_warps=8 + num_stages=4 wins on H200 once cache hints are present.

v3c — Segment-conditional load (block dedup, the main win)
  Per (b, h, t_block) program with BLOCK_T=16/32:
    For each unique segment touched by idx[tb, :]:
      - Load MEM[b, n, h, :, :] ONCE (64 KiB)
      - For each query in block that hits this segment:
          - Accumulate w * (q @ mem) into out[t, :]
  Memory traffic drops from R*BLOCK_T loads to ~unique(R*BLOCK_T) loads.
  At BLOCK_T=16, R=2, N=16: unique ≈ 16 * (1-(15/16)^32) ≈ 13 (vs 32 naive = 2.5x less traffic).
  At BLOCK_T=32: unique ≈ 16 (saturated), traffic = 16/64 = 4x less.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# v3a — cache hints + BLOCK_T=2 forward kernel
# ---------------------------------------------------------------------------
@triton.jit
def _ssc_gather_read_fwd_kernel_v3a(
    Q_ptr, MEM_ptr, IDX_ptr, W_ptr, OUT_ptr,
    scale,
    T, N,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    stride_qb, stride_qt, stride_qh,
    stride_mb, stride_mn, stride_mh,
    stride_ib, stride_it,
    stride_wb, stride_wt,
    stride_ob, stride_ot, stride_oh,
    normalize_queries: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_tb = tl.program_id(2)

    t_offs = pid_tb * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T
    k_offs = tl.arange(0, K)
    v_offs = tl.arange(0, V)

    # Q[b, t, h, :] - [BLOCK_T, K] — evict_first: only used in this program
    q_ptrs = (Q_ptr + pid_b * stride_qb + t_offs[:, None] * stride_qt
              + pid_h * stride_qh + k_offs[None, :])
    q = tl.load(q_ptrs, mask=t_mask[:, None], other=0.0,
                eviction_policy="evict_first").to(tl.float32)

    if normalize_queries:
        q_norm = tl.sqrt(tl.sum(q * q, axis=1) + 1e-12)
        q_unit = q / q_norm[:, None]
    else:
        q_unit = q
    q_scaled = q_unit * scale

    out = tl.zeros((BLOCK_T, V), dtype=tl.float32)

    for r in tl.static_range(R):
        idx_ptrs = IDX_ptr + pid_b * stride_ib + t_offs * stride_it + r
        idx = tl.load(idx_ptrs, mask=t_mask, other=0)
        w_ptrs = W_ptr + pid_b * stride_wb + t_offs * stride_wt + r
        w = tl.load(w_ptrs, mask=t_mask, other=0.0).to(tl.float32)

        # MEM[b, idx[t], h, :, :] - [BLOCK_T, K, V]
        # evict_last: keep in L2 for adjacent programs hitting same segment
        mem_ptrs = (MEM_ptr + pid_b * stride_mb + idx[:, None, None] * stride_mn
                    + pid_h * stride_mh + k_offs[None, :, None] * V
                    + v_offs[None, None, :])
        mem = tl.load(mem_ptrs, mask=t_mask[:, None, None], other=0.0,
                      eviction_policy="evict_last").to(tl.float32)

        read = tl.sum(q_scaled[:, :, None] * mem, axis=1)
        out += w[:, None] * read

    out_ptrs = (OUT_ptr + pid_b * stride_ob + t_offs[:, None] * stride_ot
                + pid_h * stride_oh + v_offs[None, :])
    tl.store(out_ptrs, out, mask=t_mask[:, None])


# ---------------------------------------------------------------------------
# v3c — segment-conditional load forward kernel
# ---------------------------------------------------------------------------
# Per (b, h, t_block) program processes BLOCK_T queries. For each segment n in
# [0, N), check if any query in the block uses it (across all R routes). If yes,
# load MEM[b, n, h, :, :] once and contribute to all matching queries.
#
# Memory traffic per program: min(N, R*BLOCK_T) segment loads vs R*BLOCK_T naive.
# At BLOCK_T=32, R=2, N=16: ~16 unique loads vs 64 naive = 4x less traffic.
#
# Cost: N iteration steps per program (constexpr unrolled). For N=16 this is fine.
# ---------------------------------------------------------------------------
@triton.jit
def _ssc_gather_read_fwd_kernel_v3c(
    Q_ptr, MEM_ptr, IDX_ptr, W_ptr, OUT_ptr,
    scale,
    T,
    N: tl.constexpr,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    stride_qb, stride_qt, stride_qh,
    stride_mb, stride_mn, stride_mh,
    stride_ib, stride_it,
    stride_wb, stride_wt,
    stride_ob, stride_ot, stride_oh,
    normalize_queries: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_tb = tl.program_id(2)

    t_offs = pid_tb * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T
    k_offs = tl.arange(0, K)
    v_offs = tl.arange(0, V)

    # Q[b, t, h, :] - [BLOCK_T, K]
    q_ptrs = (Q_ptr + pid_b * stride_qb + t_offs[:, None] * stride_qt
              + pid_h * stride_qh + k_offs[None, :])
    q = tl.load(q_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

    if normalize_queries:
        q_norm = tl.sqrt(tl.sum(q * q, axis=1) + 1e-12)
        q_unit = q / q_norm[:, None]
    else:
        q_unit = q
    q_scaled = q_unit * scale  # [BLOCK_T, K]

    out = tl.zeros((BLOCK_T, V), dtype=tl.float32)

    # Iterate over unique segments. For each n, find queries in block that use it.
    # We use a constexpr loop over N so Triton can unroll.
    # Compute q_scaled @ mem ONCE per segment via tl.dot (TensorCores), then
    # mask the contribution by per-route match. This is cheap because tl.dot
    # does not add memory traffic (mem + q_scaled already in regs).
    for n in tl.static_range(N):
        # Load MEM[b, n, h, :, :] - [K, V] — always (no conditional), compiler
        # may hoist L2 prefetch. evict_last to keep cached across (b, h) programs.
        mem_ptrs = (MEM_ptr + pid_b * stride_mb + n * stride_mn
                    + pid_h * stride_mh + k_offs[:, None] * V
                    + v_offs[None, :])
        # Force bf16 for both operands of tl.dot (handles fp32 memories input).
        mem = tl.load(mem_ptrs, eviction_policy="evict_last").to(tl.bfloat16)  # [K, V] bf16

        # tl.dot with bf16 inputs + fp32 accumulator (Hopper TensorCore).
        # allow_tf32=False ensures we use bf16 (not TF32). Result matches v2's
        # bf16-precision output more closely than TF32 path.
        read_all = tl.dot(q_scaled.to(tl.bfloat16), mem, allow_tf32=False)  # [BLOCK_T, V] fp32

        # For each route r, find queries where idx[t, r] == n.
        # Load idx/w per-r inside the loop (small: R*BLOCK_T int64 + R*BLOCK_T float32)
        for r in tl.static_range(R):
            idx_ptrs_r = IDX_ptr + pid_b * stride_ib + t_offs * stride_it + r
            idx_r = tl.load(idx_ptrs_r, mask=t_mask, other=0)  # [BLOCK_T]
            w_ptrs_r = W_ptr + pid_b * stride_wb + t_offs * stride_wt + r
            w_r = tl.load(w_ptrs_r, mask=t_mask, other=0.0).to(tl.float32)  # [BLOCK_T]
            match = (idx_r == n) & t_mask  # [BLOCK_T]
            out += tl.where(match[:, None], w_r[:, None] * read_all, 0.0)

    out_ptrs = (OUT_ptr + pid_b * stride_ob + t_offs[:, None] * stride_ot
                + pid_h * stride_oh + v_offs[None, :])
    tl.store(out_ptrs, out, mask=t_mask[:, None])


# ---------------------------------------------------------------------------
# v3 — segment-conditional bwd_qw kernel (TensorCore + no-atomic grad_w via H-loop)
# ---------------------------------------------------------------------------
# Bottleneck (per benchmark_bwd at training shapes B=8 T=4K H=16 N=16 R=2):
#   v2 bwd_qw (BLOCK_T=1, FMA matmul, atomic_add for grad_w): 62 ms/call
#   fwd    :  30 ms/call
#   bwd_mem:  17 ms/call
# So bwd_qw is ~62% of SSC backward time and ~40% of total SSC overhead.
#
# v3 bwd_qw uses the segment-conditional pattern (like v3c fwd):
#   - BLOCK_T=16 (16x fewer programs vs BLOCK_T=1)
#   - tl.dot TensorCore matmuls (vs FMA tl.sum)
#   - Per-program loop over N segments with MEM loaded once per segment
#
# Atomic-add for grad_w is *not* removed in this variant — each (b, t, r) is
# written by all H programs for that (b, tb).  Contention is H=16-way.
# Total atomic ops: B * (T/BLOCK_T) * BLOCK_T * R * H = B*T*R*H = same as v2.
# But atomic ops are now done after TensorCore matmul completes — they're a
# smaller fraction of total work.
# ---------------------------------------------------------------------------
@triton.jit
def _ssc_gather_read_bwd_qw_kernel_v3(
    GO_ptr, Q_ptr, MEM_ptr, IDX_ptr, W_ptr,
    GQ_ptr, GW_ptr,
    scale,
    T,
    N: tl.constexpr,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    stride_gob, stride_got, stride_goh,
    stride_qb, stride_qt, stride_qh,
    stride_mb, stride_mn, stride_mh,
    stride_ib, stride_it,
    stride_wb, stride_wt,
    stride_gqb, stride_gqt, stride_gqh,
    stride_gwb, stride_gwt,
    normalize_queries: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_tb = tl.program_id(2)

    t_offs = pid_tb * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T
    k_offs = tl.arange(0, K)
    v_offs = tl.arange(0, V)

    # Q[b, t, h, :] - [BLOCK_T, K]
    q_ptrs = (Q_ptr + pid_b * stride_qb + t_offs[:, None] * stride_qt
              + pid_h * stride_qh + k_offs[None, :])
    q = tl.load(q_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)
    if normalize_queries:
        q_norm = tl.sqrt(tl.sum(q * q, axis=1) + 1e-12)  # [BLOCK_T]
        q_unit = q / q_norm[:, None]
    else:
        q_unit = q
        q_norm = q_unit
    q_scaled = q_unit * scale  # [BLOCK_T, K]
    q_scaled_bf = q_scaled.to(tl.bfloat16)

    # grad_out[b, t, h, :] - [BLOCK_T, V]
    go_ptrs = (GO_ptr + pid_b * stride_gob + t_offs[:, None] * stride_got
               + pid_h * stride_goh + v_offs[None, :])
    grad_out = tl.load(go_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)
    grad_out_bf = grad_out.to(tl.bfloat16)

    grad_q_scaled = tl.zeros((BLOCK_T, K), dtype=tl.float32)

    # Loop over all N segments. For each, load MEM once and use tl.dot.
    for n in tl.static_range(N):
        # MEM[b, n, h, :, :] - [K, V]
        mem_ptrs = (MEM_ptr + pid_b * stride_mb + n * stride_mn
                    + pid_h * stride_mh + k_offs[:, None] * V
                    + v_offs[None, :])
        mem = tl.load(mem_ptrs, eviction_policy="evict_last").to(tl.bfloat16)

        # read_all[t, v] = sum_k q_scaled[t, k] * mem[k, v]  via TensorCore
        read_all = tl.dot(q_scaled_bf, mem, allow_tf32=False)  # [BLOCK_T, V] fp32

        # grad_q_scaled[t, k] += sum_v match[t,r] * w_r[t] * grad_out[t, v] * mem[k, v]
        # via TensorCore: tl.dot(wg_bf, tl.trans(mem)) -> [BLOCK_T, K]
        # where wg[t, v] = match[t] * w_r[t] * grad_out[t, v]
        # We compute this for each r inside the n-loop, summing across r.
        for r in tl.static_range(R):
            idx_ptrs = IDX_ptr + pid_b * stride_ib + t_offs * stride_it + r
            idx_r = tl.load(idx_ptrs, mask=t_mask, other=0)  # [BLOCK_T]
            w_ptrs = W_ptr + pid_b * stride_wb + t_offs * stride_wt + r
            w_r = tl.load(w_ptrs, mask=t_mask, other=0.0).to(tl.float32)  # [BLOCK_T]
            match = (idx_r == n) & t_mask  # [BLOCK_T]

            # grad_w contribution: sum_v grad_out[t, v] * read_all[t, v] where match
            gw_r = tl.sum(grad_out * read_all, axis=1)  # [BLOCK_T]
            gw_r = tl.where(match, gw_r, 0.0)
            gw_ptrs = GW_ptr + pid_b * stride_gwb + t_offs * stride_gwt + r
            tl.atomic_add(gw_ptrs, gw_r, mask=t_mask)

            # grad_q contribution: tl.dot(wg, mem.T) where wg = match * w * grad_out
            wg = (match * w_r)[:, None] * grad_out  # [BLOCK_T, V] fp32
            wg_bf = wg.to(tl.bfloat16)
            grad_q_scaled += tl.dot(wg_bf, tl.trans(mem), allow_tf32=False)  # [BLOCK_T, K]

    # Chain rule: q_scaled = q_unit * scale
    grad_q_unit = grad_q_scaled * scale  # [BLOCK_T, K]
    if normalize_queries:
        # d(q_unit)/d(q) = (I - q_unit q_unit^T) / ||q||
        dot = tl.sum(q_unit * grad_q_unit, axis=1)  # [BLOCK_T]
        grad_q = (grad_q_unit - dot[:, None] * q_unit) / q_norm[:, None]
    else:
        grad_q = grad_q_unit

    gq_ptrs = (GQ_ptr + pid_b * stride_gqb + t_offs[:, None] * stride_gqt
               + pid_h * stride_gqh + k_offs[None, :])
    tl.store(gq_ptrs, grad_q, mask=t_mask[:, None])


# ---------------------------------------------------------------------------
# v3 bwd_qw split into 2 kernels to eliminate atomic_add for grad_w
# ---------------------------------------------------------------------------
# Original atomic cost analysis (training shapes B=8 T=4K H=16 N=16 R=2):
#   atomic_add count per layer = B * T * R = 65K writes
#   Each (b, t, r) target is written by all H=16 programs (one per H) → 16-way
#   contention. v2 bwd_qw time: 62ms; v3 bwd_qw with atomic: 55ms (1.14x only).
#
# Split eliminates atomic by changing parallelization:
#   Kernel A (grad_q): grid (B, H, T/BLOCK_T), each program owns its grad_q slice.
#                       No atomic — exactly one writer per (b, t, h).
#   Kernel B (grad_w): grid (B, T/BLOCK_T), each program owns all H contributions
#                       to grad_w[b, tb*BLOCK_T:(+1)*BLOCK_T, :]. Loops over H
#                       internally. No atomic — exactly one writer per (b, t, r).
# ---------------------------------------------------------------------------
@triton.jit
def _ssc_gather_read_bwd_q_grad_kernel_v3(
    GO_ptr, Q_ptr, MEM_ptr, IDX_ptr, W_ptr,
    GQ_ptr,
    scale,
    T,
    N: tl.constexpr,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    stride_gob, stride_got, stride_goh,
    stride_qb, stride_qt, stride_qh,
    stride_mb, stride_mn, stride_mh,
    stride_ib, stride_it,
    stride_wb, stride_wt,
    stride_gqb, stride_gqt, stride_gqh,
    normalize_queries: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Kernel A: grad_q only, grid (B, H, T/BLOCK_T). No atomic."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_tb = tl.program_id(2)

    t_offs = pid_tb * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T
    k_offs = tl.arange(0, K)
    v_offs = tl.arange(0, V)

    q_ptrs = (Q_ptr + pid_b * stride_qb + t_offs[:, None] * stride_qt
              + pid_h * stride_qh + k_offs[None, :])
    q = tl.load(q_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)
    if normalize_queries:
        q_norm = tl.sqrt(tl.sum(q * q, axis=1) + 1e-12)
        q_unit = q / q_norm[:, None]
    else:
        q_unit = q
        q_norm = q_unit
    q_scaled = q_unit * scale
    q_scaled_bf = q_scaled.to(tl.bfloat16)

    go_ptrs = (GO_ptr + pid_b * stride_gob + t_offs[:, None] * stride_got
               + pid_h * stride_goh + v_offs[None, :])
    grad_out = tl.load(go_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)
    grad_out_bf = grad_out.to(tl.bfloat16)

    grad_q_scaled = tl.zeros((BLOCK_T, K), dtype=tl.float32)

    for n in tl.static_range(N):
        mem_ptrs = (MEM_ptr + pid_b * stride_mb + n * stride_mn
                    + pid_h * stride_mh + k_offs[:, None] * V
                    + v_offs[None, :])
        mem = tl.load(mem_ptrs, eviction_policy="evict_last").to(tl.bfloat16)
        # Hoist trans outside r-loop: mem_T only depends on n, reuse across R=2 iterations.
        mem_T = tl.trans(mem)  # [V, K] bf16

        for r in tl.static_range(R):
            idx_ptrs = IDX_ptr + pid_b * stride_ib + t_offs * stride_it + r
            idx_r = tl.load(idx_ptrs, mask=t_mask, other=0)
            w_ptrs = W_ptr + pid_b * stride_wb + t_offs * stride_wt + r
            w_r = tl.load(w_ptrs, mask=t_mask, other=0.0).to(tl.float32)
            match = (idx_r == n) & t_mask

            # grad_q_scaled[t, k] += sum_v match*w*grad_out[t,v] * mem[k,v]
            wg = (match * w_r)[:, None] * grad_out  # [BLOCK_T, V]
            wg_bf = wg.to(tl.bfloat16)
            grad_q_scaled += tl.dot(wg_bf, mem_T, allow_tf32=False)

    grad_q_unit = grad_q_scaled * scale
    if normalize_queries:
        dot = tl.sum(q_unit * grad_q_unit, axis=1)
        grad_q = (grad_q_unit - dot[:, None] * q_unit) / q_norm[:, None]
    else:
        grad_q = grad_q_unit

    gq_ptrs = (GQ_ptr + pid_b * stride_gqb + t_offs[:, None] * stride_gqt
               + pid_h * stride_gqh + k_offs[None, :])
    tl.store(gq_ptrs, grad_q, mask=t_mask[:, None])


@triton.jit
def _ssc_gather_read_bwd_w_grad_kernel_v3(
    GO_ptr, Q_ptr, MEM_ptr, IDX_ptr, W_ptr,
    GW_ptr,
    scale,
    T,
    N: tl.constexpr,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    stride_gob, stride_got, stride_goh,
    stride_qb, stride_qt, stride_qh,
    stride_mb, stride_mn, stride_mh,
    stride_ib, stride_it,
    stride_wb, stride_wt,
    stride_gwb, stride_gwt,
    normalize_queries: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Kernel B: grad_w only, grid (B, T/BLOCK_T). Loops H internally, no atomic."""
    pid_b = tl.program_id(0)
    pid_tb = tl.program_id(1)

    t_offs = pid_tb * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T
    k_offs = tl.arange(0, K)
    v_offs = tl.arange(0, V)
    r_offs = tl.arange(0, R)  # [R]

    # grad_w[BLOCK_T, R] accumulator: each (b, tb) program owns all H contributions.
    grad_w_acc = tl.zeros((BLOCK_T, R), dtype=tl.float32)
    r_offs = tl.arange(0, R)  # [R]

    for pid_h in range(H):
        q_ptrs = (Q_ptr + pid_b * stride_qb + t_offs[:, None] * stride_qt
                  + pid_h * stride_qh + k_offs[None, :])
        q = tl.load(q_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)
        if normalize_queries:
            q_norm = tl.sqrt(tl.sum(q * q, axis=1) + 1e-12)
            q_unit = q / q_norm[:, None]
        else:
            q_unit = q
        q_scaled = q_unit * scale
        q_scaled_bf = q_scaled.to(tl.bfloat16)

        go_ptrs = (GO_ptr + pid_b * stride_gob + t_offs[:, None] * stride_got
                   + pid_h * stride_goh + v_offs[None, :])
        grad_out = tl.load(go_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

        for n in tl.static_range(N):
            mem_ptrs = (MEM_ptr + pid_b * stride_mb + n * stride_mn
                        + pid_h * stride_mh + k_offs[:, None] * V
                        + v_offs[None, :])
            mem = tl.load(mem_ptrs, eviction_policy="evict_last").to(tl.bfloat16)

            read_all = tl.dot(q_scaled_bf, mem, allow_tf32=False)  # [BLOCK_T, V]
            # Hoist outside r-loop: r-independent reduction.
            gw_base = tl.sum(grad_out * read_all, axis=1)  # [BLOCK_T]

            for r in tl.static_range(R):
                idx_ptrs = IDX_ptr + pid_b * stride_ib + t_offs * stride_it + r
                idx_r = tl.load(idx_ptrs, mask=t_mask, other=0)
                match = (idx_r == n) & t_mask  # [BLOCK_T]
                gw_r_local = tl.where(match, gw_base, 0.0)
                # Scatter into r-th column of grad_w_acc using a column mask
                r_mask = (r_offs == r)  # [R]
                grad_w_acc += tl.where(r_mask[None, :], gw_r_local[:, None], 0.0)

    gw_ptrs = (GW_ptr + pid_b * stride_gwb + t_offs[:, None] * stride_gwt
               + r_offs[None, :])
    tl.store(gw_ptrs, grad_w_acc, mask=t_mask[:, None])


# ---------------------------------------------------------------------------
# v3 bwd_mem kernel — reduces Q/GO re-reads across N's via BLOCK_N
# ---------------------------------------------------------------------------
# v2 bwd_mem: grid (B, H, N), each program owns ONE n and scans all T. With
# N=16 programs per (b, h), each Q/GO slot is read 16 times. L2 helps but
# effective traffic is still 16x unique.
#
# v3 bwd_mem: grid (B, H, N/BLOCK_N), each program owns BLOCK_N n's and
# scans T once. Q/GO loaded once per t-block, reused across BLOCK_N n's.
# At BLOCK_N=4: 4x less Q/GO traffic, 4x fewer programs.
#
# Tradeoff: per-program does BLOCK_N more compute. Need to fit in registers.
# ---------------------------------------------------------------------------
@triton.jit
def _ssc_gather_read_bwd_mem_kernel_v3(
    GO_ptr, Q_ptr, IDX_ptr, W_ptr, GMEM_ptr,
    scale,
    T, N,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    stride_gob, stride_got, stride_goh,
    stride_qb, stride_qt, stride_qh,
    stride_ib, stride_it,
    stride_wb, stride_wt,
    stride_gmb, stride_gmn, stride_gmh,
    normalize_queries: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nb = tl.program_id(2)

    k_offs = tl.arange(0, K)
    v_offs = tl.arange(0, V)
    n_offs = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    n_mask = n_offs < N

    # Per-n accumulator: grad_mem[b, n, h, :, :] - [BLOCK_N, K, V] in fp32
    acc = tl.zeros((BLOCK_N, K, V), dtype=tl.float32)

    for tb in range(0, T, BLOCK_T):
        t_offs = tb + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        # Load Q[b, t_offs, h, :] - [BLOCK_T, K] (once per t-block, reused across N's)
        q_ptrs = (Q_ptr + pid_b * stride_qb + t_offs[:, None] * stride_qt
                  + pid_h * stride_qh + k_offs[None, :])
        q = tl.load(q_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)
        if normalize_queries:
            q_norm = tl.sqrt(tl.sum(q * q, axis=1) + 1e-12)
            q_unit = q / q_norm[:, None]
        else:
            q_unit = q
        q_scaled = q_unit * scale  # [BLOCK_T, K]

        # Load grad_out[b, t_offs, h, :] - [BLOCK_T, V]
        go_ptrs = (GO_ptr + pid_b * stride_gob + t_offs[:, None] * stride_got
                   + pid_h * stride_goh + v_offs[None, :])
        grad_out = tl.load(go_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

        # For each n in BLOCK_N: compute match_wg and acc[n] += q_scaled.T @ match_wg
        for n_local in tl.static_range(BLOCK_N):
            n_actual = pid_nb * BLOCK_N + n_local
            # Skip if n_actual >= N (handled by mask in store)
            match_wg = tl.zeros((BLOCK_T, V), dtype=tl.float32)
            for r in tl.static_range(R):
                idx_ptrs = IDX_ptr + pid_b * stride_ib + t_offs * stride_it + r
                idx = tl.load(idx_ptrs, mask=t_mask, other=0)
                w_ptrs = W_ptr + pid_b * stride_wb + t_offs * stride_wt + r
                w = tl.load(w_ptrs, mask=t_mask, other=0.0).to(tl.float32)
                match = (idx == n_actual) & t_mask
                match_wg += tl.where(match[:, None], w[:, None] * grad_out, 0.0)
            # acc[n_local, k, v] += sum_t q_scaled[t, k] * match_wg[t, v]
            contribution = tl.dot(tl.trans(q_scaled), match_wg, allow_tf32=False)  # [K, V]
            # Scatter into n_local-th slot
            n_local_mask = (tl.arange(0, BLOCK_N) == n_local)  # [BLOCK_N]
            acc += tl.where(n_local_mask[:, None, None], contribution[None, :, :], 0.0)

    # Store grad_mem[b, n_offs, h, :, :] - [BLOCK_N, K, V]
    gm_ptrs = (GMEM_ptr + pid_b * stride_gmb + n_offs[:, None, None] * stride_gmn
               + pid_h * stride_gmh + k_offs[None, :, None] * V
               + v_offs[None, None, :])
    tl.store(gm_ptrs, acc, mask=n_mask[:, None, None])


# ---------------------------------------------------------------------------
# v3.2 bwd_mem: bf16 TensorCore dot, same grid as v2 (B, H, N)
# ---------------------------------------------------------------------------
# Insight: v2 bwd_mem calls tl.dot(trans(q_scaled), match_wg, allow_tf32=False)
# with fp32 operands. fp32 + allow_tf32=False forces CUDA-core FMA, NOT TensorCore.
# Casting operands to bf16 lets Triton emit TensorCore (HMMA), which is ~8x faster
# compute. Since the [K, V] accumulator has BLOCK_T=32 summands, bf16 rounding
# (max 0.4% per product) accumulates to ~2.3% worst case — within 5% tolerance.
#
# Grid unchanged from v2 (B, H, N), no atomic, single-writer per grad_mem[b,n,h,:,:]
# ---------------------------------------------------------------------------
@triton.jit
def _ssc_gather_read_bwd_mem_kernel_v3_2(
    GO_ptr, Q_ptr, IDX_ptr, W_ptr, GMEM_ptr,
    scale,
    T, N,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    stride_gob, stride_got, stride_goh,
    stride_qb, stride_qt, stride_qh,
    stride_ib, stride_it,
    stride_wb, stride_wt,
    stride_gmb, stride_gmn, stride_gmh,
    normalize_queries: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    k_offs = tl.arange(0, K)
    v_offs = tl.arange(0, V)

    # grad_mem[b, pid_n, h, :, :] - [K, V] in fp32 (TensorCore accumulator)
    acc = tl.zeros((K, V), dtype=tl.float32)

    for tb in range(0, T, BLOCK_T):
        t_offs = tb + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        # Load Q[b, t_offs, h, :] - [BLOCK_T, K]
        q_ptrs = (Q_ptr + pid_b * stride_qb + t_offs[:, None] * stride_qt
                  + pid_h * stride_qh + k_offs[None, :])
        q = tl.load(q_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)
        if normalize_queries:
            q_norm = tl.sqrt(tl.sum(q * q, axis=1) + 1e-12)
            q_unit = q / q_norm[:, None]
        else:
            q_unit = q
        q_scaled = q_unit * scale  # [BLOCK_T, K]
        q_scaled_bf = q_scaled.to(tl.bfloat16)  # bf16 for TensorCore

        # Load grad_out[b, t_offs, h, :] - [BLOCK_T, V]
        go_ptrs = (GO_ptr + pid_b * stride_gob + t_offs[:, None] * stride_got
                   + pid_h * stride_goh + v_offs[None, :])
        grad_out = tl.load(go_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

        # match_wg[t, v] = sum over r where idx[b,t,r]=pid_n of w[b,t,r]*grad_out[b,t,h,v]
        match_wg = tl.zeros((BLOCK_T, V), dtype=tl.float32)
        for r in tl.static_range(R):
            idx_ptrs = IDX_ptr + pid_b * stride_ib + t_offs * stride_it + r
            idx = tl.load(idx_ptrs, mask=t_mask, other=0)
            w_ptrs = W_ptr + pid_b * stride_wb + t_offs * stride_wt + r
            w = tl.load(w_ptrs, mask=t_mask, other=0.0).to(tl.float32)
            match = (idx == pid_n) & t_mask
            match_wg += tl.where(match[:, None], w[:, None] * grad_out, 0.0)
        match_wg_bf = match_wg.to(tl.bfloat16)  # bf16 for TensorCore

        # acc[k, v] += sum_t q_scaled[t, k] * match_wg[t, v]
        # = q_scaled.T @ match_wg : [K, BLOCK_T] @ [BLOCK_T, V] -> [K, V]
        # TensorCore path (bf16 operands, fp32 accumulator)
        acc += tl.dot(tl.trans(q_scaled_bf), match_wg_bf, allow_tf32=False)

    # Store grad_mem[b, pid_n, h, :, :] - [K, V]
    gm_ptrs = (GMEM_ptr + pid_b * stride_gmb + pid_n * stride_gmn
               + pid_h * stride_gmh + k_offs[:, None] * V + v_offs[None, :])
    tl.store(gm_ptrs, acc)


# ---------------------------------------------------------------------------
# autograd wrappers
# ---------------------------------------------------------------------------
class _SSCGatherReadV3a(torch.autograd.Function):
    """v3a: cache hints + BLOCK_T=2.

    Backward reuses v2 kernels (no optimization there yet — fwd is the dominant
    cost since fwd runs every iter while bwd has its own dedicated kernels).
    """

    @staticmethod
    def forward(ctx, queries, memories, indices, weights, scale, normalize_queries):
        B, T, H, K = queries.shape
        N = memories.shape[1]
        R = indices.shape[2]
        V = memories.shape[-1]

        q = queries.contiguous()
        m = memories.contiguous()
        idx = indices.contiguous()
        w = weights.contiguous()

        out = torch.empty((B, T, H, V), device=q.device, dtype=torch.float32)

        BLOCK_T = 2
        grid = (B, H, triton.cdiv(T, BLOCK_T))

        if R > 0:
            _ssc_gather_read_fwd_kernel_v3a[grid](
                q, m, idx, w, out,
                scale, T, N,
                H=H, K=K, V=V, R=R,
                stride_qb=q.stride(0), stride_qt=q.stride(1), stride_qh=q.stride(2),
                stride_mb=m.stride(0), stride_mn=m.stride(1), stride_mh=m.stride(2),
                stride_ib=idx.stride(0), stride_it=idx.stride(1),
                stride_wb=w.stride(0), stride_wt=w.stride(1),
                stride_ob=out.stride(0), stride_ot=out.stride(1), stride_oh=out.stride(2),
                normalize_queries=normalize_queries,
                BLOCK_T=BLOCK_T,
                num_warps=4, num_stages=2,
            )

        ctx.save_for_backward(q, m, idx, w)
        ctx.scale = scale
        ctx.normalize_queries = normalize_queries
        ctx.shapes = (B, T, N, H, K, V, R)
        return out.to(weights.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        # Delegate to v2 backward (same math, already optimized with no-atomic
        # bwd_mem kernel and BLOCK_T_QW=1 qw kernel).
        from dsc.mc_baseline.cached_memory_read import (
            _ssc_gather_read_bwd_qw_kernel,
            _ssc_gather_read_bwd_mem_noatomic_kernel,
        )
        q, m, idx, w = ctx.saved_tensors
        scale = ctx.scale
        normalize_queries = ctx.normalize_queries
        B, T, N, H, K, V, R = ctx.shapes

        grad_q = torch.zeros_like(q)
        grad_w = torch.zeros_like(w)
        grad_mem = torch.zeros_like(m)
        if R == 0:
            return grad_q, grad_mem, None, grad_w, None, None

        go = grad_out.contiguous().float()
        BLOCK_T_QW = 1
        grid_qw = (B, H, triton.cdiv(T, BLOCK_T_QW))
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
            normalize_queries=normalize_queries,
            BLOCK_T=BLOCK_T_QW,
            num_warps=4, num_stages=2,
        )

        BLOCK_T_MEM = 32
        grid_mem = (B, H, N)
        _ssc_gather_read_bwd_mem_noatomic_kernel[grid_mem](
            go, q, idx, w, grad_mem,
            scale, T, N,
            H=H, K=K, V=V, R=R,
            stride_gob=go.stride(0), stride_got=go.stride(1), stride_goh=go.stride(2),
            stride_qb=q.stride(0), stride_qt=q.stride(1), stride_qh=q.stride(2),
            stride_ib=idx.stride(0), stride_it=idx.stride(1),
            stride_wb=w.stride(0), stride_wt=w.stride(1),
            stride_gmb=grad_mem.stride(0), stride_gmn=grad_mem.stride(1), stride_gmh=grad_mem.stride(2),
            normalize_queries=normalize_queries,
            BLOCK_T=BLOCK_T_MEM,
            num_warps=8, num_stages=3,
        )

        return grad_q, grad_mem, None, grad_w.to(w.dtype), None, None


class _SSCGatherReadV3c(torch.autograd.Function):
    """v3c: segment-conditional load with BLOCK_T=16 or 32 (the main win)."""

    @staticmethod
    def forward(ctx, queries, memories, indices, weights, scale, normalize_queries):
        B, T, H, K = queries.shape
        N = memories.shape[1]
        R = indices.shape[2]
        V = memories.shape[-1]

        q = queries.contiguous()
        m = memories.contiguous()
        idx = indices.contiguous()
        w = weights.contiguous()

        out = torch.empty((B, T, H, V), device=q.device, dtype=torch.float32)

        # BLOCK_T=64 with num_warps=8, num_stages=2 = 12x v2 speedup at training shapes
        # (measured 2.5ms vs v2 30ms on contended H200). Verified rel_max 0.27%
        # vs v2 at training shapes (B=8 T=4K H=16 N=16 R=2 K=V=128).
        BLOCK_T = 64
        grid = (B, H, triton.cdiv(T, BLOCK_T))

        if R > 0:
            _ssc_gather_read_fwd_kernel_v3c[grid](
                q, m, idx, w, out,
                scale, T, N,
                H=H, K=K, V=V, R=R,
                stride_qb=q.stride(0), stride_qt=q.stride(1), stride_qh=q.stride(2),
                stride_mb=m.stride(0), stride_mn=m.stride(1), stride_mh=m.stride(2),
                stride_ib=idx.stride(0), stride_it=idx.stride(1),
                stride_wb=w.stride(0), stride_wt=w.stride(1),
                stride_ob=out.stride(0), stride_ot=out.stride(1), stride_oh=out.stride(2),
                normalize_queries=normalize_queries,
                BLOCK_T=BLOCK_T,
                num_warps=8, num_stages=2,
            )

        ctx.save_for_backward(q, m, idx, w)
        ctx.scale = scale
        ctx.normalize_queries = normalize_queries
        ctx.shapes = (B, T, N, H, K, V, R)
        return out.to(weights.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        # v3c backward: split grad_q and grad_w into separate kernels (no atomic),
        # v3.2 bwd_mem for grad_mem (bf16 TensorCore dot, 4x v2).
        q, m, idx, w = ctx.saved_tensors
        scale = ctx.scale
        normalize_queries = ctx.normalize_queries
        B, T, N, H, K, V, R = ctx.shapes

        grad_q = torch.zeros_like(q)
        grad_w = torch.zeros_like(w)
        grad_mem = torch.zeros_like(m)
        if R == 0:
            return grad_q, grad_mem, None, grad_w, None, None

        go = grad_out.contiguous().float()

        # Kernel A: grad_q (grid B,H,T/BLOCK_T, no atomic)
        # BLOCK_T=128 with num_warps=8, num_stages=1 — sweep winner at training
        # shapes: 12.3ms total for split kernels (5x v2 bwd_qw).
        BLOCK_T_Q = 128
        grid_q = (B, H, triton.cdiv(T, BLOCK_T_Q))
        _ssc_gather_read_bwd_q_grad_kernel_v3[grid_q](
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
            normalize_queries=normalize_queries,
            BLOCK_T=BLOCK_T_Q,
            num_warps=8, num_stages=1,
        )

        # Kernel B: grad_w (grid B,T/BLOCK_T, loops H internally, no atomic)
        # BLOCK_T=64, num_stages=1.
        BLOCK_T_W = 64
        grid_w = (B, triton.cdiv(T, BLOCK_T_W))
        _ssc_gather_read_bwd_w_grad_kernel_v3[grid_w](
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
            normalize_queries=normalize_queries,
            BLOCK_T=BLOCK_T_W,
            num_warps=4, num_stages=1,
        )

        # v3.2 bwd_mem: bf16 TensorCore dot (4.05x v2 at training shapes).
        # BLOCK_T=32, num_warps=8, num_stages=1 — sweep winner.
        BLOCK_T_MEM = 32
        grid_mem = (B, H, N)
        _ssc_gather_read_bwd_mem_kernel_v3_2[grid_mem](
            go, q, idx, w, grad_mem,
            scale, T, N,
            H=H, K=K, V=V, R=R,
            stride_gob=go.stride(0), stride_got=go.stride(1), stride_goh=go.stride(2),
            stride_qb=q.stride(0), stride_qt=q.stride(1), stride_qh=q.stride(2),
            stride_ib=idx.stride(0), stride_it=idx.stride(1),
            stride_wb=w.stride(0), stride_wt=w.stride(1),
            stride_gmb=grad_mem.stride(0), stride_gmn=grad_mem.stride(1), stride_gmh=grad_mem.stride(2),
            normalize_queries=normalize_queries,
            BLOCK_T=BLOCK_T_MEM,
            num_warps=8, num_stages=1,
        )

        return grad_q, grad_mem, None, grad_w.to(w.dtype), None, None


# Alias v3b to v3a with a different default BLOCK_T — same kernel, different tuning.
# v3b will be tuned in benchmarking; for now points to v3a at BLOCK_T=4.
class _SSCGatherReadV3b(_SSCGatherReadV3a):
    pass


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def ssc_gather_read_v3a(queries, memories, indices, weights, *, scale, normalize_queries):
    return _SSCGatherReadV3a.apply(queries, memories, indices, weights, scale, normalize_queries)


def ssc_gather_read_v3b(queries, memories, indices, weights, *, scale, normalize_queries):
    return _SSCGatherReadV3b.apply(queries, memories, indices, weights, scale, normalize_queries)


def ssc_gather_read_v3c(queries, memories, indices, weights, *, scale, normalize_queries):
    return _SSCGatherReadV3c.apply(queries, memories, indices, weights, scale, normalize_queries)


# Default: v3c (the main win). Override via environment variable MC_V3_VARIANT.
import os
_DEFAULT_VARIANT = os.environ.get("MC_V3_VARIANT", "v3c")
ssc_gather_read = {
    "v3a": ssc_gather_read_v3a,
    "v3b": ssc_gather_read_v3b,
    "v3c": ssc_gather_read_v3c,
}[_DEFAULT_VARIANT]


# Backward-compat alias
SSCGatherReadV3 = _SSCGatherReadV3c
