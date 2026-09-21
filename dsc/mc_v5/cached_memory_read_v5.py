"""MC SSC v5 kernels — all-TF32 TensorCore (full precision recovery).

v4 only switched bwd_mem to TF32. PPL still regressed (18.73 vs v2=17.36) — the
remaining bf16 dots in fwd (v3c) + bwd_q (v3) + bwd_w (v3) still leak noise.

v5 = full conversion:
  - fwd (v3c → v5): drop `.to(bf16)` on Q + mem, allow_tf32=True
  - bwd_q (v3 → v5): drop `.to(bf16)` on q_scaled/wg/mem, allow_tf32=True
  - bwd_w (v3 → v5): drop `.to(bf16)` on q_scaled/mem, allow_tf32=True
  - bwd_mem: reuse v4 (already TF32)

Why TF32 vs bf16:
  - bf16: 7 mantissa bits → ~1% relative error per dot
  - TF32: 10 mantissa bits → ~0.1% relative error per dot (4096× more precision)
  - Same Hopper HMMA throughput (989 TFLOPS) for both
  - For long-context retrieval, gradient noise from bf16 accumulates over
    1B+ tokens and corrupts the routing → recovery fails on RULER S-NIAH.

Goal: numerical parity with v2 (PPL ≤ 17.5, RULER S-NIAH-1 4K ≥ 7) AND faster
than v4 (no extra casts → less register pressure).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# v5 fwd kernel — segment-conditional, all TF32 TensorCore
# ---------------------------------------------------------------------------
@triton.jit
def _ssc_gather_read_fwd_kernel_v5(
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
    q_scaled = q_unit * scale  # [BLOCK_T, K] fp32

    out = tl.zeros((BLOCK_T, V), dtype=tl.float32)

    for n in tl.static_range(N):
        mem_ptrs = (MEM_ptr + pid_b * stride_mb + n * stride_mn
                    + pid_h * stride_mh + k_offs[:, None] * V
                    + v_offs[None, :])
        # v5: load as fp32 (no bf16 cast) — TF32 TensorCore path
        mem = tl.load(mem_ptrs, eviction_policy="evict_last").to(tl.float32)

        # v5 KEY: fp32 operands + input_precision="tf32" → TF32 TensorCore
        # (10-bit mantissa vs bf16's 7-bit, same HMMA throughput on Hopper)
        read_all = tl.dot(q_scaled, mem, input_precision="tf32")  # [BLOCK_T, V] fp32

        for r in tl.static_range(R):
            idx_ptrs_r = IDX_ptr + pid_b * stride_ib + t_offs * stride_it + r
            idx_r = tl.load(idx_ptrs_r, mask=t_mask, other=0)
            w_ptrs_r = W_ptr + pid_b * stride_wb + t_offs * stride_wt + r
            w_r = tl.load(w_ptrs_r, mask=t_mask, other=0.0).to(tl.float32)
            match = (idx_r == n) & t_mask
            out += tl.where(match[:, None], w_r[:, None] * read_all, 0.0)

    out_ptrs = (OUT_ptr + pid_b * stride_ob + t_offs[:, None] * stride_ot
                + pid_h * stride_oh + v_offs[None, :])
    tl.store(out_ptrs, out, mask=t_mask[:, None])


# ---------------------------------------------------------------------------
# v5 bwd_q kernel — no atomic, all TF32 TensorCore
# ---------------------------------------------------------------------------
@triton.jit
def _ssc_gather_read_bwd_q_grad_kernel_v5(
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
    """v5 bwd_q: same structure as v3 but all dots are TF32."""
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
    q_scaled = q_unit * scale  # fp32 [BLOCK_T, K]

    go_ptrs = (GO_ptr + pid_b * stride_gob + t_offs[:, None] * stride_got
               + pid_h * stride_goh + v_offs[None, :])
    grad_out = tl.load(go_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

    grad_q_scaled = tl.zeros((BLOCK_T, K), dtype=tl.float32)

    for n in tl.static_range(N):
        mem_ptrs = (MEM_ptr + pid_b * stride_mb + n * stride_mn
                    + pid_h * stride_mh + k_offs[:, None] * V
                    + v_offs[None, :])
        # v5: load as fp32 (no bf16 cast)
        mem = tl.load(mem_ptrs, eviction_policy="evict_last").to(tl.float32)
        mem_T = tl.trans(mem)  # [V, K] fp32

        for r in tl.static_range(R):
            idx_ptrs = IDX_ptr + pid_b * stride_ib + t_offs * stride_it + r
            idx_r = tl.load(idx_ptrs, mask=t_mask, other=0)
            w_ptrs = W_ptr + pid_b * stride_wb + t_offs * stride_wt + r
            w_r = tl.load(w_ptrs, mask=t_mask, other=0.0).to(tl.float32)
            match = (idx_r == n) & t_mask

            # wg = match * w * grad_out — fp32 [BLOCK_T, V]
            wg = (match * w_r)[:, None] * grad_out
            # v5: fp32 operands + input_precision="tf32" → TF32 TensorCore
            grad_q_scaled += tl.dot(wg, mem_T, input_precision="tf32")

    grad_q_unit = grad_q_scaled * scale
    if normalize_queries:
        dot = tl.sum(q_unit * grad_q_unit, axis=1)
        grad_q = (grad_q_unit - dot[:, None] * q_unit) / q_norm[:, None]
    else:
        grad_q = grad_q_unit

    gq_ptrs = (GQ_ptr + pid_b * stride_gqb + t_offs[:, None] * stride_gqt
               + pid_h * stride_gqh + k_offs[None, :])
    tl.store(gq_ptrs, grad_q, mask=t_mask[:, None])


# ---------------------------------------------------------------------------
# v5 bwd_w kernel — no atomic, all TF32 TensorCore
# ---------------------------------------------------------------------------
@triton.jit
def _ssc_gather_read_bwd_w_grad_kernel_v5(
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
    """v5 bwd_w: same structure as v3 but all dots are TF32."""
    pid_b = tl.program_id(0)
    pid_tb = tl.program_id(1)

    t_offs = pid_tb * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T
    k_offs = tl.arange(0, K)
    v_offs = tl.arange(0, V)
    r_offs = tl.arange(0, R)

    grad_w_acc = tl.zeros((BLOCK_T, R), dtype=tl.float32)

    for pid_h in range(H):
        q_ptrs = (Q_ptr + pid_b * stride_qb + t_offs[:, None] * stride_qt
                  + pid_h * stride_qh + k_offs[None, :])
        q = tl.load(q_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)
        if normalize_queries:
            q_norm = tl.sqrt(tl.sum(q * q, axis=1) + 1e-12)
            q_unit = q / q_norm[:, None]
        else:
            q_unit = q
        q_scaled = q_unit * scale  # fp32

        go_ptrs = (GO_ptr + pid_b * stride_gob + t_offs[:, None] * stride_got
                   + pid_h * stride_goh + v_offs[None, :])
        grad_out = tl.load(go_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

        for n in tl.static_range(N):
            mem_ptrs = (MEM_ptr + pid_b * stride_mb + n * stride_mn
                        + pid_h * stride_mh + k_offs[:, None] * V
                        + v_offs[None, :])
            # v5: load as fp32 (no bf16 cast)
            mem = tl.load(mem_ptrs, eviction_policy="evict_last").to(tl.float32)

            # v5: fp32 operands + input_precision="tf32" → TF32 TensorCore
            read_all = tl.dot(q_scaled, mem, input_precision="tf32")  # [BLOCK_T, V]
            gw_base = tl.sum(grad_out * read_all, axis=1)  # [BLOCK_T]

            for r in tl.static_range(R):
                idx_ptrs = IDX_ptr + pid_b * stride_ib + t_offs * stride_it + r
                idx_r = tl.load(idx_ptrs, mask=t_mask, other=0)
                match = (idx_r == n) & t_mask
                gw_r_local = tl.where(match, gw_base, 0.0)
                r_mask = (r_offs == r)
                grad_w_acc += tl.where(r_mask[None, :], gw_r_local[:, None], 0.0)

    gw_ptrs = (GW_ptr + pid_b * stride_gwb + t_offs[:, None] * stride_gwt
               + r_offs[None, :])
    tl.store(gw_ptrs, grad_w_acc, mask=t_mask[:, None])


# ---------------------------------------------------------------------------
# v5 bwd_mem kernel — same as v4 (already TF32)
# ---------------------------------------------------------------------------
from dsc.mc_v4.cached_memory_read_v4 import _ssc_gather_read_bwd_mem_kernel_v4 as _ssc_gather_read_bwd_mem_kernel_v5


# ---------------------------------------------------------------------------
# Autograd glue
# ---------------------------------------------------------------------------
class _SSCGatherReadV5(torch.autograd.Function):

    @staticmethod
    def forward(ctx, queries, memories, indices, weights, scale, normalize_queries):
        B, T, H, K = queries.shape
        N = memories.shape[1]
        R = indices.shape[2]
        V = memories.shape[-1]
        out = torch.empty(B, T, H, V, dtype=torch.float32, device=queries.device)

        # v5: TF32 path requires BLOCK_T=64 (vs v3c's 32). At BLOCK_T=32 the fp32
        # operands double register pressure and TC tiles under-utilize → 5x slower.
        # BLOCK_T=64 amortizes the tile overhead → 3x slower than bf16 (acceptable
        # for correctness-first version). BLOCK_T=128 spills again → 5x slower.
        BLOCK_T = 64 if T >= 512 else 32
        grid = (B, H, triton.cdiv(T, BLOCK_T))
        _ssc_gather_read_fwd_kernel_v5[grid](
            queries, memories, indices, weights, out,
            scale, T, N,
            H=H, K=K, V=V, R=R,
            stride_qb=queries.stride(0), stride_qt=queries.stride(1), stride_qh=queries.stride(2),
            stride_mb=memories.stride(0), stride_mn=memories.stride(1), stride_mh=memories.stride(2),
            stride_ib=indices.stride(0), stride_it=indices.stride(1),
            stride_wb=weights.stride(0), stride_wt=weights.stride(1),
            stride_ob=out.stride(0), stride_ot=out.stride(1), stride_oh=out.stride(2),
            normalize_queries=normalize_queries,
            BLOCK_T=BLOCK_T,
            num_warps=8, num_stages=2,
        )

        ctx.save_for_backward(queries, memories, indices, weights)
        ctx.scale = scale
        ctx.normalize_queries = normalize_queries
        ctx.shapes = (B, T, N, H, K, V, R)
        return out

    @staticmethod
    def backward(ctx, grad_out):
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

        # v5 bwd_q: TF32 TensorCore
        BLOCK_T_Q = 128
        grid_q = (B, H, triton.cdiv(T, BLOCK_T_Q))
        _ssc_gather_read_bwd_q_grad_kernel_v5[grid_q](
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

        # v5 bwd_w: TF32 TensorCore
        BLOCK_T_W = 64
        grid_w = (B, triton.cdiv(T, BLOCK_T_W))
        _ssc_gather_read_bwd_w_grad_kernel_v5[grid_w](
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

        # v5 bwd_mem: reuse v4 (already TF32)
        BLOCK_T_MEM = 32
        grid_mem = (B, H, N)
        _ssc_gather_read_bwd_mem_kernel_v5[grid_mem](
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


def ssc_gather_read_v5(queries, memories, indices, weights, *, scale, normalize_queries):
    return _SSCGatherReadV5.apply(queries, memories, indices, weights, scale, normalize_queries)


sscgatherread_v5 = ssc_gather_read_v5
SSCGatherReadV5 = _SSCGatherReadV5
