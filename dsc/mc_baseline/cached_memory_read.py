"""Triton-fused gather + matmul + weighted sum for SSC Equation (17).

Replaces the PyTorch ``mems[batch_idx, indices]`` gather path that materialises
a ``[B, T, R, H, K, V]`` tensor (~64 GB at B=8 T=4096 R=2 H=16 K=V=128) per
layer.  The kernel reads selected segment memories on the fly inside the matmul
loop, so peak SRAM is one ``[BLOCK_T, K, V]`` tile per program.

Forward  (per (b, h, t_block) program):
    out[b, t, h, v] = sum_r w[b, t, r] * sum_k (q_norm[b, t, h, k] * scale)
                                  * mem[b, idx[b, t, r], h, k, v]

Backward (one kernel for grad_q and grad_w, one for grad_mem atomic scatter):
    grad_q_scaled[b, t, h, k] = sum_{r,v} w[b, t, r] * grad_out[b, t, h, v]
                                              * mem[b, idx[b, t, r], h, k, v]
    grad_w[b, t, r]           = sum_{h,v} grad_out[b, t, h, v]
                                          * sum_k q_scaled[b, t, h, k]
                                                  * mem[b, idx[b, t, r], h, k, v]
    grad_mem[b, n, h, k, v]   = sum_{(t,r): idx[b,t,r]=n} grad_out[b, t, h, v]
                                              * w[b, t, r] * q_scaled[b, t, h, k]

Memory traffic per layer (B=8 T=4096 N=16 H=16 K=V=128 R=2):
  Forward: ~68 GB read (mem) + 268 MB write (out) -> ~17 ms on H200 4 TB/s.
  16 layers x 3 (fwd + checkpoint recompute + bwd) -> ~0.8 s SCC cost per iter,
  vs ~4 s gather cost in the PyTorch path.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------
@triton.jit
def _ssc_gather_read_fwd_kernel(
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

    # Q[b, t, h, :] - [BLOCK_T, K]
    q_ptrs = (Q_ptr + pid_b * stride_qb + t_offs[:, None] * stride_qt
              + pid_h * stride_qh + k_offs[None, :])
    q = tl.load(q_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

    if normalize_queries:
        q_norm = tl.sqrt(tl.sum(q * q, axis=1) + 1e-12)  # [BLOCK_T]
        q_unit = q / q_norm[:, None]
    else:
        q_unit = q
    q_scaled = q_unit * scale  # [BLOCK_T, K]

    out = tl.zeros((BLOCK_T, V), dtype=tl.float32)

    for r in range(R):
        # IDX[b, t, r] - [BLOCK_T]
        idx_ptrs = IDX_ptr + pid_b * stride_ib + t_offs * stride_it + r
        idx = tl.load(idx_ptrs, mask=t_mask, other=0)
        # W[b, t, r] - [BLOCK_T]
        w_ptrs = W_ptr + pid_b * stride_wb + t_offs * stride_wt + r
        w = tl.load(w_ptrs, mask=t_mask, other=0.0).to(tl.float32)

        # MEM[b, idx[t], h, k, v] - [BLOCK_T, K, V]
        mem_ptrs = (MEM_ptr + pid_b * stride_mb + idx[:, None, None] * stride_mn
                    + pid_h * stride_mh + k_offs[None, :, None] * V
                    + v_offs[None, None, :])
        mem = tl.load(mem_ptrs, mask=t_mask[:, None, None], other=0.0).to(tl.float32)

        # read[t, v] = sum_k q_scaled[t, k] * mem[t, k, v]
        read = tl.sum(q_scaled[:, :, None] * mem, axis=1)  # [BLOCK_T, V]
        out += w[:, None] * read

    # OUT[b, t, h, :] - [BLOCK_T, V]
    out_ptrs = (OUT_ptr + pid_b * stride_ob + t_offs[:, None] * stride_ot
                + pid_h * stride_oh + v_offs[None, :])
    tl.store(out_ptrs, out, mask=t_mask[:, None])


# ---------------------------------------------------------------------------
# Backward kernel for grad_q and grad_w (no scatter)
# ---------------------------------------------------------------------------
# Per (b, h, t_block) program computes grad_q for K dim and contributes to
# grad_w via atomic_add across H.  Memory is processed in BLOCK_K K-tiles to
# avoid materialising a [BLOCK_T, K, V] intermediate which would spill to L2.
# With BLOCK_T=8, BLOCK_K=16, BLOCK_V=128, the per-tile working set is
# 8*16*128 = 16 K floats = 64 KiB, fitting in registers comfortably.
# ---------------------------------------------------------------------------
@triton.jit
def _ssc_gather_read_bwd_qw_kernel(
    GO_ptr, Q_ptr, MEM_ptr, IDX_ptr, W_ptr,
    GQ_ptr, GW_ptr,
    scale,
    T, N,
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

    # Q[b, t, h, :]
    q_ptrs = (Q_ptr + pid_b * stride_qb + t_offs[:, None] * stride_qt
              + pid_h * stride_qh + k_offs[None, :])
    q = tl.load(q_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)
    if normalize_queries:
        q_norm = tl.sqrt(tl.sum(q * q, axis=1) + 1e-12)  # [BLOCK_T]
        q_unit = q / q_norm[:, None]
    else:
        q_unit = q
        q_norm = q_unit  # placeholder; unused when normalize_queries=False
    q_scaled = q_unit * scale  # [BLOCK_T, K]

    # grad_out[b, t, h, :] - [BLOCK_T, V]
    go_ptrs = (GO_ptr + pid_b * stride_gob + t_offs[:, None] * stride_got
               + pid_h * stride_goh + v_offs[None, :])
    grad_out = tl.load(go_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

    grad_q_scaled = tl.zeros((BLOCK_T, K), dtype=tl.float32)

    for r in range(R):
        idx_ptrs = IDX_ptr + pid_b * stride_ib + t_offs * stride_it + r
        idx = tl.load(idx_ptrs, mask=t_mask, other=0)
        w_ptrs = W_ptr + pid_b * stride_wb + t_offs * stride_wt + r
        w = tl.load(w_ptrs, mask=t_mask, other=0.0).to(tl.float32)

        mem_ptrs = (MEM_ptr + pid_b * stride_mb + idx[:, None, None] * stride_mn
                    + pid_h * stride_mh + k_offs[None, :, None] * V
                    + v_offs[None, None, :])
        mem = tl.load(mem_ptrs, mask=t_mask[:, None, None], other=0.0).to(tl.float32)

        # read[t, v] = sum_k q_scaled[t, k] * mem[t, k, v]
        read = tl.sum(q_scaled[:, :, None] * mem, axis=1)  # [BLOCK_T, V]

        # grad_w[b, t, r] = sum_{h,v} grad_out[b,t,h,v] * read[b,t,r,h,v].
        # Each (b,t,h) program contributes its h-slice; atomic_add accumulates
        # across the H programs that share the same (b,t,r) target.
        grad_w_r = tl.sum(grad_out * read, axis=1)  # [BLOCK_T]
        gw_ptrs = GW_ptr + pid_b * stride_gwb + t_offs * stride_gwt + r
        tl.atomic_add(gw_ptrs, grad_w_r, mask=t_mask)

        # grad_reads[t, v] = w[t] * grad_out[t, v]
        # grad_q_scaled[t, k] += sum_v grad_reads[t, v] * mem[t, k, v]
        grad_q_scaled += tl.sum((w[:, None] * grad_out)[:, None, :] * mem, axis=2)

    # Chain rule: q_scaled = q_unit * scale
    grad_q_unit = grad_q_scaled * scale
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
# Backward kernel for grad_mem (atomic scatter) — LEGACY PATH
# ---------------------------------------------------------------------------
@triton.jit
def _ssc_gather_read_bwd_mem_kernel(
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
    pid_tb = tl.program_id(2)

    t_offs = pid_tb * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T
    k_offs = tl.arange(0, K)
    v_offs = tl.arange(0, V)

    q_ptrs = (Q_ptr + pid_b * stride_qb + t_offs[:, None] * stride_qt
              + pid_h * stride_qh + k_offs[None, :])
    q = tl.load(q_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)
    if normalize_queries:
        q_norm = tl.sqrt(tl.sum(q * q, axis=1) + 1e-12)  # [BLOCK_T]
        q_unit = q / q_norm[:, None]
    else:
        q_unit = q
    q_scaled = q_unit * scale  # [BLOCK_T, K]

    go_ptrs = (GO_ptr + pid_b * stride_gob + t_offs[:, None] * stride_got
               + pid_h * stride_goh + v_offs[None, :])
    grad_out = tl.load(go_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

    for r in range(R):
        idx_ptrs = IDX_ptr + pid_b * stride_ib + t_offs * stride_it + r
        idx = tl.load(idx_ptrs, mask=t_mask, other=0)
        w_ptrs = W_ptr + pid_b * stride_wb + t_offs * stride_wt + r
        w = tl.load(w_ptrs, mask=t_mask, other=0.0).to(tl.float32)

        # grad_sel[t, k, v] = w[t] * grad_out[t, v] * q_scaled[t, k]
        grad_sel = (w[:, None, None] * grad_out[:, None, :] * q_scaled[:, :, None])

        # Atomic add into GMEM[b, idx[t], h, k, v]
        target_ptrs = (GMEM_ptr + pid_b * stride_gmb + idx[:, None, None] * stride_gmn
                       + pid_h * stride_gmh + k_offs[None, :, None] * V
                       + v_offs[None, None, :])
        tl.atomic_add(target_ptrs, grad_sel, mask=t_mask[:, None, None])


# ---------------------------------------------------------------------------
# Backward kernel for grad_mem (target-parallel, no atomics)
# ---------------------------------------------------------------------------
# Each program owns ONE (b, n, h) target tile and scans all (t, r) source
# tuples for matches.  Since each target has exactly one writer, no atomics
# are needed.  At B=8 T=4096 N=16 H=16 R=2: 17B atomic_adds/layer -> 0 writes,
# replaced by 2048 programs each scanning 8K tuples and emitting one [K, V]
# tile.  No-atomic cost is dominated by Q/GO re-reads (16x duplicate per
# (b,h) across n's), ~8 GB traffic per layer -> ~2 ms on H200 vs 349 ms atomic.
# ---------------------------------------------------------------------------
@triton.jit
def _ssc_gather_read_bwd_mem_noatomic_kernel(
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

    # Accumulator for grad_mem[b, pid_n, h, :, :] - [K, V] in float32
    acc = tl.zeros((K, V), dtype=tl.float32)

    for tb in range(0, T, BLOCK_T):
        t_offs = tb + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        # Load grad_out[b, t_offs, h, :] - [BLOCK_T, V]
        go_ptrs = (GO_ptr + pid_b * stride_gob + t_offs[:, None] * stride_got
                   + pid_h * stride_goh + v_offs[None, :])
        grad_out = tl.load(go_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

        # Load Q[b, t_offs, h, :] - [BLOCK_T, K] (full K for L2-norm)
        q_ptrs = (Q_ptr + pid_b * stride_qb + t_offs[:, None] * stride_qt
                  + pid_h * stride_qh + k_offs[None, :])
        q = tl.load(q_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)
        if normalize_queries:
            q_norm = tl.sqrt(tl.sum(q * q, axis=1) + 1e-12)  # [BLOCK_T]
            q_unit = q / q_norm[:, None]
        else:
            q_unit = q
        q_scaled = q_unit * scale  # [BLOCK_T, K]

        # Weighted grad_out accumulated over matching r's
        # match_wg[t, v] = sum over r where idx[b,t,r]=pid_n of w[b,t,r] * grad_out[b,t,h,v]
        match_wg = tl.zeros((BLOCK_T, V), dtype=tl.float32)
        for r in tl.static_range(R):
            idx_ptrs = IDX_ptr + pid_b * stride_ib + t_offs * stride_it + r
            idx = tl.load(idx_ptrs, mask=t_mask, other=0)
            w_ptrs = W_ptr + pid_b * stride_wb + t_offs * stride_wt + r
            w = tl.load(w_ptrs, mask=t_mask, other=0.0).to(tl.float32)
            match = (idx == pid_n) & t_mask  # [BLOCK_T]
            match_wg += tl.where(match[:, None], w[:, None] * grad_out, 0.0)

        # acc[k, v] += sum_t q_scaled[t, k] * match_wg[t, v]
        # = q_scaled.T @ match_wg : [K, BLOCK_T] @ [BLOCK_T, V] -> [K, V]
        acc += tl.dot(tl.trans(q_scaled), match_wg, allow_tf32=False)

    # Store grad_mem[b, pid_n, h, :, :] - [K, V]
    gm_ptrs = (GMEM_ptr + pid_b * stride_gmb + pid_n * stride_gmn
               + pid_h * stride_gmh + k_offs[:, None] * V + v_offs[None, :])
    tl.store(gm_ptrs, acc)


# ---------------------------------------------------------------------------
# autograd.Function wrapper
# ---------------------------------------------------------------------------
def _pick_block_t(T: int, K: int, V: int) -> int:
    """Pick power-of-2 BLOCK_T balancing SRAM footprint vs launch overhead.

    Goal: keep the per-program [BLOCK_T, K, V] mem tile small enough to fit
    H200 SRAM (L1 cache) — empirically BLOCK_T=4 for K=V=128 is the sweet
    spot; larger BLOCK_T causes register spilling that explodes compile time
    and runtime.  For smaller K, V we scale up.
    """
    cap_numel = 1 << 16  # 64 KiB float32 tile
    max_block = max(1, cap_numel // (K * V))
    block = 1
    while block * 2 <= min(64, max_block, T):
        block *= 2
    return block


# ---------------------------------------------------------------------------
# autograd.Function wrapper
# ---------------------------------------------------------------------------


class _SSCGatherRead(torch.autograd.Function):
    """Fused gather + matmul + weighted sum with on-the-fly memory reads.

    Inputs (any dtype, internally cast to float32 for numerical stability):
        queries:  [B, T, H, K]
        memories: [B, N, H, K, V]  (float32 from chunk_gdn2 final_state)
        indices:  [B, T, R]        (int64)
        weights:  [B, T, R]
    Output:
        out:      [B, T, H, V]     (same dtype as weights)
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

        BLOCK_T = _pick_block_t(T, K, V)
        grid = (B, H, triton.cdiv(T, BLOCK_T))

        # Skip kernel entirely when there are no cached routes (R == 0).
        if R > 0:
            _ssc_gather_read_fwd_kernel[grid](
                q, m, idx, w, out,
                scale,
                T, N,
                H=H, K=K, V=V, R=R,
                stride_qb=q.stride(0), stride_qt=q.stride(1), stride_qh=q.stride(2),
                stride_mb=m.stride(0), stride_mn=m.stride(1), stride_mh=m.stride(2),
                stride_ib=idx.stride(0), stride_it=idx.stride(1),
                stride_wb=w.stride(0), stride_wt=w.stride(1),
                stride_ob=out.stride(0), stride_ot=out.stride(1), stride_oh=out.stride(2),
                normalize_queries=normalize_queries,
                BLOCK_T=BLOCK_T,
                num_warps=4,
                num_stages=2,
            )

        ctx.save_for_backward(q, m, idx, w)
        ctx.scale = scale
        ctx.normalize_queries = normalize_queries
        ctx.shapes = (B, T, N, H, K, V, R)
        return out.to(weights.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        q, m, idx, w = ctx.saved_tensors
        scale = ctx.scale
        normalize_queries = ctx.normalize_queries
        B, T, N, H, K, V, R = ctx.shapes

        grad_q = torch.zeros_like(q)
        # float32 accumulator: bwd_qw atomic-adds the H per-head contributions
        # into the same (b, t, r) slot, and ``w`` is bf16 during training, so a
        # bf16 buffer would round on every one of those H additions.  Cast back
        # to ``w.dtype`` only on return.  Cost is negligible: grad_w is [B,T,R]
        # (~256 KB at B=8 T=4096 R=2) against the [K,V] memory tile each of the
        # B*H*T programs already loads.
        grad_w = torch.zeros_like(w, dtype=torch.float32)
        grad_mem = torch.zeros_like(m)
        if R == 0:
            return grad_q, grad_mem, None, grad_w.to(w.dtype), None, None

        go = grad_out.contiguous().float()
        BLOCK_T = _pick_block_t(T, K, V)
        grid = (B, H, triton.cdiv(T, BLOCK_T))

        # bwd_qw: BLOCK_T=1 with 4 warps is 12x faster than BLOCK_T=4 because
        # the [BLOCK_T, K, V]=[1, 128, 128] = 64 KiB intermediate fits in
        # registers; at BLOCK_T=4 (256 KiB) it spills to L2.
        BLOCK_T_QW = 1
        grid_qw = (B, H, triton.cdiv(T, BLOCK_T_QW))
        _ssc_gather_read_bwd_qw_kernel[grid_qw](
            go, q, m, idx, w,
            grad_q, grad_w,
            scale,
            T, N,
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
            num_warps=4,
            num_stages=2,
        )

        # No-atomic kernel: parallelize over (b, h, n) — each program owns one
        # target tile.  BLOCK_T_MEM=32 with 8 warps + 3 stages is the empirical
        # sweet spot on H200 (6.6 ms/call at training shapes vs 205 ms atomic).
        BLOCK_T_MEM = 32
        grid_mem = (B, H, N)
        _ssc_gather_read_bwd_mem_noatomic_kernel[grid_mem](
            go, q, idx, w, grad_mem,
            scale,
            T, N,
            H=H, K=K, V=V, R=R,
            stride_gob=go.stride(0), stride_got=go.stride(1), stride_goh=go.stride(2),
            stride_qb=q.stride(0), stride_qt=q.stride(1), stride_qh=q.stride(2),
            stride_ib=idx.stride(0), stride_it=idx.stride(1),
            stride_wb=w.stride(0), stride_wt=w.stride(1),
            stride_gmb=grad_mem.stride(0), stride_gmn=grad_mem.stride(1), stride_gmh=grad_mem.stride(2),
            normalize_queries=normalize_queries,
            BLOCK_T=BLOCK_T_MEM,
            num_warps=8,
            num_stages=3,
        )

        return grad_q, grad_mem, None, grad_w.to(w.dtype), None, None


def ssc_gather_read(queries, memories, indices, weights, *, scale, normalize_queries):
    """Public entry point — see :class:`_SSCGatherRead` for shapes.

    Falls back to a pure-PyTorch implementation when K or V are not powers of 2
    (Triton requires power-of-2 tile sizes for ``tl.arange``), or when the
    tensors are not on CUDA.  Training shapes (K=V=128 on CUDA) take the fast
    Triton path; odd shapes / CPU tensors used only by unit tests fall back.
    """
    K = queries.shape[-1]
    V = memories.shape[-1]
    if (queries.is_cuda and (K & (K - 1) == 0) and (V & (V - 1) == 0)
            and K > 0 and V > 0):
        return _SSCGatherRead.apply(queries, memories, indices, weights, scale, normalize_queries)
    return _ssc_gather_read_torch(queries, memories, indices, weights, scale, normalize_queries)


def _ssc_gather_read_torch(queries, memories, indices, weights, scale, normalize_queries):
    """Pure-PyTorch fallback (used when K or V are not powers of 2)."""
    from torch.utils.checkpoint import checkpoint as _checkpoint

    def _gather_and_read(qs, mems, idxs, rws):
        B, T, H, K = qs.shape
        R = idxs.shape[2]
        V = mems.shape[-1]
        batch_idx = torch.arange(B, device=qs.device)[:, None, None]
        sel = mems[batch_idx.expand(B, T, R), idxs]
        q_f = qs.float()
        if normalize_queries:
            q_unit = torch.nn.functional.normalize(q_f, p=2, dim=-1)
        else:
            q_unit = q_f
        q_scaled = q_unit * scale
        q_exp = q_scaled.unsqueeze(2).expand(B, T, R, H, K).reshape(B * T * R * H, 1, K)
        m = sel.reshape(B * T * R * H, K, V)
        reads = torch.bmm(q_exp, m).reshape(B, T, R, H, V)
        return torch.einsum("btr,btrhv->bthv", rws.float(), reads)

    return _checkpoint(_gather_and_read, queries, memories, indices, weights, use_reentrant=False)
