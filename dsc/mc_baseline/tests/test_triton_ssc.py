"""Numerical correctness tests for the Triton-fused SSC gather+read kernel.

Compares against a pure-PyTorch reference at small and training-like shapes.
The Triton kernel operates in float32 internally, so we compare with float32
tolerance.  Verifies forward output and all three gradients (grad_q,
grad_memories, grad_weights), both with and without query L2-norm.
"""
from __future__ import annotations

import torch

from dsc.mc_baseline.cached_memory_read import ssc_gather_read


def py_ref(queries, memories, indices, weights, *, scale, normalize_queries):
    """Pure-PyTorch reference implementing the same math as the Triton kernel."""
    B, T, H, K = queries.shape
    N = memories.shape[1]
    R = indices.shape[2]
    V = memories.shape[-1]
    q_f = queries.float()
    if normalize_queries:
        q_unit = torch.nn.functional.normalize(q_f, p=2, dim=-1)
    else:
        q_unit = q_f
    q_scaled = q_unit * scale
    batch_idx = torch.arange(B, device=queries.device)[:, None, None]
    sel = memories.float()[batch_idx.expand(B, T, R), indices]  # [B,T,R,H,K,V]
    q_exp = q_scaled.unsqueeze(2).expand(B, T, R, H, K).reshape(B * T * R * H, 1, K)
    m = sel.reshape(B * T * R * H, K, V)
    reads = torch.bmm(q_exp, m).reshape(B, T, R, H, V)
    out = torch.einsum("btr,btrhv->bthv", weights.float(), reads)
    return out.to(weights.dtype)


def case_fwd_bwd(B, T, N, H, K, V, R, normalize, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    # Use float32 throughout (matches kernel internals).
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float32, requires_grad=True)
    mem = torch.randn(B, N, H, K, V, device=device, dtype=torch.float32, requires_grad=True)
    idx = torch.randint(0, N, (B, T, R), device=device, dtype=torch.int64)
    w = torch.randn(B, T, R, device=device, dtype=torch.float32, requires_grad=True)
    scale = 0.1

    # Reference (autograd in float32)
    out_ref = py_ref(q, mem, idx, w, scale=scale, normalize_queries=normalize)
    g = torch.randn_like(out_ref)
    gq_ref, gmem_ref, gw_ref = torch.autograd.grad(out_ref, [q, mem, w], grad_outputs=g)

    # Triton (clone inputs so leaf graphs don't conflict)
    q2 = q.detach().clone().requires_grad_(True)
    mem2 = mem.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    out_tri = ssc_gather_read(q2, mem2, idx, w2, scale=scale, normalize_queries=normalize)
    gq_tri, gmem_tri, gw_tri = torch.autograd.grad(out_tri, [q2, mem2, w2], grad_outputs=g)

    out_diff = (out_ref - out_tri).abs().max().item()
    gq_diff = (gq_ref - gq_tri).abs().max().item()
    gmem_diff = (gmem_ref - gmem_tri).abs().max().item()
    gw_diff = (gw_ref - gw_tri).abs().max().item()
    # Relative error scale: reference values can be O(1)-O(10) depending on K*V
    ref_scale = max(out_ref.abs().max().item(), gq_ref.abs().max().item(),
                    gmem_ref.abs().max().item(), gw_ref.abs().max().item(), 1e-6)
    return out_diff, gq_diff, gmem_diff, gw_diff, ref_scale


def main():
    print("=" * 80)
    print("Triton SSC gather+read — numerical correctness (vs PyTorch reference, float32)")
    print("=" * 80)
    cases = [
        # (B, T,   N, H, K,  V,  R, normalize, label)
        (1, 64,  4, 2, 16, 16, 1, True,  "tiny  R=1 normalize=True"),
        (1, 64,  4, 2, 16, 16, 1, False, "tiny  R=1 normalize=False"),
        (2, 128, 8, 4, 32, 32, 2, True,  "small R=2 normalize=True"),
        (2, 128, 8, 4, 32, 32, 2, False, "small R=2 normalize=False"),
        (2, 256, 8, 8, 64, 64, 3, True,  "med  R=3 normalize=True"),
        (2, 256, 8, 8, 64, 64, 3, False, "med  R=3 normalize=False"),
        (1, 512, 16, 16, 128, 128, 2, True, "training-shape (B=1) normalize=True"),
        (1, 512, 16, 16, 128, 128, 2, False, "training-shape (B=1) normalize=False"),
    ]
    all_pass = True
    for B, T, N, H, K, V, R, norm, label in cases:
        out_d, gq_d, gm_d, gw_d, ref_scale = case_fwd_bwd(B, T, N, H, K, V, R, norm)
        # Use relative tolerance: 1e-3 of reference scale.
        tol = max(1e-5, 1e-3 * ref_scale)
        worst = max(out_d, gq_d, gm_d, gw_d)
        status = "PASS" if worst < tol else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {label:46s}  out={out_d:.2e}  gq={gq_d:.2e}  "
              f"gmem={gm_d:.2e}  gw={gw_d:.2e}  ref_scale={ref_scale:.2e}  tol={tol:.1e}")
    print("=" * 80)
    print("ALL PASS" if all_pass else "SOME FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
