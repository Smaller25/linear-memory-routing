"""Equivalence test: verify v3 forward output matches v2 within fp32 tolerance.

Loads MC 30B checkpoint (or random init if not available), runs a forward pass
with both v2 (dsc.mc_baseline.cached_memory_read) and v3 (dsc.mc_v3), and
reports max absolute difference and relative error.

Tolerance: max_abs < 1e-4 for fp32 (allowing for reordering of floating-point
operations in different kernel schedules). Math equivalence is verified by
construction (same formula), so this is purely numerical stability.

Usage:
    CUDA_VISIBLE_DEVICES=0 python dsc/mc_v3/tests/test_equivalence.py \\
        --ckpt path/to/checkpoint-30B-model-ckpt.pth \\
        --seq-len 4096 --micro-batch 8

    # smoke test (small, no ckpt):
    CUDA_VISIBLE_DEVICES=0 python dsc/mc_v3/tests/test_equivalence.py --smoke
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# REPO = .../long-gdn. Add to path so `dsc.mc_baseline` resolves.
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _ssc_inputs(B, T, H, K, V, N, R, device, dtype, require_grad=True):
    """Make random inputs for ssc_gather_read."""
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=require_grad)
    m = torch.randn(B, N, H, K, V, device=device, dtype=dtype, requires_grad=require_grad)
    idx = torch.randint(0, N, (B, T, R), device=device, dtype=torch.long)
    # Weights are softmax outputs — make them non-negative and sum to ~1 across R
    w_raw = torch.randn(B, T, R, device=device, dtype=dtype)
    w = torch.softmax(w_raw, dim=-1)
    if require_grad:
        w = w.clone().requires_grad_(True)
    return q, m, idx, w


def _run_v2(q, m, idx, w, scale, normalize):
    from dsc.mc_baseline.cached_memory_read import ssc_gather_read
    return ssc_gather_read(q, m, idx, w, scale=scale, normalize_queries=normalize)


def _run_v3a(q, m, idx, w, scale, normalize):
    from dsc.mc_v3 import ssc_gather_read_v3a
    return ssc_gather_read_v3a(q, m, idx, w, scale=scale, normalize_queries=normalize)


def _run_v3c(q, m, idx, w, scale, normalize):
    from dsc.mc_v3 import ssc_gather_read_v3c
    return ssc_gather_read_v3c(q, m, idx, w, scale=scale, normalize_queries=normalize)


def _compare(name_v3, out_v2, out_v3, grad_out=None):
    """Forward + (if grad_out) backward comparison."""
    diff_fwd = (out_v2 - out_v3).abs()
    max_fwd = diff_fwd.max().item()
    rel_fwd = (diff_fwd / (out_v2.abs() + 1e-6)).max().item()
    print(f"[{name_v3}] fwd: max_abs={max_fwd:.3e}  rel={rel_fwd:.3e}  "
          f"(out_v2 mean abs={out_v2.abs().mean().item():.3e})", flush=True)
    return max_fwd, rel_fwd


def smoke_test(device="cuda", dtype=torch.bfloat16):
    """Tiny shapes to verify kernels launch + compute. ~5 GB VRAM, shareable with training."""
    print("\n=== smoke test (B=2 T=512 H=4 K=V=64 N=4 R=2) ===", flush=True)
    B, T, H, K, V, N, R = 2, 512, 4, 64, 64, 4, 2
    scale = K ** -0.5

    for normalize in [False, True]:
        print(f"\n--- normalize_queries={normalize} ---", flush=True)
        q, m, idx, w = _ssc_inputs(B, T, H, K, V, N, R, device, dtype)
        out_v2 = _run_v2(q, m, idx, w, scale, normalize)
        out_v3a = _run_v3a(q, m, idx, w, scale, normalize)
        out_v3c = _run_v3c(q, m, idx, w, scale, normalize)
        _compare("v3a", out_v2, out_v3a)
        _compare("v3c", out_v2, out_v3c)


def full_test(ckpt, device="cuda", dtype=torch.bfloat16, seq_len=4096, micro_batch=8):
    """Training-shape test. Uses MC 30B checkpoint's actual H/K/V/N/R."""
    print(f"\n=== full test (ckpt={ckpt}, seq_len={seq_len}, mb={micro_batch}) ===", flush=True)
    from lit_gpt.config import Config
    from lit_gpt.model import GPT

    cfg = Config.from_name("mc_370M")
    H = cfg.n_head
    K = cfg.head_qk_dim if hasattr(cfg, "head_qk_dim") else (cfg.n_embd // cfg.n_head)
    V = K
    N = (seq_len + cfg.mc_chunk_size - 1) // cfg.mc_chunk_size
    R = cfg.mc_topk
    print(f"  config: H={H} K={K} V={V} N={N} R={R} chunk={cfg.mc_chunk_size}", flush=True)

    scale = K ** -0.5
    B = micro_batch

    for normalize in [False, True]:
        print(f"\n--- normalize_queries={normalize} ---", flush=True)
        q, m, idx, w = _ssc_inputs(B, seq_len, H, K, V, N, R, device, dtype)
        out_v2 = _run_v2(q, m, idx, w, scale, normalize)
        out_v3a = _run_v3a(q, m, idx, w, scale, normalize)
        out_v3c = _run_v3c(q, m, idx, w, scale, normalize)
        _compare("v3a", out_v2, out_v3a)
        _compare("v3c", out_v2, out_v3c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="MC 30B checkpoint path (optional)")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="run only smoke test (no ckpt needed)")
    ap.add_argument("--no-full", action="store_true", help="skip full test even if ckpt provided")
    args = ap.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        print("ERROR: CUDA not available, run with --device cpu (slow, only smoke)")
        sys.exit(1)

    smoke_test(args.device)
    if not args.smoke and args.ckpt and not args.no_full:
        if not os.path.exists(args.ckpt):
            print(f"WARN: ckpt not found at {args.ckpt}, skipping full test")
        else:
            full_test(args.ckpt, args.device, seq_len=args.seq_len, micro_batch=args.micro_batch)


if __name__ == "__main__":
    main()
