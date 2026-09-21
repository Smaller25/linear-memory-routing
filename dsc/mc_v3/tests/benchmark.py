"""Benchmark v2 vs v3 forward + backward at training shapes.

Reports ms/iter (mean ± std over N iters after warmup), speedup ratio, and
verifies v3 output matches v2 within tolerance.

Usage:
    CUDA_VISIBLE_DEVICES=0 python dsc/mc_v3/tests/benchmark.py \\
        --seq-len 4096 --micro-batch 8 --n-iter 20 --n-warmup 5

    # smaller shapes (shares GPU with training):
    CUDA_VISIBLE_DEVICES=0 python dsc/mc_v3/tests/benchmark.py --seq-len 1024 --micro-batch 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# REPO = .../long-gdn. Add to path so `dsc.mc_baseline` and `dsc.mc_v3` resolve.
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _make_inputs(B, T, H, K, V, N, R, device, dtype):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    m = torch.randn(B, N, H, K, V, device=device, dtype=dtype, requires_grad=True)
    idx = torch.randint(0, N, (B, T, R), device=device, dtype=torch.long)
    w_raw = torch.randn(B, T, R, device=device, dtype=dtype)
    w = torch.softmax(w_raw, dim=-1).clone().requires_grad_(True)
    return q, m, idx, w


def _bench_variant(name, fn, q, m, idx, w, scale, normalize, n_warmup, n_iter):
    """Time fwd+bwd over n_iter after n_warmup. Returns ms stats + output sample."""
    # warmup
    for _ in range(n_warmup):
        q.grad = None; m.grad = None; w.grad = None
        out = fn(q, m, idx, w, scale=scale, normalize_queries=normalize)
        loss = out.float().sum()
        loss.backward()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iter):
        q.grad = None; m.grad = None; w.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(q, m, idx, w, scale=scale, normalize_queries=normalize)
        loss = out.float().sum()
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times_t = torch.tensor(times)
    out_final = fn(q, m, idx, w, scale=scale, normalize_queries=normalize)
    return {
        "name": name,
        "mean_ms": times_t.mean().item(),
        "std_ms": times_t.std().item(),
        "min_ms": times_t.min().item(),
        "max_ms": times_t.max().item(),
        "n_iter": n_iter,
        "out_sample": out_final.detach().clone(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--k-dim", type=int, default=128)
    ap.add_argument("--num-segments", type=int, default=16)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    B = args.micro_batch
    T = args.seq_len
    H = args.heads
    K = args.k_dim
    V = K
    N = args.num_segments
    R = args.topk
    scale = K ** -0.5

    print(f"=== benchmark shapes: B={B} T={T} H={H} K=V={K} N={N} R={R} dtype={args.dtype} ===", flush=True)
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

    q, m, idx, w = _make_inputs(B, T, H, K, V, N, R, args.device, dtype)

    from dsc.mc_baseline.cached_memory_read import ssc_gather_read as v2
    from dsc.mc_v3 import ssc_gather_read_v3a as ssc_v3a
    from dsc.mc_v3 import ssc_gather_read_v3c as ssc_v3c

    # Equivalence check first (catch correctness issues before timing)
    print("\n--- equivalence check ---", flush=True)
    with torch.no_grad():
        out_v2 = v2(q, m, idx, w, scale=scale, normalize_queries=False)
        out_v3a = ssc_v3a(q, m, idx, w, scale=scale, normalize_queries=False)
        out_v3c = ssc_v3c(q, m, idx, w, scale=scale, normalize_queries=False)
        d_v3a = (out_v2 - out_v3a).abs().float()
        d_v3c = (out_v2 - out_v3c).abs().float()
        diff_v3a_max = d_v3a.max().item()
        diff_v3c_max = d_v3c.max().item()
        # p99 on subsample (67M elements too large for quantile)
        n_sub = min(100_000, d_v3a.numel())
        diff_v3a_p99 = torch.topk(d_v3a.flatten()[:n_sub], max(1, n_sub // 100)).values[-1].item()
        diff_v3c_p99 = torch.topk(d_v3c.flatten()[:n_sub], max(1, n_sub // 100)).values[-1].item()
        out_scale = out_v2.abs().float().mean().item()
        print(f"  out mean abs = {out_scale:.3e}", flush=True)
        print(f"  v3a vs v2: max={diff_v3a_max:.3e}  p99={diff_v3a_p99:.3e}  rel_max={diff_v3a_max/out_scale:.3e}", flush=True)
        print(f"  v3c vs v2: max={diff_v3c_max:.3e}  p99={diff_v3c_p99:.3e}  rel_max={diff_v3c_max/out_scale:.3e}", flush=True)
        diff_v3a = diff_v3a_max
        diff_v3c = diff_v3c_max
    # Acceptance: rel_max < 0.05 (5%) is fine for bf16 training (loss noise dominates)
    if diff_v3a_max / out_scale > 0.05 or diff_v3c_max / out_scale > 0.05:
        print(f"\nERROR: equivalence FAIL — rel_max > 5% — aborting", flush=True)
        sys.exit(1)

    print(f"\n--- timing ({args.n_warmup} warmup + {args.n_iter} measured iters, fwd+bwd) ---", flush=True)
    res_v2 = _bench_variant("v2", v2, q, m, idx, w, scale, False, args.n_warmup, args.n_iter)
    print(f"  v2 : {res_v2['mean_ms']:7.2f} ± {res_v2['std_ms']:5.2f} ms (min {res_v2['min_ms']:.2f}, max {res_v2['max_ms']:.2f})", flush=True)

    res_v3a = _bench_variant("v3a", ssc_v3a, q, m, idx, w, scale, False, args.n_warmup, args.n_iter)
    speedup_v3a = res_v2["mean_ms"] / res_v3a["mean_ms"]
    print(f"  v3a: {res_v3a['mean_ms']:7.2f} ± {res_v3a['std_ms']:5.2f} ms  ({speedup_v3a:.2f}× speedup)", flush=True)

    res_v3c = _bench_variant("v3c", ssc_v3c, q, m, idx, w, scale, False, args.n_warmup, args.n_iter)
    speedup_v3c = res_v2["mean_ms"] / res_v3c["mean_ms"]
    print(f"  v3c: {res_v3c['mean_ms']:7.2f} ± {res_v3c['std_ms']:5.2f} ms  ({speedup_v3c:.2f}× speedup)", flush=True)

    # Per-layer extrapolation (16 layers in mc_370M)
    n_layers = 16
    print(f"\n--- 16-layer extrapolation (single SSC kernel cost, not full model) ---", flush=True)
    print(f"  v2: {res_v2['mean_ms'] * n_layers:.2f} ms (fwd+bwd, 16 layers, no proj)", flush=True)
    print(f"  v3a: {res_v3a['mean_ms'] * n_layers:.2f} ms ({res_v2['mean_ms']/res_v3a['mean_ms']:.2f}× speedup)", flush=True)
    print(f"  v3c: {res_v3c['mean_ms'] * n_layers:.2f} ms ({res_v2['mean_ms']/res_v3c['mean_ms']:.2f}× speedup)", flush=True)

    result = {
        "shapes": {"B": B, "T": T, "H": H, "K": K, "V": V, "N": N, "R": R, "dtype": args.dtype},
        "device": torch.cuda.get_device_name(0),
        "v2": {k: v for k, v in res_v2.items() if k != "out_sample"},
        "v3a": {k: v for k, v in res_v3a.items() if k != "out_sample"},
        "v3c": {k: v for k, v in res_v3c.items() if k != "out_sample"},
        "speedup_v3a": speedup_v3a,
        "speedup_v3c": speedup_v3c,
        "equivalence": {"v3a_max_abs": diff_v3a, "v3c_max_abs": diff_v3c},
    }
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[done] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
