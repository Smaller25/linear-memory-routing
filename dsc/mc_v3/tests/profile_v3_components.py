"""Per-component profiling: vanilla vs MC v3 — exact timing breakdown.

Runs on 1 GPU, single layer, single fwd+bwd iteration, with torch.profiler
+ manual cuda.Event timing. Reports ms for every kernel.

Usage:
    python -m dsc.mc_v3.tests.profile_v3_components
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Vanilla vs MC
from dsc.mc_baseline.cached_memory_read import _SSCGatherRead as V2
from dsc.mc_v3 import _SSCGatherReadV3c as V3C


def fmt_ms(name, t, indent=2):
    print(f"{' ' * indent}{name:<40s}: {t * 1000:>8.3f} ms", flush=True)


def time_block(name, fn, warmup=3, iters=20):
    """Median time over iters, each measured with cuda.Event."""
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    median = times[len(times) // 2]
    return median, out


def profile_kernel_only(device="cuda:0"):
    """Time kernel alone, single call, training shape."""
    print(f"\n{'=' * 70}")
    print("PART 1: KERNEL-ONLY TIMING (single call, training shape)")
    print(f"{'=' * 70}")
    B, T, H, K, V, N, R = 8, 4096, 16, 128, 128, 16, 2
    scale = 1.0
    normalize = True

    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=False)
    m = torch.randn(B, N, H, K, V, device=device, dtype=torch.float32, requires_grad=False)
    idx = torch.randint(0, N, (B, T, R), device=device, dtype=torch.long)
    w = torch.randn(B, T, R, device=device, dtype=torch.bfloat16, requires_grad=False)

    # V2 fwd
    def v2_fwd():
        q_ = q.detach().requires_grad_(True)
        m_ = m.detach().requires_grad_(True)
        w_ = w.detach().requires_grad_(True)
        return V2.apply(q_, m_, idx, w_, scale, normalize)

    # V3 fwd
    def v3_fwd():
        q_ = q.detach().requires_grad_(True)
        m_ = m.detach().requires_grad_(True)
        w_ = w.detach().requires_grad_(True)
        return V3C.apply(q_, m_, idx, w_, scale, normalize)

    med_v2_fwd, _ = time_block("v2 fwd", v2_fwd)
    med_v3_fwd, _ = time_block("v3 fwd", v3_fwd)

    # fwd+bwd: get a graph to call backward on
    q_ = q.detach().requires_grad_(True)
    m_ = m.detach().requires_grad_(True)
    w_ = w.detach().requires_grad_(True)
    out_v2 = V2.apply(q_, m_, idx, w_, scale, normalize)
    grad = torch.ones_like(out_v2)

    def v2_bwd():
        q2 = q.detach().requires_grad_(True)
        m2 = m.detach().requires_grad_(True)
        w2 = w.detach().requires_grad_(True)
        out2 = V2.apply(q2, m2, idx, w2, scale, normalize)
        gq, gm, gw = torch.autograd.grad(out2, [q2, m2, w2], grad_outputs=torch.ones_like(out2), retain_graph=False)
        return gq

    def v3_bwd():
        q3 = q.detach().requires_grad_(True)
        m3 = m.detach().requires_grad_(True)
        w3 = w.detach().requires_grad_(True)
        out3 = V3C.apply(q3, m3, idx, w3, scale, normalize)
        gq, gm, gw = torch.autograd.grad(out3, [q3, m3, w3], grad_outputs=torch.ones_like(out3), retain_graph=False)
        return gq

    med_v2_bwd, _ = time_block("v2 bwd", v2_bwd)
    med_v3_bwd, _ = time_block("v3 bwd", v3_bwd)

    print(f"\n  shape: B={B} T={T} H={H} K=V={K} N={N} R={R}")
    print(f"  v2 fwd       : {med_v2_fwd:>8.3f} ms")
    print(f"  v3 fwd       : {med_v3_fwd:>8.3f} ms   ({med_v2_fwd / med_v3_fwd:.2f}x faster)")
    print(f"  v2 fwd+bwd   : {med_v2_bwd:>8.3f} ms")
    print(f"  v3 fwd+bwd   : {med_v3_bwd:>8.3f} ms   ({med_v2_bwd / med_v3_bwd:.2f}x faster)")
    print(f"  inferred v2 bwd-only: {med_v2_bwd - med_v2_fwd:>8.3f} ms")
    print(f"  inferred v3 bwd-only: {med_v3_bwd - med_v3_fwd:>8.3f} ms")


def profile_full_layer(device="cuda:0"):
    """Build a single MC SSC layer + a single vanilla GDN-2 layer, time fwd+bwd."""
    print(f"\n{'=' * 70}")
    print("PART 2: FULL LAYER fwd+bwd (1 layer, training shape)")
    print(f"{'=' * 70}")

    from dsc.lit_gpt.config import Config
    from dsc.lit_gpt.model import Block

    B, T = 8, 4096
    config_name_mc = "mc_370M"
    config_name_va = "gdn2_370M"

    cfg_mc = Config.from_name(config_name_mc)
    cfg_va = Config.from_name(config_name_va)

    # Build MC block (one layer)
    torch.manual_seed(0)
    mc_block = Block(cfg_mc, 0).to(device).to(torch.bfloat16)
    mc_block.train()

    # Build vanilla block (one layer)
    va_block = Block(cfg_va, 0).to(device).to(torch.bfloat16)
    va_block.train()

    # Block.forward signature requires rope + max_seq_length. Build a rope cache
    # matching the config's block_size and slice [:T].
    def build_rope(cfg):
        if cfg.nope:
            return None
        from dsc.lit_gpt.model import build_rope_cache
        head_size = cfg.n_embd // cfg.n_head
        cos, sin = build_rope_cache(
            seq_len=cfg.block_size,
            n_elem=int(cfg.rotary_percentage * head_size),
            dtype=torch.bfloat16,
            device=device,
        )
        return (cos[:T], sin[:T])

    rope_mc = build_rope(cfg_mc)
    rope_va = build_rope(cfg_va)

    x_mc = torch.randn(B, T, cfg_mc.n_embd, device=device, dtype=torch.bfloat16, requires_grad=True)
    x_va = torch.randn(B, T, cfg_va.n_embd, device=device, dtype=torch.bfloat16, requires_grad=True)

    def mc_fwd_bwd():
        x = x_mc.detach().requires_grad_(True)
        out, _ = mc_block(x, rope_mc, T)
        out.sum().backward()
        return out

    def va_fwd_bwd():
        x = x_va.detach().requires_grad_(True)
        out, _ = va_block(x, rope_va, T)
        out.sum().backward()
        return out

    med_mc, _ = time_block("MC layer fwd+bwd", mc_fwd_bwd)
    med_va, _ = time_block("VA layer fwd+bwd", va_fwd_bwd)

    print(f"\n  MC  layer fwd+bwd: {med_mc:>8.3f} ms")
    print(f"  VA  layer fwd+bwd: {med_va:>8.3f} ms")
    print(f"  MC/VA ratio     : {med_mc / med_va:.2f}x")
    print(f"  overhead (ms)   : {med_mc - med_va:>8.3f} ms per layer")


def profile_16layer_iter(device="cuda:0"):
    """Simulate full 16-layer fwd+bwd (no checkpoint) — close to single-GPU iter."""
    print(f"\n{'=' * 70}")
    print("PART 3: 16-LAYER fwd+bwd (no ckpt, no DDP, single GPU)")
    print(f"{'=' * 70}")

    from dsc.lit_gpt.model import Block, build_rope_cache
    from dsc.lit_gpt.config import Config

    cfg_mc = Config.from_name("mc_370M")
    cfg_va = Config.from_name("gdn2_370M")
    n_layers = cfg_mc.n_layer  # 16

    B, T = 8, 4096

    torch.manual_seed(0)
    mc_blocks = torch.nn.ModuleList([Block(cfg_mc, i) for i in range(n_layers)]).to(device).to(torch.bfloat16)
    va_blocks = torch.nn.ModuleList([Block(cfg_va, i) for i in range(n_layers)]).to(device).to(torch.bfloat16)
    mc_blocks.train(); va_blocks.train()

    def build_rope(cfg):
        if cfg.nope:
            return None
        head_size = cfg.n_embd // cfg.n_head
        cos, sin = build_rope_cache(
            seq_len=cfg.block_size,
            n_elem=int(cfg.rotary_percentage * head_size),
            dtype=torch.bfloat16,
            device=device,
        )
        return (cos[:T], sin[:T])

    rope_mc = build_rope(cfg_mc)
    rope_va = build_rope(cfg_va)

    x_mc = torch.randn(B, T, cfg_mc.n_embd, device=device, dtype=torch.bfloat16)
    x_va = torch.randn(B, T, cfg_va.n_embd, device=device, dtype=torch.bfloat16)

    def mc_iter():
        x = x_mc.detach()
        for blk in mc_blocks:
            x, _ = blk(x, rope_mc, T)
        x.sum().backward()
        return x

    def va_iter():
        x = x_va.detach()
        for blk in va_blocks:
            x, _ = blk(x, rope_va, T)
        x.sum().backward()
        return x

    # warmup + time
    print("  warming up...", flush=True)
    med_mc, _ = time_block("MC 16-layer", mc_iter, warmup=2, iters=5)
    med_va, _ = time_block("VA 16-layer", va_iter, warmup=2, iters=5)

    print(f"\n  MC 16-layer fwd+bwd: {med_mc:>8.2f} ms")
    print(f"  VA 16-layer fwd+bwd: {med_va:>8.2f} ms")
    print(f"  MC/VA ratio       : {med_mc / med_va:.2f}x")
    print(f"  per-layer overhead: {(med_mc - med_va) / n_layers:>8.2f} ms")


def profile_with_torch_profiler(device="cuda:0"):
    """torch.profiler breakdown of 16-layer iter — per kernel CUDA time."""
    print(f"\n{'=' * 70}")
    print("PART 4: torch.profiler per-kernel CUDA time (16-layer iter)")
    print(f"{'=' * 70}")

    from dsc.lit_gpt.model import Block, build_rope_cache
    from dsc.lit_gpt.config import Config

    cfg_mc = Config.from_name("mc_370M")
    cfg_va = Config.from_name("gdn2_370M")
    n_layers = cfg_mc.n_layer

    B, T = 8, 4096
    torch.manual_seed(0)
    mc_blocks = torch.nn.ModuleList([Block(cfg_mc, i) for i in range(n_layers)]).to(device).to(torch.bfloat16)
    va_blocks = torch.nn.ModuleList([Block(cfg_va, i) for i in range(n_layers)]).to(device).to(torch.bfloat16)
    mc_blocks.train(); va_blocks.train()

    def build_rope(cfg):
        if cfg.nope:
            return None
        head_size = cfg.n_embd // cfg.n_head
        cos, sin = build_rope_cache(
            seq_len=cfg.block_size,
            n_elem=int(cfg.rotary_percentage * head_size),
            dtype=torch.bfloat16,
            device=device,
        )
        return (cos[:T], sin[:T])

    rope_mc = build_rope(cfg_mc)
    rope_va = build_rope(cfg_va)

    x_mc = torch.randn(B, T, cfg_mc.n_embd, device=device, dtype=torch.bfloat16)
    x_va = torch.randn(B, T, cfg_va.n_embd, device=device, dtype=torch.bfloat16)

    # warmup
    for _ in range(2):
        x = x_mc.detach()
        for blk in mc_blocks: x, _ = blk(x, rope_mc, T)
        x.sum().backward()
        torch.cuda.synchronize()
    for _ in range(2):
        x = x_va.detach()
        for blk in va_blocks: x, _ = blk(x, rope_va, T)
        x.sum().backward()
        torch.cuda.synchronize()

    from torch.profiler import profile, ProfilerActivity

    print("\n  --- MC SSC v3 (16-layer iter) ---", flush=True)
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof_mc:
        for _ in range(3):
            x = x_mc.detach()
            for blk in mc_blocks: x, _ = blk(x, rope_mc, T)
            x.sum().backward()
            torch.cuda.synchronize()

    print("\n  --- Vanilla GDN-2 (16-layer iter) ---", flush=True)
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof_va:
        for _ in range(3):
            x = x_va.detach()
            for blk in va_blocks: x, _ = blk(x, rope_va, T)
            x.sum().backward()
            torch.cuda.synchronize()

    print("\n  === MC TOP kernels (CUDA time, sorted, avg over 3 iters) ===", flush=True)
    print(prof_mc.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("\n  === VANILLA TOP kernels ===", flush=True)
    print(prof_va.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def main():
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    profile_kernel_only()
    profile_full_layer()
    profile_16layer_iter()
    profile_with_torch_profiler()


if __name__ == "__main__":
    main()
