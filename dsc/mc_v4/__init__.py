"""MC SSC v4 — kernel-identical to v3, adds training infra optimizations.

v4 keeps the exact same Triton kernels as v3 (selected via MC_KERNEL_VERSION=v3c
env var, dispatched in dsc.mc_baseline.mc_ssc). What's new is in the training loop:

  1. FSDP strategy: forward_prefetch=True (overlap next layer's all-gather with
     current layer's forward). gradient_as_bucket_view and static_graph were
     legacy-FSDP knobs and are no longer exposed in torch 2.12 (always-on / removed).
  2. In-training profiler (samples 5 iters every 200, dumps torch.profiler table +
     per-component cuda.Event breakdown to log_dir/v4_profile/).
  3. Per-iter component timing (always on): fwd / bwd / optim breakdown logged
     every 50 iters via [v4-perf] tag.

Math is bit-exact with v3 — no kernel changes. Realistic v4 iter ≈ v3 (855ms)
plus/minus forward_prefetch savings (~30ms typical). Profiler output gives the
ground-truth per-component breakdown needed to plan v5 (fuse MC layer ops beyond
the gather kernel, where most of the ~460ms MC-overhead-vs-vanilla lives).

See MC_SSC_V3_SPEED_KO.md for the v2→v3→v4 trajectory and bottleneck analysis.
"""
