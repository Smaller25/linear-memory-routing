"""MC SSC v3 — mathematically-equivalent but faster implementation.

Drop-in replacement for dsc.mc_baseline.cached_memory_read.ssc_gather_read.

Optimizations vs v2 (all preserve bit-exact math):
  v3a: cache hints (eviction_policy) on tl.load
  v3b: BLOCK_T=2 to halve launch overhead
  v3c: segment-conditional loading (process unique segments per block)
  v3d: target-parallel forward (one program per (b,h,n), atomic_add to out)

Variant selection: import the appropriate symbol.
  - ssc_gather_read_v3a   : safe incremental speedup
  - ssc_gather_read_v3b   : BLOCK_T sweep
  - ssc_gather_read_v3c   : block-dedup, the main win
  - ssc_gather_read       : best of (currently aliases v3c)
"""
from dsc.mc_v3.cached_memory_read_v3 import (
    ssc_gather_read,
    ssc_gather_read_v3a,
    ssc_gather_read_v3b,
    ssc_gather_read_v3c,
    _SSCGatherReadV3a,
    _SSCGatherReadV3b,
    _SSCGatherReadV3c,
    SSCGatherReadV3,
)
