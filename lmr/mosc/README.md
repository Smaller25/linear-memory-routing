# Dynamic-MoSC — from-scratch track (GDN2 backbone)

Co-trained sibling of the frozen `lmr/` retrofit track. Where the frozen track adds a hard-top-k
read-out (SSC) over a **frozen** backbone's cached state — and wins only single-needle long-context
recall (report 0005/0007), failing on multi-key interference (report 0010) — this track trains
**from scratch** so the memory can specialise, targeting the multi-key/interference regime SSC can't
reach.

## Why from-scratch (not frozen + PEFT)
The frozen ceiling is measured: SSC wins single-needle saturation but its multi-key training does
**not converge** (report 0010), and segment-level routing over a frozen single state can't
disambiguate keys sharing a segment. The parallel-memory axis needs co-training. So this track drops
the frozen retrofit as the primary path; frozen SSC remains the published **ablation** (quantifies
what co-training buys).

## Backbone: GDN2
`fla.layers.gdn2.GatedDeltaNet2` — "Gated DeltaNet-2: Decoupling Erase and Write". GDN-2 is the
**general** form: collapse the gate to a scalar → Gated DeltaNet v1; the vector-gate form → KDA. So
KDA and GDN-v1 are ablation points within one family. Pure-Triton → trains from scratch on
Blackwell/sm_120 (no mamba_ssm/tilelang; verified `scripts/sh_check_backbones.py`).
(Mamba3 — the SSM-axis alternative — needs `mamba_ssm` with Mamba-3 SISO kernels; deferred.)

## The two axes
- **Temporal — surprisal-driven dynamic chunking** (`dynamic_chunk.py`): place state checkpoints at
  high-surprisal tokens (≈ per-fact) instead of fixed 256-token segments — directly attacks report
  0010 ("one state per segment, not per fact"). Modes: `fixed` (=SSC control), `oracle` (boundaries
  at the keys — the Phase-0 upper bound), `surprisal` (NLL peaks + min-gap, to test the R1 collapse).
- **Spatial — mixture of segment-cache** (`router.py`): write-route segments to M expert pools; read
  via hard top-k over the (pool × segment) bank. `num_pools=1` == temporal-only (SSC).

## Files
| file | role | status |
|------|------|--------|
| `backbone.py` | GDN2 causal LM (vanilla baseline) | working |
| `dynamic_chunk.py` | boundary modes + surprisal + MQAR oracle helper | working |
| `router.py` | hard-top-k segment-cache router | skeleton (Phase 2) |
| `mosc_model.py` | integrated Dynamic-MoSC | runnable skeleton (segment summary = pooled-hidden PROXY) |
| `train_mqar.py` | from-scratch MQAR trainer | working |

## Experiment plan (kill-test first)
**Phase 0 — oracle-chunking kill-test.** `vanilla` vs `mosc --chunk-mode oracle` on multi-key MQAR.
If oracle (≈per-fact) boundaries don't beat vanilla, segment-level routing is dead → pivot to
bounded-memory consolidation (report 0011). PASS → continue.
**Phase 1 — surprisal chunking** approximates oracle on single-needle (does it match SSC recall with
fewer snapshots?) + replace the pooled-hidden proxy with the true GDN2 recurrent state at boundaries.
**Phase 2 — mixture of segment-cache** (M pools), routing-granularity ablation (segment vs token).
**Phase 3 — consolidation / bounded cache.** **Phase 4 — LongBench/RULER scale.**

## Run
```bash
# Phase-0 kill-test (Blackwell, via Slurm). Use an EASY curriculum (train-kv 4 8) — training
# directly on kv 16/32 fails to bootstrap (loss pinned at random; not a kernel bug — fla's gated-
# delta op test passes on sm_120). >=6000 steps.
sbatch scripts/sh_slurm_run.sh python -m lmr.mosc.train_mqar \
    --model gdn2 --train-kv 4 8 --eval-kv 4 8 16 32 64 128 --steps 6000              # vanilla baseline
sbatch scripts/sh_slurm_run.sh python -m lmr.mosc.train_mqar \
    --model mosc --chunk-mode oracle --train-kv 4 8 --eval-kv 4 8 16 32 64 128 --steps 6000
```

## Phase-0 results (2026-06-19, RTX PRO 6000, train-kv 4/8, 6000 steps) — recall acc vs #kv
| model | kv4 | kv8 | kv16 | kv32 | kv64 | kv128 |
|-------|----|----|-----|-----|-----|------|
| vanilla GDN2     | 1.00 | 1.00 | 0.93 | 0.55 | 0.28 | —    |
| mosc **oracle**  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| mosc fixed       | 1.00 | 1.00 | 0.97 | 0.64 | 0.34 | 0.17 |
| mosc surprisal   | 1.00 | 1.00 | 0.91 | 0.50 | 0.25 | 0.12 |

**Verdict — PASS, but the whole win is boundary placement.** Oracle (a boundary at each context
value = per-fact segments) gives near-perfect recall at every kv (0.997 @ kv128 vs vanilla 0.28 @
kv64) — segment-level caching + hard-top-k read *does* solve multi-key recall (the regime SSC failed
in report 0010). BUT fixed-chunk barely beats vanilla and the current surprisal heuristic (NLL peak +
min-gap) is no better than vanilla — it does not find fact boundaries. So Phase-1's single question:
**learn boundaries that approximate the oracle.** Caveats: segment summary is still the pooled-hidden
PROXY (not the true recurrent state); the read-out is attention-lite over segments (a compressed-cache
point on the RNN↔attention spectrum, cf. report 0011), so "mosc >> vanilla" is partly expected — the
*scientific* signal is oracle ≫ fixed/surprisal (same read-out, only boundaries differ).

NOTE: the legacy `lmr/scripts/train_mocm_mqar.py` (MoCMMixer) does NOT learn under fla 0.5.2 (silent,
likely API drift) — not on this track's path; GDN2LM here learns fine.

## Phase-1 status (IN PROGRESS — learn the oracle's boundaries)
`chunk-mode learned`: a `nn.Linear(d_model,1)` boundary head (`DynamicMoSC.boundary_predictor`),
distilled from oracle positions via pos-weighted BCE during training (`--boundary-distill`), using
its own hard (sigmoid>0.5) boundaries at eval.

**First attempt FAILED** (train-kv 4/8, 6000 steps, distill 1.0) — recall vs #kv:
| kv4 | kv8 | kv16 | kv32 | kv64 | kv128 |
|----|----|-----|-----|-----|------|
| 1.00 | 1.00 | 0.50 | 0.25 | 0.13 | 0.06 |

learned is **worse than vanilla** (kv16 0.50 vs 0.93) — predicted boundaries are bad enough that the
read-out injects noise. So naive per-token BCE distillation does NOT recover oracle boundaries.

**Diagnosed (job 1101) — the failure is UNDER-FIRING.** Predicted-boundary count + precision/recall
vs oracle:
| kv | recall-acc | pred/seq (oracle) | precision | boundary-recall |
|----|-----------|-------------------|-----------|-----------------|
| 4  | 1.00 | 4.6 (4)  | 0.87 | 1.00 |
| 8  | 1.00 | 8.2 (8)  | 0.98 | 1.00 |
| 16 | 0.52 | 8.2 (16) | 1.00 | **0.52** |
| 32 | 0.28 | 10.3 (32)| 0.96 | **0.31** |
| 64 | 0.51 | 36.5 (64)| 0.97 | 0.56 |
Precision is ~1.0 (a fired boundary is almost always a true fact), but the sigmoid>0.5 threshold is
too conservative → it MISSES ~half the facts at mid-kv → un-checkpointed facts can't be recalled.

**SOLVED — it was positional overfitting, not the threshold.** The threshold sweep barely moved recall
(kv16 0.61→0.78 from thr 0.5→0.05) because the head fired only ~10 boundaries *regardless of kv*
(pred/seq capped ~10): trained on kv 4/8 only (context 8–16 tokens), it learned "fire on the first
~8 value-looking tokens" — a POSITION bias, not "is this token a value". Widening the training
curriculum to **train-kv 4 8 16 32 64** (8000 steps) fixes it completely:

| model | kv4 | kv8 | kv16 | kv32 | kv64 | kv128 |
|-------|----|----|-----|-----|-----|------|
| vanilla            | 1.00 | 1.00 | 0.93 | 0.55 | 0.28 | —    |
| oracle (ceiling)   | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| learned (kv 4/8)   | 1.00 | 1.00 | 0.61 | 0.31 | 0.15 | 0.08 |
| **learned (kv 4–64)** | 1.00 | 1.00 | **1.00** | **1.00** | **1.00** | **1.00** |

Boundary count now scales with kv (pred/seq 16.9/33.0/65.2/128.7 vs oracle 16/32/64/128; precision
0.84–0.99, recall 1.00; threshold-insensitive 0.05–0.5). **=> boundaries ARE learnable**: a Linear
head over the model's own hidden states recovers oracle-level recall. The Dynamic-MoSC core mechanism
works.

### RESUME HERE (Phase-1 done w/ supervision; remaining)
1. **Remove the oracle crutch**: train boundaries end-to-end from the TASK loss only (`--boundary-
   distill 0`), or with a sparsity/budget penalty — does the head still find facts without BCE
   supervision? (distillation proved the signal is *present*; this proves it's *learnable unsupervised*.)
2. **Validity**: replace the pooled-hidden segment-summary PROXY in `mosc_model._segment_summaries`
   with the TRUE GDN2 recurrent state at each boundary (segment-wise run with `output_final_state`,
   cf. frozen-track `lmr/segment_runner.py`).
3. Then Phase 2 (M parallel pools) and real-data / LongBench (Phase 4). Caveat throughout: the
   read-out is attention-lite over segments (RNN↔attention spectrum, report 0011) — the scientific
   claim is about boundary learnability, not constant memory.

Env: `conda activate sh_routing`; run via `sbatch scripts/sh_slurm_run.sh python -m lmr.mosc.train_mqar ...`.
