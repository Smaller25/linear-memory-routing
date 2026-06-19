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
# Phase-0 kill-test (Blackwell, via Slurm; MQAR needs >=3000 steps — delayed phase transition)
sbatch scripts/sh_slurm_run.sh python -m lmr.mosc.train_mqar \
    --model gdn2 --train-kv 16 32 --eval-kv 16 32 64 --steps 3000              # baseline
sbatch scripts/sh_slurm_run.sh python -m lmr.mosc.train_mqar \
    --model mosc --chunk-mode oracle --train-kv 16 32 --eval-kv 16 32 64 --steps 3000
```
