# Linear-Memory-Routing — Interim Report (2026-06-22)

A consolidated account of the project so far: the initial hypothesis, the method family, every
experiment track and its result, and where we stand. Per-experiment detail lives in `report/0001`–
`0012`; operational state in `SESSION_HANDOFF.md` and `lmr/mosc/README.md`. This document is the
single narrative.

---

## 1. Initial hypothesis & motivation

Linear-recurrent sequence models (Mamba-2, Gated DeltaNet, …) run in O(1) state but a **fixed-size
recurrent state saturates** on long context, and **mixes co-occurring facts** in one state matrix.
Two failure modes, on two axes:
- **Temporal (saturation):** over a long horizon the fixed state overwrites old information →
  fails Needle-in-a-Haystack / passkey at length.
- **Spatial (interference):** many facts accumulated into one state overwrite each other →
  fails multi-key recall (MQAR).

**Core hypothesis.** If we **cache recurrent-state checkpoints** along the sequence and add a small
**trained read-out router** that selects among them at query time, a model can **exceed its
fixed-state recall** — recovering information the live state has lost — without growing the state it
runs with.

Two references frame the axes: **Memory Caching** (temporal checkpointing) and **Mixture-of-Memories
/ MoM** (parallel memories for interference).

---

## 2. Two tracks

| | **Frozen retrofit** (Track A, reports 0001–0011) | **From-scratch co-train** (Track B, report 0012) |
|---|---|---|
| Idea | freeze a pretrained LM; train ONLY a read-out router over its cached states | train a small model end-to-end with caching + routing built in |
| Backbone | Mamba-2 1.3B/2.7B, GDN 1.3B (pretrained → FLA) | GDN-2 (2-layer, from scratch) |
| Target | single-needle long-context (saturation) | multi-key recall (interference) |
| Outcome | **net win, validated** | **multi-key solved (this session)** |

---

## 3. Track A — frozen retrofit (reports 0001–0011)

Read-out mechanisms tried over the per-segment recurrent-state cache:
- **RM** (training-free read mix) — neutral-to-negative; frozen ≠ trained-for (0002).
- **GRM** (trained gate) — recovers RM's collapse, but generalises only ~2× the trained #segments
  (0003, 0004).
- **SSC** (hard top-k selection over snapshots) — the win.

**Headline results.**
- **Net win at the saturation length.** Natural-language passkey @8k: vanilla→+SSC =
  **0.738→0.986 (+0.25) @mamba2-1.3b**, **0.500→1.000 (+0.50) @2.7b** — the win **grows with model
  size** (0005, 0007).
- **Transfers to the standard benchmark.** SSC trained on our passkey is **zero-shot on RULER
  `niah_single`** and beats vanilla (+0.06–0.18 @4k/8k) (0009).
- **Only hard top-k generalises.** RM / GRM / AoM (dense) / MoM-slot-merge / hierarchical all collapse
  at long context; only sparse hard selection survives (k∈[2,8]; k=1 brittle) (0008, 0009).
- **GDN** is a strong long-context recaller (vanilla passkey 8k≈0.93 vs mamba2 0.74), but GDN **router
  training is kernel-blocked** at head_dim=256 (chunk-bwd shared-mem > A100; tilelang gap) (0006).

**Honest negatives (the boundary of Track A).**
- **Multi-key is out of scope (0010).** Training SSC on multi-key did not converge. Root cause:
  *"MC caches one state per **segment**, not per **fact**; keys sharing a segment can't be
  disambiguated."* This is the *interference* regime, not *saturation*.
- **Not constant-memory (0011).** The win needs ~all O(N) snapshots; capping the cache to B degrades
  recall ∝ B/N. Top-k cuts the *read* to O(N·k) but the *cache* stays O(N) → SSC is a compressed-cache
  point on the RNN↔attention spectrum, not a constant-memory linear model.

---

## 4. Track B — from-scratch Dynamic-MoSC (report 0012, this session)

**Idea.** Attack the 0010 interference regime by co-training from scratch with **content-adaptive
segment boundaries** so a segment is ≈ *per-fact* (not per-256-tokens), plus the hard-top-k
segment-cache read-out from Track A. Backbone: **GDN-2** (`fla.layers.gdn2`; the general gated-delta
form — scalar gate → Gated DeltaNet v1, vector gate → KDA). Task: Zoology-faithful **MQAR**.

### Phase 0 — does segment routing solve multi-key at all? (recall vs #kv)
| model | kv4 | kv8 | kv16 | kv32 | kv64 | kv128 |
|-------|----|----|-----|-----|-----|------|
| vanilla GDN-2 | 1.00 | 1.00 | 0.93 | 0.55 | 0.28 | — |
| **oracle** (per-fact boundaries) | 1.00 | 1.00 | **1.00** | **1.00** | **1.00** | **1.00** |
| fixed / surprisal | 1.00 | 1.00 | ~0.95 | ~0.6 | ~0.3 | ~0.15 |

→ **PASS.** With per-fact boundaries the segment-cache + hard-top-k read **solves** multi-key (the
0010 regime). The entire win, though, is **boundary placement** — fixed/surprisal ≈ vanilla.

### Phase 1 — are the oracle boundaries learnable?
A `Linear(d,1)` head over the model's own hidden states, distilled from oracle positions.
- First attempt failed (worse than vanilla) — diagnosed as **curriculum overfit** (trained on kv 4/8
  only, it learned a *position* bias and fired ~8 boundaries regardless of kv), **not a kernel bug**
  (fla gated-delta op test passes on sm_120, 10/10).
- Widening the curriculum to **train-kv 4–64** fixed it: **learned ≈ oracle** (kv128 0.998),
  generalising to a kv count not seen in training. → **boundaries are learnable.**

### True-state validity check — CONFIRMED
The read-out had summarised segments by a pooled-hidden **proxy** (attention-lite). Re-running with
the **true GDN-2 recurrent state** at each boundary (`backbone.run_segmented`):
| model | kv16 | kv32 | kv64 | kv128 |
|-------|-----|-----|-----|------|
| oracle (proxy)      | 1.00 | 1.00 | 1.00 | 1.00 |
| **oracle (true-state)** | 1.00 | 0.999 | **0.998** | **0.992** |
| fixed (proxy)       | 0.97 | 0.64 | 0.34 | 0.17 |
| **fixed (true-state)**  | 0.999 | 0.988 | **0.882** | **0.468** |

→ (1) **The multi-key win is genuine recurrent-state recall**, not a pooled-hidden artifact. (2) The
true state is a strong lever on its own (fixed 0.34→0.88 @ kv64); boundary placement still wins at the
highest density (kv128).

---

## 5. Process notes (hard-won; don't re-discover)

- **Server migration.** A100 (VESSL) → 2× RTX PRO 6000 Blackwell (sm_120, Slurm). Built a GPU-aware
  env (`scripts/setup_env.sh`) + a conda env **`sh_routing`** + Slurm entrypoint
  (`scripts/sh_slurm_run.sh`); both A100 and Blackwell paths kept alive. All GPU work goes through
  `sbatch`.
- **fla 0.5.1 → 0.5.2** to vendor **GDN-2** (`fla/ops/gdn2`, `fla/layers/gdn2.py`); re-applied the
  transformers-5.12 `_tied_weights_keys` patch; frozen-track gates still pass (18/18).
- **Blackwell kernels are sound.** Mamba-3 needs `mamba_ssm` (SISO kernels) — deferred; GDN-2 / KDA /
  GDN-v1 are pure-Triton and train from scratch on sm_120 (no tilelang). Tilelang is unavailable under
  py3.13/Blackwell — irrelevant for the GDN-2 path.
- **MQAR gotchas.** Needs high lr (3e-3) + grad-clip + a curriculum that includes both easy and hard
  kv; training straight on kv 16/32 pins at random loss (curriculum, not a bug). Legacy MoCMMixer
  trainer does not learn under fla 0.5.2 (off-path).

---

## 6. Where we stand

**Established.**
- Track A: a trained hard-top-k read-out over a frozen model's cached state gives a real, size-scaling
  net win on single-needle long-context, and transfers to RULER — but is single-needle-only and not
  constant-memory.
- Track B: the multi-key regime Track A could not reach **is solved from scratch** by GDN-2 + a
  hard-top-k read-out over the **true** cached recurrent state, **when boundaries are per-fact**, and
  those boundaries are **learnable** (learned ≈ oracle, generalising).

**Open (next milestones).**
1. **Unsupervised boundaries** — learn per-fact boundaries from the task loss / a budget penalty,
   without oracle distillation. (Distillation proved the signal is present; this proves it's learnable
   unsupervised.)
2. **learned × true-state together** (so far measured separately).
3. **Scale the true-state read-out** — `run_segmented` is a sequential per-segment loop (~5–6× slower);
   needs a batched / intermediate-state kernel.
4. **Generalisation** — single-needle long-context (passkey/RULER), parallel-pool routing (Mixture of
   Segment-Cache), longer contexts (16k/32k).

**Map to reports:** Track A = 0001–0011; Track B = 0012; this file = the synthesis.
