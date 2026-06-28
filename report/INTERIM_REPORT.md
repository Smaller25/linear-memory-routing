# Linear-Memory-Routing — Interim Report (updated 2026-06-28)

Consolidated account through report 0019: the goal, the experimental setup, and every result. Per-
experiment detail is in `report/0001`–`0019`; this is the single narrative. (Supersedes the earlier
2026-06-28 cut at 0016, and the 2026-06-22 version at 0012.)

---

## 1. Goal & hypothesis
Linear-recurrent models (Mamba-2, Gated DeltaNet/GDN-2, KDA) run in O(1) state, but the fixed state
(a) **saturates** over long context and (b) **mixes co-occurring facts**. Two failure axes:
- **Temporal (saturation):** old info is overwritten → fails long-context single-needle retrieval.
- **Spatial (interference):** many facts in one state overwrite each other → fails multi-key recall.

**Hypothesis.** Cache recurrent-state checkpoints along the sequence and add a small **trained
hard-top-k read-out router** over them, so the model can **exceed its fixed-state recall** without
growing the state it runs with.

**Two tracks** (cannot be one checkpoint — synthetic-from-scratch vs real-pretrained):
| | Track A — frozen retrofit (0001–0011, 0014) | Track B — from-scratch co-train (0012–0016) |
|---|---|---|
| idea | freeze a pretrained LM; train ONLY the read-out router | train a small model end-to-end with caching + routing |
| target | single-needle long-context retrieval | multi-key recall + adaptive segmentation |

---

## 2. Experimental setup
- **Models.** Track A: pretrained `state-spaces/mamba2-{370m,1.3b,2.7b}` and GDN-1.3b → FLA, frozen,
  + ~30M SSC router. Track B: from-scratch GDN-2 (d_model 256, 4 layers, ~6M).
- **Benchmarks (standardized suite).**
  - **MQAR** (Zoology-faithful, `lmr/tasks/mqar.py`) — token-level multi-key recall; the controlled
    in-house probe. **Irregular variant** (`make_mqar_gapped`): random filler between facts → facts at
    non-periodic positions (tests adaptivity).
  - **RULER** (vendored NVIDIA, real-text long-context) — `niah_single`, `niah_multikey`; free
    generation + official string-match.
  - **flip-flop** (FFLM, `lmr/tasks/flipflop.py`) — state-tracking do-no-harm check.
  - **Selective Copying** (Gu & Dao 2023, `lmr/tasks/selective_copying.py`, **run 0017**) — M data
    tokens at random positions among 8M noise, reproduced in order; the standard content-vs-position
    task and external analog of our MQAR+filler probe. (**MAD** noisy/fuzzy recall still recommended.)
- **Recipe.** lr 3e-3, AdamW (wd 0.1, β 0.9/0.95), grad-clip 1.0; MQAR needs a curriculum spanning
  easy+hard kv (training straight on hard kv fails to bootstrap).
- **Hardware.** Migrated A100(VESSL) → 2× RTX PRO 6000 Blackwell (sm_120), Slurm, conda env
  `sh_routing` (torch 2.11+cu128, FLA 0.5.2). All GPU work via `sbatch scripts/sh_slurm_run.sh`.
- **Read-out modes** (`lmr/mosc/`): `fixed` (every C), `oracle` (boundary at each fact), `surprisal`,
  `learned` (boundary head distilled from oracle), `unsup`/`unsup_ste` (no oracle), `density` (axis-1:
  boundaries from an intrinsic info-density signal — surprisal/entropy/cosine-distance — at a target
  rate). Segment summary = the **boundary-token hidden** (mean-pool dilutes facts; see 0015/§4). True
  recurrent-state cache via `backbone.run_segmented`. **Bounded-memory cache** (axis-2, `cache_mode`):
  `full` (O(N)) | `capped` (keep recent B, drop older) | `hier` (recent fine + older merged to ≤B/2 via
  a learned conv).

---

## 3. Results

### Track A — frozen retrofit
- **SSC net win, single-needle (0005/0007).** Passkey @8k vanilla→+SSC: **0.738→0.986 (+0.25) @1.3b**,
  **0.500→1.000 (+0.50) @2.7b** — win grows with model size.
- **Transfers to RULER (0009).** Passkey-trained SSC is zero-shot on RULER `niah_single` (+0.06–0.18).
- **Only hard top-k generalizes (0008).** RM/GRM/AoM/MoM-merge/hierarchical collapse; SSC k∈[2,8].
- **RULER free-generation, mamba2-370m (0014, this session).** Official metric, matched n=30:
  niah_single @2048 **vanilla 0.00 → +SSC 36.7** (zero-shot); multikey 0.00 → 3.3. So the retrofit
  gives a **real net win on the standard free-gen protocol** at 2048. Caveats: 370m is OOD past its
  ~2k pretraining (floors at 4k/8k); **+SSC free-gen is too slow past 2k** (4h timeout — the decode
  loop re-runs cached segments per token; needs a batched kernel).
- **Negatives.** Multi-key out of scope (0010); **not constant-memory** — needs ~all O(N) snapshots,
  capping degrades ∝ B/N (0011).

### Track B — from-scratch Dynamic-MoSC
- **Multi-key solved, scales (0012/0013).** GDN-2 + hard-top-k segment-cache read-out: vanilla
  saturates (regular 0.92@kv64→0.002@kv512), while **oracle and learned boundaries hit ~1.0 to
  kv512**, generalizing past the kv≤128 training range. Boundary precision/recall ~1.0.
- **True recurrent state confirms it (0012).** Replacing the pooled-hidden proxy with the actual GDN-2
  state at each boundary keeps oracle ~1.0 (kv128 0.99) — the win is genuine state recall, not
  attention over pooled activations. True state is itself a strong lever (fixed boundaries 0.34→0.88
  @kv64).
- **Adaptivity — the decisive test (0015).** Regular MQAR is **degenerate**: fixed `chunk=2` == oracle
  == 1.0 (facts at a fixed period). On **irregular** MQAR the rules diverge:

  | irregular | kv64 | kv128 | kv256 | kv512 |
  |---|---|---|---|---|
  | vanilla | 0.82 | 0.30 | 0.06 | 0.01 |
  | **fixed chunk=2** | **0.00** | **0.00** | **0.00** | **0.00** |
  | oracle | 1.00 | 1.00 | 1.00 | 0.97 |
  | **learned** | ~1.0 | ~1.0 | ~1.0 | **1.00** |

  Fixed stride **fails**; the learned head matches the oracle (precision/recall **1.0**) and its
  segment lengths become a **real distribution** (median 4@kv64→2@kv512, tracking the random gaps) vs
  the regular spike-at-2. → boundaries are **content-adaptive and learnable** — *with supervision*.
- **Unsupervised boundary learning fails three ways (0016).** Without the oracle: **soft** landmark-
  attention recalls ~1.0 but via full attention (boundaries never form); **STE hard-cache + L1**
  collapses to 0; **warm-start** (distill then drop oracle) **fully drifts** — threshold-free top-k(p)∩
  facts = **0.00** vs supervised **1.00**. The boundary signal needs continuous supervision; unsup
  discrete-boundary discovery is open.
- **State-tracking do-no-harm (flip-flop).** vanilla GDN-2 and Dynamic-MoSC(fixed) both 1.00 at
  n_instr 128/256/512 — segmenting does not break native state-tracking.

### This session (0017–0019): standard benchmark + the two new directions
- **Selective Copying validates the supervised method, with a caveat (0017).** Read-out turns vanilla's
  collapse at M128 (**0.59**) into ~1.0; the **learned head recovers the data positions perfectly**
  (precision = recall = top-k(p) = **1.00**, exactly #data boundaries) with variable adaptive segment
  lengths (mean 8.9, tracking the 1/8 density) — the 0015 adaptivity claim reproduces on a recognized
  task. *But* fixed chunk=2 **also** ~1.0: with an unconstrained cache, over-segmentation is free, so
  the task discriminates *read-out vs vanilla*, not *where you cut*. Adaptivity-necessity needs a budget.
- **Axis-1 (information-density segmentation) — a real but fragile signal (0018).** A task-trained
  backbone's **raw surprisal ranks facts at top-k(p) = 0.40** (vs 0016 unsup 0.00, random ~0.16) — the
  intrinsic signal genuinely carries partial fact-location info. But it fails as a *mechanism*: a fixed
  0.5 cutoff fires nothing (p diffuse), and a quantile cutoff fires the right count but recall stays 0
  **and co-training drifts the signal 0.40 → 0.01** — using density as a hard selector destroys the
  correlation it relies on. Entropy/cosine-distance weaker. Supervised remains the only method on facts.
- **Axis-2 (hierarchical re-compression) — lossy merge loses to dropping (0019).** Under a cache budget
  B, `capped` reproduces 0011's ∝B/N degradation (capped-64: 0.98→0.49→0.24 as N grows). `hier` (learned
  merge of old segments) is **worse than capped at every budget** (hier-64 0.73/0.26/0.03): merging two
  distinct facts makes neither recoverable, so trading exact slots for lossy coarse ones strictly hurts
  *exact* recall. **0011's ~O(N) for exact multi-key stands** — compression only fits lossy-tolerant
  tasks. (Density boundaries → ~0 under every policy: axis-2 presupposes good cuts.)

---

## 4. Honest scope / what is NOT shown
- **Adaptivity is supervised-only.** The learned per-fact boundaries (0015) require oracle distillation;
  unsupervised discovery failed three ways (0016) and the axis-1 information-density route (0018) gives
  only a partial, drift-prone signal (surprisal 0.40, collapses under co-training). Still the main open
  problem.
- **A sparse-cache budget is essential to the claim.** Given full (soft) attention or an unbounded cache,
  these recall tasks are trivially solved and prove nothing about caching — shown both ways: 0016 (soft
  attention) and 0017 (unbounded cache makes fixed chunk=2 tie the oracle on Selective Copying). Every
  claim about *where to cut* must hold the cache/read budget fixed.
- **You cannot compress to constant memory for exact recall (0019).** Hierarchical re-compression loses
  to plain dropping because merging distinct facts makes them unrecoverable; exact multi-key recall is
  intrinsically ~O(N). Compression only fits lossy-tolerant tasks (summarization/semantic retrieval).
- **Small scale.** Track B is ~6M from-scratch on synthetic MQAR; Track A RULER is 370m (OOD past 2k).
- **Free-gen + read-out is slow** past 2k (RULER 4k/8k +SSC timed out) — needs a batched/intermediate-
  state kernel.
- Mamba-3 deferred (needs `mamba_ssm` SISO kernels); GDN router *training* at head_dim=256 kernel-blocked.

## 5. Where we stand
**Established.** (A) A trained hard-top-k read-out over a frozen model's cached state gives a real,
size-scaling net win on single-needle long-context, transfers to RULER, and shows a net win even under
real-text free generation at 2048 — but is single-needle-only and not constant-memory. (B) Multi-key
recall is solved from scratch by GDN-2 + read-out over the **true** cached state when boundaries are
per-fact, and those boundaries are **learnable and content-adaptive with supervision** — now also
**validated on the standard Selective Copying benchmark** (0017: perfect boundary recovery). (C) Two
limits are now pinned down empirically: unsupervised segmentation does not yet work (0016/0018), and
exact-recall memory cannot be compressed below ~O(N) (0019).

**Open.** Unsupervised boundary learning — the live lead is a **frozen/stop-grad density signal** (0018:
protect the 0.40 surprisal signal from co-training drift); scaling free-gen+read-out past 2k (kernel);
**retrieval-not-merge** bounded memory (0019: compress the index, materialize top-B values on demand);
MAD noisy/fuzzy recall; true per-row recurrent-state cache; LongBench.

**Report map.** Track A = 0001–0011, 0014; Track B = 0012, 0013, 0015–0019; figures = artifacts
`seg-length-dist`, `adaptive-boundaries`, `selcopy-seg-length-dist`; this file = synthesis.
