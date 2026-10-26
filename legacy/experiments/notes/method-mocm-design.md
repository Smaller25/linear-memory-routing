# Method design — MoCM: Mixture of Cached Memories (parallel × temporal)

_Design doc, 2026-06-18. From-scratch method combining MoM's parallel-memory axis with MC's
temporal-caching axis, routed by the hard top-k read-out we found to be the only one that
generalizes (report 0008/0009)._

## Motivation (why a 2-axis method)
The two prior mechanisms attack different failure modes on *different axes*, and each leaves the
other unsolved:
- **MoM (parallel axis):** M independent memories + a write-router → reduces **interference** between
  concurrent facts (multi-key / MQAR). Does nothing for length.
- **MC (temporal axis):** cache the memory state per segment → recover **old info under long-context
  saturation** (single-needle). Does nothing for interference (one state still mixes co-occurring
  facts — exactly our multi-key failure, report 0010).

Our own results pin this: temporal-only SSC wins single-needle long-context (passkey 8k +0.25→0.50,
RULER niah_single) but collapses on multi-key (MQAR-like). The gap is precisely the parallel axis.
**MoCM unifies both** so one model handles long-context *and* multi-fact interference.

## The mechanism
Base update rule: **gated delta rule** (GDN — strongest single-memory recaller on MQAR, RESULTS.md).
A MoCM layer keeps a **2-D memory bank**: `M` parallel memories, each cached at every segment
boundary → an `M × N` grid of state snapshots (`N` = #segments so far), plus the `M` online states
and one shared always-on memory.

1. **Write-routing (parallel axis, MoM).** Router `W_w: x_t → ℝ^M`; each token writes to its top-`k_w`
   memories (e.g. k_w=2 of M=4) with the gated-delta update; non-selected memories are unchanged
   (interference isolation). One **shared memory** receives every token (global context). Per-memory
   k/v/β/g projections (or shared, ablate `single_kv_proj`).
2. **Temporal caching (MC axis).** Sequence split into segments of size `C` (e.g. 256). At each
   boundary cache each memory's final state → bank `{h^{(s,m)}}`, s=1..N, m=1..M. Within a segment the
   M memories run from their cached state (online); the bank is frozen (detached), as in our runner.
3. **Read-routing (the crux — hard top-k over the 2-D bank).** For query token `t`, score all
   `M·N` cached snapshots + the M online states by `⟨W_r x_t, descriptor(h)⟩`, take **hard top-`k_r`**
   (SSC-style; our evidence: dense AoM / slot-merge MoM / coarse-hierarchical all collapse, only hard
   top-k generalizes — reports 0008/0009; sweet spot k∈[2,8], report 0009). Read-out = the online
   contribution + Σ over the selected snapshots, `o_t = Σ_{(s,m)∈TopK} w · (q_t · h^{(s,m)})` + shared.
   Switch load-balance aux on both routers (**small `aux_scale`** — report 0010 showed a layer-summed
   aux dominates and blocks selectivity; use ~1e-4 or average over layers).
4. **Trained FROM SCRATCH end-to-end**: recurrence + both routers + memories jointly. This is what
   lets the parallel memories specialize (impossible to bolt onto a frozen single-state backbone —
   report's parallel-axis analysis).

## Complexity
- Parallel: `M×` state (constant in length). Shared: +1.
- Temporal bank: `M·N` snapshots, but read-out fan-in is **constant `k_r`** (hard top-k), so read cost
  is O(k_r), not O(M·N). Cache memory grows O(M·N); bound with hierarchical/segment-size or a cap.
- Training stays linear-time per segment; bank reads are top-k. (Watch the O(N²) read-recompute that
  bit us at fine chunk — score via cheap pooled descriptors, read only the k_r selected.)

## Why it should beat both parents
- vs **MC/SSC**: adds parallel memories → handles multi-key interference (the regime SSC failed).
- vs **MoM**: adds temporal cache → handles long-context saturation (the regime MoM ignores).
- vs both: the **unified hard-top-k router over the (parallel × temporal) bank** is the new object;
  our ablations already show hard-top-k is the right selector.

## Evaluation (MQAR = main figure; frozen-SSC = ablation)
- **Main: MQAR from-scratch** (standard protocol: T∈{512,1024,2048,4096}, K≈T/4), small models, compare
  **MoCM vs vanilla {mamba2, GDN}, vs MoM (parallel-only), vs MC/SSC (temporal-only)**. Reuse the
  sibling `evaluation/zoology_mqar` Track-B infra (2-layer from-scratch mamba2+GDN already wired).
- **Long-context:** passkey / RULER niah_single (single-needle) AND niah_multikey (multi-key) — MoCM
  should win *both*, unlike SSC (single only) or MoM (multi only).
- **Ablations:** M=1 → MC/SSC (temporal only); N=1 → MoM (parallel only); read-router = top-k vs dense
  vs merge; k_w, k_r, M, C sweeps; shared-memory on/off; aux_scale.
- **Frozen retrofit** (our prior setting) appears as a *special-case ablation*: temporal axis (SSC) can
  be added post-hoc to a frozen backbone (our +0.25/+0.50 results), but the parallel axis needs
  from-scratch — quantifies what co-training buys.

## Build plan (sketch)
- New `lmr/layers/mocm.py`: MoCM layer (write-router + M gated-delta memories + shared) reusing FLA's
  `chunk_gated_delta_rule` per memory; a 2-D read-router head reusing `lmr/readout.py` SSC logic over
  the M·N+M bank.
- From-scratch trainer on MQAR: extend `evaluation/zoology_mqar` Track-B (2-layer model, d_model sweep)
  to drop in a MoCM mixer; compare vs mamba2/GDN/MoM baselines.
- Start tiny (2-layer, d_model 128, M=4, C=64, MQAR T=512→2048) to validate the mechanism cheaply
  before scaling.

## Open risks
- Read-recompute cost over the bank (O(N²)) — score by pooled descriptor, read only top-k_r; cap N.
- Two routers + aux balancing can be finicky (report 0010) — keep aux tiny, maybe warm-up.
- Novelty hinges on the *2-D bank + unified hard-top-k router*; MoM(parallel) and MC(temporal) alone
  are prior work — the combination + the selector-ablation evidence is the contribution.
