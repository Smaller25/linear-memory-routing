# Adaptive segment boundaries — research question, methods, benchmarks

_Design note, 2026-06-27. Sets the current goal, a menu of methods to test it, and whether our task
is the right one._

## The research question (now the goal) — REWRITTEN 2026-06-28

**Old framing (0012–0016).** "Can the model *learn where to place boundaries* when information sits at
irregular positions?" — answered: yes with oracle distillation (0015), no unsupervised (0016, three
failures). The trouble is the framing itself: a per-token **boundary classifier distilled from oracle
positions memorizes the *position distribution* of facts** — brittle to any shift in gap statistics or
content, and undefined without labels. That is exactly the fragility intuition flagged on 2026-06-28.

**New framing — information-density bit allocation.** Treat the segment cache as a **fixed bit budget**
and allocate it by *information content*, not by a learned position label. Stop asking "is token t a
boundary?" and start asking "where is the information, and how should a bounded memory spend its bits on
it?" Two orthogonal axes:

> **(1) Spatial — *where to cut*.** Segment by an **intrinsic information-density signal** of the
> backbone — token surprisal (NLL), recurrent **‖Δstate‖**, or adjacent-state dissimilarity —
> thresholded to a **target firing rate**. Each segment then carries ≈constant *information* (variable
> length); the rule is content-driven, **transfers across distributions, and needs no oracle**.
>
> **(2) Temporal — *how to age*.** As a segment recedes from the current token, **re-compress it
> hierarchically** (recent = fine, distant = progressively coarser), so memory stays bounded (≈log N)
> instead of the flat O(N) cache that **0011** showed degrades ∝ B/N.

Unifying principle: **fine bits to recent/high-information regions, coarse bits to distant/low-
information regions.** (1) is spatial allocation by density; (2) is temporal allocation by recency.
They compose into one multi-resolution, information-equalized recurrent-state memory.

Status from reports 0012–0016 (what this reframing inherits):
- Regular MQAR is **degenerate** (facts at a fixed period → fixed `chunk=2` == oracle == 1.0).
- Irregular MQAR (`make_mqar_gapped`): **fixed stride fails (0.00)**, **oracle ≈ 1.0**, boundary head
  **distilled from oracle** recovers it (≈1.0, prec/rec 1.0, variable lengths) — but **supervised-only**.
- **0016 unsupervised failed three ways**; STE+L1 collapsed to 0 because L1 has no floor. → axis (1)'s
  **target-rate** loss (two-sided, = H-Net's ratio loss) is the direct fix; density signals also give a
  gradient *before any boundary forms*, which the oracle-distilled head never had.
- **0010/0011**: multi-key out of scope, not constant-memory (∝ B/N). → axis (2) is the direct answer.

## Axis-3 — segment by STATE SATURATION (the leading direction, added 2026-06-28)
**Origin.** 0019 killed axis-2's *merge*: a learned conv that fuses two cached states loses to plain
dropping, because **the recurrent update rule (GDN/Mamba) is already the right way to combine two
states into one** — a post-hoc conv is strictly cruder. The fix is not a better merge but **never
splitting then merging**: let one recurrent state accumulate over a *variable-length* span and only
decide **when to cut**. And 0018 showed the cut signal should not be output-side surprisal (it drifts
under co-training) — it should be read off the **state itself**.

**Mechanism.** Run the backbone's recurrent state; monitor a **fullness signal**; when the state
*saturates*, emit a boundary (checkpoint the full state into the cache, optionally reset), then start a
fresh state. Each cached state is thus a "full but not overflowing" unit built by the model's own
update rule — no lossy merge (fixes 0019), cut by an intrinsic structural signal (more drift-robust
than 0018), variable-length by information content (axis-1's goal, achieved structurally), and memory
≈ #facts / capacity ≪ O(N).

**Fullness = effective rank (connects to the earlier rank experiment).** Linear-attention / DeltaNet /
GDN states are `S = Σ kᵢ vᵢᵀ` — a sum of rank-1 updates. **effective rank(S) ≈ # distinct associations
stored**, and capacity ≈ head_dim. So "state is full" = rank approaches head_dim / rank-growth → 0 →
**cut here**. This turns the project's founding motivation (fixed state saturates at high kv) into a
*quantitative cut criterion*. Cheap proxies (no per-token SVD): **stable rank** `‖S‖_F²/‖S‖₂²`,
effective rank (singular-value entropy `exp(H(σ̃))`), nuclear norm, or the increment `‖ΔS‖/‖S‖`.

**Already measured — `Smaller25/SSM_Rank_Analysis`.** That repo establishes the exact signal axis-3
needs: the **effective rank of the SSM hidden state saturates with context length** — rank rises then
**plateaus at a per-head threshold T\***. So "state is full" is not hypothetical; it is the measured
plateau, and **T\* is the cut point**. It also finds (i) **head heterogeneity** (Type A/B/C heads with
different saturation curves → the cut must aggregate across heads, or trigger on a chosen head set),
and (ii) **state injection replicates oracle retrieval** — independent evidence that a cached state is
a usable context proxy (grounds our read-out). Caveat: it's **Mamba-2 (370m)**; we must **port the
effective-rank measurement to GDN-2's state** (`S = Σ kᵢvᵢᵀ`, heads × headdim × d_state).

**Open questions (for ultraplan).** (a) GDN-2 state object + which fullness proxy (align with the repo's
**effective rank**, plus cheap stable-rank / `‖ΔS‖` variants); (b) cut trigger parameter-free (detect
the rank plateau T\*) vs lightly learned — parameter-free sidesteps the 0018 co-training drift; (c)
head heterogeneity: per-head vs aggregated cut (Type A/B/C); (d) post-cut policy (hard reset vs
carry/decay residual); (e) reuse the repo's effective-rank-vs-T curves / T\* as the calibration for
"full", and its state-injection result as the read-out sanity check.

## Two new directions vs the current method (comparison)
| | current (boundary head) | (1) info-density segmentation | (2) hierarchical re-compression |
|---|---|---|---|
| decides | where to cut (per-token classify) | where to cut (threshold a signal) | how to age old segments |
| signal | oracle-position labels → BCE | **intrinsic**: surprisal / ‖Δstate‖ / adjacent cos-dist | recency + over-budget merge |
| supervision | **needs oracle** | **self-supervised** | unsup (a policy) |
| d.o.f. | a full per-position classifier | **one scalar threshold** (+target rate) | #levels, merge operator |
| shift-robust | weak (memorizes positions) | **strong** (content-driven, transfers) | neutral |
| fixes which negative | — | **0016** (unsup *where*) | **0010/0011** (not constant-memory) |
| prior art | — | **BLT** entropy byte-patching (Meta'24); **H-Net** dynamic chunking, ratio-loss+STE ('25) | **Compressive Transformer** (Rae'20); **∞-former** sticky memory |
| cost / risk | oracle required | tune threshold/target-rate | lossy → far-needle recall drop (must quantify) |

Axis (1) directly attacks the **live** failure (unsupervised segmentation, 0016) and is testable on the
existing MQAR/selcopy harness → **do it first**. Axis (2) attacks the memory-scaling negative (0011),
needs a multi-segment cache + a learned merge operator, and is measured by the B/N degradation curve.

## Methods to learn the boundaries (a menu)
**Supervised reference (done, 0015).** Boundary head distilled from oracle positions (BCE). Upper
bound on learnability; not deployable (needs labels).

**Unsupervised (the real target):**
1. **Landmark-biased read + L1 budget** (running now, `--chunk-mode unsup`). Head predicts a per-token
   landmark prob `p`; the read-out is a causal attention over all tokens **biased by `log p`** (so the
   task loss gradient reaches the head), with an L1 penalty `budget·mean(p)` for sparsity. Train soft,
   threshold `p` for hard boundaries at inference. No oracle. *Tests: does task-loss + sparsity alone
   put boundaries on facts?*
2. **Straight-through hard boundaries + budget.** Forward hard (threshold), backward via the sigmoid
   (STE); budget penalty toward a target boundary rate. Keeps the discrete segment-cache read; cheaper
   read than (1) but weaker gradient.
3. **Gumbel-sigmoid boundaries.** Stochastic, reparameterized boundary draws + budget; anneal
   temperature. More faithful gradient than STE, more variance.
4. **Surprisal / state-change trigger (parameter-free or lightly trained).** Fire a boundary when the
   backbone's own signal spikes: token NLL (frozen lm_head), or `‖Δ recurrent-state‖`, or input-gate
   magnitude. Cheap, interpretable; test whether "high information ⇒ boundary" suffices. (Our earlier
   fixed-threshold surprisal failed on *regular* MQAR; revisit on irregular with a learned threshold.)
5. **Auxiliary reconstruction objective.** Train the head so each segment's cached summary can
   reconstruct/predict its own tokens — a boundary is "good" if segments are self-contained. Unsup,
   no task-recall signal needed.

**Architectural alternatives to "predict a boundary":**
6. **Soft / overlapping segments.** Drop hard cuts: each token contributes to a cache slot with a
   learned soft assignment (differentiable end-to-end) — sidesteps the discreteness entirely.
7. **Top-m landmark selection.** Instead of per-token boundaries, directly select the m most
   "cache-worthy" tokens (differentiable top-k / perturbed-topk). m is the budget.
8. **True recurrent-state cache (per-row).** Orthogonal to *where*: replace the boundary-hidden proxy
   with the actual GDN-2 state at each (irregular, per-row) boundary — needs a batched/intermediate-
   state kernel (0012 milestone 3). Pairs with any of 1–7.

Evaluation for all: recall vs kv on irregular MQAR (vs fixed-stride floor and oracle ceiling) **plus**
boundary precision/recall against the true value positions **plus** the segment-length distribution
(should be variable and track the gaps).

## Is MQAR + random-filler the right benchmark? (alternatives)
Our hand-rolled "MQAR + random filler" is a reasonable probe, but it reinvents two **standard**
synthetic tasks that target exactly "find content at irregular positions among noise":

- **Selective Copying** (Mamba, Gu & Dao 2023; impl `MinhZou/selective-copying-mamba`). Copy a set of
  marked tokens scattered at **random spacing** among noise — the canonical content-vs-position test;
  LTI/fixed models fail, selective models pass. **The closest match to our question**, and standard.
- **MAD suite** (Poli et al. 2024, `athms/mad-lab`): synthetic tasks (compression, in-context recall,
  **noisy recall**, **fuzzy recall**, selective copying, memorization) with an explicit **noise-ratio**
  knob — the established battery for probing exactly these skills, designed to predict at-scale
  behavior. **Adopt this** rather than our ad-hoc generator: comparable, knobbed, and credible.
- **RULER `variable_tracking`**: chains of variable assignments at irregular positions (state must be
  tracked through them) — a real-text-ish irregular-structure task we already vendor.
- **Real long-context (LongBench/SCROLLS):** natural variable structure; the ultimate test, but
  confounded (many skills at once).

**Recommendation.** Keep `make_mqar_gapped` as the controlled in-house probe (we own every knob), but
**validate the method on Selective Copying + MAD noisy/fuzzy-recall** (standard, with noise-ratio
sweeps) so the adaptivity claim rests on recognized benchmarks, not a bespoke one. RULER-VT for the
state-tracking flavor; LongBench last.

## Plan (reframed 2026-06-28 — information-density allocation)
**In flight:** validate the *supervised* method on Selective Copying (jobs `selcopy-{vanilla,fixed,
oracle,learned}`) → report 0017. This is the upper-bound/transfer check; the directions below replace
the supervised head with self-supervised allocation.

**Axis (1) — info-density segmentation (do first; attacks 0016):**
1. Add density signals to the trainer: token surprisal (backbone NLL), ‖Δ recurrent-state‖, adjacent-
   state cosine distance. Emit a per-token density `d_t`; boundary = `d_t` over a threshold.
2. **Target-rate loss** instead of L1 (the 0016 collapse fix): penalize `(mean(fire) − ρ)²` toward a
   target rate ρ = budget/seq-len, two-sided so it can't collapse to 0. STE/smoothing for the discrete
   cut (cf. H-Net ratio loss).
3. Evaluate on irregular MQAR + Selective Copying: recall vs kv (between fixed-floor and oracle-ceil),
   boundary precision/recall vs true facts, **threshold-free top-k(p)∩facts** (the 0016 metric), and
   the seg-length distribution. Crucial: **does it find facts with NO oracle?** (the 0016 open problem).
4. Distribution-shift test (the robustness claim): train on one gap statistic, eval on another; the
   density rule should transfer where the distilled head (0015) does not.

**Axis (2) — hierarchical re-compression (attacks 0011):**
5. Multi-level cache: a fine FIFO of recent segments; when over budget B, merge the oldest adjacent
   pair via a learned coarsen operator → next level (recursively → ≈log N levels).
6. Measure the **recall–memory tradeoff**: needle-vs-distance accuracy as a function of B and #levels,
   against the 0011 flat-cache ∝B/N curve. Honest cost = far-needle degradation from lossy coarsening.

**Shared:**
7. Pair the winner with the true per-row recurrent-state cache (method 8) and port to **MAD
   noisy/fuzzy recall** for a standard, noise-knobbed benchmark.
