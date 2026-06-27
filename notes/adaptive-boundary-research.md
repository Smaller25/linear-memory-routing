# Adaptive segment boundaries — research question, methods, benchmarks

_Design note, 2026-06-27. Sets the current goal, a menu of methods to test it, and whether our task
is the right one._

## The research question (now the goal)
> Given a linear-RNN backbone and a segment-cache read-out, **can the model learn *where* to place
> segment boundaries when the information sits at irregular, content-dependent positions** — i.e.
> recover variable-length "fact" segments rather than a fixed stride?

Status from reports 0012–0015:
- Regular MQAR is **degenerate** (facts at a fixed period → fixed `chunk=2` == oracle == 1.0).
- Irregular MQAR (`make_mqar_gapped`): **fixed stride fails (0.00)**, **oracle ≈ 1.0**, and a boundary
  head **distilled from oracle** recovers it (≈1.0, precision/recall 1.0, variable segment lengths).
- Open: that was **supervised** (oracle distillation) and a **single** method. We want unsupervised,
  and a menu of methods, and confidence the task is right.

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

## Plan
1. Unsupervised method (1) running on irregular MQAR — does it find facts without oracle?
2. If yes, sweep methods (2–5) and report boundary precision/recall + seg-length distribution.
3. Port the winning method to **Selective Copying** and **MAD noisy/fuzzy recall** (standard).
4. Pair with true per-row recurrent-state cache (method 8).
