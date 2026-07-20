# Research state — where we are (2026-07-12)

Honest, calibrated summary of the memory-routing investigation. Separates what is **established**
(in-distribution, defensible) from what is **confounded** (discarded). Detail: `report/0001`–`0021`,
`notes/research-center.md`, `notes/established-facts.md`, and `SSM_Rank_Analysis/REPORT.md` (upstream
diagnostics).

## The problem
Fixed-state linear-recurrent LMs (Mamba-2, GDN) run in O(1) state — the efficiency win over attention —
but a bounded state holds only ~C key–value associations before interference. Long context carries more
content ⇒ more keys ⇒ long-context recall collapses. (`SSM_Rank_Analysis/REPORT`: **capacity = recall**,
load-limited; **eRank ≠ capacity**, anti-correlated with recall under load.)

**Question.** Can we extend a fixed-state recurrent LM's effective long-context recall **without giving
up its efficiency** (without reverting to O(N) attention)?

## The approach (SSC)
Freeze a pretrained recurrent LM; cache its recurrent state at checkpoints along the sequence; add a
small **trained hard-top-k read-out router** that retrieves from the cached states at query time. No
backbone retraining. (From-scratch co-trained variant = DSC, Track B.)

## Established (in-distribution, defensible)
1. **Single-fact long-context: SSC helps.** Passkey@8k vanilla→+SSC 0.738→0.986 (1.3b), 0.500→1.000
   (2.7b); win grows with size; transfers zero-shot to RULER `niah_single`; only hard top-k generalizes
   (0005/0007/0008/0009). Reproduced on real mamba2-370m: niah_single 0.69→0.73/0.78 (0021).
2. **Multi-fact: SSC HURTS.** Real mamba2-370m, RULER `niah_multikey`: vanilla 0.45 → +SSC 0.13 @2048
   (0021). Segmenting hurts where vanilla is competent (0010).
3. **The multi-fact failure is a ROUTING failure (mechanism, the key finding).** Routing probe on the
   real frozen model (0021): for the queried key among K fact-chunks, the router's top-1 is **1.0 when
   K=1** (single) but only **~0.27 (≈2× chance) when K≈8** (multi); the confusion is with **other
   fact-chunks, not filler** (`top1_amongkeys == top1_all`). So single works only because there is one
   candidate; multi fails because the router cannot select *which* cached fact the query wants. This is
   **query-conditioned selection failure**, not a per-chunk capacity limit (each chunk holds few keys).
4. **Not constant-memory.** The win needs ~O(N) snapshots; capping to B degrades ∝ B/N (0011). SSC
   trades attention's O(N) compute for a cache's O(N) memory — not the O(1) dream.
5. **From-scratch co-training fails.** Unsupervised boundary / density / hierarchical-merge all fail
   (0016/0018/0019/0020); DSC's memory gate collapses at scale (long-gdn). Cause: the next-token
   objective doesn't reward the memory/routing path, so it atrophies (avoided only by freezing the
   backbone, or by making memory the recurrent substrate à la MoM).

## Confounded / discarded (prove nothing — do not cite)
- **qa_1 (natural QA), SSC 0.53→0.41:** the head is passkey-trained, so natural QA is **OOD** for it.
  Tells us nothing about the method on natural language.
- **chunk-size sweep (128/64/32):** the head was trained at chunk 256, so other sizes are **OOD**; also
  finer chunks just multiply candidates. Inconclusive.
- Both share the same flaw (out-of-distribution head). The **chunk-256 routing probe is the only
  in-distribution** measurement and is what finding #3 rests on.

## The crystallized open problem
**Query-conditioned selection of the *right* fact from a compressed/cached memory.** Every failure has
this shape: a cache entry says "a fact is here" but not "this is the fact you asked for." It works for
one fact (K=1) or discrete-token facts (synthetic RULER); it breaks when the query must pick among many
**semantically** similar facts. Known fixes exit the premise:
- finer chunks / per-key vectors → synthetic-specific; `SSM_Rank_Analysis/REPORT` §3 shows natural-text
  chunking is degenerate;
- exact per-token retrieval (late-interaction / kNN) → works on natural text but is O(N) exact storage
  = Memorizing-Transformers, abandoning the efficiency thesis.

So the honest frontier is a **tension**, not a bug: *sub-O(N) memory × exact multi-fact recall ×
natural semantics* may not be jointly achievable in the snapshot-route paradigm.

## Scope / limits of everything above
Synthetic RULER, small models, a synthetic (passkey)-trained head. **Natural-language transfer has
never been fairly tested** (would require a head trained in-distribution for the natural target). The
single-fact win may also be partly inflated by low-information filler (`REPORT` §2) — not cleanly
controlled here.

## What a fair next experiment needs (not yet run)
To ask "does caching help *natural-language* recall," the router must be **trained on data matched to
the evaluation** (natural or synthetic+natural mix) so the eval is not OOD; then compare vs vanilla with
identical scoring, without cross-comparing mismatched tasks. This is a training experiment, not a
zero-shot eval. Open decision: pursue (a) that natural-language test, or (b) further harden the
in-distribution H2 mechanism finding.

## Not the goal
"Beat RULER-multikey." That is a benchmark, not the problem. The problem is the capacity limit and
whether query-conditioned retrieval from compressed memory can overcome it for real content.
