# Research center — what problem is this? (north star, 2026-07-10)

_Re-anchor after many threads (0012–0021, axes 1–3, DSC, SSM_Rank_Analysis). One problem, one figure._

## The one problem
**Fixed-state linear-recurrent LMs (Mamba-2, GDN) have an associative-capacity limit.** They run in
O(1) state (the efficiency win over attention), but a bounded state holds only ~C key–value
associations before interference; long context carries more content ⇒ more keys ⇒ recall collapses.
(`SSM_Rank_Analysis/REPORT`: capacity = recall, load-limited; eRank ≠ capacity.)

**The question:** can we extend the *effective* long-context recall of a fixed-state recurrent LM
**without giving up its efficiency** (i.e. without reverting to O(N) attention)?

## The approach under test
**Snapshot → route → reuse.** Cache the recurrent state at checkpoints along the sequence; add a small
trained read-out router that, at query time, retrieves from the cached states. (SSC = Sparse Selective
Caching, Track A. from-scratch co-trained variant = DSC, Track B.)

## What we've answered
- **Single-fact long-context: it works** — trained hard-top-k read-out recovers the one needle from a
  cached state; net win grows with model size (0.738→0.986; up to 0.969@512K), transfers zero-shot
  across RULER lengths. **Caveat under test (qa_1):** RULER niah pads with *inert* filler, so the win
  may be partly "the needle isn't interfered with." qa_1 (natural high-information distractors) is the
  honest test of whether the win survives real content.
- **It is not constant-memory (0011):** needs ~O(N) snapshots; capping degrades ∝ B/N. So the
  efficiency dream (O(1) inference) is *not* achieved by SSC; it trades attention's O(N) compute for a
  cache's O(N) memory.
- **Multi-fact: it fails** (SSC < vanilla; 0021 on real mamba2-370m: 0.452→0.129). Mechanism: the
  router cannot select *which* cached checkpoint holds the queried fact — among the K fact-chunks it
  lands right only ~2× chance, and the confusion is with *other fact-chunks*, not filler (H2,
  query-unconditioned pooled descriptor). Single is immune only because K=1.
- **Learning the memory path from scratch fails** (0016/0018/0019/0020, DSC alpha-collapse): the
  short-range next-token objective doesn't reward the memory/routing path, so it decays. Avoided only
  by freezing the backbone (Track A) or making memory the substrate (MoM).

## The crystallized open problem (the real center)
> **Query-conditioned selection of the RIGHT fact from a compressed/cached memory.**

Every failure is the same shape: a cache entry (pooled state descriptor) says *"a fact is here"* but
not *"this is the fact you asked for."* It works when there is one fact (single-needle) or the facts
are discrete easily-separated tokens (RULER); it breaks when the query must pick one among many
**semantically** similar facts. And the known fixes all exit the premise:
- finer chunks / per-key vectors → RULER-specific, doesn't transfer to natural text (no discrete facts;
  `SSM_Rank_Analysis/REPORT` §3: natural-passage chunking degenerate);
- exact per-token retrieval (late-interaction / kNN) → works on natural text but is O(N) exact storage
  = Memorizing-Transformers, abandoning the O(1)/constant-memory thesis and offering thin novelty.

So the honest frontier is a genuine tension, not a bug to patch:
**extend fixed-state recall for multi-fact natural-language queries under a sub-O(N) memory budget** —
and our evidence says the second and third clauses are in conflict for *exact* multi-fact recall.

## What the current experiments feed into this center
- **qa_1 (natural single-fact):** does the single-fact win survive natural high-info content, or was it
  a filler artifact? (Tests the caveat on our one positive result.)
- **routing probe (which-key):** localised the multi-fact failure to query-conditioned selection (H2) —
  the center of the open problem.
- **chunk-size sweep:** diagnostic only — confirms (and REPORT §3 predicts) that granularity tricks are
  RULER-specific, not a natural-text solution.

## Honest framings available for a paper
1. **Positive + characterized limit (safe/strong):** cache-augmented read-out extends *single-fact*
   long-context recall with size-scaling gains; multi-fact is bounded by query-conditioned addressing of
   lossy memory — characterized, with the routing probe as evidence. Scope: not constant-memory.
2. **The tension as the contribution:** formalize why sub-O(N) memory + exact multi-fact recall +
   natural semantics cannot be had together in the snapshot-route paradigm (0011/0019/0021 + REPORT).

Not the goal: "beat RULER-multikey." That is a benchmark, not the problem.
