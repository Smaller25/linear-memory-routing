# linear-memory-routing — report index & narrative

**Thesis (frozen story).** Take an *already-pretrained, frozen* linear-recurrent LM (Mamba2, Gated
DeltaNet) and add a small **trained read-out router over cached recurrent-state checkpoints** — no
backbone retraining. A hard top-k router (**SSC**) lets the frozen model **exceed its fixed-state
recall exactly where that state saturates** (long-context single-needle), at the cost of a ~30M-param
router instead of retraining a 1.3–2.7B model.

## Headline results
- **Net win at the collapse length.** Natural-language passkey @8k: vanilla→+SSC = **0.738→0.986
  (+0.25) @mamba2-1.3b**, **0.500→1.000 (+0.50) @mamba2-2.7b** — the win **grows with model size**
  (0005, 0007).
- **Validated on the standard benchmark.** SSC trained on our passkey **transfers zero-shot to RULER
  `niah_single`** and beats vanilla (+0.06–0.18 @4k/8k) (0009).
- **Only hard top-k generalizes.** RM(no-train)/GRM/AoM(dense)/MoM(slot-merge)/hierarchical all
  collapse at long context; **only SSC's sparse hard selection** survives (k∈[2,8]; k=1 too brittle)
  (0008, 0009).
- **GDN is a strong long-context recaller** (vanilla passkey 8k≈0.93 vs mamba2 0.74); GDN forward/RM
  measured, GDN router *training* blocked on this hardware (kernel) (0006).

## Honest scope (where it does NOT help)
- **Multi-key recall (MQAR-style):** SSC does not help — where vanilla is competent, segmenting hurts;
  where it's hard, both floor. Segment-level routing can't disambiguate keys clustered in a segment;
  this is the *interference* regime (MoM's domain), not the *saturation* regime (0010).
- **Not constant-memory:** the win needs ~all O(N) segment snapshots; capping the cache to a constant
  B degrades recall ∝ B/N (an evicted needle is unrecoverable). Top-k cuts the *read* to O(N·k) but
  the cache stays O(N) — SSC is a compressed-cache point on the RNN↔attention spectrum, not a
  constant-memory linear model (0011).

## Reports
| # | topic | result |
|---|-------|--------|
| 0001 | scaffold (cloud) | MC library on FLA Mamba2 |
| 0002 | A100 validation + Phase 0 | training-free MC-RM is neutral-to-negative (frozen ≠ trained-for) |
| 0003 | Phase 1 GRM | trained gate recovers RM's collapse |
| 0004 | longer-context GRM | generalizes only ~2× trained #segments |
| 0005 | **SSC** | **first net win @8k (+0.25)** |
| 0006 | GDN backbone | GDN long-context-robust; router training kernel-blocked |
| 0007 | **size scaling** | **+0.50 @2.7b — win grows with size** |
| 0008 | mechanism comparison | only SSC generalizes; AoM/MoM/hier collapse |
| 0009 | top-k sweep + **RULER** | k∈[2,8] robust; **SSC zero-shot transfers to RULER niah_single** |
| 0010 | multi-key | SSC's win is scoped to single-needle (multi-key = out of scope) |
| 0011 | bounded cache | SSC's win needs ~full O(N) cache; capping degrades ∝ B/N → NOT constant-memory |
| 0012 | **from-scratch Dynamic-MoSC** | **GDN-2 + learned per-fact boundaries solve multi-key MQAR (learned ≈ oracle, kv128 ~1.0)** — the 0010 regime, from scratch |
| 0013 | learned segment-length distribution | learned head segments **per fact (median len 2)**, count scales with kv (→512), recovering the oracle structure |
| 0015 | **adaptive vs fixed boundaries** | regular MQAR is degenerate (fixed chunk=2 == oracle == 1.0); on **irregular** MQAR fixed-stride **fails (0.00)** while learned ≈ oracle (~1.0, prec/rec 1.0) — boundaries are genuinely content-adaptive |

## Future work
- **MoCM (Mixture of Cached Memories)** — combine MoM's *parallel* memory axis with MC's *temporal*
  caching, trained **from scratch**, to cover the multi-key/interference regime SSC can't. Design in
  `notes/method-mocm-design.md`; a parallel-memory mixer + from-scratch MQAR trainer exist
  (`lmr/layers/mocm.py`, `lmr/scripts/train_mocm_mqar.py`). The standalone `lmr/tasks/mqar.py` is now
  **Zoology-faithful and validated**: a 2-layer FLA GDN learns it (recall k=8 0.999 → k=64 0.83) via a
  **delayed phase transition at ~2000 steps** (loss sits at random until then — earlier ≤1500-step runs
  hadn't transitioned, which is why they looked stuck). Future MoCM runs need ≥~3000 steps.
- GDN router training on H100 (tilelang) or a differentiable chunked scan; full RULER free-generation
  metric; longer contexts (16k/32k).
