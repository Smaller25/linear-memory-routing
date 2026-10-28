# linear-memory-routing — report index & narrative

> **Current honest state (start here):** [`RESEARCH_STATE.md`](RESEARCH_STATE.md) — established vs
> confounded as of 2026-07-12, the crystallized open problem (query-conditioned selection from
> compressed memory), and scope. The index below is the Track-A frozen-story narrative (0001–0021).

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
| 0014 | **RULER free-gen** (frozen mamba2-370m) | **+SSC net win on real RULER free-gen**: niah_single @2048 vanilla 0.00 → +SSC **36.7** (zero-shot, matched n); multikey near-floor (0010); 4k/8k +SSC timed out (free-gen too slow) |
| 0015 | **adaptive vs fixed boundaries** | regular MQAR is degenerate (fixed chunk=2 == oracle == 1.0); on **irregular** MQAR fixed-stride **fails (0.00)** while learned ≈ oracle (~1.0, prec/rec 1.0) — boundaries are genuinely content-adaptive |
| 0016 | unsupervised boundary learning | **fails 3 ways** (soft=attention-bypass, STE+L1=collapse, warm-start=drift to 0.00 top-k(p) overlap) — boundary signal needs oracle supervision; unsupervised is open |
| 0022 | **decorrelated-write (design idea A)** | write-into-idle-directions **helps multi-key recall ONLY under overload** (H1×16: kv64 0.081→**0.181** w/ decorrelation-reg C4, 2.2×); **neutral w/ slack, hurts near-saturation**. Winner=C4 (loss-only, free). erank↔recall co-move only under load (eRank≠capacity). Testbed has no decay → targets secondary lever only |
| 0023 | **idea A / C4 on REAL GDN-2 (w/ decay)** | **does NOT transfer — C4 hurts recall in every regime** (hd32 0.497→0.430; λ=0.1 0.497→0.480; hd16 blocks learning) and **collapses state erank (16→3–6)**, opposite of the decay-free 0022 toy. Decay is the dominant rank-limiter (F7); forcing isotropic keys fights the learned key↔gate coupling. **(a) fails → (b) finetuning not run.** Pivot to the **decay/retention** lever |
| 0024 | **MC-SSC multi-NIAH failure decomposition** (write/read/route/gen, GDN2-370M mean-pool) | **routing is the causal bottleneck**: write fidelity intact (single≈multi), routing broken worst for cross-segment keys (paired-D hit@2 .50–.56 vs chance .29; among key-bearing segments *below* coin-flip), and **oracle gold-chunk injection lifts all 4 cells** (mc-30B/D .062→**.562**). Read-side breaks only under same-segment key collision (readout cos vs single twin →.90–.93 late layers vs ≥.98 in D). 30B routes better (anchor MK-NIAH 2→32) but has the *most* oracle headroom → descriptors/routers for D + state-level read disambiguation for S |
| 0025 | **MC-SSC router diagnosis** — what does the router respond to? (X1 descriptor-only probe + `u:=q`, X2 multiquery/multivalue (M)-test; **X4 진행 중**) | **claim (M) "router = needle detector, not key discriminator" is REJECTED**: on multivalue (1 key in context — *nothing* to discriminate) hit@2 is .435/.442 vs a needle-null of .723, **~4 SE below** a baseline that assumes needle detection is already solved; multiquery .503/.520 vs null .750. Essay-haystack control `single_2` skill .74/.78 → multiquery .24/.29 (needle-multiplicity cost ~2.5x the haystack-type cost). Both cheap fixes die: **`u_t:=q_t` gives no gain on paired-D** (.563→.500 / .500→.500; paired-S **−18.75pp both models**) and **geometry post-hoc correction fails** (centering is an algebraic no-op for top-k; top-1 PC removal is null-to-negative on paired-D despite gram off-diag .96–1.0). Local H_blind confirmed: layers 2–13 have descriptor-only R²≥0.9 in 8/8 cells (`pos` alone, median R² .985) while stock hit@2 peaks at layers 14–15 where R² is lowest. Bottleneck is the **descriptor**, upstream of key discrimination → X3/X5 held; only contrastive router supervision + WY/UT-geometry descriptors remain |

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
