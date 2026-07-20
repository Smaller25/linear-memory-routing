# Descriptor / multi-key failure — hypotheses & experiment design

_2026-07-09. Why frozen SSC fails multi-key. Hypotheses (H1–H8) + per-hypothesis experiments.
Facts in `established-facts.md`; this is the hypothesis+design layer._

## The pivotal observation (routing vs extraction)
Multi-key @4K: **any-layer-hit 0.95** yet **end-to-end 0.42** (SSC). The gap has two causes, to be
separated:
- **any-hit ≠ clean routing.** any-hit = "≥1 of 24 layers had the right chunk in top-k". But per-layer
  max rate is **0.65** for multi (single has 4 layers at **1.0**). The read-out aggregates ALL layers,
  so most-layers-wrong pollutes the read.
- **extraction loss.** Even given the right chunk, the pooled read can't isolate the *queried* key →
  the 0.65→0.42 gap.
So multi-key fails on BOTH (a) noisy cross-layer routing and (b) within-chunk pooled extraction. The
first experiment must isolate these.

## Hypotheses
- **H1. Multiplicity collapse (pooling).** Pooling a chunk → 1 vector loses per-key identity. AUC 0.92
  (binary needle-chunk) but retrieval 0.65 (specific key). within-chunk state unsaturated (facts D2) →
  info is there, pool kills it. Fix candidate: per-key multi-vector.
- **H2. Query-independence.** Descriptor fixed before the query → chunk with K1 scores as high for a
  K2 query. Fix candidate: late-interaction (MaxSim) at read time.
- **H3. State-derived vs raw-KV.** Descriptor from the mixed recurrent state / mixer output `h_t`
  inherits mixing; pre-mixer `k_t/v_t` projections keep per-token resolution.
- **H4. Key-vs-value encoding.** Routing needs query→KEY match; DSC 3-view are V-axis statistics
  (value-ish). Descriptor may under-encode keys.
- **H5. Chunk granularity (c=256).** Multiple keys per chunk compete; F1-G size-sweep failed (partial
  counter-evidence) — re-examine with per-key metrics.
- **H6. Layer specialization/aggregation.** Only some layers route well (multi max 0.65 @layer 15);
  aggregate dilutes. Best-layer may ≫ aggregate.
- **H7. Softmax/top-k dilution.** Many chunks → routing prob spreads thin (needle prob 0.19→0.028);
  top-k=2 misses even a rank-1 needle.
- **H8. Non-metric space / magnitude.** cos-sim random (disc 1.002); signal only in a learned
  direction; dot-product may let high-norm filler dominate.

## Experiments (shared frozen GDN-1.3B + trained SSC head; RULER niah_multikey_1 @4K unless noted)
Most are **learning-free** (forward + probe); one forward-collection feeds several.

- **E0 — route-vs-extract split (answers the pivotal Q; gates the rest).**
  Compare end-to-end multi-key acc under: (a) normal SSC routing, (b) **oracle routing** (force the true
  needle chunk into top-k, keep the pooled read), (c) oracle routing + **per-position/attention read**
  within the chunk (bypass pooling). (a→b) isolates routing loss; (b→c) isolates extraction loss.
- **E1 (H1) — pooled vs per-position separability.** In the oracle chunk, linear-probe / MaxSim for the
  queried key's value from (i) the pooled descriptor vs (ii) per-position states. Δ = multiplicity loss.
- **E2 (H2) — query-conditioned vs independent.** On the same cached descriptors, routing by
  query-independent dot-product vs query-conditioned MaxSim; measure needle-in-topk.
- **E3 (H3) — descriptor source.** Per-key separability (AUC/MaxSim) from pre-mixer `k/v` vs
  state-derived, same positions.
- **E4 (H4) — key vs value probe.** Probe the chunk descriptor for its keys vs its values; report which
  is more decodable.
- **E5 (H6) — per-layer decomposition.** Per-layer routing rate AND per-layer oracle-chunk extraction;
  is best-layer ≫ 24-layer aggregate? (reuses 7Q's per-layer capture.)
- **E6 (H7) — top-k / temperature sweep.** Needle rank distribution; fraction "rank-1 but outside
  top-2"; recall vs k and softmax temperature.
- **E7 (H5) — chunk-size, per-key.** Re-analyze F1-G size-sweep with per-key retrieval/extraction, not
  aggregate acc.
- **E8 (H8) — normalized / metric.** Routing with L2-normalized descriptors vs raw; cos vs learned proj.

Priority: **E0 first** (splits routing/extraction → tells us whether to chase H1/H6 (extraction) or
H2/H7 (routing)). Then E1–E3 (the multi-vector premise), E5 (layers). E4/E6/E7/E8 as follow-ups.

## Report target
`report/0021` — per-hypothesis verdict table (supported / rejected / partial + the number), with E0 as
the headline decomposition. Facts feed back into `established-facts.md`.
