# Task 6 report — E1: per-layer routing accuracy

## Status: DONE (revised after review — see "Revision 1" and "Revision 2" below)

## Revision 2 (e2_join model-collision bug, found during Task 7 review)
`_load_e2_baseline` built its `(condition, pair_id) -> correct` map from
`results/e2_oracle.json`'s flat top-level `"rows"` list without a model key.
That list concatenates both models' baseline rows back-to-back
(model-grouped: all mc-30B rows, then all mc-5B rows), so keying only by
`(condition, pair_id)` meant every collision was won by whichever model's
row appeared later in the list — mc-5B, since it's appended second. Both
models' `e2_join` tables in `e1_routing.json` were silently reflecting
mc-5B's baseline correctness, even under the `"mc-30B"` key. (Task 7's own
report had guessed the opposite direction — that mc-30B would win the
collision — which was checked and confirmed wrong; it's mc-5B.)

Fixed by reading the model-scoped nested structure directly —
`e2_oracle.json["results"][model][condition]["baseline"]["rows"]` — inside
`_compute_e2_join`, loading a separate baseline map per model instead of
one shared flat map. Verified against `e2_oracle.json` directly: mc-5B
baseline S 2/16, D 3/16; mc-30B baseline S 0/16, D 1/16. Re-ran
`sbatch/e1.sbatch` (~2 min); the regenerated `e1_routing.json["e2_join"]`
now shows, per model, hit/miss bucket totals that sum to exactly these
counts:
- mc-30B: `paired_S_multi` hit n=12 (0 correct) + miss n=4 (0 correct) = 0/16; `paired_D_multi` hit n=8 (0 correct) + miss n=8 (1 correct) = 1/16.
- mc-5B: `paired_S_multi` hit n=13 (1 correct) + miss n=3 (1 correct) = 2/16; `paired_D_multi` hit n=9 (0 correct) + miss n=7 (3 correct) = 3/16.

hit@2/gold_rank/amongkeys/chance_hit2 numbers in this rerun are byte-for-byte
unchanged from Revision 1 (the bug only ever touched `e2_join`).
Commits: `1f5a0e02` (code), `5a84af7a` (results).

## Revision 1 (post-review fixes)
A reviewer confirmed the routing math itself is exact but flagged three
issues, all fixed and re-run (job took ~70s):
1. **Bug** — `amongkeys` was being computed even for structurally-ineligible
   samples (`gold_seg >= cur_seg`), where it is deterministically `False`
   (gold's own segment is never in `eligible_key_segs`). Fixed:
   `amongkeys` is now `None` whenever `eligible=False`, in addition to the
   existing `<2` eligible-key-segment gate. This moved
   `niah_multikey_1` `amongkeys_n` from 50→47 and, at mc-5B's best layer,
   accuracy from a (wrong, deflated) value to 25/47 ≈ 0.532.
2. **Meta** — chance level for hit@2 is dataset-specific (depends on how
   many past segments exist), not a fixed ≈0.29. Added `chance_hit2` (mean
   over eligible samples of `min(1.0, 2/cur_seg)`) and `mean_cur_seg` per
   model × dataset to `e1_routing.json`.
3. **Report prose** — the original findings section claimed an ordering
   ("single > multikey ≈ paired_S > paired_D at essentially every layer")
   that the per-layer data contradicts: for layers 1–13, `niah_multikey_1`
   is *higher* than `niah_single_1`, not lower. Rewritten below with actual
   per-layer numbers and per-dataset chance comparisons — nothing massaged.

## What was built
- `lmr/analysis/260725_mc_niah_analysis/routing_stats.py`: recomputes the SSC
  routing score at the answer position for every MC layer
  (`u = ssc.connector(h)`, `summaries = segment_key_sums(normalize(k), chunk)`,
  `score = <u, summary_i>`), and reports per-layer `hit@2`, `gold_rank`, and
  `amongkeys` accuracy, plus per-dataset `chance_hit2`/`mean_cur_seg`, across:
  - Dataset A: `niah_single_1`, `niah_multikey_1` (50 samples each, `/data2/sohyung/mc_niah/data/2048/...`)
  - Dataset B: `paired_S_multi`, `paired_D_multi` (16 `variant=="multi"` rows each, `/data2/sohyung/mc_niah/data/paired/{S,D}.jsonl`)
  - Both `mc-5B` and `mc-30B` checkpoints.
  - The script is invoked once per `--model`; each run loads any existing
    `results/e1_routing.json` / `e1_per_sample.json`, merges in the current
    model's results, and rewrites both files + the figure — so
    `sbatch/e1.sbatch`'s `mc-5B && mc-30B` chain ends with one combined
    artifact set.
- `lmr/analysis/260725_mc_niah_analysis/sbatch/e1.sbatch`: `-p main --gres=gpu:rtx6000:1 -t 02:00:00`, sources `env_common.sh`, runs both model passes.

## Edge cases handled
1. **Un-routable gold (current-segment case).** `routing_scores_at` masks
   the current/future segments to `-inf`; when the answer position `t=T-1`
   falls in the same segment as the gold needle (`gold_seg == t//256`),
   routing is structurally impossible. Each sample is tagged
   `eligible = gold_seg < cur_seg`; `hit`/`gold_rank` are set to `None` (not
   0/last-rank) for ineligible samples so they don't pollute the aggregate
   means. Measured ineligible counts: `niah_single_1` 5/50, `niah_multikey_1`
   3/50, both paired datasets 0/16 (Dataset B's gold is always placed well
   before the end of the body, so this only bites Dataset A as anticipated
   in the brief).
2. `gold_rank` is computed by sorting the full (masked) score vector
   descending — since eligible gold scores are always finite and mask
   entries are `-inf`, the eligible gold's rank is always a genuine rank
   among past segments; only used when `eligible=True`.
3. `amongkeys` restricts the key-segment set to eligible (past) key segments
   only (`eligible_key_segs = [s in key_segs if s < cur_seg]`), **and** is
   only computed when the sample itself is `eligible` (fixed per review —
   see "Revision" above; the un-gated version double-counted 3
   `niah_multikey_1` samples where gold was un-routable, deflating the
   reported accuracy). If fewer than 2 eligible key segments qualify (true
   for essentially all `niah_single_1` samples — single needle → never ≥2
   key segments), `amongkeys=None`.
4. **E2 cross-table.** `_load_e2_baseline()` looks for
   `results/e2_oracle.json` (repo or `$MC_OUT`), tries a few permissive
   schemas, and returns `None` on any parse failure or missing file. Task 7
   hasn't run yet, so `e1_routing.json["e2_join"]` is currently `null`, as
   specified — the join logic is in place and will populate on a future
   re-run of `routing_stats.py` once `e2_oracle.json` exists (no code change
   needed, just needs a rerun after Task 7).

## Outputs
- `results/e1_routing.json` (both `$MC_OUT/results/` and repo
  `lmr/analysis/260725_mc_niah_analysis/results/`, byte-identical): per
  model × dataset × layer `hit_at_2`, `gold_rank_mean`,
  `amongkeys_acc`/`amongkeys_n`, `n_eligible`/`n_total`; per model × dataset
  (dataset-level, not per-layer) `mean_cur_seg` and `chance_hit2`; plus
  `best_layer`/`best_layer_hit_at_2`.
- `results/e1_per_sample.json` (`$MC_OUT/results/` only, ~300KB): per-sample
  per-layer hit/gold_rank/amongkeys + `eligible` flag + the dataset's
  `best_layer_hit`.
- `results/e1_routing.png` (both locations): two panels (mc-30B, mc-5B),
  x=layer, y=hit@2, one line per dataset, ylim [0,1].

## Chance level is dataset-specific (measured, not assumed)

| dataset | mean_cur_seg (eligible) | chance_hit2 = mean(min(1, 2/cur_seg)) |
|---|---|---|
| niah_single_1 | 6.96 | 0.288 |
| niah_multikey_1 | 4.19 | 0.486 |
| paired_S_multi | 7.00 | 0.286 |
| paired_D_multi | 7.00 | 0.286 |

`niah_multikey_1` contexts have noticeably fewer past segments at the
answer position (RULER's multikey template front-loads the query earlier
relative to the needle set), so its chance hit@2 (≈0.49) is nearly double
that of the other three datasets (≈0.29). Any comparison of raw hit@2
across datasets has to account for this — a flat 0.66 on multikey is a much
smaller effect than the same 0.66 would be on single/paired.

## Headline numbers (hit@2 at each dataset's best layer, eligible samples only)

| model | niah_single_1 | niah_multikey_1 | paired_S_multi | paired_D_multi |
|---|---|---|---|---|
| mc-5B  | 0.933 (layer 14, n=45/50) | 0.660 (layer 2, n=47/50) | 0.812 (layer 14, n=16/16) | 0.562 (layer 14, n=16/16) |
| mc-30B | 1.000 (layer 15, n=45/50) | 0.809 (layer 14, n=47/50) | 0.750 (layer 14, n=16/16) | 0.500 (layer 0, n=16/16) |

`amongkeys` accuracy at each dataset's own best-hit@2 layer (multikey/paired
only — single never has ≥2 eligible key segments): mc-5B `niah_multikey_1`
25/47 ≈ 0.532 (layer 2); mc-30B `niah_multikey_1` 22/47 ≈ 0.468 (layer 14).
Both are close to a 50/50 coin flip and stay roughly flat across most
layers (mc-5B: 0.53 at layers 2/3/5/7/9–13; mc-30B fluctuates 0.30–0.62 with
no clear layer trend) — see the JSON for exact per-layer values.

## Findings — actual per-layer shape (corrected)

Per-layer hit@2, mc-5B (chance in parentheses):
- `niah_single_1` (chance 0.288): `[0.29, 0.00, 0.29, 0.24, 0.47, 0.24, 0.60, 0.24, 0.76, 0.24, 0.24, 0.24, 0.24, 0.33, 0.93, 0.91]` (layers 0–15)
- `niah_multikey_1` (chance 0.486): `[0.32, 0.49, 0.66, 0.66, 0.43, 0.66, 0.66, 0.66, 0.30, 0.66, 0.66, 0.66, 0.66, 0.66, 0.62, 0.60]`

Per-layer hit@2, mc-30B:
- `niah_single_1` (chance 0.288): `[0.00, 0.07, 0.27, 0.24, 0.38, 0.24, 0.67, 0.24, 0.96, 0.24, 0.24, 0.24, 0.24, 0.24, 0.96, 1.00]`
- `niah_multikey_1` (chance 0.486): `[0.49, 0.60, 0.64, 0.66, 0.79, 0.66, 0.57, 0.66, 0.45, 0.66, 0.66, 0.70, 0.66, 0.66, 0.81, 0.64]`

**The originally-reported ordering ("single > multikey at essentially every
layer") was wrong and is retracted.** Averaged over layers 1–13 (excluding
the layer-0 embedding-adjacent layer and the final 2 layers where single
spikes): `niah_single_1` sits at ≈0.32 (mc-5B) / ≈0.33 (mc-30B) — close to
or below its own chance (0.288) for most of that range, with occasional
spikes (e.g. layer 6, layer 8). `niah_multikey_1` sits at ≈0.60 (mc-5B) /
≈0.64 (mc-30B) across the *same* layer range — consistently and
substantially higher than single, and also clearly above its own (higher)
chance of 0.486, though the margin (0.66 vs 0.486, ≈1.36×) is far more
modest than single's eventual separation from chance.

`niah_single_1`'s profile is the opposite of flat: it is noisy and mostly
near-chance through layers 1–13, then jumps sharply at layers 14–15 to
0.91–1.00 — routing-relevant information for the single-needle case appears
concentrated almost entirely in the last two MC layers. `niah_multikey_1`,
by contrast, has a flatter, plateaued profile that is already substantially
above its own chance from layer 2 onward and does not show the same
late-layer takeoff (it actually recovers to ~0.6 by layers 14–15, similar
to or slightly below its own mid-layer plateau).

`amongkeys` (fixed n=47 for multikey) sits at ≈0.53 for most layers at
mc-5B — near a 50/50 coin flip between the (typically 2) eligible key
segments, i.e. essentially uninformative on top of the segment-level hit@2
signal, even though hit@2 itself is well above chance on this dataset. This
suggests that when the model does route to a past segment (hit@2 succeeds),
it is not reliably distinguishing the gold key's segment from a *sibling*
key segment specifically — the routing signal discriminates key-bearing
segments from non-key segments better than it discriminates among
key-bearing segments.

`paired_S_multi` (best layer 0.75–0.81) is consistently higher than
`paired_D_multi` (best layer 0.50–0.56) across models, with `paired_D`
spending most mid-layers essentially flat at its own chance (0.286) —
placing the distractor in a *different* segment than gold is measurably
harder for the routing head to resolve than a same-segment distractor,
at every layer except layer 0.

## Sanity gate
Per the plan: single/eligible hit@2 must be clearly above chance, and NOT
flat-at-chance across all layers (which would suggest a routing/eligibility
bug), since the 30B model scores 92 on S-NIAH-1@2K. Observed: mc-30B
`niah_single_1` reaches hit@2=1.0 at layer 15 (mc-5B: 0.933 at layer 14),
both far above the 0.288 chance baseline for this dataset, and the layer
curve is clearly non-random (sharp rise at layers 14–15 rather than sitting
flat at chance everywhere). Gate passes — no bug suspected. (The mid-layer
near-chance dip for single is a genuine, separately-noted finding, not a
failure of this gate, since the gate is about the *existence* of a clear
above-chance signal somewhere in the layer stack, which is present.)

## Concerns / notes for downstream tasks
- **Retracted claim:** the previous version of this report claimed
  "single > multikey ≈ paired_S > paired_D at essentially every layer."
  That is false for layers 1–13, where multikey > single. See "Findings"
  above for the corrected, per-layer-accurate description. Apologies for
  not verifying this against the actual per-layer arrays before reporting.
- The single-needle routing signal is concentrated in the last ~2 MC
  layers rather than distributed across depth; multikey's signal is present
  earlier but flatter and only modestly above its (higher) own chance level
  — worth flagging in the Task 9 synthesis, since these are qualitatively
  different profiles, not just a magnitude difference.
- `amongkeys` ≈0.53 (near coin-flip) on multikey indicates the routing head
  is much better at "past segment with a key vs. without" than at
  discriminating between multiple key-bearing past segments — a
  finer-grained failure mode worth surfacing separately from the headline
  hit@2 numbers.
- `e2_join` is `null` for now by design; re-run `routing_stats.py --model
  <kind>` after Task 7 produces `results/e2_oracle.json` to populate the
  hit-vs-baseline-correctness cross-table (no code change required).
