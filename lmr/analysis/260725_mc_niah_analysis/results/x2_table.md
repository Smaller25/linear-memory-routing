# X2 results: multi-query/multivalue per-key routing hit@2

**Corrections round 2 (adversarial review)**: niah_single_1 is NOISE-haystack (not directly comparable to the essay-haystack tasks below) — niah_single_2 (essay-haystack, single-needle) is the clean control. The "monotonic decline" claim across the needle-count sweep is dropped (not monotone in the raw numbers; the q=2 sweep also changes total needle count IN CONTEXT, not just queried count, so it isn't a clean k-only manipulation). "Clears the noise floor Nx" language is replaced by sampling SE (`se_hit2` = sqrt(p(1-p)/n) at each row's own n_eligible); best-layer selection (max over 16 layers) is upward-biased, so treat se_hit2 as a lower bound on the true uncertainty.

## Table 1 — hit@2, chance, needle-null, structural ceiling, skill

`needle_null_hit2` = reviewer's stronger null: top-2 uniform over *needle-bearing* eligible segments (any key), not all past segments — the "perfect needle detector, zero key discrimination" baseline. `ceiling` = max achievable macro hit@2 given ONE shared top-2 per sample (routing score is computed once per sample, not independently per needle) — can exceed 2/(#queried needles) when needles happen to collide into the same segment (plausible here: cur_seg is often only 4-7 at length=2048). `skill` = (macro_hit2 - chance_hit2) / (ceiling - chance_hit2) — 0 means no better than chance, 1 means saturating the structural ceiling. `se_hit2` = sqrt(p(1-p)/n_eligible), a rough sampling SE for macro_hit2.

| task | length | model | layer | hit@2 | chance | needle-null | ceiling | skill | hit2/chance | se_hit2 | n (elig/tot) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| niah_single_1 (E1 ref, NOISE-haystack — not directly comparable) | 2048 | mc-30B | 15 | 1.000 | 0.288 | - | - | - | 3.47x | 0.0pp | 45/50 |
| niah_single_1 (E1 ref, NOISE-haystack — not directly comparable) | 2048 | mc-5B | 14 | 0.933 | 0.288 | - | - | - | 3.24x | 3.7pp | 45/50 |
| niah_single_2 (essay-haystack, single-needle control) | 2048 | mc-30B | 14 | 0.875 | 0.425 | 1.000 | 1.000 | 0.78 | 2.06x | 5.2pp | 40/50 |
| niah_single_2 (essay-haystack, single-needle control) | 2048 | mc-5B | 15 | 0.850 | 0.425 | 1.000 | 1.000 | 0.74 | 2.00x | 5.6pp | 40/50 |
| niah_multiquery | 2048 | mc-30B | 4 | 0.520 | 0.414 | 0.750 | 0.783 | 0.29 | 1.26x | 7.1pp | 50/50 |
| niah_multiquery | 2048 | mc-5B | 15 | 0.503 | 0.414 | 0.750 | 0.783 | 0.24 | 1.21x | 7.1pp | 50/50 |
| niah_multiquery_q2 (q=2 sweep — ALSO only 2 needles in context, not 4) | 2048 | mc-30B | 14 | 0.552 | 0.348 | 1.000 | 1.000 | 0.31 | 1.59x | 7.2pp | 48/50 |
| niah_multiquery_q2 (q=2 sweep — ALSO only 2 needles in context, not 4) | 2048 | mc-5B | 15 | 0.490 | 0.348 | 1.000 | 1.000 | 0.22 | 1.41x | 7.2pp | 48/50 |
| niah_multivalue | 2048 | mc-30B | 4 | 0.435 | 0.346 | 0.723 | 0.762 | 0.21 | 1.26x | 7.0pp | 50/50 |
| niah_multivalue | 2048 | mc-5B | 4 | 0.442 | 0.346 | 0.723 | 0.762 | 0.23 | 1.28x | 7.0pp | 50/50 |
| niah_multikey_1 | 2048 | mc-30B | 4 | 0.809 | 0.486 | 0.773 | 1.000 | 0.63 | 1.66x | 5.7pp | 47/50 |
| niah_multikey_1 | 2048 | mc-5B | 2 | 0.660 | 0.486 | 0.773 | 1.000 | 0.34 | 1.36x | 6.9pp | 47/50 |
| niah_multiquery | 4096 | mc-30B | 4 | 0.177 | 0.161 | 0.607 | 0.623 | 0.03 | 1.10x | 5.4pp | 50/50 |
| niah_multiquery | 4096 | mc-5B | 4 | 0.213 | 0.161 | 0.607 | 0.623 | 0.11 | 1.32x | 5.8pp | 50/50 |
| niah_multivalue | 4096 | mc-30B | 4 | 0.233 | 0.143 | 0.550 | 0.568 | 0.21 | 1.64x | 6.0pp | 50/50 |
| niah_multivalue | 4096 | mc-5B | 14 | 0.217 | 0.143 | 0.550 | 0.568 | 0.17 | 1.52x | 5.8pp | 50/50 |
| niah_multikey_1 | 4096 | mc-30B | 15 | 0.286 | 0.161 | 0.592 | 1.000 | 0.15 | 1.77x | 6.5pp | 49/50 |
| niah_multikey_1 | 4096 | mc-5B | 15 | 0.429 | 0.161 | 0.592 | 1.000 | 0.32 | 2.66x | 7.1pp | 49/50 |
| niah_multiquery | 8192 | mc-30B | 8 | 0.100 | 0.067 | 0.530 | 0.532 | 0.07 | 1.50x | 4.2pp | 50/50 |
| niah_multiquery | 8192 | mc-5B | 11 | 0.145 | 0.067 | 0.530 | 0.532 | 0.17 | 2.17x | 5.0pp | 50/50 |
| niah_multivalue | 8192 | mc-30B | 4 | 0.128 | 0.066 | 0.520 | 0.523 | 0.14 | 1.94x | 4.7pp | 50/50 |
| niah_multivalue | 8192 | mc-5B | 15 | 0.130 | 0.066 | 0.520 | 0.523 | 0.14 | 1.96x | 4.8pp | 50/50 |
| niah_multikey_1 | 8192 | mc-30B | 0 | 0.080 | 0.070 | 0.530 | 1.000 | 0.01 | 1.14x | 3.8pp | 50/50 |
| niah_multikey_1 | 8192 | mc-5B | 1 | 0.160 | 0.070 | 0.530 | 1.000 | 0.10 | 2.28x | 5.2pp | 50/50 |

## Table 2 — hit@k detail (k = number of queried needles) and saturation

`hit@k (all)` uses every eligible sample at that row's best layer (same as Table 1's companion column in the previous round). `hit@k (cur_seg>k only)` restricts to samples where cur_seg > k — i.e. NOT structurally saturated (when cur_seg <= k, selecting the top-k literally selects every available past segment, so hit@k is trivially forced high regardless of routing quality). `frac_saturated` discloses what fraction of eligible samples were in that trivial regime — item (ii) of the corrections: a large fraction of multiquery@2048 rows are saturated (cur_seg=4=k), so the unrestricted hit@k column overstates the routing signal at k.

| task | length | model | hit@k (all) | chance hit@k (all) | hit@k (cur_seg>k only) | chance hit@k (cur_seg>k only) | frac_saturated |
|---|---|---|---|---|---|---|---|
| niah_single_2 (essay-haystack, single-needle control) | 2048 | mc-30B | 0.325 | 0.212 | 0.325 | 0.213 | 0.00 |
| niah_single_2 (essay-haystack, single-needle control) | 2048 | mc-5B | 0.325 | 0.212 | 0.325 | 0.213 | 0.00 |
| niah_multiquery | 2048 | mc-30B | 0.807 | 0.829 | 0.517 | 0.571 | 0.60 |
| niah_multiquery | 2048 | mc-5B | 0.850 | 0.829 | 0.625 | 0.571 | 0.60 |
| niah_multiquery_q2 (q=2 sweep — ALSO only 2 needles in context, not 4) | 2048 | mc-30B | 0.552 | 0.348 | 0.552 | 0.348 | 0.00 |
| niah_multiquery_q2 (q=2 sweep — ALSO only 2 needles in context, not 4) | 2048 | mc-5B | 0.490 | 0.348 | 0.490 | 0.348 | 0.00 |
| niah_multivalue | 2048 | mc-30B | 0.743 | 0.691 | 0.644 | 0.571 | 0.28 |
| niah_multivalue | 2048 | mc-5B | 0.738 | 0.691 | 0.637 | 0.571 | 0.28 |
| niah_multikey_1 | 2048 | mc-30B | 0.574 | 0.243 | 0.574 | 0.243 | 0.00 |
| niah_multikey_1 | 2048 | mc-5B | 0.383 | 0.243 | 0.383 | 0.243 | 0.00 |
| niah_multiquery | 4096 | mc-30B | 0.413 | 0.323 | 0.413 | 0.323 | 0.00 |
| niah_multiquery | 4096 | mc-5B | 0.353 | 0.323 | 0.353 | 0.323 | 0.00 |
| niah_multivalue | 4096 | mc-30B | 0.408 | 0.285 | 0.408 | 0.285 | 0.00 |
| niah_multivalue | 4096 | mc-5B | 0.390 | 0.285 | 0.390 | 0.285 | 0.00 |
| niah_multikey_1 | 4096 | mc-30B | 0.163 | 0.081 | 0.163 | 0.081 | 0.00 |
| niah_multikey_1 | 4096 | mc-5B | 0.306 | 0.081 | 0.306 | 0.081 | 0.00 |
| niah_multiquery | 8192 | mc-30B | 0.187 | 0.134 | 0.187 | 0.134 | 0.00 |
| niah_multiquery | 8192 | mc-5B | 0.160 | 0.134 | 0.160 | 0.134 | 0.00 |
| niah_multivalue | 8192 | mc-30B | 0.195 | 0.133 | 0.195 | 0.133 | 0.00 |
| niah_multivalue | 8192 | mc-5B | 0.182 | 0.133 | 0.182 | 0.133 | 0.00 |
| niah_multikey_1 | 8192 | mc-30B | 0.040 | 0.035 | 0.040 | 0.035 | 0.00 |
| niah_multikey_1 | 8192 | mc-5B | 0.100 | 0.035 | 0.100 | 0.035 | 0.00 |
