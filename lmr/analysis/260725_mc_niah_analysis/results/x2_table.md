# X2 results: multi-query/multivalue per-key routing hit@2

task × length × model. `macro_hit2`/`macro_hitk` are best-layer values (layer chosen by highest macro_hit2 for that task/length/model); `k` for hit@k = number of queried needles in that task (4 for multiquery/multivalue, 2 for multiquery_q2, 1 for multikey_1/single). chance columns are re-derived per §7 (multi-gold: mean over eligible samples of min(1, m/cur_seg), m=2 or k). **`hit2/chance` is the key comparison column** — task chance levels differ substantially (more needle sentences -> smaller haystack budget -> fewer past segments -> higher chance), so raw macro_hit2 is not directly comparable across tasks; the ratio to that task's own chance is.

| task | length | model | best layer | macro hit@2 | chance hit@2 | hit2/chance | macro hit@k | chance hit@k | hitk/chance | n (eligible/total) |
|---|---|---|---|---|---|---|---|---|---|---|
| niah_single_1 (E1 ref) | 2048 | mc-30B | 15 | 1.000 | 0.288 | 3.47x | - | - | - | 45/50 |
| niah_single_1 (E1 ref) | 2048 | mc-5B | 14 | 0.933 | 0.288 | 3.24x | - | - | - | 45/50 |
| niah_multiquery | 2048 | mc-30B | 4 | 0.520 | 0.414 | 1.26x | 0.807 | 0.829 | 0.97x | 50/50 |
| niah_multiquery | 2048 | mc-5B | 15 | 0.503 | 0.414 | 1.21x | 0.850 | 0.829 | 1.03x | 50/50 |
| niah_multiquery_q2 | 2048 | mc-30B | 14 | 0.552 | 0.348 | 1.59x | 0.552 | 0.348 | 1.59x | 48/50 |
| niah_multiquery_q2 | 2048 | mc-5B | 15 | 0.490 | 0.348 | 1.41x | 0.490 | 0.348 | 1.41x | 48/50 |
| niah_multivalue | 2048 | mc-30B | 4 | 0.435 | 0.346 | 1.26x | 0.743 | 0.691 | 1.08x | 50/50 |
| niah_multivalue | 2048 | mc-5B | 4 | 0.442 | 0.346 | 1.28x | 0.738 | 0.691 | 1.07x | 50/50 |
| niah_multikey_1 | 2048 | mc-30B | 4 | 0.809 | 0.486 | 1.66x | 0.574 | 0.243 | 2.36x | 47/50 |
| niah_multikey_1 | 2048 | mc-5B | 2 | 0.660 | 0.486 | 1.36x | 0.383 | 0.243 | 1.57x | 47/50 |
| niah_multiquery | 4096 | mc-30B | 4 | 0.177 | 0.161 | 1.10x | 0.413 | 0.323 | 1.28x | 50/50 |
| niah_multiquery | 4096 | mc-5B | 4 | 0.213 | 0.161 | 1.32x | 0.353 | 0.323 | 1.10x | 50/50 |
| niah_multivalue | 4096 | mc-30B | 4 | 0.233 | 0.143 | 1.64x | 0.408 | 0.285 | 1.43x | 50/50 |
| niah_multivalue | 4096 | mc-5B | 14 | 0.217 | 0.143 | 1.52x | 0.390 | 0.285 | 1.37x | 50/50 |
| niah_multikey_1 | 4096 | mc-30B | 15 | 0.286 | 0.161 | 1.77x | 0.163 | 0.081 | 2.03x | 49/50 |
| niah_multikey_1 | 4096 | mc-5B | 15 | 0.429 | 0.161 | 2.66x | 0.306 | 0.081 | 3.80x | 49/50 |
| niah_multiquery | 8192 | mc-30B | 8 | 0.100 | 0.067 | 1.50x | 0.187 | 0.134 | 1.40x | 50/50 |
| niah_multiquery | 8192 | mc-5B | 11 | 0.145 | 0.067 | 2.17x | 0.160 | 0.134 | 1.20x | 50/50 |
| niah_multivalue | 8192 | mc-30B | 4 | 0.128 | 0.066 | 1.94x | 0.195 | 0.133 | 1.47x | 50/50 |
| niah_multivalue | 8192 | mc-5B | 15 | 0.130 | 0.066 | 1.96x | 0.182 | 0.133 | 1.37x | 50/50 |
| niah_multikey_1 | 8192 | mc-30B | 0 | 0.080 | 0.070 | 1.14x | 0.040 | 0.035 | 1.14x | 50/50 |
| niah_multikey_1 | 8192 | mc-5B | 1 | 0.160 | 0.070 | 2.28x | 0.100 | 0.035 | 2.85x | 50/50 |
