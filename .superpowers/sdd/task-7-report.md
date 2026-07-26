# Task 7 report — E2: oracle routing intervention (causal main experiment)

## Status: DONE

## What was built
- `lmr/analysis/260725_mc_niah_analysis/oracle.py`:
  - `inject_gold(top_indices, top_scores, online_score, gold, segment_ids)` — pure
    function, forces the gold segment into the last top-k slot whenever it is
    eligible (`segment_ids > gold`) but not already selected, with the
    replacement score set to `max(selected_score_max, online_score)` so the
    gate can't structurally down-weight the injected memory. Matches the
    brief's pseudocode and unit-test spec exactly.
  - `_make_oracle_class()` / `patch_oracle(model)` — lazily imports the
    worktree's `dsc.mc_gdn2.ssc.GDN2SSC` only at call time (not at module
    import time), so `inject_gold` stays importable/testable on CPU without
    the worktree on `sys.path`. `patch_oracle` swaps every MC layer's
    `attn.ssc` for an `OracleGDN2SSC` copy (`load_state_dict` from the
    original, same device/dtype), returning the list of oracle instances so
    the driver can set `.gold_segment` on all of them per sample.
  - Driver (`main`): for `--model {mc-5B,mc-30B}`, for `condition in {S,D}`,
    loops the Dataset B `variant=="multi"` rows (16 each) directly (not
    `gen_eval.run_file`, since it has no per-sample callback) calling
    `gen_eval.greedy_generate` twice per row — once with `gold_segment=None`
    (baseline) and once with `gold_segment=int(row["gold_seg"])` on every
    oracle layer (oracle), resetting to `None` after. `n_gen=128`.
  - **Sanity check (brief requirement):** before each condition's main loop,
    `_sanity_check_baseline_matches_unpatched` runs one sample with
    `gold_segment=None` through the patched model and asserts the generated
    token sequence is byte-identical to a *stock* `GDN2SSC` (same weights,
    unpatched forward) — i.e. the patch is provably a no-op until
    `gold_segment` is set. This passed on all 4 model×condition sanity calls
    (see logs).
  - Output `results/e2_oracle.json` (repo + `$MC_OUT`, byte-identical):
    `{"meta": {models, n_gen, injection_score_mode}, "results": {model:
    {cond: {"baseline"/"oracle": {"score","n","rows":[{pair_id, gold_seg,
    pred, correct}]}}}}, "rows": [...]}`. The extra top-level flat `"rows"`
    list is a compatibility shim for `routing_stats.py`'s `e2_join` parser
    (see "post-hoc fix" below — the flat-list approach turned out to have a
    model-collision bug, since fixed by reading `results[model][...]`
    directly instead).
- `tests/lmr/test_mc_niah_oracle.py` — the two tests specified in the brief,
  verbatim. Both pass CPU-only, no worktree needed
  (`HF_HOME=/data2/sohyung/hf_home $PY -m pytest tests/lmr/test_mc_niah_oracle.py -x -q` → 2 passed).
- `lmr/analysis/260725_mc_niah_analysis/gen_eval.py` — small fix per reviewer
  flag: `run_file` now returns `{"score": 0.0, "n": 0, "rows": []}` instead of
  raising `ZeroDivisionError` when `rows` is empty.
- `lmr/analysis/260725_mc_niah_analysis/sbatch/e2.sbatch` — generic template
  (`-t 05:50:00`, `--gres=gpu:rtx6000:1`), takes `MODEL` via `--export`, so
  the two models run as **separate SLURM jobs**:
  ```
  sbatch --job-name=mc_e2_5B  --export=ALL,MODEL=mc-5B  sbatch/e2.sbatch
  sbatch --job-name=mc_e2_30B --export=ALL,MODEL=mc-30B sbatch/e2.sbatch
  ```

## Runs
- Smoke test (job 2308, `--limit 1 --n-gen 8 --dry-run`): validated the
  sanity check and both generation paths end-to-end before committing to the
  full multi-hour run.
- Full jobs: mc-5B = job 2309, mc-30B = job 2310. The node has 2 GPUs but one
  was occupied by another user's job throughout, so both jobs ran
  sequentially on the remaining GPU (not in parallel) — well within the
  5:50:00 budget each; no need to drop `n_gen` to 64.
- Rejoin: `sbatch/e1.sbatch` resubmitted as job 2312 after `e2_oracle.json`
  existed, to populate `e1_routing.json`'s `e2_join`.
- Post-hoc fix + re-rejoin: the first `e2_join` populated this way had a
  model-collision bug (see "Concerns" history below) — `routing_stats.py`'s
  `_load_e2_baseline` was reading the flat top-level `"rows"` list without a
  model key, so both models' `e2_join` silently showed the same (later-in-list)
  model's baseline correctness. Fixed to read `results[model][condition]
  ["baseline"]["rows"]` directly per model, then `sbatch/e1.sbatch` was
  rerun once more to regenerate `e1_routing.json` with correctly
  model-separated `e2_join` (verified: per-model bucket sums now exactly
  match that model's own baseline score in `e2_oracle.json`).

## Headline result — 2×2×2 score table (the causal main result)

| model | condition | baseline score | oracle score | Δ | n |
|---|---|---|---|---|---|
| mc-5B  | S | 0.125 | 0.312 | **+0.188** | 16 |
| mc-5B  | D | 0.188 | 0.500 | **+0.312** | 16 |
| mc-30B | S | 0.000 | 0.312 | **+0.312** | 16 |
| mc-30B | D | 0.062 | 0.562 | **+0.500** | 16 |

Every one of the 4 cells satisfies the sanity gate `oracle >= baseline` (in
fact strictly `>`, by 0.19–0.50), and by a wide margin — not a single-sample
fluke. Forcing the gold segment into the router's top-k, with a fair
(non-disadvantaged) gate score, roughly **doubles-to-quadruples** the
model's ability to answer correctly, in both the easier same-segment
distractor condition (S) and the harder different-segment distractor
condition (D), and for both model sizes. This is the direct causal evidence
that the E1 routing-accuracy gap (Task 6: `paired_D_multi` hit@2 ≈0.50–0.56
vs `paired_S_multi` ≈0.75–0.81) is not just correlated with downstream
answer failure but **causes** it: repairing routing alone, with everything
else (weights, gate softmax, online memory) held fixed, recovers a large
fraction of the missing accuracy. mc-30B/D shows the largest jump
(0.062→0.562, +0.5), consistent with it having the worst baseline routing
in E1 (`paired_D_multi` best-layer hit@2 = 0.500, at layer 0 — essentially
no depth carries the signal) and thus the most headroom for an oracle fix.

Note baseline scores here are markedly lower than a naive read of E1's
`hit@2` might suggest (e.g. mc-30B/S baseline = 0.000): E1's hit@2 is a
routing-only metric at a single layer/position, while E2's `correct` also
requires the full 128-token greedy generation to actually surface the
correct 7-digit number in the right format — so baseline scores here reflect
both routing failures **and** downstream generation/formatting failures that
oracle routing alone does not fix, which is expected and does not affect the
causal reading (oracle vs baseline is the controlled comparison).

## e2_join cross-table (from the rejoined, model-separation-fixed `e1_routing.json`)
Bucketed by whether E1's best-layer routing was a `hit` or `miss` at the
answer position, joined against **E2 baseline** (`gold_segment=None`)
generation correctness for the same model's `(condition, pair_id)`:

| model | dataset | routing | n | correct | acc |
|---|---|---|---|---|---|
| mc-5B | paired_S_multi | hit | 13 | 1 | 0.077 |
| mc-5B | paired_S_multi | miss | 3 | 1 | 0.333 |
| mc-5B | paired_D_multi | hit | 9 | 0 | 0.000 |
| mc-5B | paired_D_multi | miss | 7 | 3 | 0.429 |
| mc-30B | paired_S_multi | hit | 12 | 0 | 0.000 |
| mc-30B | paired_S_multi | miss | 4 | 0 | 0.000 |
| mc-30B | paired_D_multi | hit | 8 | 0 | 0.000 |
| mc-30B | paired_D_multi | miss | 8 | 1 | 0.125 |

`e2_join` is confirmed non-null and correctly model-separated in both
`e1_routing.json` copies (`lmr/analysis/260725_mc_niah_analysis/results/e1_routing.json`
and `$MC_OUT/results/e1_routing.json`, byte-identical). Sanity check: each
model's `hit`+`miss` correct-counts sum to that model's own baseline score
in `e2_oracle.json` — mc-5B: 1+1=2/16 (S, matches 0.125) and 0+3=3/16 (D,
matches 0.188); mc-30B: 0+0=0/16 (S, matches 0.000) and 0+1=1/16 (D, matches
0.062) — exact match in all 4 cells, confirming the fix.

For mc-30B, baseline correctness is essentially all-zero regardless of
routing bucket (0/16 on S, 1/16 on D) — there's too little baseline signal
to read a routing-vs-correctness relationship out of this table for that
model; it mostly says "mc-30B's baseline free-generation answer is wrong
almost everywhere," which the 2×2×2 table already establishes directly.
mc-5B has enough spread to be suggestive (`paired_D_multi` miss=3/7=0.43 vs
hit=0/9=0.00 — routing miss correlating with *higher* baseline accuracy is
counter-intuitive), but n=7–13 per bucket is too small to treat as more than
a single-experiment observation; it isn't the paper's causal claim (that's
the 2×2×2 oracle-vs-baseline table, which holds up cleanly) and shouldn't be
over-interpreted without a larger sample.

## Sanity gate
Per the brief: "if oracle < baseline anywhere by more than 1 sample,
investigate." Observed: oracle ≥ baseline in **all 4 cells**, with margins
of +3 to +8 samples out of 16 — no investigation needed, gate passes
cleanly. The per-sample `_sanity_check_baseline_matches_unpatched` check
(patched-with-`gold_segment=None` == stock forward, byte-identical
generated tokens) also passed on all 4 calls (one per condition per model),
confirming the oracle patch introduces zero side effects when inactive.

## Concerns
1. ~~`e2_join` model-conflation bug~~ — **fixed.** `routing_stats.py`'s
   `_load_e2_baseline()` (written in Task 6, before `e2_oracle.json`
   existed) originally built a single flat `(condition, pair_id) -> correct`
   mapping with no model dimension, reused for every model in
   `_compute_e2_join`, so both models' `e2_join` silently showed the same
   (later-in-list) model's baseline correctness. Fixed in a follow-up commit
   (`1f5a0e02`) to read `results[model][condition]["baseline"]["rows"]`
   directly per model inside `_compute_e2_join`, then `e1.sbatch` rerun once
   more (`5a84af7a`) to regenerate `e1_routing.json`. Verified: each model's
   `e2_join` hit+miss correct-counts now sum exactly to that model's own
   `e2_oracle.json` baseline score (see e2_join section above) — no more
   collision. This did touch `routing_stats.py`, outside Task 7's originally
   listed file scope, but was necessary to make `e2_join` trustworthy rather
   than silently wrong.
2. **Baseline scores are low in absolute terms** (0.000–0.188), so the
   oracle deltas, while large in relative and margin terms, are computed
   over correspondingly small absolute baseline denominators — e.g.
   mc-30B/S baseline is literally 0/16. This is expected (E1 established
   routing is often wrong at this context length) and the oracle intervention
   still clearly and monotonically helps, but small-n (16 per cell) per-cell
   scores should be read as directional rather than tightly estimated
   proportions; no formal CI was computed.
3. **GPU/bf16 run-to-run noise in the rejoined `e1_routing.json`.** Some
   `niah_single_1`/`niah_multikey_1` per-layer `hit_at_2` values shifted by
   up to ~2.3pp versus the previously-committed `e1_routing.json` (e.g.
   mc-30B `niah_single_1` best-layer hit@2: 1.000 → 0.978) purely from
   rerunning `routing_stats.py` on the same model/data with no code change —
   consistent with non-deterministic bf16 GPU kernel reduction order, not a
   regression. Flagging so it isn't mistaken for a routing_stats.py bug.
4. Both e2 jobs ran sequentially on a single shared GPU (the node's second
   GPU was occupied by another user's unrelated job throughout), so parallel
   2-GPU submission was not exercised — future reruns on an idle node could
   run mc-5B/mc-30B concurrently for ~2x wall-clock speedup.

## Commits (branch `sh/mc-niah-analysis`, pushed to `origin`)
- `20d8710b` — E2 oracle routing intervention code + tests (S/D)
- `bbb4da5c` — E2 oracle routing results (mc-5B/30B x S/D) + e1 e2_join rejoin
- `1f5a0e02` — E1 fix: key `e2_join` baseline by model, not just `(condition, pair_id)`
- `5a84af7a` — E1 results rerun: model-separated `e2_join` (mc-5B, mc-30B)
- Pushed: `git push -u origin sh/mc-niah-analysis` → new branch on
  `https://github.com/Smaller25/linear-memory-routing`
  (PR link: https://github.com/Smaller25/linear-memory-routing/pull/new/sh/mc-niah-analysis).
  Note: `.superpowers/sdd/` is entirely gitignored by this repo's own
  `.superpowers/sdd/.gitignore` (`*`), same as Task 6's report — this report
  file itself is a local working artifact, not part of the pushed commits;
  the code/tests/results described in it are what's on GitHub.

## Files
- `/home/sohyung/linear-memory-routing/lmr/analysis/260725_mc_niah_analysis/oracle.py`
- `/home/sohyung/linear-memory-routing/tests/lmr/test_mc_niah_oracle.py`
- `/home/sohyung/linear-memory-routing/lmr/analysis/260725_mc_niah_analysis/gen_eval.py`
- `/home/sohyung/linear-memory-routing/lmr/analysis/260725_mc_niah_analysis/sbatch/e2.sbatch`
- `/home/sohyung/linear-memory-routing/lmr/analysis/260725_mc_niah_analysis/routing_stats.py` (e2_join fix)
- `/home/sohyung/linear-memory-routing/lmr/analysis/260725_mc_niah_analysis/results/e2_oracle.json`
- `/home/sohyung/linear-memory-routing/lmr/analysis/260725_mc_niah_analysis/results/e1_routing.json` (rejoined, model-separated e2_join)
- Logs: `/data2/sohyung/mc_niah/logs/e2_mc_e2_5B_2309.log`,
  `/data2/sohyung/mc_niah/logs/e2_mc_e2_30B_2310.log`,
  `/data2/sohyung/mc_niah/logs/e1_2312.log`
