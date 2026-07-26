# Task 4 Report: Dataset B paired-controlled generator

## Summary

Implemented `lmr/analysis/260725_mc_niah_analysis/paired_gen.py` per the brief, with one
required deviation: the Paul Graham essays JSON path. Task 3's actual download location
is `src/ruler/gen/synthetic/json/PaulGrahamEssays.json` (confirmed 3.1MB file present),
not `data/PaulGrahamEssays.json` as in the brief's literal code block. Confirmed its
structure first:

```
python -c "import json; d=json.load(open('src/ruler/gen/synthetic/json/PaulGrahamEssays.json')); print(type(d), list(d)[:3])"
# <class 'dict'> ['text']
```

Structure matched the brief's assumption (`dict` with a `"text"` key) — only the path
needed correction. `_essay_sentences` now loads from:

```python
os.path.join(mcdata.REPO, "src", "ruler", "gen", "synthetic", "json", "PaulGrahamEssays.json")
```

Everything else in `paired_gen.py` (WORDS, NEEDLE_FMT, TEMPLATE, `_compose`, `_make_one`,
`build_pairs`, `prepare_b`) was implemented exactly as specified in the brief.

## TDD steps followed

1. Appended `test_paired_gen_invariants` to `tests/lmr/test_mc_niah_data.py` exactly as
   given in the brief.
2. Confirmed it failed first: `ModuleNotFoundError` for `paired_gen` (module didn't exist
   yet) — verified via
   `HF_HOME=/data2/sohyung/hf_home /data2/sohyung/conda-envs/sh_infocap/bin/python -m pytest tests/lmr/test_mc_niah_data.py::test_paired_gen_invariants -x -q`.
3. Wrote `paired_gen.py` (with the essay-path fix above).
4. Re-ran the full suite — all 3 tests passed on the first attempt, no need to touch the
   8-attempt placement retry loop or insertion offsets; the brief's invariants held as-is:

```
tests/lmr/test_mc_niah_data.py::test_annotate_finds_gold_needle PASSED
tests/lmr/test_mc_niah_data.py::test_annotate_multikey_picks_queried PASSED
tests/lmr/test_mc_niah_data.py::test_paired_gen_invariants PASSED
3 passed, 15 warnings in 1.16s
```

## Deliverable run: `prepare-b`

```
source lmr/analysis/260725_mc_niah_analysis/env_common.sh
$PY $ANA/data.py prepare-b
```

Output:
```
[prepare-b] S: 32 rows (16 pairs) -> /data2/sohyung/mc_niah/data/paired/S.jsonl
[prepare-b] D: 32 rows (16 pairs) -> /data2/sohyung/mc_niah/data/paired/D.jsonl
```

Matches expected `S: 32 rows`, `D: 32 rows` (16 pairs × 2 variants each).

## Full validation of all generated rows

Loaded both `S.jsonl` and `D.jsonl` (64 rows total, 32 pairs), re-ran `mcdata.annotate`
on every row's `input`, and checked every invariant from the brief (query_key match,
n_tok ≤ 1920, needle counts, gold_seg/distractor_segs match between stored metadata and
actual re-annotation, S ⇒ gold_seg in distractor_segs + codist_tok_dist set, D ⇒
gold_seg not in distractor_segs + codist_tok_dist is None, and single/multi pairing
consistency — same needle_key/outputs/gold_seg across variants). No exceptions raised;
all assertions passed:

```
ALL VALIDATED OK: 64 rows checked, 32 pairs across S/D
```

## Files changed

- Created: `lmr/analysis/260725_mc_niah_analysis/paired_gen.py`
- Modified: `tests/lmr/test_mc_niah_data.py` (added `test_paired_gen_invariants`)

## Commit

```
47f94ebd mc-niah: Dataset B paired-controlled generator (S/D placement)
 2 files changed, 158 insertions(+)
 create mode 100644 lmr/analysis/260725_mc_niah_analysis/paired_gen.py
```

## Notes / concerns

- The 8-attempt verify-and-retry placement loop was never stressed — every `_make_one`
  call succeeded on its first attempt across both the 3-pair test fixture and the full
  16-pair × 2-condition `prepare-b` run (32 pairs, 64 rows). No adjustment to attempt
  count or insertion offsets was needed.
- One pre-existing untracked file, `src/ruler/gen/synthetic/json/squad.json`, is present
  in the working tree (leftover from Task 3's RULER benchmark file downloads) but was
  deliberately left unstaged as it's unrelated to this task.
- `data.py`'s CLI dispatch (`if a.cmd == "prepare-b": from paired_gen import prepare_b; prepare_b(n_pairs_per_cond=a.n_pairs)`) was already present from Task 3/prior work and required no changes — it correctly imports the newly created module.

---

## Fix report: reviewer-found Important issues (post-initial-implementation)

The reviewer found 2 Important issues in the first version of `paired_gen.py`. Both are
fixed below, the test suite is strengthened, data is regenerated, and everything is
re-validated.

### Issue 1 (core): single/multi did not share the same haystack

**Root cause confirmed.** The original `_make_one` built `ctx_single` by calling
`_compose(sents, ins[:1], tokenizer, body)` — a fresh, independent recomposition with
only the gold needle. `_compose`'s cumulative token counter (`cum`) advances by the
needle sentence's own token count whenever an insert is placed, so with 1 insert
(single) vs 4 inserts (multi) the counter drifts differently through the sentence list
from the very first needle onward. Concretely, single consumes *more* haystack
sentences before hitting `body_budget` (since it has fewer/shorter insert tokens
"stealing" from the budget), so single and multi end up built from different subsets
of Paul Graham sentences after the first insertion point — confirmed empirically before
the fix: for `S.jsonl` pid=0, the raw token sequences diverged after only 394 tokens
even though gold sits at token ~1360, meaning roughly 70% of the "shared" context was
in fact different text. This confounds any comparison between single/multi under
Dataset B's design (E3).

**Fix implemented** (deviates from the literal 8-line `_compose(ins[:1])` call in the
original brief, per the reviewer's explicit required-fix spec): construct `ctx_single`
*from* the already-composed `ctx_multi`, not by recomposing from scratch.

1. `ctx_multi` is composed exactly as before (`_compose(sents, ins, tokenizer, body)`,
   all 4 needle sentences).
2. New `_neutralize(tokenizer, ctx_multi, distractor_sentences)` replaces each of the 3
   distractor needle sentences in `ctx_multi` with a neutral filler whose **in-context
   token span is exactly the same length** as the distractor sentence it replaces
   (measured via `tokenizer(..., return_offsets_mapping=True)` + a `tok_at` char→token
   lookup, i.e. the same technique `annotate()` already uses). Replacement proceeds
   rightmost-sentence-first so earlier character offsets stay valid across the three
   substitutions.
3. New `_fit_filler(tokenizer, target_n, max_tries=50)` builds the filler from a fixed
   neutral word pool (`NEUTRAL_ATOMS`, plain lowercase words, no digits/punctuation
   inside words). Empirically verified with the real TinyLlama tokenizer (see below)
   that each pool word contributes exactly +1 token when appended (with a leading
   space) — so `target_n - 1` words + a trailing period lands on `target_n` tokens
   almost always on the first attempt; the function still retries up to 50 times with
   reshuffled word order as a safety net, then raises (which `_make_one` catches and
   treats as "resample the whole sample" by `continue`-ing to the next of the existing
   8 attempts).

   Empirical check (`bin/python` + TinyLlama tokenizer) that motivated this design:
   ```
   needle standalone: 19 tokens
   1 'grass.'                                          -> 2 tokens (+2)
   2 'grass green.'                                     -> 3 tokens (+1)
   3 'grass green sky.'                                  -> 4 tokens (+1)
   ... (every additional word: +1 token, consistently)
   in-context span (needle embedded in surrounding prose): 19 tokens == standalone 19
   ```
4. Because the filler's *token* count (not character count) exactly matches the
   replaced distractor's, all token positions after each replacement stay aligned
   between single and multi — including the gold needle's absolute token position,
   regardless of whether a given distractor appears before or after the gold needle in
   the composed text.
5. Row schema is unchanged (`_neutralize` only changes what `context` string is used to
   fill `TEMPLATE`).

### Issue 2: inconsistent acceptance-check branch (pre-annotate vs. annotated gold_seg)

Original code:
```python
ok_cond = ((gold_seg in d_actual) if condition == "S"
           else (am["gold_seg"] not in d_actual))
```
The S-branch checked the pre-annotate *target* `gold_seg` (the value chosen before
composing/annotating), while the D-branch checked the *actual annotated* `am["gold_seg"]`.
Fixed to use `am["gold_seg"]` (the real, re-measured value) in both branches:
```python
ok_cond = ((am["gold_seg"] in d_actual) if condition == "S"
           else (am["gold_seg"] not in d_actual))
```

### Additional acceptance checks added to `_make_one`

To make the haystack-sharing invariant self-enforcing (not just test-covered), the
accept condition in `_make_one` now also requires, using the annotated needle records
(`gold_m`, `gold_s`):
```python
gold_m["tok_start"] == gold_s["tok_start"]   # identical absolute token position
am["n_tok"] == asg["n_tok"]                  # identical total token count
```
in addition to the pre-existing `am["gold_seg"] == asg["gold_seg"]`. Any mismatch
(e.g. if `_fit_filler`'s in-context substitution shifted tokenization at a boundary in
a way the standalone check didn't anticipate) causes the whole attempt to be
resampled, same as any other placement failure.

### Test strengthened

Added `test_paired_gen_shared_haystack(tok)` to `tests/lmr/test_mc_niah_data.py`,
covering exactly what the reviewer specified:
- (a) `len(enc_s.input_ids) == len(enc_m.input_ids)` — equal total token count.
- (b) `gold_s["tok_start"] == gold_m["tok_start"]` and `gold_s["seg"] == gold_m["seg"]`
  — gold at the identical token position and segment in both variants.
- (c) locate the 3 distractor needle sentences in the multi input via
  `mcdata.NEEDLE_RE` char spans → token spans via offset mapping → build the
  complement (all token indices *not* in any distractor span) → assert
  `enc_s.input_ids[i] == enc_m.input_ids[i]` for every index `i` in the complement.

**Confirmed the test fails against the pre-fix code** (ran it before applying the
fix, to prove it catches the real bug):
```
HF_HOME=/data2/sohyung/hf_home .../python -m pytest tests/lmr/test_mc_niah_data.py::test_paired_gen_shared_haystack -x -q
FAILED — AssertionError: assert 1866 == 1860   (len(enc_s.input_ids) == len(enc_m.input_ids))
```

**After the fix**, full suite (existing 3 tests + new test) — all green:
```
tests/lmr/test_mc_niah_data.py::test_annotate_finds_gold_needle PASSED
tests/lmr/test_mc_niah_data.py::test_annotate_multikey_picks_queried PASSED
tests/lmr/test_mc_niah_data.py::test_paired_gen_invariants PASSED
tests/lmr/test_mc_niah_data.py::test_paired_gen_shared_haystack PASSED
4 passed, 15 warnings in 1.40s
```
Re-ran 3 more times (fresh interpreter, different random module state each run) to
check for flakiness in the filler-search / 8-attempt retry — consistently `4 passed`
every time.

### Data regenerated

```
source lmr/analysis/260725_mc_niah_analysis/env_common.sh
$PY $ANA/data.py prepare-b
[prepare-b] S: 32 rows (16 pairs) -> /data2/sohyung/mc_niah/data/paired/S.jsonl
[prepare-b] D: 32 rows (16 pairs) -> /data2/sohyung/mc_niah/data/paired/D.jsonl
```

### Full re-validation of all 64 regenerated rows

Extended the validation script to also check, per pair, the reviewer's 3
haystack-sharing conditions (not just the brief's original invariants):

```
ALL VALIDATED OK: 64 rows checked, 32 pairs across S/D
  - equal total token count: PASS (all pairs)
  - gold tok_start & seg identical across single/multi: PASS (all pairs)
  - token sequences identical outside distractor spans: PASS (all pairs)
```

Spot-checked `S.jsonl` pid=0 directly (the same pair used to illustrate the original
bug) to confirm the fix concretely:
```
S pid=0: total tokens single/multi = 1858/1858 (equal)
gold tok_start = 1360 (identical in both variants)
divergent token blocks between single/multi: 3 (expected — exactly one per distractor)
  block (394, 412)   len 19
  block (1131, 1149) len 19
  block (1329, 1346) len 18
gold tok_start 1360 lies outside all 3 divergent blocks -> True
```
This is the same pair the reviewer used as an example (previously: common prefix only
394 tokens vs. gold at ~1360, i.e. everything after token 394 differed). Now: exactly
3 short (~18-19 token) divergent blocks — one per distractor sentence, replaced
1:1 by an equal-length filler — and every other token, including the entire region
around and after the gold needle, is byte-for-byte identical between single and multi.

### Files changed (this fix round)

- Modified: `lmr/analysis/260725_mc_niah_analysis/paired_gen.py`
  (added `NEUTRAL_ATOMS`, `_fit_filler`, `_neutralize`; rewrote `_make_one`'s
  single-context construction and acceptance checks)
- Modified: `tests/lmr/test_mc_niah_data.py`
  (added `test_paired_gen_shared_haystack`)
- Regenerated: `/data2/sohyung/mc_niah/data/paired/{S,D}.jsonl` (32 rows each)

### Concerns

- `_fit_filler`'s word-growth assumption (each `NEUTRAL_ATOMS` word = +1 token) was
  verified empirically against the actual TinyLlama tokenizer for the specific pool
  used, both standalone and in-context; it is not a mathematical guarantee for
  arbitrary target token counts, which is why the retry-with-reshuffle (up to 50) and
  the outer 8-attempt resample-the-whole-sample fallback are both still in place. In
  the actual `prepare-b` run (96 filler insertions: 32 pairs × 3 distractors), every
  filler search succeeded — no retries or resamples were observed to be necessary in
  practice.
- The filler text is semantically neutral ("grass green sky ...") but not
  grammatically fluent; this is intentional (matches the reviewer's specified neutral
  filler pool) and only affects the single-needle variant's non-gold filler content,
  never the gold needle sentence itself or the query.
