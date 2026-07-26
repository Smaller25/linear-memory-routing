# Task 3 Report: Dataset A prepare + gold-chunk annotation

## Status: DONE

## Commit
`8d73ed0e` — "mc-niah: Dataset A prepare (RULER, TinyLlama tok) + needle annotation"
(branch `sh/mc-niah-analysis`)

Files added:
- `lmr/analysis/260725_mc_niah_analysis/data.py`
- `tests/lmr/test_mc_niah_data.py`

## TDD flow followed
1. Wrote `tests/lmr/test_mc_niah_data.py` exactly per brief.
2. Confirmed failure: `AttributeError: module 'data' has no attribute 'annotate'`
   (note: without a real `data.py`, `import data` resolved to the repo-root
   `data/` directory as an implicit namespace package rather than raising
   `ModuleNotFoundError` — expected/harmless, just a slightly different
   failure signature than "no module named data").
3. Wrote `data.py` per brief's Step 3 code, with two deltas from the literal
   brief text (both required to make `prepare-a` actually work — see below):
   - `prepare_a`'s essay-existence check now points at
     `src/ruler/gen/synthetic/json/PaulGrahamEssays.json` (where the vendored
     RULER scripts actually read/write it) instead of the brief's
     `data/PaulGrahamEssays.json` at repo root, which is not a path any RULER
     code reads. The essay file already existed there from prior work, so
     with the brief's literal path check `prepare_a` would have redundantly
     re-triggered `scripts/ruler.py download-essays` on every run.
   - `prepare_a`'s subprocess call now passes `env=` with this interpreter's
     `bin/` directory prepended to `PATH`. Root-caused via
     `systematic-debugging`-style bisection: `src/ruler/gen/prepare.py`
     re-invokes each task's generator script via a **hardcoded shell string
     `"python {script} ..."`** (not `sys.executable`), so it silently picks
     up whatever `python` resolves to on the *caller's* `PATH` — in this
     environment that's `/home/compu/anaconda3/bin/python` (3.13, missing
     `wonderwords`/`tenacity`), not the `sh_infocap` env `data.py` itself runs
     under. The brief's assumption that `sys.executable` propagates through
     was wrong for this one hop; fixed entirely in `data.py`, no vendored
     file touched.
4. Confirmed 2/2 tests pass.
5. Ran `prepare-a`; hit two missing deps surfaced via the above PATH fix
   (previously masked because the wrong interpreter was silently used and
   failures only showed up as empty output files, not raised exceptions):
   - `tenacity` — not installed in `sh_infocap` env. Fixed with
     `pip install tenacity` directly into the env (not a `--target` pydeps
     shim; it's a plain third-party dep, no fla/env pinning concerns).
   - `pytest` itself was also missing from `sh_infocap` (`No module named
     pytest`) — installed via `pip install pytest` into the same env before
     any test could run.
   - `wonderwords` was already present in `sh_infocap` (pre-existing from
     earlier task setup); no action needed.
   - nltk `punkt` was already cached at `/home/sohyung/nltk_data` (not under
     `/data2`), found via nltk's default search path with no `NLTK_DATA`
     override needed — verified directly, so I did **not** add the
     speculative `NLTK_DATA=/data2/...` env var the brief anticipated as a
     "common cause": it wasn't a real failure in this environment, and
     adding an unused, nonexistent-directory fallback would have been
     speculative noise. If a fresh environment lacks `~/nltk_data/punkt`,
     that fix path (download to `/data2/sohyung/cache/nltk_data` +
     `NLTK_DATA` export) remains the correct one, just not needed here.
   - No changes needed to `env_common.sh` (no new required env var — the
     PATH fix lives inside `data.py`'s subprocess call, and `data.py` runs
     correctly both with and without sourcing `env_common.sh`, verified both
     ways).
6. Re-ran `prepare-a` clean (removed prior partial output first) — both
   tasks produced 50/50 samples.
7. Ran the annotate-validation snippet from Step 5 of the brief over all 100
   generated samples (50 + 50) — zero exceptions.

**No files under `src/ruler/` or `scripts/ruler.py` were modified.** All
fixes are contained in `lmr/analysis/260725_mc_niah_analysis/data.py` (PATH
prepend for the subprocess call, corrected essay-path check) plus two `pip
install`s into the `sh_infocap` conda env (`tenacity`, `pytest`).

## Test / run summary
- `pytest tests/lmr/test_mc_niah_data.py -x -q` → **2 passed**
- `prepare-a` → `niah_single_1: 50 samples`, `niah_multikey_1: 50 samples`
  → `$MC_OUT/data/2048/{niah_single_1,niah_multikey_1}/validation.jsonl`
- annotate validation over all 100 samples → no exceptions;
  `niah_single_1`: n=50, gold_seg range 0–7, n_tok max 1900;
  `niah_multikey_1`: n=50, gold_seg range 0–5, n_tok max 1905
  (both within the expected ≤~1920 token / 0–7 segment envelope for a
  2048-token 256-chunk layout)

## Concerns for later tasks
- `sh_infocap` env now has `tenacity` and `pytest` installed that weren't
  there before (not pinned in any lockfile I could find) — worth capturing
  in whatever env-reproducibility notes Task 1's scaffold keeps, since a
  fresh clone of that env would hit the same two missing-dep failures.
- The vendored `prepare.py`'s hardcoded `"python ..."` re-invocation is a
  latent footgun for any future caller that doesn't route through
  `data.py`'s `prepare_a` (e.g. calling `prepare.py` directly with a
  differently-ordered `PATH`) — flagging in case Task 4's `paired_gen.py`
  needs the same subprocess call, so it should reuse (or replicate) this
  `PATH`-prepend rather than re-discovering the same bug.
- `json` is imported in `data.py` per the brief's exact Step 3 code but is
  currently unused (no `json.` calls in this file). Left as-is to match the
  brief verbatim since Task 4/`paired_gen.py` may end up needing it in this
  module too; flagging in case a later lint pass wants it removed if it
  stays unused.
