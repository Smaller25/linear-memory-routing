# Task 1: Pinned Worktree + 실험 환경 스캐폴드

## Summary

Successfully completed Task 1. Created two analysis scaffold files and pinned the long-gdn worktree at commit e71713e.

## Step 1: Worktree Creation & Verification

### Command Execution
```bash
mkdir -p /data2/sohyung/worktrees
git -C /home/sohyung/long-gdn worktree add /data2/sohyung/worktrees/long-gdn-e71713e e71713e
```

### Verification Output

#### File Existence Check
```
/data2/sohyung/worktrees/long-gdn-e71713e/dsc/mc_gdn2/ssc.py
/data2/sohyung/worktrees/long-gdn-e71713e/dsc/mc_baseline/mc_ssc.py
```
✓ Both files exist

#### Docstring Verification (mean of L2-normalized keys)
```
43:    """Return mean of L2-normalized keys per segment (paper-faithful MeanPooling).
```
✓ Found at line 43 in mc_ssc.py

#### Config Verification (mc_370M)
```
221:        name="mc_370M",
```
✓ Found at line 221 in config.py

All Step 1 verifications PASSED.

## Step 2: Environment Scaffold Files Created

### File 1: `lmr/analysis/260725_mc_niah_analysis/env_common.sh`
- Location: `/home/sohyung/linear-memory-routing/lmr/analysis/260725_mc_niah_analysis/env_common.sh`
- Content: 15-line bash script with environment variable exports and directory initialization
- Key exports:
  - `MC_LONGGDN_WORKTREE=/data2/sohyung/worktrees/long-gdn-e71713e`
  - `MC_OUT=/data2/sohyung/mc_niah`
  - `PY=/data2/sohyung/conda-envs/sh_infocap/bin/python`
  - `LMR=/home/sohyung/linear-memory-routing`
  - HuggingFace, Triton, PyTorch CUDA settings

### File 2: `lmr/analysis/260725_mc_niah_analysis/__init__.py`
- Location: `/home/sohyung/linear-memory-routing/lmr/analysis/260725_mc_niah_analysis/__init__.py`
- Content: Empty Python module initialization file

Note: env_common.sh was force-added due to .gitignore/*.sh pattern, but this is appropriate given the brief explicitly requires it.

## Step 3: Commit

```
git add lmr/analysis/260725_mc_niah_analysis/
git commit -m "mc-niah: env scaffold + pinned long-gdn worktree recipe"
```

**Commit Hash:** `27ba5601`

```
27ba5601 mc-niah: env scaffold + pinned long-gdn worktree recipe
```

Files committed:
- `lmr/analysis/260725_mc_niah_analysis/__init__.py` (new file)
- `lmr/analysis/260725_mc_niah_analysis/env_common.sh` (new file with -f flag due to .gitignore)

## Verification Summary

✓ All Step 1 verification greps matched
✓ Both required files created with correct content
✓ Commit successful
✓ No issues or blockers

## Completion Status

**DONE** — All requirements met, all verifications passed, ready for downstream tasks that import model code from the worktree.
