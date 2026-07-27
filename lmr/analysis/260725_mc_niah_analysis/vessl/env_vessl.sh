#!/usr/bin/env bash
# 0025 VESSL job env — source this ON VESSL before every job.
# Mirrors env_common.sh (greenbeard) but rooted at /root/smaller (geesefs) for
# persistent artifacts and container-local /tmp for anything geesefs must
# never receive scratch writes for (HF cache, triton cache, tmpdir).

ROOT=/root/smaller/mc_niah

export MC_LONGGDN_WORKTREE=/root/work/long-gdn   # 컨테이너 로컬 (geesefs git 불안정 — bundle에서 복원)
export MC_CKPT_DIR="$ROOT/ckpts"
export MC_OUT="$ROOT"

# geesefs trap: never point these at /root/smaller — container-local disk only
export HF_HOME=/tmp/hf_home
export TMPDIR=/tmp
export TRITON_CACHE_DIR=/tmp/.triton
export XDG_CACHE_HOME=/tmp/cache

export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# fla pinned at 4b02d15d (worktree kernels need pre-0d0a2f9a GLA API) — see
# lmr/analysis/260725_mc_niah_analysis/env_common.sh for the greenbeard twin.
export PYTHONPATH="/root/work/pydeps:$MC_LONGGDN_WORKTREE:$MC_LONGGDN_WORKTREE/dsc${PYTHONPATH:+:$PYTHONPATH}"

PY=/opt/conda/bin/python
LMR="$ROOT/code/linear-memory-routing"
ANA="$LMR/lmr/analysis/260725_mc_niah_analysis"

mkdir -p "$ROOT"/{data,logs,results} /tmp/hf_home /tmp/.triton /tmp/cache
