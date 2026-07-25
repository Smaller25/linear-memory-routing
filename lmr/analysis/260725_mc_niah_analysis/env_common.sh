#!/usr/bin/env bash
# MC NIAH 분석 공통 환경 — 모든 sbatch가 source
export HF_HOME=/data2/sohyung/hf_home TMPDIR=/data2/sohyung/tmp XDG_CACHE_HOME=/data2/sohyung/cache
export TRITON_CACHE_DIR=/data2/sohyung/tmp/.triton
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MC_LONGGDN_WORKTREE=/data2/sohyung/worktrees/long-gdn-e71713e
export MC_OUT=/data2/sohyung/mc_niah
PY=/data2/sohyung/conda-envs/sh_infocap/bin/python
LMR=/home/sohyung/linear-memory-routing
ANA=$LMR/lmr/analysis/260725_mc_niah_analysis
mkdir -p "$MC_OUT"/{data,logs,results} /data2/sohyung/tmp/.triton
