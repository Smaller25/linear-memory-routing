#!/usr/bin/env bash
# AC-8: 라우팅 단계 벽시계 시간 + 토큰당 FLOPs + 캐시 메모리.
# meank vs ssketch(r ∈ {4,8,16}) vs full-state, context 2K/8K/32K, chunk=256, topk=2.
#
#   sbatch sbatch/ssketch_bench.sbatch
#
# batch는 1이 기본이다. 32K × B=8 셀은 layer당 memories가 ~2.1 GB(fp32)라
# 유일한 OOM 후보 — 늘리려면 MC_BENCH_BATCH로 명시적으로 올릴 것.
#SBATCH -J mc_ssketch_bench
#SBATCH -p main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -t 02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -o /data2/sohyung/mc_niah/logs/ssketch_bench_%j.log
#SBATCH -e /data2/sohyung/mc_niah/logs/ssketch_bench_%j.log
set -eo pipefail
source /home/sohyung/linear-memory-routing/lmr/analysis/260725_mc_niah_analysis/env_common.sh

WT="${MC_SKETCH_WORKTREE:-/home/sohyung/long-gdn/.sh_exp_worktrees/ssketch-router-descriptor}"
export PYTHONPATH="$WT:$WT/dsc:$PYTHONPATH"
export MC_KERNEL_VERSION="${MC_KERNEL_VERSION:-v2}"

echo "[bench] host=$(hostname) job=${SLURM_JOB_ID:-none} worktree=$WT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PY "$WT/dsc/mc_sketch/scripts/bench_router.py" \
    --device cuda --dtype bf16 \
    --contexts 2048 8192 32768 \
    --ranks 4 8 16 \
    --batch "${MC_BENCH_BATCH:-1}" \
    --iters 20 --warmup 5 \
    --out "$MC_OUT/results/ssketch_bench.json"

cp -v "$MC_OUT/results/ssketch_bench.json" "$ANA/results/ssketch_bench.json"
