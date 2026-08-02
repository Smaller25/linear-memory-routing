#!/usr/bin/env bash
# [C] 선행 차단 항목 + mc_sketch 단위 테스트.
#
# 1) test_triton_ssc.py — 커밋 4d08c73의 grad_w fp32 수정이 아직 GPU에서
#    검증되지 않았다. CUDA 전용이라 sbatch로 한 번 통과시켜야 학습 arm으로 갈 수
#    있다. (추론 전용 hit@2 진단만 돌릴 거면 병렬로 처리해도 된다.)
# 2) dsc/mc_sketch 4종 테스트 (CPU에서도 도는 것들; 여기서는 GPU 노드에서 한 번 더).
# 3) AC-10 커널 무변경 확인.
#
#   sbatch sbatch/ssketch_tests.sbatch
#SBATCH -J mc_ssketch_tests
#SBATCH -p main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -t 00:45:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -o /data2/sohyung/mc_niah/logs/ssketch_tests_%j.log
#SBATCH -e /data2/sohyung/mc_niah/logs/ssketch_tests_%j.log
set -eo pipefail
source /home/sohyung/linear-memory-routing/lmr/analysis/260725_mc_niah_analysis/env_common.sh

WT="${MC_SKETCH_WORKTREE:-/home/sohyung/long-gdn/.sh_exp_worktrees/ssketch-router-descriptor}"
export PYTHONPATH="$WT:$WT/dsc:$PYTHONPATH"
export MC_KERNEL_VERSION=v2

echo "=== [3] AC-10: kernel / backbone unchanged ==="
bash "$WT/dsc/mc_sketch/scripts/check_no_kernel_change.sh"

echo "=== [2] mc_sketch unit tests ==="
$PY -m dsc.mc_sketch.tests.run_all

echo "=== [1] grad_w fp32 GPU verification (blocks the training arm only) ==="
cd "$WT"
$PY dsc/mc_baseline/tests/test_triton_ssc.py
