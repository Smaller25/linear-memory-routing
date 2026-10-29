#!/bin/bash
# 연기 시험 작업(GPU 점검·학습·재개·평가). 직접 sbatch 하지 말고 scripts/sbatch_cmd.sh smoke 가 만든 명령을 쓰세요:
#   CONFIG=configs/smoke_kisti.env bash scripts/sbatch_cmd.sh smoke
#SBATCH --ntasks-per-node=1
set -euo pipefail
: "${MSLM_REPO:?MSLM_REPO 가 없습니다. scripts/sbatch_cmd.sh 가 만든 명령으로 제출하세요}"
: "${CONFIG:?CONFIG 가 없습니다}"
echo "[slurm] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST host=$(hostname) repo=$MSLM_REPO config=$CONFIG"
exec bash "$MSLM_REPO/scripts/smoke.sh"
