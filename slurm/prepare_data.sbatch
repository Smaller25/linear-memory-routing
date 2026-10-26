#!/bin/bash
# 학습 데이터 사전 토큰화 작업. 직접 sbatch 하지 말고 scripts/sbatch_cmd.sh prepare_data 이 만든 명령을 쓰세요:
#   CONFIG=configs/mc_50m.env bash scripts/sbatch_cmd.sh prepare_data
# (파티션·GPU·시간·--comment 등은 그 명령의 옵션으로 config 에서 채워진다.)
#SBATCH --ntasks-per-node=1
set -euo pipefail
: "${MSLM_REPO:?MSLM_REPO 가 없습니다. scripts/sbatch_cmd.sh 가 만든 명령으로 제출하세요}"
: "${CONFIG:?CONFIG 가 없습니다}"
echo "[slurm] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST host=$(hostname) repo=$MSLM_REPO config=$CONFIG"
exec bash "$MSLM_REPO/scripts/prepare_data.sh"
