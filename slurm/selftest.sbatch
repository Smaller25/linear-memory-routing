#!/bin/bash
# GPU 점검 작업. 직접 sbatch 하지 말고 scripts/sbatch_cmd.sh selftest 가 만든 명령을 쓰세요:
#   CONFIG=configs/smoke_kisti.env bash scripts/sbatch_cmd.sh selftest
#SBATCH --ntasks-per-node=1
set -euo pipefail
: "${MSLM_REPO:?MSLM_REPO 가 없습니다. scripts/sbatch_cmd.sh 가 만든 명령으로 제출하세요}"
: "${CONFIG:?CONFIG 가 없습니다}"
echo "[slurm] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST host=$(hostname) repo=$MSLM_REPO config=$CONFIG"
exec bash "$MSLM_REPO/scripts/selftest.sh"
