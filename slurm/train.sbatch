#!/bin/bash
# 학습 작업. 직접 sbatch 하지 말고 scripts/sbatch_cmd.sh train 이 만든 명령을 쓰세요:
#   CONFIG=configs/mc_50m.env bash scripts/sbatch_cmd.sh train
# (파티션·GPU·시간·--comment 등은 그 명령의 옵션으로 config 에서 채워진다.)
#SBATCH --ntasks-per-node=1
set -euo pipefail
: "${MSLM_REPO:?MSLM_REPO 가 없습니다. scripts/sbatch_cmd.sh 가 만든 명령으로 제출하세요}"
: "${CONFIG:?CONFIG 가 없습니다}"
echo "[slurm] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST host=$(hostname) repo=$MSLM_REPO config=$CONFIG"
# 단일 노드면 NCCL 네트워크 플러그인을 끈다(KISTI 안내: Slingshot VNI가 없어 초기화 실패 가능).
if [[ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]]; then unset NCCL_NET; export NCCL_NET_PLUGIN=none; fi
# 다중 노드: KISTI 안내대로 LL128 끄기. TODO(확인 필요): pip torch 의 NCCL 과 aws-ofi-nccl 플러그인 호환.
if [[ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]]; then export NCCL_PROTO=^LL128; fi
srun --ntasks-per-node=1 --kill-on-bad-exit=1 bash "$MSLM_REPO/scripts/train.sh"
