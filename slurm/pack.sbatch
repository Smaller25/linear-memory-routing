#!/bin/bash
# GPU 1장짜리 학습 여러 개를 한 노드에서 함께 돌리는 작업. 직접 sbatch 하지 말고 scripts/run_pack.sh 로 내세요:
#   MAX_STEPS=5700 bash scripts/run_pack.sh stage1a mc:RUN_TAG=mon ce:RUN_TAG=ce_w003,MC_AUX_WEIGHT=0.03
# (파티션·GPU·시간·--comment 등은 첫 run 의 config 에서 채워진다. 어떤 run 을 돌릴지는 PACK_FILE.)
#SBATCH --ntasks-per-node=1
set -euo pipefail
: "${MSLM_REPO:?MSLM_REPO 가 없습니다. scripts/run_pack.sh 로 제출하세요}"
: "${PACK_FILE:?PACK_FILE 이 없습니다}"
echo "[slurm] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST host=$(hostname) repo=$MSLM_REPO pack=$PACK_FILE"
unset NCCL_NET; export NCCL_NET_PLUGIN=none   # 단일 노드(train.sbatch 와 같은 이유)
srun --ntasks-per-node=1 --kill-on-bad-exit=1 bash "$MSLM_REPO/scripts/pack_train.sh"
