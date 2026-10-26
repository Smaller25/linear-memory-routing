#!/usr/bin/env bash
# 학습 실행. config 파일과 덮어쓰기 변수만 받는다.
#   CONFIG=configs/mc_50m.env bash scripts/train.sh
#   CONFIG=configs/mc_50m.env MAX_TOKENS=1e9 SEED=1 bash scripts/train.sh
# SLURM 안에서는 slurm/train.sbatch 가 노드마다 이 스크립트를 한 번 실행한다(srun).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/config.sh"
mslm_load_config "${CONFIG:?CONFIG=configs/<name>.env 를 지정하세요}"
mslm_activate_env
mslm_offline

NNODES="${SLURM_JOB_NUM_NODES:-1}"
NODE_RANK="${SLURM_NODEID:-0}"
echo "[train] config=$CONFIG run=$(python -c 'from mslm.settings import Settings; print(Settings.from_env().run_dir())') nodes=$NNODES gpus/node=$NGPU_PER_NODE overrides=[${MSLM_OVERRIDES}]"

if [[ "$NGPU_PER_NODE" -le 1 && "$NNODES" -le 1 ]]; then
  exec python -m mslm.train
fi

if [[ "$NNODES" -gt 1 ]]; then
  MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
  MASTER_PORT="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"
  exec torchrun --nnodes="$NNODES" --nproc-per-node="$NGPU_PER_NODE" \
    --rdzv-id="${SLURM_JOB_ID:-mslm}" --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m mslm.train
fi
exec torchrun --standalone --nproc-per-node="$NGPU_PER_NODE" -m mslm.train
