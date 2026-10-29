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
RUN_DIR="$(python -c 'from mslm.settings import Settings; print(Settings.from_env().run_dir())')"
echo "[train] config=$CONFIG run=$RUN_DIR nodes=$NNODES gpus/node=$NGPU_PER_NODE overrides=[${MSLM_OVERRIDES}]"

run_train() {
  if [[ "$NGPU_PER_NODE" -le 1 && "$NNODES" -le 1 ]]; then
    python -m mslm.train
  elif [[ "$NNODES" -gt 1 ]]; then
    MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
    MASTER_PORT="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"
    torchrun --nnodes="$NNODES" --nproc-per-node="$NGPU_PER_NODE" \
      --rdzv-id="${SLURM_JOB_ID:-mslm}" --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
      -m mslm.train
  else
    torchrun --standalone --nproc-per-node="$NGPU_PER_NODE" -m mslm.train
  fi
}

# wandb: train.jsonl 을 옆에서 올리는 별도 프로세스(학습과 무관하게 실패해도 학습은 계속). 첫 노드에서만.
WB_PID=""
if [[ -n "$WANDB_PROJECT" && "$NODE_RANK" == 0 ]]; then
  if python -c "import wandb" 2>/dev/null; then
    mkdir -p "$RUN_DIR"
    python -m mslm.wandb_sync --follow "$RUN_DIR" >> "$RUN_DIR/wandb_sync.log" 2>&1 &
    WB_PID=$!
    echo "[train] wandb project=$WANDB_PROJECT log=$RUN_DIR/wandb_sync.log"
  else
    echo "[train] WANDB_PROJECT=$WANDB_PROJECT 이지만 wandb 가 설치돼 있지 않아 건너뜀(pip install wandb)"
  fi
fi

rc=0
run_train || rc=$?
if [[ -n "$WB_PID" ]]; then
  kill -TERM "$WB_PID" 2>/dev/null || true   # 남은 줄을 마저 올리고 끝낸다
  wait "$WB_PID" 2>/dev/null || true
fi
exit "$rc"
