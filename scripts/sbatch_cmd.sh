#!/usr/bin/env bash
# config 로부터 sbatch 명령을 만들어 화면에 보여 준다(제출은 하지 않음 — 복사해서 직접 실행).
#   CONFIG=configs/mc_50m.env bash scripts/sbatch_cmd.sh train
#   CONFIG=configs/mc_50m.env SEED=1 bash scripts/sbatch_cmd.sh train
#   CONFIG=configs/mc_50m.env SB_GPU_OPT=--gpus-per-node=1 SB_EXCLUSIVE=0 bash scripts/sbatch_cmd.sh eval
# JOB: train | eval | prepare_data | prepare_eval
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/config.sh"
JOB="${1:?usage: sbatch_cmd.sh train|eval|prepare_data|prepare_eval}"
[[ -f "$MSLM_REPO/slurm/$JOB.sbatch" ]] || { echo "no slurm/$JOB.sbatch" >&2; exit 1; }
mslm_load_config "${CONFIG:?CONFIG=configs/<name>.env 를 지정하세요}"
mslm_sbatch_cmd "$JOB"
