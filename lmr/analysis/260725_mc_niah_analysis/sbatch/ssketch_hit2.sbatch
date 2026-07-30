#!/usr/bin/env bash
# ssketch vs meank routing hit@2 @2048 (chunk=256, topk=2).
#
# 한 잡 = arm 하나 × 모델 하나. 잡을 쪼개는 이유는 partition `main`의 6시간
# 제한 때문이다 (arm 5 × 모델 2를 한 잡에 넣으면 8~10시간이라 넘는다).
#
# 사용 (submit_ssketch.sh 가 이걸 감싼다):
#   sbatch --export=ALL,MC_MODEL=mc-5B,MC_DESC=ssketch,MC_RANK=8,MC_SEED=0 \
#          sbatch/ssketch_hit2.sbatch
#
# 변수 (기본값): MC_MODEL=mc-5B MC_DESC=meank MC_RANK=8 MC_SEED=0
#                MC_RQ=q MC_ONLINE=read_norm MC_TAG=ssketch_routing
#
# 주 표 10잡은 `bash sbatch/submit_ssketch.sh` 로 넣는다. 손으로 10번 sbatch 하면
# 안 된다 — 이 클러스터는 큐에 5잡까지만 받아서(QOS normal MaxSubmitPU=5,
# assoc MaxJobs=5) 6번째부터 거절당하고, 그러면 mc-30B arm이 통째로 빠진다.
# submit_ssketch.sh 는 squeue 를 폴링해 한도 아래로 내려갈 때까지 기다렸다 넣는다.
# (그 파일은 repo .gitignore 의 `*.sh` 에 걸려 추적되지 않으므로 이 주석을 남긴다.)
#
#   arm 구성: meank + ssketch r8 seed 0/1/2 + full-state(r=128), × mc-5B/mc-30B
#   부록:     MC_RANK ∈ {4,16} × 시드 3개, MC_RQ=wu, MC_ONLINE=raw_norm|legacy
#             (APPENDIX=1 bash sbatch/submit_ssketch.sh)
#
# 결과는 arm별 shard 파일에 따로 쓰인다(동시 실행 안전, 병합 없음):
#   results/$MC_TAG/<arm_key>.json, $MC_OUT/results/$MC_TAG/<arm_key>_per_sample.json
# 전부 끝난 뒤 GPU 없이 병합 + 그림 + 표:
#   python ssketch_table.py --tag $MC_TAG
#SBATCH -J mc_ssketch
#SBATCH -p main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -t 03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -o /data2/sohyung/mc_niah/logs/ssketch_%j.log
#SBATCH -e /data2/sohyung/mc_niah/logs/ssketch_%j.log
set -eo pipefail
source /home/sohyung/linear-memory-routing/lmr/analysis/260725_mc_niah_analysis/env_common.sh

# 새 모듈은 실험용 worktree에 있다. 모델/커널은 여전히 pinned worktree(e71713e)를
# 쓰고, 여기서 오는 건 점수 수식 파일뿐이다 (sketch_bridge.py 주석 참고).
export MC_SKETCH_ROOT="${MC_SKETCH_ROOT:-/home/sohyung/long-gdn/.sh_exp_worktrees/ssketch-router-descriptor/dsc/mc_sketch}"
export MC_KERNEL_VERSION="${MC_KERNEL_VERSION:-v2}"   # PINNED: 비교는 v2에서만

MC_MODEL="${MC_MODEL:-mc-5B}"
MC_DESC="${MC_DESC:-meank}"
MC_RANK="${MC_RANK:-8}"
MC_SEED="${MC_SEED:-0}"
MC_RQ="${MC_RQ:-q}"
MC_ONLINE="${MC_ONLINE:-read_norm}"
MC_TAG="${MC_TAG:-ssketch_routing}"

echo "[ssketch] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[ssketch] model=$MC_MODEL descriptor=$MC_DESC rank=$MC_RANK seed=$MC_SEED"
echo "[ssketch] router_query=$MC_RQ online_score=$MC_ONLINE tag=$MC_TAG"
echo "[ssketch] worktree=$MC_LONGGDN_WORKTREE sketch_root=$MC_SKETCH_ROOT kernel=$MC_KERNEL_VERSION"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PY "$ANA/routing_stats.py" \
    --model "$MC_MODEL" \
    --descriptor "$MC_DESC" \
    --rank "$MC_RANK" \
    --seed "$MC_SEED" \
    --router-query "$MC_RQ" \
    --online-score "$MC_ONLINE" \
    --out-tag "$MC_TAG"
