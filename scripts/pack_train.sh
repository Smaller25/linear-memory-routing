#!/usr/bin/env bash
# pack 작업 안에서 실행된다(slurm/pack.sbatch). PACK_FILE 의 줄(<run>\t<config>\t<KEY=val ...>)마다
# GPU 하나를 주고 scripts/train.sh 를 띄운 뒤, 모두 끝날 때까지 기다린다. 하나라도 실패하면 exit 1.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PACK_FILE:?PACK_FILE 이 없습니다. scripts/run_pack.sh 로 제출하세요}"
TRAIN="${PACK_TRAIN:-bash $REPO/scripts/train.sh}"   # PACK_JOB=eval 이면 scripts/eval.sh (테스트에서는 stub)
LOG_DIR="$(dirname "$PACK_FILE")"

mapfile -t RUNS < <(grep -v '^[[:space:]]*\(#\|$\)' "$PACK_FILE")
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then IFS=, read -ra DEVS <<<"$CUDA_VISIBLE_DEVICES"
else mapfile -t DEVS < <(seq 0 $((${#RUNS[@]} - 1))); fi
[[ ${#RUNS[@]} -le ${#DEVS[@]} ]] || { echo "[pack] run ${#RUNS[@]}개 > GPU ${#DEVS[@]}개 ($CUDA_VISIBLE_DEVICES)" >&2; exit 1; }

PIDS=() NAMES=()
for i in "${!RUNS[@]}"; do
  IFS=$'\t' read -r name cfg kv <<<"${RUNS[$i]}"
  log="$LOG_DIR/${name}_${SLURM_JOB_ID:-local}.log"
  echo "[pack] gpu ${DEVS[$i]}  $name  config=$cfg${kv:+  [$kv]}  log=$log"
  # shellcheck disable=SC2086
  ( export CUDA_VISIBLE_DEVICES="${DEVS[$i]}" NGPU_PER_NODE=1 CONFIG="$cfg"
    for a in $kv; do export "$a"; done
    exec $TRAIN ) >"$log" 2>&1 &
  PIDS+=($!); NAMES+=("$name")
done

rc=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then echo "[pack] 끝남  ${NAMES[$i]}"
  else r=$?; rc=1; echo "[pack] 실패  ${NAMES[$i]} (exit $r, 로그 $LOG_DIR/${NAMES[$i]}_${SLURM_JOB_ID:-local}.log)"; fi
done
exit "$rc"
