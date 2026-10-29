#!/usr/bin/env bash
# run 의 train.jsonl 을 wandb 에 올린다. 학습 작업은 스스로 올리므로(WANDB_PROJECT), 이건
# 그 기능이 생기기 전에 시작한 run 이나 계산 노드에서 인터넷이 안 될 때 로그인 노드에서 쓴다.
#   bash scripts/wandb_sync.sh                         # RUNS_ROOT 아래 모든 run 을 지금까지 내용만 올리고 끝
#   FOLLOW=1 bash scripts/wandb_sync.sh                # 안 끝난 run 마다 하나씩 띄워 끝날 때까지 계속 올림(tmux 에서)
#   bash scripts/wandb_sync.sh /scratch/$USER/mslm/runs/mc_50m_ctx8192_s0   # 특정 run 만
# 키는 먼저 한 번: conda activate sh_mslm && wandb login   (~/.netrc 에 저장, 파일에 직접 쓰지 않는다)
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/config.sh"
mslm_load_config "${CONFIG:-$MSLM_REPO/configs/base.env}" >/dev/null || exit 1
mslm_activate_env || exit 1
python -c "import wandb" 2>/dev/null || { echo "wandb 가 없습니다: pip install wandb" >&2; exit 1; }
: "${WANDB_PROJECT:=linear-memory-routing}"; export WANDB_PROJECT

if [[ $# -gt 0 ]]; then runs=("$@")
else mapfile -t runs < <(ls -d "$RUNS_ROOT"/*/ 2>/dev/null | sed 's:/$::' | while read -r d; do [[ -f "$d/train.jsonl" ]] && echo "$d"; done)
fi
[[ ${#runs[@]} -eq 0 ]] && { echo "올릴 run 이 없습니다($RUNS_ROOT)"; exit 0; }

if [[ "${FOLLOW:-0}" != 1 ]]; then
  exec python -m mslm.wandb_sync "${runs[@]}"
fi
pids=()
for d in "${runs[@]}"; do
  if grep -q '"event": "end"' "$d/train.jsonl" 2>/dev/null; then   # 끝난 run: 한 번만
    python -m mslm.wandb_sync "$d"
  else
    python -m mslm.wandb_sync --follow "$d" &
    pids+=($!)
  fi
done
echo "[wandb_sync] 따라가는 run ${#pids[@]}개. 끝날 때까지 대기(Ctrl-c 로 멈춰도 학습에는 영향 없음)"
trap 'kill -TERM "${pids[@]}" 2>/dev/null' INT TERM
wait
