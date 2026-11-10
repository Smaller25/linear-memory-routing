#!/usr/bin/env bash
# 평가 실행(RULER NIAH 공식 생성 채점 + 라우팅 적중률). config 와 덮어쓰기 변수만 받는다.
#   CONFIG=configs/mc_50m.env bash scripts/eval.sh                 # 그 run 의 마지막 체크포인트
#   CONFIG=configs/mc_50m.env SEED=1 EVAL_LENGTHS=8192 bash scripts/eval.sh
#   CONFIG=configs/mc_50m.env EVAL_CKPT=/path/step_0010000.pt bash scripts/eval.sh
# 평가 문항이 없으면 먼저 만든다(scripts/prepare_eval.sh 와 같은 일).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/config.sh"
mslm_load_config "${CONFIG:?CONFIG=configs/<name>.env 를 지정하세요}"
mslm_activate_env
mslm_offline
export NLTK_DATA="$HF_CACHE_DIR/nltk_data"

python -m mslm.eval.ruler_prepare
python -m mslm.eval.run_eval
# 평가 결과(summary.json)를 그 run 의 wandb 요약에 올린다. 실패해도 평가 결과 파일은 남는다.
if [[ -n "${WANDB_PROJECT:-}" && -z "${EVAL_OUT:-}" ]]; then
  run_dir="$(python -c 'from mslm.settings import Settings; print(Settings.from_env().run_dir())')"
  python -m mslm.wandb_sync "$run_dir" >> "$run_dir/wandb_sync.log" 2>&1 \
    || echo "[eval] wandb 업로드 실패($run_dir/wandb_sync.log). 결과 파일은 그대로 있음"
fi
