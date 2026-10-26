#!/usr/bin/env bash
# RULER NIAH 평가 문항 생성(학습과 같은 토크나이저, 고정 시드). 결과는 EVAL_DATA_DIR(jsonl).
#   CONFIG=configs/mc_50m.env bash scripts/prepare_eval.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/config.sh"
mslm_load_config "${CONFIG:?CONFIG=configs/<name>.env 를 지정하세요}"
mslm_activate_env
mslm_offline
export NLTK_DATA="$HF_CACHE_DIR/nltk_data"
exec python -m mslm.eval.ruler_prepare
