#!/usr/bin/env bash
# FineWeb-Edu 사전 토큰화(학습 shard 만들기). 인터넷 없이 HF_CACHE_DIR 의 원본만 읽는다.
# 먼저 인터넷 되는 노드에서 setup/download_assets.sh 를 돌려 원본을 받아 두세요.
#   CONFIG=configs/mc_50m.env bash scripts/prepare_data.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/config.sh"
mslm_load_config "${CONFIG:?CONFIG=configs/<name>.env 를 지정하세요}"
mslm_activate_env
mslm_offline
exec python -m mslm.data.pretokenize
