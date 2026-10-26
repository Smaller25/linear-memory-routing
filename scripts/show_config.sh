#!/usr/bin/env bash
# 최종 변수값 확인(아무것도 실행하지 않음).
#   CONFIG=configs/mc_50m.env SEED=2 bash scripts/show_config.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/config.sh"
mslm_load_config "${CONFIG:-}"
mslm_print_config
echo "# overrides: ${MSLM_OVERRIDES:-none}"
