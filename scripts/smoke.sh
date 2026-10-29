#!/usr/bin/env bash
# 연기 시험 한 바퀴를 작업 하나 안에서: GPU 점검 → 학습 MAX_STEPS → 재개해서 +10 step → 평가.
# 모두 통과하면 run 폴더에 SMOKE_OK 를 남긴다(submit_pipeline.sh 가 이것을 보고 본학습을 건다).
#   CONFIG=configs/smoke_kisti.env bash scripts/smoke.sh                 # GPU 노드 안에서
#   CONFIG=configs/smoke_kisti.env bash scripts/sbatch_cmd.sh smoke
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/config.sh"
mslm_load_config "${CONFIG:?CONFIG=configs/<name>.env 를 지정하세요}"
mslm_activate_env
RUN_DIR="$(python -c 'from mslm.settings import Settings; print(Settings.from_env().run_dir())')"
first="$MAX_STEPS"
[[ "$first" -gt 0 ]] || { echo "[smoke] MAX_STEPS 가 0 입니다(smoke config 를 쓰세요)" >&2; exit 1; }
rm -f "$RUN_DIR/SMOKE_OK"
echo "[smoke] 1/4 GPU 점검";               bash "$HERE/selftest.sh"
echo "[smoke] 2/4 학습 → step $first";     MAX_STEPS="$first" bash "$HERE/train.sh"
echo "[smoke] 3/4 재개 → step $((first + 10))"; MAX_STEPS=$((first + 10)) bash "$HERE/train.sh"
grep -q "\"resumed_from\": \"" "$RUN_DIR/train.jsonl" || { echo "[smoke] 재개 기록이 없습니다" >&2; exit 1; }
echo "[smoke] 4/4 평가";                   bash "$HERE/eval.sh"
date -Is > "$RUN_DIR/SMOKE_OK"
echo "[smoke] 통과: $RUN_DIR/SMOKE_OK"
