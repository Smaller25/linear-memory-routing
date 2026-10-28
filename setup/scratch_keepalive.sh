#!/usr/bin/env bash
# /scratch 는 15일 동안 접근(atime)이 없는 파일을 지운다(대상은 ToBeDelete_ 접두사가 붙은 뒤 20~30일 후 일괄 삭제).
# 오래 쉬는 동안 데이터·체크포인트를 지키려면 이 스크립트로 접근 시각을 갱신하거나, 백업을 받아 둔다.
#   CONFIG=configs/mc_50m.env bash setup/scratch_keepalive.sh            # 접근 시각 갱신(touch -a)
#   CONFIG=configs/mc_50m.env BACKUP=1 bash setup/scratch_keepalive.sh   # 각 run 의 마지막 체크포인트를 BACKUP_ROOT 로 복사
# 접근 시각 확인: ls -lu <파일> 또는 stat <파일>.  ToBeDelete_* 파일이 보이면 이름을 되돌리고 이 스크립트를 실행.
# 외부(연구실 서버) 백업은 데이터무버로: sftp <ID>@hangang-dm.ksc.re.kr  (get -r <run>/ckpt)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/lib/config.sh"
mslm_load_config "${CONFIG:?CONFIG=configs/<name>.env 를 지정하세요}"

for d in "$TRAIN_DATA_DIR" "$EVAL_DATA_DIR" "$RUNS_ROOT" "$HF_CACHE_DIR"; do
  [[ -d "$d" ]] || continue
  n=$(find "$d" -type f -exec touch -a {} + -print | wc -l)
  echo "[keepalive] $d: $n files touched"
done
found=$(find "$SCRATCH_ROOT" -name 'ToBeDelete_*' 2>/dev/null | head -5)
[[ -n "$found" ]] && echo "[keepalive] 경고: 삭제 예정 표시된 파일이 있습니다:"$'\n'"$found"

if [[ "${BACKUP:-0}" == "1" ]]; then
  [[ -n "$BACKUP_ROOT" ]] || { echo "BACKUP_ROOT 가 비어 있습니다"; exit 1; }
  for latest in "$RUNS_ROOT"/*/ckpt/latest; do
    [[ -f "$latest" ]] || continue
    run=$(basename "$(dirname "$(dirname "$latest")")")
    mkdir -p "$BACKUP_ROOT/$run"
    cp -p "$(dirname "$latest")/$(cat "$latest")" "$(dirname "$latest")/../config.env" "$(dirname "$latest")/../train.jsonl" "$BACKUP_ROOT/$run/"
    echo "[backup] $run -> $BACKUP_ROOT/$run"
  done
fi
