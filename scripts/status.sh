#!/usr/bin/env bash
# 진행 상황 한 화면: 내 SLURM 작업, run 별 마지막 학습/val 기록과 평가 결과, 최근 실패 로그.
#   bash scripts/status.sh
#   CONFIG=configs/smoke_greenbeard.env bash scripts/status.sh   # 다른 SCRATCH_ROOT 를 쓰는 config
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/lib/config.sh"
mslm_load_config "${CONFIG:-$REPO/configs/base.env}" >/dev/null || exit 1

echo "== SLURM 작업 ($USER)"
if command -v squeue >/dev/null; then
  squeue -u "$USER" -o "%.10i %.24j %.9T %.10M %.11l %.6D %R" 2>/dev/null || true
else
  echo "  squeue 없음"
fi

echo; echo "== run ($RUNS_ROOT)"
python3 - "$RUNS_ROOT" <<'EOF'
import glob, json, os, sys
root = sys.argv[1]
runs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isfile(os.path.join(d, "train.jsonl")))
if not runs:
    print("  아직 run 없음")
for d in runs:
    last = val = start = end = None
    for line in open(os.path.join(d, "train.jsonl")):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "loss" in r: last = r
        if "val_loss" in r: val = r
        if r.get("event") == "start": start = r
        if r.get("event") == "end": end = r
    total = start.get("total_steps") if start else "?"
    s = f"  {os.path.basename(d):32s}"
    if last:
        s += f" step {last['step']}/{total} loss {last['loss']:.4f} {last.get('tok_per_s', 0)/1e3:.0f}k tok/s"
    if val:
        s += f" | val {val['val_loss']:.4f}@{val['step']}"
    if end and last and end["step"] == last["step"]:
        s += " | 끝"
    evals = sorted(glob.glob(os.path.join(d, "eval", "*", "summary.json")))
    if evals:
        s += f" | 평가 {len(evals)}개(마지막 {os.path.basename(os.path.dirname(evals[-1]))})"
    print(s)
EOF

echo; echo "== 데이터"
[[ -f "$TRAIN_DATA_DIR/meta.json" ]] && echo "  [ok] 학습 데이터 $TRAIN_DATA_DIR" || echo "  [--] 학습 데이터 아직 없음($TRAIN_DATA_DIR)"
ls -d "$EVAL_DATA_DIR"/*/ >/dev/null 2>&1 && echo "  [ok] 평가 문항 $EVAL_DATA_DIR" || echo "  [--] 평가 문항 아직 없음($EVAL_DATA_DIR)"

echo; echo "== 최근 로그 ($SB_LOG_DIR)"
if ls "$SB_LOG_DIR"/*.err >/dev/null 2>&1; then
  for f in $(ls -t "$SB_LOG_DIR"/*.err | head -5); do
    if grep -qiE "error|traceback|exception|cancelled|killed" "$f"; then
      echo "  !! $(basename "$f")"; grep -iE "error|exception|cancelled|killed" "$f" | tail -3 | sed 's/^/     /'
    else
      echo "  ok $(basename "$f")"
    fi
  done
else
  echo "  로그 없음"
fi
ls -t "$SB_SUBMIT_DIR"/pipeline_*.txt 2>/dev/null | head -1 | sed 's/^/  최근 제출 목록: /'
