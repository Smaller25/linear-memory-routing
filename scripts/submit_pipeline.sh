#!/usr/bin/env bash
# 실험 전체를 SLURM 에 건다. 같은 명령을 다시 치면 이어서 진행한다(끝난 단계·큐에 있는 작업은 건너뜀).
#
#   prepare_data ──→ smoke(작업 1개: GPU 점검·학습·재개·평가) ──→ [MAIN=1] 본학습 config×seed
#   prepare_eval    (본평가 문항, 독립)
#
#   cd ~/linear-memory-routing && DRY_RUN=1 bash scripts/submit_pipeline.sh   # 무엇을 낼지 보기만
#   bash scripts/submit_pipeline.sh                                           # 준비 + smoke
#   MAIN=1 bash scripts/submit_pipeline.sh                                    # smoke 통과 조건으로 본학습까지
#
# 다시 실행하면: 학습 데이터(meta.json)·평가 문항이 있으면 준비 생략, SMOKE_OK 가 있으면 smoke 생략,
#   끝난 본학습 run 은 생략, 끊긴(시간 초과 등) run 은 다시 내서 마지막 체크포인트부터 잇는다(RESUME=auto).
# 사용자당 제출 수 한도(QOS)에 걸리면 거기서 멈춘다. 앞 작업이 끝난 뒤 같은 명령을 다시 치면 나머지를 낸다.
#
# 바꿀 수 있는 값(환경변수):
#   SMOKE_CONFIG=configs/smoke_kisti.env   DATA_CONFIG=configs/mc_50m.env
#   MAIN_CONFIGS="configs/vanilla_50m.env configs/mc_50m.env"   SEEDS="0 1 2"
#   SKIP_SMOKE=1    smoke 없이 본학습을 건다      PREP_DATA_TIME=06:00:00  PREP_EVAL_TIME=01:00:00
# 그 밖의 변수(SB_COMMENT 등)는 평소처럼 앞에 붙이면 모든 작업에 적용된다. 진행 확인: bash scripts/status.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
SMOKE_CONFIG="${SMOKE_CONFIG:-configs/smoke_kisti.env}"
DATA_CONFIG="${DATA_CONFIG:-configs/mc_50m.env}"
MAIN_CONFIGS="${MAIN_CONFIGS:-configs/vanilla_50m.env configs/mc_50m.env}"
SEEDS="${SEEDS:-0 1 2}"
MAIN="${MAIN:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
PREP_DATA_TIME="${PREP_DATA_TIME:-06:00:00}"
PREP_EVAL_TIME="${PREP_EVAL_TIME:-01:00:00}"

# cfg_get CONFIG KEY [VAR=val ...] : resolved config value (same rules as every script)
cfg_get() {
  local cfg="$1" key="$2"; shift 2
  env "$@" bash -c 'source "$1/scripts/lib/config.sh"; mslm_load_config "$2" >/dev/null || exit 1; printf "%s" "${!3}"' _ "$REPO" "$cfg" "$key"
}
# run_dir CONFIG [VAR=val ...]
run_dir() {
  local cfg="$1"; shift
  env "$@" bash -c 'source "$1/scripts/lib/config.sh"; mslm_load_config "$2" >/dev/null || exit 1
    PYTHONPATH="$1" python3 -c "from mslm.settings import Settings; print(Settings.from_env().run_dir())"' _ "$REPO" "$cfg"
}
# run_finished RUN_DIR : train.jsonl has an "end" event at total_steps
run_finished() {
  python3 - "$1/train.jsonl" <<'EOF'
import json, sys
total = end = None
try:
    for line in open(sys.argv[1]):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("event") == "start": total = r.get("total_steps")
        if r.get("event") == "end": end = r.get("step")
except FileNotFoundError:
    sys.exit(1)
sys.exit(0 if total is not None and end == total else 1)
EOF
}

# queued job name -> id (pending/running)
declare -A QUEUED=()
if command -v squeue >/dev/null; then
  while read -r id name; do [[ -n "$name" ]] && QUEUED["$name"]="$id"; done < <(squeue -u "$USER" -h -o "%i %j" 2>/dev/null)
fi

SUBMITTED=() PENDING_LEFT=() LAST_ID="" STOPPED=0
# submit NAME CONFIG JOB DEPENDENCY [VAR=val ...] : sets LAST_ID (queued id, new id, or "" if not submitted)
submit() {
  local name="$1" cfg="$2" job="$3" dep="$4"; shift 4
  local jobname="mslm-$name" extra cmd out
  LAST_ID=""
  if [[ -n "${QUEUED[$jobname]:-}" ]]; then
    LAST_ID="${QUEUED[$jobname]}"; echo "[pipeline] 큐에 있음  $LAST_ID  $name"; return
  fi
  if [[ "$STOPPED" == 1 ]]; then PENDING_LEFT+=("$name"); return; fi
  extra="$(cfg_get "$cfg" SB_EXTRA "$@")"
  # 앞 작업이 실패하면 기다리지 말고 같이 취소(--kill-on-invalid-dep)
  [[ -n "$dep" ]] && extra+=" --dependency=$dep --kill-on-invalid-dep=yes"
  cmd="$(env "$@" SB_JOB_NAME="$jobname" SB_EXTRA="--parsable${extra:+ $extra}" CONFIG="$cfg" \
         bash "$REPO/scripts/sbatch_cmd.sh" "$job" | sed "s/--job-name=$jobname-$job/--job-name=$jobname/")"
  if [[ "$DRY_RUN" == 1 ]]; then
    LAST_ID="<$name>"; echo "[dry] $name${dep:+  (after $dep)}"; echo "      $cmd"; return
  fi
  if out="$(bash -c "$cmd" 2>&1)"; then
    LAST_ID="${out%%;*}"; SUBMITTED+=("$LAST_ID $name"); echo "[pipeline] 제출      $LAST_ID  $name${dep:+  (after $dep)}"
  elif grep -qiE "QOSMaxSubmitJob|job submit limit" <<<"$out"; then
    STOPPED=1; PENDING_LEFT+=("$name"); echo "[pipeline] 제출 한도에 걸림 → 여기서 멈춤 ($name)"
  else
    echo "[pipeline] 제출 실패: $name" >&2; echo "$out" >&2; echo "$cmd" >&2; exit 1
  fi
}
afterok() { local out="" i; for i in "$@"; do [[ -n "$i" ]] && out+="${out:+:}$i"; done; [[ -n "$out" ]] && echo "afterok:$out"; }

echo "[pipeline] smoke=$SMOKE_CONFIG data=$DATA_CONFIG main=$MAIN dry_run=$DRY_RUN"

# 1) 준비
data_id=""
if [[ -f "$(cfg_get "$DATA_CONFIG" TRAIN_DATA_DIR)/meta.json" ]]; then
  echo "[pipeline] 끝남      prepare_data (학습 데이터 있음)"
else
  submit prepare_data "$DATA_CONFIG" prepare_data "" SB_TIME="$PREP_DATA_TIME"; data_id="$LAST_ID"
  [[ -z "$data_id" ]] && { echo "[pipeline] 학습 데이터 준비가 아직 제출되지 않아 뒤 단계는 다음 실행으로 미룹니다"; STOPPED=1; }
fi
eval_root="$(cfg_get "$DATA_CONFIG" EVAL_DATA_DIR)"
if ls "$eval_root"/*/meta.json >/dev/null 2>&1; then
  echo "[pipeline] 끝남      prepare_eval (평가 문항 있음)"
else
  submit prepare_eval "$DATA_CONFIG" prepare_eval "" SB_TIME="$PREP_EVAL_TIME"
fi

# 2) smoke (작업 1개)
gate="$(afterok "$data_id")"
if [[ "$SKIP_SMOKE" != 1 ]]; then
  smoke_dir="$(run_dir "$SMOKE_CONFIG")"
  if [[ -f "$smoke_dir/SMOKE_OK" ]]; then
    echo "[pipeline] 끝남      smoke ($smoke_dir/SMOKE_OK)"
  else
    submit smoke "$SMOKE_CONFIG" smoke "$gate"
    if [[ -n "$LAST_ID" ]]; then gate="afterok:$LAST_ID"; else STOPPED=1; fi
  fi
fi

# 3) 본학습: 끝난 run 은 건너뛰고, 끊긴 run 은 다시 내서 이어 간다
if [[ "$MAIN" == 1 ]]; then
  for cfg in $MAIN_CONFIGS; do
    for seed in $SEEDS; do
      name="$(basename "$cfg" .env)_s$seed"
      if run_finished "$(run_dir "$cfg" SEED="$seed")"; then echo "[pipeline] 끝남      $name"; continue; fi
      submit "$name" "$cfg" train "$gate" SEED="$seed"
    done
  done
else
  echo "[pipeline] 본학습은 MAIN=1 일 때만 겁니다"
fi

echo
if [[ "$DRY_RUN" != 1 && ${#SUBMITTED[@]} -gt 0 ]]; then
  dir="$(cfg_get "$SMOKE_CONFIG" SB_SUBMIT_DIR)"; mkdir -p "$dir"
  f="$dir/pipeline_$(date +%Y%m%d_%H%M%S).txt"
  printf '%s\n' "${SUBMITTED[@]}" > "$f"
  echo "[pipeline] 이번에 낸 작업 ${#SUBMITTED[@]}개: $f   (전부 취소: scancel \$(cut -d' ' -f1 $f))"
fi
if [[ ${#PENDING_LEFT[@]} -gt 0 || "$STOPPED" == 1 ]]; then
  echo "[pipeline] 아직 안 낸 것: ${PENDING_LEFT[*]:-(앞 단계 대기)}"
  echo "[pipeline] 앞 작업이 끝난 뒤 같은 명령을 다시 실행하세요."
fi
echo "[pipeline] 진행 확인: bash $REPO/scripts/status.sh"
