#!/usr/bin/env bash
# GPU 1장짜리 학습 run 여러 개(노드 GPU 수까지)를 SLURM 작업 하나로 묶어 낸다. 계정당 작업 수 한도가
# 빠듯할 때 50M run 을 작업 한 칸에 넷씩 넣으려는 것. step 당 토큰 수와 데이터 순서는 GPU 수와 무관하다
# (GPU 1장이면 누적 횟수만 늘어난다).
#
#   bash scripts/run_pack.sh list                                         # 아이디어 목록(run_idea.sh 와 같음)
#   DRY_RUN=1 bash scripts/run_pack.sh stage1a mc:RUN_TAG=mon ce:RUN_TAG=ce_w003,MC_AUX_WEIGHT=0.03
#   MAX_STEPS=5700 ROUTE_MONITOR_FILE=... bash scripts/run_pack.sh stage1a mc:RUN_TAG=mon ce ...
#
# 인자: 묶음 이름, 그다음 run 마다 <아이디어>[:KEY=val,KEY=val]. 앞에 붙인 config 변수는 모든 run 에 적용된다.
# 같은 명령을 다시 내면 끝난 run 은 빼고, 끊긴 run 은 마지막 체크포인트부터 잇는다(RESUME=auto).
# run 별 로그: SB_SUBMIT_DIR/packs/<run>_<jobid>.log. 바꿀 수 있는 값: SIZE=50m  SEED=0  DRY_RUN=1
#   AFTER=<jobid>  그 작업이 끝난 뒤(성공·실패·시간 초과 무관) 시작. 24시간에 못 끝나는 run 을 미리 이어 걸 때
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/scripts/lib/ideas.sh"   # CFG / DESC / ORDER

if [[ $# -eq 0 || "$1" == list ]]; then
  for i in $ORDER; do printf '  %-22s %s\n' "$i" "${DESC[$i]}"; done
  exit 0
fi
PACK="$1"; shift
[[ $# -gt 0 ]] || { echo "run 이 없습니다: bash scripts/run_pack.sh $PACK <아이디어>[:KEY=val,...] ..." >&2; exit 1; }
SIZE="${SIZE:-50m}"
DRY_RUN="${DRY_RUN:-0}"
AFTER="${AFTER:-}"

# cfg_get CONFIG KEY [VAR=val ...] : resolved config value (same rules as every script)
cfg_get() {
  local cfg="$1" key="$2"; shift 2
  env "$@" bash -c 'source "$1/scripts/lib/config.sh"; mslm_load_config "$2" >/dev/null || exit 1; printf "%s" "${!3}"' _ "$REPO" "$cfg" "$key"
}
run_dir() {
  local cfg="$1"; shift
  env "$@" bash -c 'source "$1/scripts/lib/config.sh"; mslm_load_config "$2" >/dev/null || exit 1
    PYTHONPATH="$1" python3 -c "from mslm.settings import Settings; print(Settings.from_env().run_dir())"' _ "$REPO" "$cfg"
}
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
        if r.get("event") == "start": total = r.get("stop_step", r.get("total_steps"))
        if r.get("event") == "end": end = r.get("step")
except FileNotFoundError:
    sys.exit(1)
sys.exit(0 if total is not None and end == total else 1)
EOF
}

PLAN=() DIRS=() FIRST_CFG=""
for spec in "$@"; do
  idea="${spec%%:*}"; kv=""
  [[ "$spec" == *:* ]] && kv="${spec#*:}"; kv="${kv//,/ }"
  [[ -n "${CFG[$idea]:-}" ]] || { echo "모르는 아이디어: $idea  (bash scripts/run_pack.sh list)" >&2; exit 1; }
  cfg="configs/${CFG[$idea]}_$SIZE.env"
  [[ -f "$cfg" ]] || { echo "config 없음: $cfg" >&2; exit 1; }
  for a in $kv; do [[ "$a" =~ ^[A-Z_][A-Z0-9_]*=[^[:space:]]*$ ]] || { echo "잘못된 덮어쓰기: $a ($spec)" >&2; exit 1; }; done
  # shellcheck disable=SC2086
  dir="$(run_dir "$cfg" $kv)" || exit 1
  for d in "${DIRS[@]}"; do [[ "$d" == "$dir" ]] && { echo "같은 run 폴더가 두 번: $dir (RUN_TAG 로 나누세요)" >&2; exit 1; }; done
  DIRS+=("$dir")
  name="$(basename "$dir")"
  if run_finished "$dir"; then echo "[run_pack] 학습 끝남  $name"; continue; fi
  PLAN+=("$name"$'\t'"$REPO/$cfg"$'\t'"$kv")
  echo "[run_pack] $name  ($idea: ${DESC[$idea]})${kv:+  [$kv]}"
  [[ -z "$FIRST_CFG" ]] && FIRST_CFG="$cfg"
done
[[ ${#PLAN[@]} -gt 0 ]] || { echo "[run_pack] 낼 run 이 없습니다(모두 끝남)"; exit 0; }
ngpu="$(cfg_get "$FIRST_CFG" NGPU_PER_NODE)"
[[ ${#PLAN[@]} -le "$ngpu" ]] || { echo "run ${#PLAN[@]}개 > 노드당 GPU $ngpu개. 묶음을 나누세요" >&2; exit 1; }

submit_dir="$(cfg_get "$FIRST_CFG" SB_SUBMIT_DIR)"
plan="$submit_dir/packs/${PACK}_$(date +%Y%m%d_%H%M%S).tsv"
# 작업 이름 mslm-pack-<이름>. 작업 안에서는 slurm/pack.sbatch → scripts/pack_train.sh 가 plan 의 줄마다 GPU 하나씩 쓴다
cmd="$(SB_JOB_NAME="mslm-pack-$PACK" SB_EXTRA="--parsable${AFTER:+ --dependency=afterany:$AFTER}" CONFIG="$FIRST_CFG" bash "$REPO/scripts/sbatch_cmd.sh" pack \
       | sed "s/--job-name=mslm-pack-$PACK-pack/--job-name=mslm-pack-$PACK/")" || exit 1
if [[ "$DRY_RUN" == 1 ]]; then
  echo "[dry] pack $PACK -> $plan"; printf '      %s\n' "${PLAN[@]}"; echo "      PACK_FILE=$plan $cmd"
  exit 0
fi
mkdir -p "$(dirname "$plan")"
printf '%s\n' "${PLAN[@]}" > "$plan"
out="$(PACK_FILE="$plan" bash -c "$cmd" 2>&1)" || { echo "[run_pack] 제출 실패" >&2; echo "$out" >&2; exit 1; }
echo "[run_pack] 제출 ${out%%;*}  pack $PACK (${#PLAN[@]} run)  plan=$plan"
echo "[run_pack] 진행 확인: bash $REPO/scripts/status.sh   run 별 로그: $submit_dir/packs/<run>_${out%%;*}.log"
