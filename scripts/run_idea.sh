#!/usr/bin/env bash
# 아이디어 이름 하나로 학습 → 평가를 SLURM 에 건다(평가는 학습이 끝나야 시작). 다시 치면 끝난 학습은 건너뛴다.
#
#   bash scripts/run_idea.sh list                         # 아이디어 목록
#   DRY_RUN=1 bash scripts/run_idea.sh ce                 # 낼 명령만 보기
#   bash scripts/run_idea.sh ce                           # 50M seed 0
#   SIZE=370m SEEDS="0 1 2" bash scripts/run_idea.sh ce sepgate maxsim
#   MAX_STEPS=5700 RUN_TAG=ce_w003 MC_AUX_WEIGHT=0.03 bash scripts/run_idea.sh ce   # config 값 덮어쓰기
#
# 바꿀 수 있는 값(환경변수):
#   SIZE=50m|370m   SEEDS="0"   DRY_RUN=1   EVAL=0(평가 안 냄)
#   TRAIN_PARTS     학습 작업을 몇 번 이어서 낼지(SB_TIME 안에 안 끝나는 run 은 체크포인트에서 이어 간다).
#                   기본 50m=1, 370m=3(MC 30B 토큰 약 45시간 / 24시간 작업)
#   EVAL_TIME=12:00:00   평가 작업 시간(1노드)
# config 의 다른 변수(MAX_STEPS, ROUTE_MONITOR_FILE, SB_COMMENT ...)는 앞에 붙이면 학습·평가 모두에 적용된다.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# 아이디어 → config 이름(configs/<이름>_<SIZE>.env)과 한 줄 설명
declare -A CFG=(
  [vanilla]=vanilla
  [mc]=mc
  [ce]=mc_ce
  [sepgate]=mc_ce_sepgate
  [maxsim]=mc_ce_sepgate_maxsim
  [share]=mc_ce_sepgate_maxsim_share
  [maxsim_share_nolabel]=mc_maxsim_share
  [chained]=mc_chained
  [chunk64]=mc_chunk64
  [dense_chained]=mc_dense_chained
)
declare -A DESC=(
  [vanilla]="설정 1. 기준선: 과거 state 저장 없는 GDN-2"
  [mc]="설정 2. MC 원본(256토큰 구간, 평균 요약, top-4, 게이트 하나)"
  [ce]="설정 3. 아이디어 a: span 복사 정답으로 라우터 CE(후보 전부에 기울기)"
  [sepgate]="설정 4. a + 아이디어 c: 고르는 점수와 섞는 비율 분리"
  [maxsim]="설정 5. a + c + 아이디어 b 일부: 32토큰 하위 블록 최댓값으로 점수"
  [share]="설정 5 + 아이디어 b 나머지: 0층이 고른 구간을 위층 전부가 읽음"
  [maxsim_share_nolabel]="아이디어 b 단독: max-sim + 층 공유, 라벨 없이 LM 손실만"
  [chained]="아이디어 d: MC 원본 + 끊지 않는 압축기(구간 state 가 앞에서 이어짐)"
  [chunk64]="아이디어 e: MC 원본 + 64토큰 구간"
  [dense_chained]="설정 6(MARCH식): 끊지 않는 압축기 + 과거 구간 전부 섞기"
)
ORDER="vanilla mc ce sepgate maxsim share maxsim_share_nolabel chained chunk64 dense_chained"

if [[ $# -eq 0 || "$1" == list ]]; then
  for i in $ORDER; do printf '  %-22s %s\n' "$i" "${DESC[$i]}"; done
  echo "  (아이디어 f 동적 구간 경계는 구현돼 있지 않음)"
  exit 0
fi

SIZE="${SIZE:-50m}"
SEEDS="${SEEDS:-0}"
DRY_RUN="${DRY_RUN:-0}"
EVAL="${EVAL:-1}"
EVAL_TIME="${EVAL_TIME:-12:00:00}"
if [[ -z "${TRAIN_PARTS:-}" ]]; then [[ "$SIZE" == 370m ]] && TRAIN_PARTS=3 || TRAIN_PARTS=1; fi

for i in "$@"; do
  [[ -n "${CFG[$i]:-}" ]] || { echo "모르는 아이디어: $i  (bash scripts/run_idea.sh list)" >&2; exit 1; }
  [[ -f "configs/${CFG[$i]}_$SIZE.env" ]] || { echo "config 없음: configs/${CFG[$i]}_$SIZE.env" >&2; exit 1; }
done

run_dir() {
  env "$@" bash -c 'source "$1/scripts/lib/config.sh"; mslm_load_config "$2" >/dev/null || exit 1
    PYTHONPATH="$1" python3 -c "from mslm.settings import Settings; print(Settings.from_env().run_dir())"' _ "$REPO" "$CONFIG"
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
# submit JOB NAME DEP [VAR=val ...] : prints the job id ("" on dry run)
submit() {
  local job="$1" name="$2" dep="$3"; shift 3
  local extra="--parsable" cmd out
  [[ -n "$dep" ]] && extra+=" --dependency=$dep --kill-on-invalid-dep=yes"
  cmd="$(env "$@" SB_JOB_NAME="mslm-$name" SB_EXTRA="$extra" CONFIG="$CONFIG" bash "$REPO/scripts/sbatch_cmd.sh" "$job")" || exit 1
  if [[ "$DRY_RUN" == 1 ]]; then echo "[dry] $job $name${dep:+ ($dep)}" >&2; echo "      $cmd" >&2; echo "<$name-$job>"; return; fi
  out="$(bash -c "$cmd" 2>&1)" || { echo "[run_idea] 제출 실패: $name $job" >&2; echo "$out" >&2; exit 1; }
  echo "[run_idea] 제출 ${out%%;*}  $job  $name${dep:+  ($dep)}" >&2
  echo "${out%%;*}"
}

for i in "$@"; do
  CONFIG="configs/${CFG[$i]}_$SIZE.env"
  echo "[run_idea] $i ($SIZE): ${DESC[$i]}"
  for seed in $SEEDS; do
    name="${i}_${SIZE}_s$seed"
    dir="$(run_dir SEED="$seed")" || exit 1
    dep=""
    if run_finished "$dir"; then
      echo "[run_idea] 학습 끝남  $dir"
    else
      for ((p = 1; p <= TRAIN_PARTS; p++)); do
        id="$(submit train "$name" "${dep:+afterany:$dep}" SEED="$seed")" || exit 1
        dep="$id"
      done
    fi
    if [[ "$EVAL" == 1 ]]; then
      # 학습이 끝난 뒤(마지막 학습 작업이 정상 종료) 1노드로 평가
      submit eval "$name" "${dep:+afterok:$dep}" SEED="$seed" SB_NODES=1 SB_TIME="$EVAL_TIME" >/dev/null || exit 1
    fi
  done
done
echo "[run_idea] 진행 확인: bash $REPO/scripts/status.sh   결과: <run>/eval/"
