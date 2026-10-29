#!/usr/bin/env bash
# KISTI 한강 로그인 노드에서 한 번만 실행: 아키텍처 확인 → --comment 허용값 확인 → conda env → 데이터 반입 → CPU 점검.
# 끝난 단계는 건너뛰므로, 중간에 끊기면 같은 명령을 다시 치면 된다. 인터넷이 필요하다(계산 노드에서 돌리지 않는다).
#
#   cd ~/linear-memory-routing && bash setup/kisti_bootstrap.sh
#   SKIP_DOWNLOAD=1 bash setup/kisti_bootstrap.sh            # 데이터 반입 건너뜀
#   REINSTALL=1 bash setup/kisti_bootstrap.sh                # env 가 멀쩡해도 다시 설치
#   CONFIG=configs/mc_50m.env bash setup/kisti_bootstrap.sh  # 기본값. CONDA_ENV·경로를 이 config 에서 읽는다
#   ALLOW_ARCH=1 ...                                         # aarch64 가 아닌 곳에서 흐름만 시험할 때
# 다음 단계: bash scripts/submit_pipeline.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
CONFIG="${CONFIG:-configs/mc_50m.env}"
source "$REPO/scripts/lib/config.sh"
mslm_load_config "$CONFIG" || exit 1

declare -a SUMMARY=()
step() { echo; echo "=== [$1] $2"; }
pass() { SUMMARY+=("✅ $1"); echo "✅ $1"; }
skip() { SUMMARY+=("⏭  $1"); echo "⏭  $1"; }
warn() { SUMMARY+=("⚠️  $1"); echo "⚠️  $1"; }
fail() { SUMMARY+=("❌ $1"); echo "❌ $1"; finish 1; }
finish() {
  echo; echo "================ 요약 ================"
  printf '%s\n' "${SUMMARY[@]}"
  if [[ "$1" -eq 0 ]]; then
    echo; echo "다음: cd $REPO && bash scripts/submit_pipeline.sh   (MAIN=1 을 붙이면 본학습까지 연쇄 제출)"
  else
    echo; echo "멈춘 단계를 고친 뒤 같은 명령을 다시 실행하세요(끝난 단계는 건너뜁니다)."
  fi
  exit "$1"
}

step 1 "아키텍처"
ARCH="$(uname -m)"
if [[ "$ARCH" == aarch64 ]]; then pass "arch=aarch64 (GH200 GPU 노드와 같음)"
elif [[ "${ALLOW_ARCH:-0}" == 1 ]]; then warn "arch=$ARCH (ALLOW_ARCH=1 로 계속)"
else fail "arch=$ARCH: 이 노드에서 깐 env 는 aarch64 GPU 노드에서 돌지 않습니다. aarch64 로그인 노드(예: ssh glogin03)에서 다시 실행"
fi

step 2 "sbatch --comment 허용값 ($SB_COMMENT)"
if [[ -z "$SB_COMMENT" ]]; then skip "SB_COMMENT 비어 있음"
elif command -v showappl >/dev/null 2>&1; then
  APPL="$(showappl 2>&1)"
  field="$(sed -nE 's/.*field=([^;]+).*/\1/p' <<<"$SB_COMMENT")"; appl="$(sed -nE 's/.*appl=([^;]+).*/\1/p' <<<"$SB_COMMENT")"
  miss=""
  for w in "$field" "$appl"; do grep -qiw -- "$w" <<<"$APPL" || miss+=" $w"; done
  if [[ -z "$miss" ]]; then pass "showappl 에 $field / $appl 있음"
  else warn "showappl 에서 못 찾음:$miss → 맞는 값으로 configs/base.env 의 SB_COMMENT 를 고치세요"; fi
else
  warn "showappl 명령 없음(로그인 노드가 아님?). SB_COMMENT 확인 못 함"
fi

step 3 "conda env ($CONDA_ENV)"
env_ok() {
  ( set +u; mslm_activate_env >/dev/null 2>&1 && python - <<'EOF'
import importlib.metadata as md, json, torch, triton, fla, transformers
c = json.loads(md.distribution("flash-linear-attention").read_text("direct_url.json") or "{}").get("vcs_info", {}).get("commit_id", "")
assert c.startswith("4b02d15d"), c
assert torch.__version__.startswith("2.9.1"), torch.__version__
print(f"torch {torch.__version__} triton {triton.__version__} fla {c[:8]} transformers {transformers.__version__}")
EOF
  )
}
if [[ "${REINSTALL:-0}" != 1 ]] && info="$(env_ok 2>/dev/null)"; then
  skip "env 이미 있음: $info"
else
  ENV_NAME="$CONDA_ENV" bash "$REPO/setup/install_env.sh" || fail "env 설치 실패(위 로그 확인)"
  info="$(env_ok)" || fail "env 설치 뒤 확인 실패(torch/fla 버전)"
  pass "env 설치: $info"
fi

step 4 "데이터 반입 (토크나이저·FineWeb-Edu ${HF_DATASET_CONFIG}·nltk → $HF_CACHE_DIR)"
if [[ "${SKIP_DOWNLOAD:-0}" == 1 ]]; then skip "SKIP_DOWNLOAD=1"
elif [[ -f "$TRAIN_DATA_DIR/meta.json" ]]; then skip "토큰화된 학습 데이터가 이미 있음($TRAIN_DATA_DIR)"
else
  CONFIG="$CONFIG" bash "$REPO/setup/download_assets.sh" || fail "다운로드 실패(다시 실행하면 받은 파일은 건너뜁니다)"
  pass "다운로드 완료"
fi

step 5 "CPU 점검 (check_env.sh)"
out="$(CONFIG="$CONFIG" bash "$REPO/setup/check_env.sh" 2>&1)"; echo "$out"
if grep -q "\[FAIL\]" <<<"$out"; then fail "check_env.sh 에 [FAIL] 있음(위 출력)"; fi
pass "CPU 점검 통과 (GPU 점검은 submit_pipeline.sh 의 smoke 작업이 한다)"

finish 0
