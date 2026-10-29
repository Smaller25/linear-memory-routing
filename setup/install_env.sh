#!/usr/bin/env bash
# conda env 설치(aarch64 GH200 / x86_64 공통). 인터넷이 되는 노드에서 실행한다.
# KISTI: 로그인 노드(hangang.ksc.re.kr / glogin03). 계산 노드는 인터넷이 막혔다고 가정.
#
#   bash setup/install_env.sh                       # env 이름 sh_mslm (이미 있으면 갱신)
#   ENV_NAME=sh_mslm2 bash setup/install_env.sh
#   ENV_PREFIX=/data2/$USER/conda-envs/sh_mslm bash setup/install_env.sh   # 경로로 만들기
#   FLA_SRC=/path/to/flash-linear-attention bash setup/install_env.sh      # 미리 clone 한 fla 사용
#   TORCH_CUDA=cu129 bash setup/install_env.sh      # cu128 wheel 이 GH200 에서 안 돌 때의 대안
#
# 패키지 목록과 버전은 setup/environment.yml 하나에만 적는다(이 스크립트는 그 파일로 env 를 만들고 fla 만 따로 깐다).
#  - fla 는 반드시 4b02d15d. 다른 버전(pip 0.5.2, master)은 vendored GDN-2 커널이 use_exp2 오류로 죽는다.
#    --no-deps 로 깐다(의존성까지 깔면 PyPI fla-core 가 커밋 코드를 덮어쓴다). 다시 깔 때는 site-packages/fla 를 먼저 지운다.
#  - torch 2.9.1+cu128 / triton 3.5.1 aarch64 wheel 존재 확인(2026-10, pip --dry-run --platform manylinux_*_aarch64).
#    실행이 안 되면 TORCH_CUDA=cu129, 또는 KISTI 컨테이너 /apps/common/kisti-container/images/pytorch-25.03-py3-aarch64.sif
#    안에서 fla@4b02d15d 만 pip install --no-deps (torch 버전이 달라지므로 check_env.sh 로 커널 시험 필수).
#  - mamba-ssm, causal-conv1d 는 설치하지 않는다(코드가 쓰지 않음, ShortConv 는 fla triton 백엔드).
set -euo pipefail

ENV_NAME="${ENV_NAME:-sh_mslm}"
ENV_PREFIX="${ENV_PREFIX:-}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
FLA_COMMIT=4b02d15d6a68700181b180235be62a9fb95d2a38
FLA_SRC="${FLA_SRC:-}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YML="$REPO/setup/environment.yml"
TORCH_VERSION="$(grep -oE 'torch==[0-9.]+' "$YML" | cut -d= -f3)"
echo "[install] arch=$(uname -m) env=${ENV_PREFIX:-$ENV_NAME} torch=$TORCH_VERSION+$TORCH_CUDA fla=${FLA_COMMIT:0:8} spec=$YML"

if ! command -v conda >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load miniforge 2>/dev/null || module load miniforge/26.1.1-3   # KISTI 한강: miniforge/26.1.1-3
  fi
fi
command -v conda >/dev/null || { echo "[install] conda 가 없습니다(module load miniforge 확인)"; exit 1; }
set +u; eval "$(conda shell.bash hook)"; set -u

# 1) environment.yml 로 env 만들기/갱신 (TORCH_CUDA 가 cu128 이 아니면 그 index 로 바꾼 사본 사용)
SPEC="$YML"
if [[ "$TORCH_CUDA" != cu128 ]]; then
  SPEC="$(mktemp --suffix=.yml)"
  sed "s#/whl/cu128#/whl/$TORCH_CUDA#; s#+cu128#+$TORCH_CUDA#" "$YML" > "$SPEC"
fi
if [[ -n "$ENV_PREFIX" ]]; then TARGET=(-p "$ENV_PREFIX"); else TARGET=(-n "$ENV_NAME"); fi
if conda env list | grep -qE "(^|/)${ENV_PREFIX:-$ENV_NAME}( |$)"; then
  conda env update "${TARGET[@]}" -f "$SPEC"
else
  conda env create "${TARGET[@]}" -f "$SPEC"
fi
set +u; conda activate "${ENV_PREFIX:-$ENV_NAME}"; set -u
PIP="python -m pip"

# 2) fla @ 4b02d15d (먼저 흔적 제거, 의존성 없이)
$PIP uninstall -y flash-linear-attention fla-core fla >/dev/null 2>&1 || true
SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
rm -rf "$SITE/fla" "$SITE"/flash_linear_attention-*.dist-info "$SITE"/fla_core-*.dist-info
if [[ -n "$FLA_SRC" ]]; then
  git -C "$FLA_SRC" checkout -q "$FLA_COMMIT"
  $PIP install --no-deps --no-build-isolation "$FLA_SRC"
else
  $PIP install --no-deps "git+https://github.com/fla-org/flash-linear-attention@$FLA_COMMIT"
fi

# 3) 확인
for bad in mamba_ssm causal_conv1d; do
  if python -c "import $bad" 2>/dev/null; then echo "[install] 경고: $bad 가 설치돼 있음(필요 없음)"; fi
done
python - <<EOF
import importlib.metadata as md, json, torch, triton, fla, sys
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda, "| triton", triton.__version__)
try:
    url = json.loads(md.distribution("flash-linear-attention").read_text("direct_url.json"))
    print("fla", fla.__version__, url.get("vcs_info", {}).get("commit_id", url))
except Exception as e:
    print("fla", fla.__version__, "(direct_url 없음:", e, ")")
assert torch.__version__.startswith("$TORCH_VERSION"), torch.__version__
EOF
python -m ipykernel install --user --name "$(basename "${ENV_PREFIX:-$ENV_NAME}")" --display-name "mslm ($(basename "${ENV_PREFIX:-$ENV_NAME}"))" >/dev/null
echo "[install] 완료. 다음: bash setup/check_env.sh (GPU 노드에서)"
