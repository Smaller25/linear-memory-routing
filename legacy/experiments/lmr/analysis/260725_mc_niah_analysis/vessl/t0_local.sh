#!/bin/bash
set -euo pipefail
PY=/opt/conda/bin/python
WORK=/root/work; VENDOR=/root/smaller/mc_niah/vendor
mkdir -p "$WORK"
log(){ echo "[t0] $*"; }
if [ ! -f "$WORK/long-gdn/dsc/mc_gdn2/ssc.py" ]; then
  rm -rf "$WORK/long-gdn"
  log "cloning long-gdn from bundle"
  GIT_LFS_SKIP_SMUDGE=1 git clone -q "$VENDOR/long-gdn-e71713e.bundle" "$WORK/long-gdn"
  cd "$WORK/long-gdn"; GIT_LFS_SKIP_SMUDGE=1 git checkout -q e71713e402fcbf63b50365ed97ab2d5555cb6710 2>/dev/null || true
fi
log "long-gdn @ $(git -C $WORK/long-gdn rev-parse --short HEAD)"
grep -q "mean of L2-normalized keys" "$WORK/long-gdn/dsc/mc_baseline/mc_ssc.py" && log "mean-pool pin OK"
if [ ! -d "$WORK/pydeps/fla" ]; then
  log "restoring pydeps from vendor tar"; tar xzf "$VENDOR/pydeps-fla-4b02d15d.tar.gz" -C "$WORK"
fi
log "fla pydeps OK"
"$PY" -m pip install -q einops transformers huggingface_hub numpy matplotlib lightning sentencepiece 2>&1 | tail -1 || true
"$PY" -c "import torch,einops,transformers,huggingface_hub,numpy,matplotlib; print('[t0] deps OK torch', torch.__version__)"
dl(){ local repo="$1" f="$2"; local d="/root/smaller/mc_niah/ckpts/$(echo $repo | tr / __)"; mkdir -p "$d"
  if [ -f "$d/$f" ] && [ "$(stat -c%s $d/$f)" -gt 1000000000 ]; then log "ckpt cached: $f"; return; fi
  log "downloading $repo/$f"
  HF_HOME=/tmp/hf_home "$PY" -c "
import shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download('$repo', '$f')
shutil.copy2(p, '$d/$f')
print('[t0] saved $d/$f')"
}
dl LLM-OS-Models2/mc-gdn2-370m-fineweb-edu-30b-v2-meanpool checkpoint-30B-model-ckpt.pth
dl LLM-OS-Models2/mc-gdn2-370m-fineweb-edu-30b-v2-meanpool checkpoint-5B-model-ckpt.pth
dl LLM-OS-Models2/gdn2-370m-fineweb-edu-5b-vanilla checkpoint-5B-model-ckpt.pth
log "ALL DONE"

# NOTE (검증된 운영 교훈, 2026-07-27):
# - geesefs(/root/smaller)에서 git clone/checkout은 소파일 대량 쓰기로 실패/증발함
#   → 코드는 반드시 컨테이너 로컬(/root/work), geesefs에는 bundle/tar/ckpt/결과만
# - long-gdn은 private + git-lfs: vendor/long-gdn-e71713e.bundle에서 클론,
#   GIT_LFS_SKIP_SMUDGE=1 필수 (코드만 필요, lfs 데이터 불필요)
# - volatile deps에 lightning, sentencepiece 포함할 것 (lit_gpt.utils가 lightning.fabric import)
# - 컨테이너 재시작 시: /root/.ssh 초기화 → 웹터미널에서
#   cat /root/smaller/ssh_authorized_keys_sohyung2 >> /root/.ssh/authorized_keys
#   후 이 스크립트 재실행 (idempotent; fla는 vendor tar에서 수 초 복원)
