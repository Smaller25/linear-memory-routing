#!/usr/bin/env bash
# 인터넷이 되는 노드에서 미리 받아 둘 것(계산 노드는 오프라인 가정).
# KISTI: 데이터무버 hangang-dm.ksc.re.kr 권장(CPU 제한 없음, wget/git/대용량 전송 권장 노드).
#   CONFIG=configs/mc_50m.env bash setup/download_assets.sh
#   CONFIG=configs/mc_50m.env SKIP_DATASET=1 bash setup/download_assets.sh   # 토크나이저·nltk 만
#   CONFIG=configs/smoke_kisti.env MAX_FILES=1 bash setup/download_assets.sh  # parquet 1개만(smoke 용, 약 2GB)
# 받는 것: TOKENIZER, HF_DATASET/HF_DATASET_CONFIG 원본 parquet, nltk punkt(RULER essay 문장 분리용)
# 모두 HF_CACHE_DIR(=HF_HOME) 아래(/scratch)에 저장된다. 15일 미접근 시 삭제되니 오래 쉬면 다시 받거나 touch.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/lib/config.sh"
mslm_load_config "${CONFIG:?CONFIG=configs/<name>.env 를 지정하세요}"
mslm_activate_env
mkdir -p "$HF_CACHE_DIR"
export NLTK_DATA="$HF_CACHE_DIR/nltk_data"

python - <<EOF
import os
from huggingface_hub import snapshot_download
print("tokenizer:", snapshot_download("$TOKENIZER", allow_patterns=["*.json", "*.model", "*.txt"]))
if os.environ.get("SKIP_DATASET") != "1":
    sub = "$HF_DATASET_CONFIG"
    # fineweb-edu: sample-10BT -> files under sample/10BT/ ; default -> data/
    prefix = f"sample/{sub.split('-', 1)[1]}/" if sub.startswith("sample-") else "data/"
    from huggingface_hub import HfApi
    files = sorted(f for f in HfApi().list_repo_files("$HF_DATASET", repo_type="dataset")
                   if f.startswith(prefix) and f.endswith(".parquet"))
    n = int(os.environ.get("MAX_FILES", "0") or 0)
    files = files[:n] if n > 0 else files
    print(f"dataset: {len(files)} parquet files under {prefix}")
    print(snapshot_download("$HF_DATASET", repo_type="dataset", allow_patterns=files + ["README.md"], max_workers=16))
import nltk
for p in ["punkt", "punkt_tab"]:
    nltk.download(p, download_dir=os.environ["NLTK_DATA"], quiet=True)
print("nltk:", os.environ["NLTK_DATA"])
EOF
du -sh "$HF_CACHE_DIR" || true
