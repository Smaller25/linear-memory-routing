#!/usr/bin/env bash
# 접속 직후 / 설치 직후 확인. GPU 검사는 GPU 노드에서만 돈다(로그인 노드에서는 CPU 검사만).
#   bash setup/check_env.sh                                  # 현재 env
#   CONFIG=configs/smoke_kisti.env bash setup/check_env.sh   # config 의 CONDA_ENV/경로까지 확인
# KISTI 에서 GPU 검사:  salloc 으로 GPU 노드를 잡은 뒤 실행, 예)
#   salloc --partition=normal --comment="field=nlp;appl=pytorch" --nodes=1 --gpus-per-node=1 --time=00:30:00
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/lib/config.sh"
mslm_load_config "${CONFIG:-$REPO/configs/base.env}" || exit 1
mslm_activate_env || exit 1

ok() { echo "  [ok]   $*"; }
warn() { echo "  [warn] $*"; }
echo "== host: $(hostname)  arch: $(uname -m)  user: $USER"
echo "== 경로"
for d in "$SCRATCH_ROOT" "$HF_CACHE_DIR" "$TRAIN_DATA_DIR" "$EVAL_DATA_DIR" "$RUNS_ROOT"; do
  [[ -d "$d" ]] && ok "$d" || warn "$d 없음(필요하면 mkdir -p)"
done
case "$REPO" in /scratch/*) warn "코드가 /scratch 에 있음: 15일 미접근 시 삭제됨. 코드는 /home01 에 두세요";; esac
command -v lfs >/dev/null && { lfs quota -h "$HOME" 2>/dev/null | tail -n +2 | head -3; lfs quota -h /scratch 2>/dev/null | tail -n +2 | head -3; }

echo "== python 패키지"
python - <<'EOF'
import importlib, json, sys
import importlib.metadata as md
print("  python", sys.version.split()[0], sys.executable)
for m in ["torch", "triton", "fla", "transformers", "datasets", "nltk", "wonderwords"]:
    try:
        mod = importlib.import_module(m)
        print(f"  [ok]   {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  [FAIL] {m}: {e}")
try:
    url = json.loads(md.distribution("flash-linear-attention").read_text("direct_url.json") or "{}")
    c = url.get("vcs_info", {}).get("commit_id", "")
    print("  [ok]  " if c.startswith("4b02d15d") else "  [FAIL]", "fla commit", c or url or "unknown (4b02d15d 이어야 함)")
except Exception as e:
    print("  [warn] fla commit 확인 불가:", e)
for bad in ["mamba_ssm", "causal_conv1d"]:
    try:
        importlib.import_module(bad); print(f"  [warn] {bad} 설치됨(필요 없음)")
    except ImportError:
        pass
EOF

echo "== CPU 모델 시험(tiny)"
python -c "
from mslm import tower
r = tower.smoke('configs/mc_50m.env', {'MODEL_SIZE':'tiny','MC_CHUNK':'16','VOCAB_SIZE':'256'})
print('  ', r); assert r['reload_loss_equal'] and r['oracle_picks_gold']
" && ok "CPU smoke" || echo "  [FAIL] CPU smoke"

if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "== GPU 시험"
  nvidia-smi -L
  python "$REPO/setup/gpu_selftest.py" && ok "GPU kernels" || echo "  [FAIL] GPU kernels"
else
  echo "== GPU 없음: GPU 시험은 GPU 노드(salloc/sbatch)에서 다시 실행"
fi
echo "== 토크나이저(오프라인)"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "
from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$TOKENIZER'); print('   vocab', len(t), 'bos', t.bos_token_id, 'eos', t.eos_token_id)
" && ok "tokenizer in HF_CACHE_DIR" || warn "토크나이저가 캐시에 없음 → setup/download_assets.sh"
