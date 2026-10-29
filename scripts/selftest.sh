#!/usr/bin/env bash
# GPU 노드 점검: 커널 정확도(GDN-2 vs 참조, gather-read) + 작은 MC 모델 fwd/bwd. 실패하면 0 이 아닌 값으로 끝난다.
#   CONFIG=configs/smoke_kisti.env bash scripts/selftest.sh        # GPU 노드 안에서
#   CONFIG=configs/smoke_kisti.env bash scripts/sbatch_cmd.sh selftest
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/config.sh"
mslm_load_config "${CONFIG:?CONFIG=configs/<name>.env 를 지정하세요}"
mslm_activate_env
mslm_offline
echo "[selftest] host=$(hostname) arch=$(uname -m)"
nvidia-smi -L
python -c "import torch, triton, fla; print('[selftest] torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__)"
exec python "$MSLM_REPO/setup/gpu_selftest.py"
