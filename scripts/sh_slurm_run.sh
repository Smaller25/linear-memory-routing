#!/usr/bin/env bash
#SBATCH --job-name=sh_routing
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/sh_%x_%j.out
#SBATCH --error=logs/sh_%x_%j.err
# ---------------------------------------------------------------------------------------------
# Slurm job entrypoint for the Pro 6000 (Blackwell) server.
#
# All GPU work on this server must go through Slurm; this script runs inside the sh_routing
# conda env (create it first with: bash scripts/sh_env_pro6000.sh). Per the owner's convention
# the job name is 'sh_routing' and all custom vars are SH_*.
#
# Submit (defaults to the GDN correctness gate — a safe smoke test):
#   sbatch scripts/sh_slurm_run.sh
#
# Submit a specific experiment (everything after the script name is run verbatim in the env):
#   sbatch scripts/sh_slurm_run.sh \
#     python -m lmr.scripts.eval_long --arch gdn --variant ssc --topk 4 \
#            --low-rank-dim 64 --lengths 4096 8192
#
# Request both cards / longer time at submit:
#   sbatch --gres=gpu:rtx6000:2 --time=6:00:00 scripts/sh_slurm_run.sh ...
#
# SH_* knobs (env, optional):
#   SH_ENV   conda env to activate (default: sh_routing)
#   SH_CMD   command to run (overridden by any positional args passed to the script)
# ---------------------------------------------------------------------------------------------
set -euo pipefail

SH_ENV="${SH_ENV:-sh_routing}"

# resolve the repo root from where sbatch was submitted (SLURM_SUBMIT_DIR), else this file's dir
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR" ]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs ckpt

echo "=== [slurm] job $SLURM_JOB_ID on $(hostname) | gres=${SLURM_JOB_GRES:-?} ==="

# activate the sh_routing conda env. Disable nounset around activation: conda's activate.d hooks
# (e.g. cuda-nvcc's NVCC_PREPEND_FLAGS) reference unset vars and trip `set -u`.
CONDA_BASE="$(conda info --base)"
set +u
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$SH_ENV"
set -u

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [env] $CONDA_PREFIX ==="
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda", ok,
      "|", torch.cuda.get_device_name(0) if ok else "CPU")
if ok:
    c = torch.cuda.get_device_capability(0); print(f"device sm_{c[0]}{c[1]}")
PY

# command: positional args win; else SH_CMD; else the GDN correctness gate (safe default)
if [ "$#" -gt 0 ]; then
  SH_RUN=( "$@" )
elif [ -n "${SH_CMD:-}" ]; then
  # shellcheck disable=SC2206
  SH_RUN=( $SH_CMD )
else
  SH_RUN=( python -m pytest tests/lmr/test_segment_runner_gdn.py -q )
fi

echo "=== [run] ${SH_RUN[*]} ==="
set -x
"${SH_RUN[@]}"
