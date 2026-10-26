#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# Pro 6000 (Blackwell) server — conda environment bring-up for linear-memory-routing.
#
# This server is Slurm-managed and requires a dedicated conda env. Per the owner's convention,
# every env / variable name this project creates carries the 'sh' signature (env = sh_routing,
# vars = SH_*). Run this ONCE on the LOGIN node to create + populate the env (no GPU needed for
# the install). GPU work (sanity gates, training, eval) goes through Slurm — see sh_slurm_run.sh.
#
# Usage (login node):
#   bash scripts/sh_env_pro6000.sh                 # create+populate sh_routing (gdn backbone)
#   SH_BACKBONE=mamba2 bash scripts/sh_env_pro6000.sh   # also source-build mamba CUDA kernels
#   SH_ENV=sh_routing SH_PY=3.11 bash scripts/sh_env_pro6000.sh
#
# SH_* knobs (all optional):
#   SH_ENV       conda env name              (default: sh_routing)
#   SH_PY        python version for the env  (default: 3.11)
#   SH_BACKBONE  gdn (default) | mamba2      -> forwarded to setup_env.sh as BACKBONE
#   SH_RECREATE  1 to delete+recreate the env if it already exists (default: 0)
# ---------------------------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SH_ENV="${SH_ENV:-sh_routing}"
SH_PY="${SH_PY:-3.11}"
SH_BACKBONE="${SH_BACKBONE:-gdn}"
SH_RECREATE="${SH_RECREATE:-0}"

echo "=== [conda] bring up '$SH_ENV' (python $SH_PY) ==="
# make `conda activate` work inside this non-interactive shell
CONDA_BASE="$(conda info --base)"
# disable nounset around conda machinery: activate.d hooks (e.g. cuda-nvcc's NVCC_PREPEND_FLAGS)
# reference unset vars and trip `set -u`.
set +u
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

env_exists() { conda env list | awk '{print $1}' | grep -qx "$SH_ENV"; }

if env_exists && [ "$SH_RECREATE" = "1" ]; then
  echo "  removing existing '$SH_ENV' (SH_RECREATE=1)"
  conda env remove -y -n "$SH_ENV"
fi
if env_exists; then
  echo "  '$SH_ENV' already exists — reusing (set SH_RECREATE=1 to rebuild)"
else
  # named env lands in /home/sohyung/.conda/envs (first entry in `conda config --show envs_dirs`),
  # since the shared /home/compu/anaconda3/envs is not writable by us.
  conda create -y -n "$SH_ENV" "python=$SH_PY" pip
fi

conda activate "$SH_ENV"
set -u
echo "  active env : $CONDA_PREFIX"
echo "  python     : $(python -c 'import sys; print(sys.version.split()[0])')"

echo "=== [deps] delegate to GPU-aware setup_env.sh (GPU=blackwell) ==="
# setup_env.sh installs into whatever python is active (= this conda env). It auto-detects
# Blackwell (sm_120) and skips tilelang under py3.13; on this env's python it installs the
# common stack (torch cu128, transformers 5.12, triton, ...).
GPU=blackwell BACKBONE="$SH_BACKBONE" bash scripts/setup_env.sh

echo ""
echo "=== DONE. To use the env: ==="
echo "  conda activate $SH_ENV"
echo "  # then submit GPU work via Slurm:  sbatch scripts/sh_slurm_run.sh"
