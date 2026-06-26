#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# GPU-aware environment setup for linear-memory-routing.
#
# Brings up the Python deps for BOTH our hardware targets from one script, branching on the
# detected CUDA compute capability:
#
#   sm_80  = A100      (py3.10 container; the original/validated setup)
#   sm_90  = H100      (GDN router TRAINING works here via tilelang)
#   sm_120 = Blackwell = RTX PRO 6000 / B-series  (the new server)
#
# What actually differs between targets (everything else is common):
#   - TORCH_CUDA_ARCH_LIST for any source builds (mamba_ssm / causal_conv1d): 8.0 / 9.0 / 12.0
#   - tilelang: needed for GDN-chunk-backward on Hopper; CRASHES on import under py3.13 (TVM-FFI),
#     so it is skipped (with a warning) on the Blackwell+py3.13 box.
# The cu128 PyTorch wheel already bundles sm_80/sm_90/sm_120, so torch itself is a common install.
#
# Usage:
#   bash scripts/setup_env.sh                 # auto-detect GPU, install common + per-GPU deps
#   BACKBONE=mamba2 bash scripts/setup_env.sh # also source-build mamba_ssm / causal_conv1d
#   GPU=a100 bash scripts/setup_env.sh        # force a target (a100|h100|blackwell) if detection is wrong
#   WITH_TILELANG=1 bash scripts/setup_env.sh # force-attempt tilelang even on Blackwell/py3.13
#
# Env knobs (all optional):
#   GPU            override detection: a100 | h100 | blackwell
#   BACKBONE       gdn (default) | mamba2     — mamba2 needs the source-built CUDA kernels; gdn does not
#   WITH_TILELANG  0/1 (default: 1 on A100/H100, 0 on Blackwell-py3.13)
#   TILELANG_VER   tilelang pin (default 0.1.9 — fla's minimum; latest crashes on import here)
#   TORCH_INDEX    pip index for torch (default the cu128 wheel index)
# Run from the repo root. Idempotent: re-runs skip already-satisfied installs.
# ---------------------------------------------------------------------------------------------
set -euo pipefail

# --- locate repo root (works whether sourced or run from anywhere) --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python}"
PYVER="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

echo "=== [0] detect hardware ==="
# ---- resolve target GPU --------------------------------------------------------------------
detect_gpu() {
  local cap name
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  case "$cap" in
    8.0|8.6|8.7|8.9) echo "a100" ;;          # Ampere/Ada — treat as the A100 (sm_80) path
    9.0)             echo "h100" ;;          # Hopper
    10.*|12.*)       echo "blackwell" ;;     # Blackwell (B-series sm_100 / workstation sm_120)
    *)               echo "" ;;
  esac
}

GPU="${GPU:-$(detect_gpu)}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo '?')"
GPU_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || echo '?')"

case "$GPU" in
  a100)      ARCH_LIST="8.0";  DEFAULT_TILELANG=1 ;;
  h100)      ARCH_LIST="9.0";  DEFAULT_TILELANG=1 ;;
  blackwell) ARCH_LIST="12.0"; DEFAULT_TILELANG=0 ;;   # see tilelang/py3.13 note below
  *) echo "!! could not map GPU (name='$GPU_NAME' cap='$GPU_CAP')."
     echo "   Set GPU=a100|h100|blackwell explicitly and re-run." ; exit 1 ;;
esac

echo "  GPU target : $GPU   (name='$GPU_NAME', compute_cap=$GPU_CAP)"
echo "  python     : $PYVER"
echo "  arch list  : $ARCH_LIST   (used for any source builds)"
export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"

# --- common deps ----------------------------------------------------------------------------
echo "=== [1] common deps ==="
"$PY" -m pip install -q --upgrade pip

# torch: the cu128 wheel covers sm_80/sm_90/sm_120, so this line is identical for every target.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
if ! "$PY" -c 'import torch' 2>/dev/null; then
  echo "  installing torch (cu128 wheel — covers A100/H100/Blackwell)"
  "$PY" -m pip install -q torch --index-url "$TORCH_INDEX"
else
  echo "  torch already present: $("$PY" -c 'import torch; print(torch.__version__)')"
fi

# transformers 5.12 is the version the loaders/converter were patched against (SESSION_HANDOFF §5).
"$PY" -m pip install -q "transformers>=5.12,<6" huggingface_hub tokenizers sentencepiece \
    einops "datasets>=3.3.0" accelerate pytest nltk
"$PY" -c 'import triton' 2>/dev/null || "$PY" -m pip install -q triton

# --- per-GPU: tilelang (GDN router TRAINING on Hopper) --------------------------------------
echo "=== [2] tilelang (GDN-chunk-backward backend) ==="
WITH_TILELANG="${WITH_TILELANG:-$DEFAULT_TILELANG}"
TILELANG_VER="${TILELANG_VER:-0.1.9}"
if [ "$WITH_TILELANG" = "1" ]; then
  if "$PY" -c 'import tilelang' 2>/dev/null; then
    echo "  tilelang already importable"
  else
    echo "  installing tilelang==$TILELANG_VER"
    "$PY" -m pip install -q "tilelang==$TILELANG_VER" || true
    # verify it actually imports — the latest releases crash with a TVM-FFI double-registration.
    if "$PY" -c 'import tilelang' 2>/dev/null; then
      echo "  tilelang OK"
    else
      echo "  !! tilelang installed but FAILS to import (TVM-FFI). GDN router TRAINING will be"
      echo "     unavailable; GDN forward/eval (fused_recurrent) still works."
    fi
  fi
else
  echo "  SKIPPED on '$GPU' (py$PYVER)."
  if [ "$GPU" = "blackwell" ]; then
    echo "     Only GDN router TRAINING (head_dim=256 chunk-backward) needs tilelang, and the"
    echo "     latest tilelang crashes on import under py3.1x (TVM-FFI). GDN here is forward/"
    echo "     eval-only for now. (Set WITH_TILELANG=1 to force-attempt.)"
  fi
fi

# --- per-GPU: mamba2 CUDA kernels (only if running a mamba2 backbone) -----------------------
BACKBONE="${BACKBONE:-gdn}"
echo "=== [3] mamba2 kernels (backbone=$BACKBONE) ==="
if [ "$BACKBONE" = "mamba2" ]; then
  echo "  source-building mamba_ssm / causal_conv1d for arch $ARCH_LIST (this is slow)"
  "$PY" -m pip install -q ninja packaging setuptools wheel

  # These compile CUDA extensions, so they need an nvcc whose MAJOR version matches torch's CUDA
  # (PyTorch's extension builder rejects a major mismatch). The Pro 6000 box ships a system CUDA 13
  # toolkit while torch here is cu12.x — install a matching toolkit into the active conda env and
  # build against that. (On an A100 container where system nvcc already matches, this is a no-op.)
  TORCH_CUDA="$("$PY" -c 'import torch; print(torch.version.cuda or "")')"   # e.g. 12.8
  TORCH_CUDA_MAJ="${TORCH_CUDA%%.*}"
  SYS_NVCC_MAJ="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\).*/\1/p' | head -1)"
  if [ -n "${CONDA_PREFIX:-}" ] && [ -n "$TORCH_CUDA_MAJ" ] && [ "$SYS_NVCC_MAJ" != "$TORCH_CUDA_MAJ" ]; then
    if [ ! -x "$CONDA_PREFIX/bin/nvcc" ]; then
      echo "  system nvcc (CUDA $SYS_NVCC_MAJ) != torch CUDA $TORCH_CUDA -> installing cuda-toolkit=$TORCH_CUDA into the env"
      conda install -y -c nvidia "cuda-toolkit=$TORCH_CUDA" >/dev/null
    fi
    export CUDA_HOME="$CONDA_PREFIX"
    export PATH="$CUDA_HOME/bin:$PATH"
    # conda's nvidia CUDA puts headers/libs under targets/<triple>/ , not $PREFIX/include|lib,
    # which torch's extension builder & gcc host-compile don't search -> add them explicitly
    # (otherwise: "fatal error: cuda_runtime_api.h: No such file or directory").
    _tgt="$CUDA_HOME/targets/x86_64-linux"
    [ -d "$_tgt/include" ] && export CPATH="$_tgt/include:${CPATH:-}"
    [ -d "$_tgt/lib" ] && export LIBRARY_PATH="$_tgt/lib:${LIBRARY_PATH:-}"
  fi
  echo "  CUDA_HOME=${CUDA_HOME:-<system>} | nvcc=$(command -v nvcc) | arch=$ARCH_LIST"
  # best-effort: these are an OPTIONAL acceleration for the pretrained mamba2 checkpoints. FLA's
  # Triton mamba2 runs without them (use FLA_CONV_BACKEND=triton), so a build failure on bleeding-
  # edge GPUs is non-fatal — we warn and continue.
  if MAX_JOBS="${MAX_JOBS:-8}" "$PY" -m pip install --no-build-isolation causal-conv1d mamba-ssm; then
    echo "  mamba_ssm / causal_conv1d built OK"
  else
    echo "  !! mamba_ssm/causal_conv1d build FAILED — continuing. The FLA Triton mamba2 path still"
    echo "     works; run mamba2 with FLA_CONV_BACKEND=triton (no CUDA kernels needed)."
  fi
else
  echo "  gdn backbone uses only this repo's FLA Triton ops — mamba_ssm/causal_conv1d NOT needed."
fi

# --- runtime env ----------------------------------------------------------------------------
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p ckpt logs

echo "=== [4] sanity ==="
"$PY" - <<'PY'
import torch
ok = torch.cuda.is_available()
print("  torch", torch.__version__, "| cuda", ok,
      "|", torch.cuda.get_device_name(0) if ok else "CPU")
if ok:
    cap = torch.cuda.get_device_capability(0)
    print(f"  device capability sm_{cap[0]}{cap[1]}")
PY

echo ""
echo "=== DONE ($GPU). Always run experiments from repo root with: ==="
echo "  export PYTHONPATH=$REPO_ROOT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  export TORCH_CUDA_ARCH_LIST=$ARCH_LIST   # only matters for source builds"
