#!/usr/bin/env bash
# 0025 VESSL bootstrap — runs ON the VESSL container. Idempotent / safe to re-run
# after a container restart (which wipes everything outside /root/smaller,
# including pip site-packages and /root/.ssh).
#
# Layout created under the persistent geesefs mount:
#   /root/smaller/mc_niah/{code,data,results,logs,ckpts,pydeps}
#
# Usage (on VESSL):
#   bash bootstrap.sh
set -euo pipefail

ROOT=/root/smaller/mc_niah
CODE="$ROOT/code"
PYDEPS="$ROOT/pydeps"
CKPTS="$ROOT/ckpts"
GH_TOKEN_FILE=/root/smaller/.gh_token_new
PY=/opt/conda/bin/python

LMR_REPO_SLUG="Smaller25/linear-memory-routing"
LMR_BRANCH="sh/mc-niah-analysis"
LONGGDN_REPO_URL="https://github.com/gyunggyung/long-gdn.git"
LONGGDN_PIN="e71713e402fcbf63b50365ed97ab2d5555cb6710"
FLA_REPO_URL="https://github.com/fla-org/flash-linear-attention.git"
FLA_PIN="4b02d15d6a68700181b180235be62a9fb95d2a38"

log() { echo "[bootstrap] $*"; }

mkdir -p "$ROOT"/{code,data,results,logs,ckpts,pydeps}

# ---------------------------------------------------------------------------
# 1. lmr repo (this repo) — clone-or-pull, private, needs token
# ---------------------------------------------------------------------------
if [ ! -f "$GH_TOKEN_FILE" ]; then
  echo "[bootstrap] BLOCKED: token file $GH_TOKEN_FILE missing" >&2
  exit 1
fi
GH_TOKEN=$(cat "$GH_TOKEN_FILE")
LMR_URL="https://${GH_TOKEN}@github.com/${LMR_REPO_SLUG}.git"
LMR_DIR="$CODE/linear-memory-routing"

if [ -d "$LMR_DIR/.git" ]; then
  log "lmr repo exists — pulling ${LMR_BRANCH}"
  git -C "$LMR_DIR" remote set-url origin "$LMR_URL"
  git -C "$LMR_DIR" fetch origin "$LMR_BRANCH"
  git -C "$LMR_DIR" checkout "$LMR_BRANCH" 2>/dev/null || git -C "$LMR_DIR" checkout -b "$LMR_BRANCH" "origin/${LMR_BRANCH}"
  git -C "$LMR_DIR" reset --hard "origin/${LMR_BRANCH}"
else
  log "cloning lmr repo (${LMR_BRANCH})"
  git clone --branch "$LMR_BRANCH" "$LMR_URL" "$LMR_DIR"
fi
# scrub token from stored remote url (best-effort hygiene; local disk only)
git -C "$LMR_DIR" remote set-url origin "https://github.com/${LMR_REPO_SLUG}.git"

# ---------------------------------------------------------------------------
# 2. long-gdn (public) — clone-or-reuse, pin to fixed commit (plain checkout)
# ---------------------------------------------------------------------------
LONGGDN_DIR="$CODE/long-gdn"
if [ ! -d "$LONGGDN_DIR/.git" ]; then
  log "cloning long-gdn"
  git clone "$LONGGDN_REPO_URL" "$LONGGDN_DIR"
fi
git -C "$LONGGDN_DIR" fetch origin
git -C "$LONGGDN_DIR" checkout "$LONGGDN_PIN"
log "long-gdn @ $(git -C "$LONGGDN_DIR" rev-parse --short HEAD)"

# ---------------------------------------------------------------------------
# 3. fla — pinned commit, installed once into pydeps (persistent, --target)
# ---------------------------------------------------------------------------
FLA_MARKER="$PYDEPS/.fla_${FLA_PIN}_installed"
if [ -f "$FLA_MARKER" ]; then
  log "fla already installed into pydeps @ ${FLA_PIN} — skipping"
else
  log "installing fla @ ${FLA_PIN} into pydeps"
  FLA_BUILD_DIR="$CODE/flash-linear-attention"
  if [ ! -d "$FLA_BUILD_DIR/.git" ]; then
    git clone "$FLA_REPO_URL" "$FLA_BUILD_DIR"
  fi
  git -C "$FLA_BUILD_DIR" fetch origin
  git -C "$FLA_BUILD_DIR" checkout "$FLA_PIN"
  # remove any stale partial install of fla itself from a previous failed attempt
  rm -rf "$PYDEPS/fla" "$PYDEPS"/fla-*.dist-info 2>/dev/null || true
  "$PY" -m pip install --no-deps --no-cache-dir --target "$PYDEPS" "$FLA_BUILD_DIR"
  touch "$FLA_MARKER"
  log "fla installed @ $(git -C "$FLA_BUILD_DIR" rev-parse --short HEAD)"
fi

# ---------------------------------------------------------------------------
# 4. volatile deps — reinstalled every boot (container restart wipes conda
#    site-packages outside /root/smaller). Image is assumed to already ship
#    torch (verified: 2.9.1+cu128 on this image) — do NOT touch torch here.
# ---------------------------------------------------------------------------
log "installing volatile deps (einops transformers huggingface_hub numpy matplotlib)"
"$PY" -m pip install -q einops transformers huggingface_hub numpy matplotlib
"$PY" - <<'EOF'
import torch, einops, transformers, huggingface_hub, numpy, matplotlib
print(f"[bootstrap] versions: torch={torch.__version__} einops={einops.__version__} "
      f"transformers={transformers.__version__} huggingface_hub={huggingface_hub.__version__} "
      f"numpy={numpy.__version__} matplotlib={matplotlib.__version__} "
      f"cuda_available={torch.cuda.is_available()}")
EOF

# ---------------------------------------------------------------------------
# 5. checkpoints — download once into persistent ckpts/, skip if plausible
#    size already present. Cache lives in container-local /tmp (geesefs trap:
#    never point HF_HOME at /root/smaller).
# ---------------------------------------------------------------------------
MIN_BYTES=$((1024 * 1024 * 1024))  # 1GB plausibility floor

download_ckpt() {
  local repo="$1" fname="$2"
  local dest_dir="$CKPTS/$(echo "$repo" | tr '/' '__')"
  local dest="$dest_dir/$fname"
  if [ -f "$dest" ]; then
    local sz
    sz=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    if [ "$sz" -gt "$MIN_BYTES" ]; then
      log "ckpt present (${sz} bytes): $dest — skipping"
      return 0
    fi
    log "ckpt present but too small (${sz} bytes) — re-downloading: $dest"
  fi
  mkdir -p "$dest_dir"
  mkdir -p /tmp/hf_home
  log "downloading $repo :: $fname"
  HF_HOME=/tmp/hf_home "$PY" - "$repo" "$fname" "$dest" <<'EOF'
import sys, os, shutil
from huggingface_hub import hf_hub_download
repo, fname, dest = sys.argv[1:4]
p = hf_hub_download(repo, fname)
shutil.copy2(p, dest)
print(f"[bootstrap]   -> {dest} ({os.path.getsize(dest)} bytes)")
EOF
}

download_ckpt "LLM-OS-Models2/mc-gdn2-370m-fineweb-edu-30b-v2-meanpool" "checkpoint-30B-model-ckpt.pth"
download_ckpt "LLM-OS-Models2/mc-gdn2-370m-fineweb-edu-30b-v2-meanpool" "checkpoint-5B-model-ckpt.pth"
download_ckpt "LLM-OS-Models2/gdn2-370m-fineweb-edu-5b-vanilla" "checkpoint-5B-model-ckpt.pth"

log "bootstrap complete."
