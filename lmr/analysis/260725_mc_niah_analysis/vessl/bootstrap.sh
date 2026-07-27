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

mkdir -p "$ROOT"/{code,data,results,logs,ckpts,pydeps,vendor}

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
# 2. long-gdn — clone-or-reuse, pin to fixed commit (plain checkout).
#    NOTE: gyunggyung/long-gdn is actually PRIVATE (confirmed via GitHub API:
#    "private": true) despite the original assumption that it was public.
#    Anonymous clone AND the lmr token both get HTTP 401 "Repository not
#    found" from VESSL's egress IP (verified 2026-07-27). Needs its own token
#    at /root/smaller/.gh_token_longgdn (a `gho_...` OAuth token with access
#    to this repo). If that token file is ever missing, fall back to a
#    pre-staged bundle of the pinned commit (built on greenbeard, scp'd once
#    to $ROOT/vendor/), then to a bare anonymous clone as a last resort.
# ---------------------------------------------------------------------------
LONGGDN_DIR="$CODE/long-gdn"
LONGGDN_BUNDLE="$ROOT/vendor/long-gdn-e71713e.bundle"
LONGGDN_TOKEN_FILE=/root/smaller/.gh_token_longgdn
# We only need the lit_gpt/dsc *code* from this repo (model defs), not the
# large git-lfs-tracked data/ files (replay-eval jsonls etc) — skip smudging
# unconditionally to save bandwidth/time; nothing here reads data/.
export GIT_LFS_SKIP_SMUDGE=1

if [ -f "$LONGGDN_TOKEN_FILE" ]; then
  LONGGDN_TOK=$(cat "$LONGGDN_TOKEN_FILE")
  LONGGDN_CLONE_URL="https://oauth2:${LONGGDN_TOK}@github.com/gyunggyung/long-gdn.git"
else
  LONGGDN_CLONE_URL="$LONGGDN_REPO_URL"
fi

# Always fetch exactly one ref (the pinned commit) via `init` + `fetch <src>
# <ref>` + `checkout FETCH_HEAD`, never a full multi-branch `git clone` — this
# repo's ref advertisement has tripped git's "multiple updates for ref ...
# not allowed" on a plain clone (observed 2026-07-27), and a single targeted
# fetch avoids enumerating all branches anyway. Works identically whether
# <src> is a real remote URL or a local bundle file.
if [ ! -d "$LONGGDN_DIR/.git" ]; then
  mkdir -p "$LONGGDN_DIR"
  git -C "$LONGGDN_DIR" init -q
fi
if git -C "$LONGGDN_DIR" cat-file -e "${LONGGDN_PIN}^{commit}" 2>/dev/null; then
  log "long-gdn pin already present locally"
elif git -C "$LONGGDN_DIR" fetch -q "$LONGGDN_CLONE_URL" "$LONGGDN_PIN" 2>/tmp/longgdn_fetch.err; then
  log "fetched long-gdn pin (token: $([ -f "$LONGGDN_TOKEN_FILE" ] && echo yes || echo no))"
elif [ -f "$LONGGDN_BUNDLE" ] && git -C "$LONGGDN_DIR" fetch -q "$LONGGDN_BUNDLE" "$LONGGDN_PIN"; then
  log "network/token fetch unavailable — used pre-staged bundle ($LONGGDN_BUNDLE)"
else
  cat /tmp/longgdn_fetch.err >&2 2>/dev/null || true
  echo "[bootstrap] BLOCKED: cannot fetch long-gdn pin ${LONGGDN_PIN} (no token, no bundle, or both failed)" >&2
  exit 1
fi
git -C "$LONGGDN_DIR" checkout -q FETCH_HEAD 2>/dev/null || git -C "$LONGGDN_DIR" checkout -q "$LONGGDN_PIN"
git -C "$LONGGDN_DIR" remote set-url origin "$LONGGDN_REPO_URL" 2>/dev/null || git -C "$LONGGDN_DIR" remote add origin "$LONGGDN_REPO_URL"
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
  # Same targeted init+fetch<pin>+checkout as long-gdn above — a plain `git
  # clone` enumerates every remote branch, and fla-org/flash-linear-attention
  # has dozens of them, so it risks the same "multiple updates for ref ...
  # not allowed" failure observed on long-gdn.
  if [ ! -d "$FLA_BUILD_DIR/.git" ]; then
    mkdir -p "$FLA_BUILD_DIR"
    git -C "$FLA_BUILD_DIR" init -q
  fi
  if ! git -C "$FLA_BUILD_DIR" cat-file -e "${FLA_PIN}^{commit}" 2>/dev/null; then
    git -C "$FLA_BUILD_DIR" fetch -q "$FLA_REPO_URL" "$FLA_PIN"
  fi
  git -C "$FLA_BUILD_DIR" checkout -q FETCH_HEAD 2>/dev/null || git -C "$FLA_BUILD_DIR" checkout -q "$FLA_PIN"
  git -C "$FLA_BUILD_DIR" remote add origin "$FLA_REPO_URL" 2>/dev/null || true
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
