#!/usr/bin/env bash
# 0025 VESSL bootstrap — runs ON the VESSL container. Idempotent / safe to re-run
# after a container restart (which wipes everything outside /root/smaller,
# including pip site-packages, /root/.ssh, AND /root/work).
#
# Layout:
#   /root/smaller/mc_niah/{data,results,logs,ckpts,vendor}  — persistent (geesefs)
#   /root/smaller/mc_niah/code/linear-memory-routing        — persistent (geesefs; this repo)
#   /root/work/{long-gdn,pydeps}                             — container-local disk, EPHEMERAL
#
# NOTE (operational lesson, verified 2026-07-27): `git clone`/`checkout` and
# `pip install --target` directly onto geesefs (/root/smaller) is unreliable
# for trees with many small files — clones can silently leave partial/corrupt
# working trees, and `rm -rf` can fail with "Directory not empty" on things
# that were supposedly just removed. The lmr repo itself (a normal git clone
# of moderate size) has been reliable so far and stays on geesefs; long-gdn's
# checked-out working tree and fla's installed package tree do NOT — they
# live on /root/work (container-local, fast, reliable) and are *restored* on
# every boot from small vendor artifacts on geesefs (a git bundle and a pip
# --target tarball), which is cheap (seconds) compared to a fresh network
# clone/install.
#
# Usage (on VESSL):
#   bash bootstrap.sh
set -euo pipefail

ROOT=/root/smaller/mc_niah
CODE="$ROOT/code"
CKPTS="$ROOT/ckpts"
VENDOR="$ROOT/vendor"
WORK=/root/work
LONGGDN_DIR="$WORK/long-gdn"
PYDEPS="$WORK/pydeps"
GH_TOKEN_FILE=/root/smaller/.gh_token_new
PY=/opt/conda/bin/python

LMR_REPO_SLUG="Smaller25/linear-memory-routing"
LMR_BRANCH="sh/mc-niah-analysis"
LONGGDN_REPO_URL="https://github.com/gyunggyung/long-gdn.git"
LONGGDN_PIN="e71713e402fcbf63b50365ed97ab2d5555cb6710"
LONGGDN_BUNDLE="$VENDOR/long-gdn-e71713e.bundle"
LONGGDN_TOKEN_FILE=/root/smaller/.gh_token_longgdn
FLA_REPO_URL="https://github.com/fla-org/flash-linear-attention.git"
FLA_PIN="4b02d15d6a68700181b180235be62a9fb95d2a38"
FLA_TARBALL="$VENDOR/pydeps-fla-4b02d15d.tar.gz"

log() { echo "[bootstrap] $*"; }

mkdir -p "$ROOT"/{code,data,results,logs,ckpts,vendor} "$WORK"

# ---------------------------------------------------------------------------
# 1. lmr repo (this repo) — clone-or-pull, private, needs token. Geesefs.
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
# 2. long-gdn — restored onto /root/work (container-local), NOT geesefs.
#    NOTE: gyunggyung/long-gdn is actually PRIVATE (confirmed via GitHub API:
#    "private": true) despite the original assumption it was public, and its
#    data/ is git-lfs tracked (we only need lit_gpt/dsc code, so smudging is
#    skipped unconditionally). Preferred source is the vendor bundle (fast,
#    no network dependency); falls back to a token-authenticated fetch
#    (/root/smaller/.gh_token_longgdn) or a bare anonymous fetch if the
#    bundle is ever missing.
# ---------------------------------------------------------------------------
export GIT_LFS_SKIP_SMUDGE=1

if [ -f "$LONGGDN_DIR/dsc/mc_baseline/mc_ssc.py" ] && git -C "$LONGGDN_DIR" cat-file -e "${LONGGDN_PIN}^{commit}" 2>/dev/null; then
  log "long-gdn already present on /root/work @ pin"
else
  rm -rf "$LONGGDN_DIR"
  if [ -f "$LONGGDN_BUNDLE" ]; then
    log "restoring long-gdn from vendor bundle ($LONGGDN_BUNDLE)"
    git clone -q "$LONGGDN_BUNDLE" "$LONGGDN_DIR" || true  # clone warns (no HEAD symref in bundle) but still populates objects
    git -C "$LONGGDN_DIR" checkout -q "$LONGGDN_PIN"
  else
    log "vendor bundle missing — fetching long-gdn from network"
    mkdir -p "$LONGGDN_DIR"
    git -C "$LONGGDN_DIR" init -q
    if [ -f "$LONGGDN_TOKEN_FILE" ]; then
      LONGGDN_TOK=$(cat "$LONGGDN_TOKEN_FILE")
      LONGGDN_CLONE_URL="https://oauth2:${LONGGDN_TOK}@github.com/gyunggyung/long-gdn.git"
    else
      LONGGDN_CLONE_URL="$LONGGDN_REPO_URL"
    fi
    # Targeted single-ref fetch, never a full multi-branch `git clone` — this
    # repo's ref advertisement has tripped git's "multiple updates for ref
    # ... not allowed" on a plain clone (observed 2026-07-27).
    if ! git -C "$LONGGDN_DIR" fetch -q "$LONGGDN_CLONE_URL" "$LONGGDN_PIN"; then
      echo "[bootstrap] BLOCKED: cannot fetch long-gdn pin ${LONGGDN_PIN} (no vendor bundle, network/token fetch failed)" >&2
      exit 1
    fi
    git -C "$LONGGDN_DIR" checkout -q FETCH_HEAD
  fi
  git -C "$LONGGDN_DIR" remote set-url origin "$LONGGDN_REPO_URL" 2>/dev/null || git -C "$LONGGDN_DIR" remote add origin "$LONGGDN_REPO_URL"
fi
grep -q "mean of L2-normalized keys" "$LONGGDN_DIR/dsc/mc_baseline/mc_ssc.py" && log "long-gdn mean-pool pin OK"
log "long-gdn @ $(git -C "$LONGGDN_DIR" rev-parse --short HEAD)"

# ---------------------------------------------------------------------------
# 3. fla — pinned commit, installed into /root/work/pydeps (container-local).
#    Restored from a vendor tarball (built once, persisted to geesefs) rather
#    than re-running `pip install --target` on every boot.
# ---------------------------------------------------------------------------
if [ -d "$PYDEPS/fla" ]; then
  log "fla already present on /root/work/pydeps — skipping"
elif [ -f "$FLA_TARBALL" ]; then
  log "restoring pydeps from vendor tarball ($FLA_TARBALL)"
  mkdir -p "$WORK"
  tar xzf "$FLA_TARBALL" -C "$WORK"
else
  log "no vendor tarball — building fla @ ${FLA_PIN} from source into /root/work/pydeps"
  FLA_BUILD_DIR="$WORK/flash-linear-attention"
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
  mkdir -p "$PYDEPS"
  "$PY" -m pip install --no-deps --no-cache-dir --target "$PYDEPS" "$FLA_BUILD_DIR"
  # persist a tarball to geesefs so future boots skip the pip install entirely
  tar czf "$FLA_TARBALL" -C "$WORK" pydeps
  log "fla installed @ $(git -C "$FLA_BUILD_DIR" rev-parse --short HEAD); vendor tarball written"
fi

# ---------------------------------------------------------------------------
# 4. volatile deps — reinstalled every boot (container restart wipes conda
#    site-packages outside /root/smaller). Image is assumed to already ship
#    torch (verified: 2.9.1+cu128 on this image) — do NOT touch torch here.
#    lightning + sentencepiece are required by lit_gpt.utils (imports
#    lightning.fabric) — lesson from the first successful live smoke run.
# ---------------------------------------------------------------------------
log "installing volatile deps (einops transformers huggingface_hub numpy matplotlib lightning sentencepiece)"
"$PY" -m pip install -q einops transformers huggingface_hub numpy matplotlib lightning sentencepiece
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
#    never point HF_HOME at /root/smaller). Large few-file downloads are fine
#    on geesefs (unlike git's many-small-files pattern above).
# ---------------------------------------------------------------------------
MIN_BYTES=$((1024 * 1024 * 1024))  # 1GB plausibility floor

download_ckpt() {
  local repo="$1" fname="$2"
  local dest_dir="$CKPTS/$(echo "$repo" | tr '/' '_')"
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
