#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# VESSL job entrypoint for linear-memory-routing.
#
# Container : quay.io/vessl-ai/torch:2.9.1-cuda12.8.1-py3.10-slim  (torch 2.9.1 / cu12.8 / py3.10)
# Secret    : GH_TOKEN  — GitHub PAT (repo scope), injected by VESSL as an env var.
# Optional  : HF_TOKEN  — for gated HF downloads (the Mistral tokenizer the GDN ckpt uses).
#             Experiment knobs (env, all optional): BRANCH ARCH VARIANT MODEL TRAIN_LEN BATCH
#             STEPS LR LOW_RANK EVAL_LENGTHS DTYPE OUT  (defaults below — GDN-1.3B SSC by default).
#
# Two ways to launch from a VESSL job "command":
#   A) repo already mounted/cloned by VESSL  ->  bash scripts/vessl_run.sh
#   B) bootstrap clone (recommended), one line:
#        git clone --branch "${BRANCH:-gdn-base-and-mechanisms}" \
#          "https://x-access-token:${GH_TOKEN}@github.com/Smaller25/linear-memory-routing.git" lmr-repo \
#        && cd lmr-repo && bash scripts/vessl_run.sh
#
# Why H100: GDN-1.3B has head_dim=256; the chunk_gated_delta_rule BACKWARD kernel needs ~225KB
# shared memory > A100's 167KB limit. H100 (228KB/SM) fits, so router TRAINING on GDN runs here.
# ---------------------------------------------------------------------------------------------
set -euo pipefail

REPO_SLUG="Smaller25/linear-memory-routing"
BRANCH="${BRANCH:-gdn-base-and-mechanisms}"

echo "=== [0] GitHub auth + sync to origin/${BRANCH} ==="
if [ -n "${GH_TOKEN:-}" ]; then
  git config --global credential.helper store
  printf 'https://x-access-token:%s@github.com\n' "$GH_TOKEN" > "$HOME/.git-credentials"
  chmod 600 "$HOME/.git-credentials"
  git config --global user.name  "${GIT_NAME:-Sohyung Kim}"
  git config --global user.email "${GIT_EMAIL:-sohyung.kim@kaist.ac.kr}"
fi
if git rev-parse --git-dir >/dev/null 2>&1; then
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git reset --hard "origin/$BRANCH"   # ckpt/ and logs/ are gitignored, so they survive
else
  echo "  not inside a git repo — clone first (see option B in the header)"; exit 1
fi
echo "  HEAD: $(git log --oneline -1)"

echo "=== [1] Python deps (GPU-aware; delegated to scripts/setup_env.sh) ==="
# All dependency logic lives in setup_env.sh, which branches on the detected GPU (A100/H100/
# Blackwell). On this H100 container it auto-enables tilelang (needed for GDN router TRAINING:
# Hopper+Triton>=3.4 miscomputes FLA's gated-delta chunk BACKWARD, fla #640, so fla dispatches
# to the tilelang backend). Set BACKBONE=mamba2 to also source-build mamba_ssm/causal_conv1d.
BACKBONE="${BACKBONE:-gdn}" bash scripts/setup_env.sh

# Optional HF auth (the GDN ckpt ships no tokenizer; loader falls back to the Mistral tokenizer,
# which may be gated — set HF_TOKEN as a VESSL secret if the download 403s).
if [ -n "${HF_TOKEN:-}" ]; then
  python -c "from huggingface_hub import login; login('${HF_TOKEN}')" || true
fi

# fla + lmr import from the repo root (pyproject sets pythonpath, but be explicit for any cwd).
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p ckpt logs

echo "=== [2] sanity: GPU + GDN correctness gates ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
python -m pytest tests/lmr/test_segment_runner_gdn.py -q || true

echo "=== [3] run experiment ==="
ARCH="${ARCH:-gdn}"
VARIANT="${VARIANT:-ssc}"
TRAIN_LEN="${TRAIN_LEN:-2048}"
BATCH="${BATCH:-2}"
STEPS="${STEPS:-250}"
LR="${LR:-1e-3}"
LOW_RANK="${LOW_RANK:-64}"
EVAL_LENGTHS="${EVAL_LENGTHS:-4096 8192 16384}"
DTYPE="${DTYPE:-float32}"   # gdn-1.3b fits fp32 on H100; use bfloat16 for big mamba2
OUT="${OUT:-ckpt/${VARIANT}_${ARCH}.pt}"
MODEL_ARG=(); [ -n "${MODEL:-}" ] && MODEL_ARG=(--model "$MODEL")

set -x
python -m lmr.scripts.train_grm_passkey \
  --arch "$ARCH" --variant "$VARIANT" --dtype "$DTYPE" "${MODEL_ARG[@]}" \
  --train-len "$TRAIN_LEN" --batch "$BATCH" --steps "$STEPS" --lr "$LR" \
  --low-rank-dim "$LOW_RANK" --eval-lengths $EVAL_LENGTHS --out "$OUT" \
  2>&1 | tee "logs/${VARIANT}_${ARCH}.log"
set +x

echo "=== [4] commit + push results (ckpt/*.pt and logs/ are gitignored, so extract to report/runs/) ==="
RUN_ID="${ARCH}_${VARIANT}_$(date +%Y%m%d_%H%M%S)"
RESULT="report/runs/${RUN_ID}.md"
mkdir -p report/runs
{
  echo "# Auto run — ${ARCH} / ${VARIANT}  (${RUN_ID})"
  echo
  echo "- model: ${MODEL:-default for ${ARCH}} | train_len ${TRAIN_LEN} | batch ${BATCH} | steps ${STEPS} | low_rank ${LOW_RANK} | dtype ${DTYPE}"
  echo "- GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo '?')"
  echo "- commit: $(git rev-parse --short HEAD)"
  echo
  echo '```'
  # final train lines + the vanilla/+RM/+variant eval table from the run log
  grep -E '\[train\]|^  step|length|vanilla|^ *[0-9]+ \||saved' "logs/${VARIANT}_${ARCH}.log" | tail -40
  echo '```'
} > "$RESULT"
echo "  wrote $RESULT"

if [ -n "${GH_TOKEN:-}" ]; then
  git add "$RESULT" report/ notes/ 2>/dev/null || true
  if ! git diff --cached --quiet; then
    git commit -q -m "vessl run: ${ARCH}/${VARIANT} results (${RUN_ID})"
    # the branch may have advanced during the (long) run — rebase before pushing
    git pull --rebase --autostash origin "$BRANCH" || true
    git push origin "$BRANCH" && echo "  pushed results to origin/$BRANCH" \
      || echo "  PUSH FAILED — commit is local ($RESULT); push manually"
  else
    echo "  nothing new to commit"
  fi
else
  echo "  GH_TOKEN unset — results saved locally at $RESULT (not pushed)"
fi

echo "=== DONE. heads -> $OUT | log -> logs/${VARIANT}_${ARCH}.log | results -> $RESULT ==="
