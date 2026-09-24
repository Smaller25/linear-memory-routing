#!/usr/bin/env bash
# Track 1, step 0: does the answer span actually carry high surprisal?
#
# This is the cheapest possible verdict on the whole track and it comes before
# any router is fitted. Weighting a descriptor by surprisal can only help if
# the tokens a diverse-key query must match are the surprising ones, and that
# is a property of the data, answerable from a capture with no routing at all.
# A ratio near 1.0 ends track 1 here.
#
# The capture also stores block-level sums per tau, so if the premise holds
# the same file feeds the tau sweep and the m sweep without a second pass over
# the GPU.
set -uo pipefail
cd /root/work/lmr
export PYTHONPATH=/root/work/lmr:/root/work/lmr/dsc
export MC_KERNEL_VERSION=v2 TOKENIZERS_PARALLELISM=false
export HF_HOME=/root/cache/hf TRITON_CACHE_DIR=/root/cache/triton
export WANDB_DIR=/root/cache/wandb
PY=/opt/conda/bin/python
CKPT=/root/smaller/mc/ckpts/LLM-OS-Models2_mc-gdn2-370m-fineweb-edu-30b-v2-meanpool/checkpoint-30B-model-ckpt.pth
DATA=/root/smaller/mc/data/dk
OUT=/root/smaller/mc/out/t1_premise
mkdir -p "$OUT"

step() { echo "=== $* ($(date +%T)) ==="; }

step "0/2 capture with surprisal and per-tau block sums"
# `| tail` makes the pipeline's status tail's, so a failed capture used to
# sail on into the probe and report a missing file instead of the real error.
$PY -u dsc/scripts/capture_key_identity.py \
    --ckpt "$CKPT" --config-name mc_370M --data-root "$DATA" \
    --cells 8192:4 8192:16 --seeds 42 43 44 --max-samples 50 \
    --blocks 8 --layers 0 --taus 0 0.5 1 2 \
    --out "$OUT/keyid.pt" > "$OUT/capture.log" 2>&1 || {
    echo "CAPTURE FAILED"; tail -25 "$OUT/capture.log"; exit 1; }
tail -6 "$OUT/capture.log"

step "1/2 premise probe (CPU, no router)"
$PY -u dsc/scripts/probe_surprisal_premise.py \
    --cache "$OUT/keyid.pt" --taus 0 0.5 1 2 \
    --out "$OUT/premise.json" --wandb-name t1-premise 2>&1 | tail -24

echo "=== T1 PREMISE DONE ($(date +%T)) ==="
