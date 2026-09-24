#!/usr/bin/env bash
# Track 2: train a per-segment soft prompt, then screen it on routing hit.
#
# Order follows the funnel this project settled on. Routing hit is one prompt
# forward per item and costs a minute for 300 items; a score arm costs hours.
# The prompt is trained to improve exactly the quantity hit measures, so if it
# does not clear the paired detection floor here there is nothing to score.
#
# Three arms, all on the fixed protocol (seeds 42/43/44, 8192, N=4 and 16, 50
# items per cell, topk 2):
#   native      no prompt at all. The only exact control, since a prefix of
#               zero vectors is still a token.
#   p8-untrained  the layout cost on its own: 8 warm-started vectors that no
#               objective has touched. Separates "inserting tokens changed
#               the routing" from "the training changed the routing".
#   p8-trained  the arm under test.
set -uo pipefail
cd /root/work/lmr
export PYTHONPATH=/root/work/lmr:/root/work/lmr/dsc
export MC_KERNEL_VERSION=v2 TOKENIZERS_PARALLELISM=false
export HF_HOME=/root/cache/hf TRITON_CACHE_DIR=/root/cache/triton
export WANDB_DIR=/root/cache/wandb
PY=/opt/conda/bin/python
CKPT=/root/smaller/mc/ckpts/LLM-OS-Models2_mc-gdn2-370m-fineweb-edu-30b-v2-meanpool/checkpoint-30B-model-ckpt.pth
DATA=/root/smaller/mc/data/dk
OUT=/root/smaller/mc/out/t2_prompt
mkdir -p "$OUT"
step() { echo "=== $* ($(date +%T)) ==="; }

hit() {
    local TAG=$1; shift
    $PY -u dsc/scripts/measure_hit.py --ckpt "$CKPT" --config-name mc_370M \
        --data-root "$DATA" --arm native --topk 2 \
        --cells 8192:4 8192:16 --seeds 42 43 44 --max-samples 50 \
        --batch-size 4 --out "$OUT/hit_$TAG.jsonl" "$@" \
        > "$OUT/hit_$TAG.log" 2>&1 || {
        echo "HIT FAILED ($TAG)"; tail -20 "$OUT/hit_$TAG.log"; return 1; }
    sed -n '/^cell /,$p' "$OUT/hit_$TAG.log" | head -9
}

step "1/4 baseline routing hit, no prompt"
hit native || exit 1

step "2/4 train the prompt (frozen backbone, 8 vectors)"
$PY -u dsc/scripts/train_soft_prompt.py \
    --ckpt "$CKPT" --config-name mc_370M --data-root "$DATA" \
    --train-cells 2048:8 4096:8 8192:8 8192:16 --train-seed 45 \
    --max-samples 50 --n-prefix 8 --n-suffix 0 --warm-start "." \
    --steps 2000 --lr 1e-3 --ppl-tolerance 0.05 \
    --out "$OUT/p8" > "$OUT/train_p8.log" 2>&1 || {
    echo "TRAIN FAILED"; tail -25 "$OUT/train_p8.log"; exit 1; }
grep -E "^\[sp\]" "$OUT/train_p8.log" | tail -14

step "3/4 routing hit, untrained layout control"
hit p8-untrained --n-prefix 8 || exit 1

step "4/4 routing hit, trained prompt"
hit p8-trained --n-prefix 8 --soft-prompt "$OUT/p8" || exit 1

step "paired comparisons on the clean seeds"
for T in p8-untrained p8-trained; do
    $PY dsc/scripts/compare_arms.py --metric hit \
        --a "$OUT/hit_native.jsonl" --b "$OUT/hit_$T.jsonl" \
        --label-a "no prompt" --label-b "$T" \
        --out "$OUT/cmp_native_vs_$T.json" 2>&1 | sed -n '2,12p'
done
echo "=== T2 PROMPT DONE ($(date +%T)) ==="
