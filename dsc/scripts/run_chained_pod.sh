#!/usr/bin/env bash
# Does un-fragmenting the recurrence recover what caching cost?
#
# The protocol table says vanilla 11.0, MC-SSC 2.7, and the code says why:
# every segment is scanned from initial_state=None, so a position sees only
# its own 256 tokens and the top-k read is the sole bridge. Chaining threads
# each segments final state into the next, which is section 3.4s other mode
# and is what the deployed path rejects.
#
# Read this in ONE direction only. The checkpoint was trained with independent
# compressors and has never seen a state cross a boundary, so a drop could be
# that shift rather than the fragmentation. A RISE, though, happens despite
# the shift and is evidence. The clean version is a from-scratch run.
set -uo pipefail
cd /root/work/lmr
export PYTHONPATH=/root/work/lmr:/root/work/lmr/dsc:/root/work/lmr/src
export MC_KERNEL_VERSION=v2 TOKENIZERS_PARALLELISM=false
export HF_HOME=/root/cache/hf TRITON_CACHE_DIR=/root/cache/triton
PY=/opt/conda/bin/python
MC=/root/smaller/mc/ckpts/LLM-OS-Models2_mc-gdn2-370m-fineweb-edu-30b-v2-meanpool/checkpoint-30B-model-ckpt.pth
DATA=/root/smaller/mc/data/dk
OUT=/root/smaller/mc/out/protocol
ROUTER=/root/smaller/mc/out/routers_maxsim8

arm() {
    local LABEL=$1; shift
    local T=$SECONDS
    echo "=== arm $LABEL ($(date +%T)) ==="
    $PY -u dsc/scripts/diverse_key_niah_eval.py \
        --backend lit_gpt --ckpt "$MC" --config-name mc_370M \
        --model-label "$LABEL" --gate-label "$LABEL" \
        --tokenizer TinyLlama/TinyLlama_v1.1 --data-root "$DATA" \
        --lengths 8192 --needles 4 16 --seeds 42 43 44 \
        --max-examples 50 --batch-size 4 --n-gen 48 --log-routing \
        --out-dir "$OUT" --summary-name "${LABEL}.json" "$@" \
        > "$OUT/${LABEL}.log" 2>&1
    echo "$LABEL rc=$? wall=$((SECONDS-T))s"
    grep -E "^\[cell\]" "$OUT/${LABEL}.log" || tail -12 "$OUT/${LABEL}.log"
}

arm chained --config-overrides mc_topk=2 mc_checkpoint_mode=chained
arm chained-maxsim --config-overrides mc_topk=2 mc_checkpoint_mode=chained \
    --broadcast-routing --broadcast-source mlp --broadcast-source-layer 0 \
    --mlp-router "$ROUTER"

echo "=== paired ==="
for PAIR in "base chained" "vanilla30b chained" "base chained-maxsim" "maxsim chained-maxsim"; do
    set -- $PAIR
    $PY dsc/scripts/compare_arms.py --a "$OUT/$1" --b "$OUT/$2" \
        --label-a "$1" --label-b "$2" --ceiling "$OUT/oracle" \
        --out "$OUT/cmp_$1_vs_$2.json" 2>&1 | sed -n "2,11p"
done
echo "=== CHAINED DONE ($(date +%T)) ==="
