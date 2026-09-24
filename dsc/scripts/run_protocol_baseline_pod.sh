#!/usr/bin/env bash
# The fixed-protocol baseline table. Every number in one setting.
#
# This exists because the same untouched baseline was recorded as 4.0 (one
# seed), 2.3 (three seeds) and 3.0 (two clean seeds), and the oracle as 82 and
# 74, across a month of arms. None of those could be put in one table. Until
# this run exists, no claim in report 0027 has a comparable denominator.
#
# Protocol, fixed and not to be varied by later arms:
#   length 8192, needles 4 and 16, seeds 42/43/44, 50 items per cell,
#   topk 2, gate native, n_gen 48, base and oracle in the SAME run,
#   router training on seed 45, which the evaluation never touches.
#
# Five arms:
#   vanilla    single-state GDN-2, no cache at all. The hook: caching has to
#              beat this, and at 8K it does not.
#   base       MC-SSC untouched.
#   maxsim     layer-shared routing from L0 + m=8 sub-block descriptors.
#   oracle     gold forced into every layer's top-1. The ceiling, and the
#              denominator for "share of achievable".
#   dense      every segment read. Not a dense ARCHITECTURE -- a model trained
#              with top-2 forced to read everything, which measures dilution.
set -uo pipefail
cd /root/work/lmr
# src/ is on the path because the official RULER scoring function
# (ruler.eval_metrics) lives there; without it every arm dies after
# loading the model, which is 15 seconds in and looks like a model bug.
export PYTHONPATH=/root/work/lmr:/root/work/lmr/dsc:/root/work/lmr/src
export MC_KERNEL_VERSION=v2 TOKENIZERS_PARALLELISM=false
export HF_HOME=/root/cache/hf TRITON_CACHE_DIR=/root/cache/triton
export WANDB_DIR=/root/cache/wandb
PY=/opt/conda/bin/python
MC=/root/smaller/mc/ckpts/LLM-OS-Models2_mc-gdn2-370m-fineweb-edu-30b-v2-meanpool/checkpoint-30B-model-ckpt.pth
VAN=/root/smaller/mc/ckpts/LLM-OS-Models2_gdn2-370m-fineweb-edu-5b-vanilla/checkpoint-5B-model-ckpt.pth
DATA=/root/smaller/mc/data/dk
OUT=/root/smaller/mc/out/protocol
ROUTER=/root/smaller/mc/out/routers_maxsim8
mkdir -p "$OUT"
step() { echo "=== $* ($(date +%T)) ==="; }

step "0/4 training cells on seed 45 (evaluation never sees this seed)"
$PY -u dsc/scripts/gen_diverse_key_niah.py --data-root "$DATA" \
    --lengths 2048 4096 8192 --needles 1 4 32 --seeds 45 \
    --haystack essay --num-samples 50 --tokenizer TinyLlama/TinyLlama_v1.1 \
    > "$OUT/gen45.log" 2>&1 || { echo "GEN FAILED"; tail -15 "$OUT/gen45.log"; exit 1; }
grep -E "done\]|FAIL" "$OUT/gen45.log" | tail -2

step "1/4 capture for the router (seed 45, 15 cells, 8 blocks)"
if [ -s "$ROUTER/router_L0.pt" ]; then
    echo "router present, skipping capture+fit"
else
    $PY -u dsc/scripts/capture_key_identity.py --ckpt "$MC" \
        --config-name mc_370M --data-root "$DATA" \
        --cells 2048:1 2048:4 2048:8 2048:16 2048:32 \
                4096:1 4096:4 4096:8 4096:16 4096:32 \
                8192:1 8192:4 8192:8 8192:16 8192:32 \
        --seeds 45 --max-samples 50 --blocks 8 --layers 0 1 --taus 0 \
        --out "$OUT/keyid_s45.pt" > "$OUT/capture.log" 2>&1 || {
        echo "CAPTURE FAILED"; tail -20 "$OUT/capture.log"; exit 1; }
    grep -E "^\[capture\]" "$OUT/capture.log"

    step "2/4 fit the m=8 router (CPU)"
    CUDA_VISIBLE_DEVICES="" $PY -u dsc/scripts/train_maxsim_router.py \
        --cache "$OUT/keyid_s45.pt" --blocks 8 --layers 0 1 --steps 12000 \
        --out "$ROUTER" > "$OUT/fit.log" 2>&1 || {
        echo "FIT FAILED"; tail -20 "$OUT/fit.log"; exit 1; }
    grep -E "^\[layer|^\[train\] wrote" "$OUT/fit.log"
fi

arm() {
    local LABEL=$1 CKPT=$2 CFG=$3; shift 3
    local T=$SECONDS
    step "arm $LABEL"
    $PY -u dsc/scripts/diverse_key_niah_eval.py \
        --backend lit_gpt --ckpt "$CKPT" --config-name "$CFG" \
        --model-label "$LABEL" --gate-label "$LABEL" \
        --tokenizer TinyLlama/TinyLlama_v1.1 --data-root "$DATA" \
        --lengths 8192 --needles 4 16 --seeds 42 43 44 \
        --max-examples 50 --batch-size 4 --n-gen 48 --log-routing \
        --out-dir "$OUT" --summary-name "${LABEL}.json" "$@" \
        > "$OUT/${LABEL}.log" 2>&1
    echo "$LABEL rc=$? wall=$((SECONDS - T))s" | tee -a "$OUT/wallclock.txt"
    grep -E "^\[cell\]" "$OUT/${LABEL}.log" || true
}

step "3/4 five arms, one protocol"
arm vanilla "$VAN" gdn2_370M
arm base    "$MC"  mc_370M --config-overrides mc_topk=2
arm maxsim  "$MC"  mc_370M --config-overrides mc_topk=2 \
    --broadcast-routing --broadcast-source mlp --broadcast-source-layer 0 \
    --mlp-router "$ROUTER"
arm oracle  "$MC"  mc_370M --config-overrides mc_topk=2 --oracle-routing
arm dense   "$MC"  mc_370M --config-overrides mc_topk=32

step "4/4 paired comparisons against the untouched baseline"
for A in vanilla maxsim oracle dense; do
    $PY dsc/scripts/compare_arms.py --a "$OUT/base" --b "$OUT/$A" \
        --label-a base --label-b "$A" --ceiling "$OUT/oracle" \
        --out "$OUT/cmp_base_vs_$A.json" 2>&1 | sed -n '2,12p'
done
echo "=== PROTOCOL BASELINE DONE ($(date +%T)) ==="
