#!/usr/bin/env bash
# Does the 50M fail the same way the 370M does?
#
# Not "does it score well" -- it will score zero. The 370M itself only
# manages 2.3 untouched on 8K diverse-key NIAH. The question is whether the
# failure has the same shape, because that is what decides whether a small
# model is a valid place to test the two interventions by training them in.
#
# Three checks, each against a known 370M number:
#
#   score   oracle far above untouched.   370M: 74.0 against 2.3.
#           If oracle is also near zero the model cannot read its own cache
#           and nothing about routing is testable at this size.
#   hit     native routing near chance.   370M: 0.033, chance about 0.065.
#           If the small model already routes well, there is no failure to
#           fix and the rung is not a proxy.
#   AUC     gold signal only in the first layers, the rest indistinguishable
#           from the original linear connector.
#           370M: L0 0.742, L1 0.706, L2-L15 0.51-0.59, control 0.507-0.554.
#
# train_mlp_router.py kills itself if the linear control leaves 0.35-0.60,
# so the capture validates itself on a model it has never seen.
set -uo pipefail
cd /root/long-gdn
export PYTHONPATH=/root/long-gdn:/root/long-gdn/dsc
export MC_KERNEL_VERSION=v2 TOKENIZERS_PARALLELISM=false
export HF_HOME=/root/dk_local/hf_home TRITON_CACHE_DIR=/root/dk_local/triton_cache
PY=/root/venv/bin/python
CKPT=/root/ladder/outputs/tsz128x4k_chinchilla_mc_50m_fineweb_edu_chinchilla/final-model-ckpt.pth
CFG=mc_50M
OUT=/root/ladder_diag_50m
ROUTER=/root/ladder_routers_50m
mkdir -p "$OUT"

step() { echo "=== $* ($(date +%T)) ==="; }

run_arm() {
    local LABEL=$1 GATE=$2 EXTRA=$3
    local T=$SECONDS
    step "arm $LABEL"
    $PY -u dsc/scripts/diverse_key_niah_eval.py \
        --backend lit_gpt --ckpt "$CKPT" --config-name "$CFG" \
        --config-overrides mc_topk=2 \
        --model-label "$LABEL" --gate-label "$GATE" \
        --tokenizer TinyLlama/TinyLlama_v1.1 \
        --data-root /root/dk_data_full \
        --lengths 8192 --needles 4 16 --seeds 42 43 44 \
        --max-examples 50 --batch-size 4 --n-gen 48 \
        --log-routing --out-dir "$OUT" --summary-name "${LABEL}.json" ${EXTRA} \
        > "$OUT/${LABEL}.log" 2>&1
    echo "$LABEL rc=$? wall=$((SECONDS - T))s" | tee -a "$OUT/wallclock.txt"
    grep -E "^\[cell\]" "$OUT/${LABEL}.log" || true
}

step "1/3 score: untouched and oracle"
run_arm "base-50m"   "hard_top2" ""
run_arm "oracle-50m" "oracle"    "--oracle-routing"

step "2/3 native routing hit, 3 seeds"
$PY -u dsc/scripts/measure_hit.py --ckpt "$CKPT" --config-name "$CFG" \
    --data-root /root/dk_data_full --arm native --topk 2 \
    --cells 8192:4 8192:16 --seeds 42 43 44 \
    --max-samples 50 --batch-size 4 --out "$OUT/hit_native.jsonl" \
    > "$OUT/hit_native.log" 2>&1 || {
    echo "HIT FAILED"; tail -20 "$OUT/hit_native.log"; }
sed -n '/^cell /,$p' "$OUT/hit_native.log" | head -9

step "3/3 per-layer gold AUC, all 16 layers, with the linear null control"
$PY -u dsc/scripts/train_mlp_router.py \
    --ckpt "$CKPT" --config-name "$CFG" --config-overrides mc_topk=2 \
    --data-root /root/dk_data_full \
    --train-cells 2048:8 2048:16 4096:8 4096:16 8192:8 8192:16 \
    --val-cells 8192:16 8192:4 --train-seed 42 --val-seed 43 \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
    --max-samples 400 --steps 4000 --out "$ROUTER" \
    > "$OUT/auc.log" 2>&1 || {
    echo "AUC FAILED (a dead linear control kills this on purpose)"
    tail -25 "$OUT/auc.log"; }
grep -E "^\[layer|control" "$OUT/auc.log" | tail -20

echo "=== LADDER DIAG 50M DONE ($(date +%T)) ==="
