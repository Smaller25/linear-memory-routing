#!/usr/bin/env bash
# The token-matched control the baseline table was missing.
#
# The first run compared MC-SSC at 30B tokens against a vanilla checkpoint at
# 5B — six times less training — and the vanilla arm scored 1.7. That gap is
# not evidence about caching, and the earlier "caching is worse than a single
# state" claim rested on an 18.7 from a different benchmark and a different
# checkpoint entirely. This arm is the same backbone, the same 30B tokens and
# the same protocol as `base`, so the comparison is finally one variable.
set -uo pipefail
cd /root/work/lmr
export PYTHONPATH=/root/work/lmr:/root/work/lmr/dsc:/root/work/lmr/src
export MC_KERNEL_VERSION=v2 TOKENIZERS_PARALLELISM=false
export HF_HOME=/root/cache/hf TRITON_CACHE_DIR=/root/cache/triton
PY=/opt/conda/bin/python
VAN30=/root/smaller/mc/ckpts/LLM-OS-Models2_gdn2-370m-fineweb-edu-30b-paper-matched/model.pth
DATA=/root/smaller/mc/data/dk
OUT=/root/smaller/mc/out/protocol

T=$SECONDS
echo "=== arm vanilla30b ($(date +%T)) ==="
$PY -u dsc/scripts/diverse_key_niah_eval.py \
    --backend lit_gpt --ckpt "$VAN30" --config-name gdn2_370M \
    --model-label vanilla30b --gate-label vanilla30b \
    --tokenizer TinyLlama/TinyLlama_v1.1 --data-root "$DATA" \
    --lengths 8192 --needles 4 16 --seeds 42 43 44 \
    --max-examples 50 --batch-size 4 --n-gen 48 --log-routing \
    --out-dir "$OUT" --summary-name vanilla30b.json \
    > "$OUT/vanilla30b.log" 2>&1
echo "vanilla30b rc=$? wall=$((SECONDS-T))s"
grep -E "^\[cell\]" "$OUT/vanilla30b.log" || tail -15 "$OUT/vanilla30b.log"

echo "=== paired: token-matched vanilla vs MC ==="
$PY dsc/scripts/compare_arms.py --a "$OUT/vanilla30b" --b "$OUT/base" \
    --label-a "vanilla 30B" --label-b "MC-SSC 30B" --ceiling "$OUT/oracle" \
    --out "$OUT/cmp_vanilla30b_vs_base.json" 2>&1 | sed -n "2,12p"
$PY dsc/scripts/compare_arms.py --a "$OUT/vanilla30b" --b "$OUT/maxsim" \
    --label-a "vanilla 30B" --label-b "MC + fixes" --ceiling "$OUT/oracle" \
    --out "$OUT/cmp_vanilla30b_vs_maxsim.json" 2>&1 | sed -n "2,12p"
echo "=== VANILLA30B DONE ($(date +%T)) ==="
