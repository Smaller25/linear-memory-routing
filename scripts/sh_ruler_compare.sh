#!/usr/bin/env bash
# RULER vanilla-vs-SSC, matched samples, OFFICIAL scoring (predict_ruler's inline metric is buggy —
# use scripts/ruler.py eval). 8k floors + SSC free-gen too slow there, so 2048/4096 only.
# (sbatch scripts/sh_slurm_run.sh bash scripts/sh_ruler_compare.sh)  needs ckpt/ssc_370m.pt + nltk/wonderwords.
set -euo pipefail
export FLA_CONV_BACKEND=triton
MODEL="state-spaces/mamba2-370m"; CKPT="ckpt/ssc_370m.pt"; N=30
TASKS_C="niah_single_1,niah_multikey_2"; TASKS="niah_single_1 niah_multikey_2"

python scripts/ruler.py prepare --lengths 2048 --tasks "$TASKS_C" --num-samples "$N"
python scripts/ruler.py prepare --lengths 4096 --tasks "$TASKS_C" --num-samples "$N"

for L in 2048 4096; do
  echo "########## length $L ##########"
  echo "--- vanilla ---"
  python -m lmr.scripts.predict_ruler --arch mamba2 --model "$MODEL" --variant vanilla \
    --dtype bfloat16 --lengths "$L" --tasks $TASKS --max-examples "$N" >/dev/null
  python scripts/ruler.py eval --lengths "$L" --tasks "$TASKS_C"
  echo "--- +SSC ---"
  python -m lmr.scripts.predict_ruler --arch mamba2 --model "$MODEL" --variant ssc --heads "$CKPT" \
    --topk 4 --low-rank-dim 64 --chunk-size 256 --dtype bfloat16 --lengths "$L" --tasks $TASKS --max-examples "$N" >/dev/null
  python scripts/ruler.py eval --lengths "$L" --tasks "$TASKS_C"
done
echo "===== DONE (official string-match; vanilla then +SSC per length) ====="
