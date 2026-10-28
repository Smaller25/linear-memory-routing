#!/usr/bin/env bash
# RULER eval-only (reuse the already-trained SSC head): prepare data -> free-gen predict -> metric.
# (sbatch scripts/sh_slurm_run.sh bash scripts/sh_ruler_eval.sh) — needs ckpt/ssc_370m.pt + nltk.
set -euo pipefail
export FLA_CONV_BACKEND=triton

MODEL="state-spaces/mamba2-370m"
CKPT="ckpt/ssc_370m.pt"
TASKS="niah_single_1 niah_multikey_2"
LENS="4096 8192"

echo "===== [1/3] prepare RULER data (vendored NVIDIA generators; needs nltk punkt) ====="
python scripts/ruler.py prepare --lengths 4096 --tasks niah_single_1,niah_multikey_2 --num-samples 50
python scripts/ruler.py prepare --lengths 8192 --tasks niah_single_1,niah_multikey_2 --num-samples 50

echo "===== [2/3] free-gen predict: vanilla baseline ====="
python -m lmr.scripts.predict_ruler --arch mamba2 --model "$MODEL" --variant vanilla \
  --dtype bfloat16 --lengths $LENS --tasks $TASKS

echo "===== [3/3] free-gen predict: +SSC (read-out in the decode loop) ====="
python -m lmr.scripts.predict_ruler --arch mamba2 --model "$MODEL" --variant ssc --heads "$CKPT" \
  --topk 4 --low-rank-dim 64 --chunk-size 256 --dtype bfloat16 --lengths $LENS --tasks $TASKS

echo "===== DONE — scores printed inline above (official string-match metric) ====="
