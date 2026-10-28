#!/usr/bin/env bash
# RULER pipeline (run via: sbatch scripts/sh_slurm_run.sh bash scripts/sh_ruler_pipeline.sh)
# frozen mamba2-370m + trained SSC router; passkey-trained -> RULER zero-shot, free-gen + official metric.
set -euo pipefail
export FLA_CONV_BACKEND=triton   # mamba2 on Blackwell has no causal_conv1d; use the Triton conv

MODEL="state-spaces/mamba2-370m"
CKPT="ckpt/ssc_370m.pt"
TASKS="niah_single_1 niah_multikey_2"
LENS="4096 8192"

echo "===== [1/4] train SSC router on passkey (backbone frozen) ====="
# memory: the read-out stacks per-checkpoint scans, so cost ~ batch x train_len x #segments. Keep
# batch=2 / train_len=1024 (4 segments @ chunk 256) — batch 4 / len 2048 OOMs a 96GB card on 370m.
python -m lmr.scripts.train_grm_passkey --arch mamba2 --model "$MODEL" --variant ssc \
  --topk 4 --low-rank-dim 64 --aux-scale 1e-4 --train-len 1024 --batch 2 --steps 500 \
  --dtype bfloat16 --eval-lengths 512 1024 2048 --out "$CKPT"

echo "===== [2/4] prepare RULER data (vendored NVIDIA generators) ====="
python scripts/ruler.py prepare --lengths 4096 --tasks niah_single_1,niah_multikey_2 --num-samples 50
python scripts/ruler.py prepare --lengths 8192 --tasks niah_single_1,niah_multikey_2 --num-samples 50

echo "===== [3/4] free-gen predict: vanilla baseline ====="
python -m lmr.scripts.predict_ruler --arch mamba2 --model "$MODEL" --variant vanilla \
  --dtype bfloat16 --lengths $LENS --tasks $TASKS

echo "===== [4/4] free-gen predict: +SSC (read-out in the decode loop) ====="
python -m lmr.scripts.predict_ruler --arch mamba2 --model "$MODEL" --variant ssc --heads "$CKPT" \
  --topk 4 --low-rank-dim 64 --chunk-size 256 --dtype bfloat16 --lengths $LENS --tasks $TASKS

echo "===== DONE — scores printed inline above (official string-match metric) ====="
