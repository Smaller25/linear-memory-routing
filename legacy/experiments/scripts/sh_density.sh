#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# Axis-1 (information-density segmentation) — the unsupervised test that attacks the 0016 failure.
# On IRREGULAR MQAR (no oracle ever), segment by the backbone's own info-density signal calibrated to
# a target firing rate, and ask the 0016 question: does it rank facts highest (top-k(p) ∩ facts)?
# Reference points already on record: supervised learned = 1.00 (0015), unsup STE+L1 = 0.00 (0016).
# Three signals; same backbone/recipe as 0015/0016 (d256/4L/4heads, ctx-filler 128, 8000 steps).
#
#   bash scripts/sh_density.sh        # submits the three density-signal jobs
# ---------------------------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

COMMON=(--model mosc --chunk-mode density --ctx-filler 128 \
        --d-model 256 --n-layers 4 --num-heads 4 \
        --train-kv 4 8 16 32 64 --eval-kv 64 128 256 \
        --steps 8000 --batch 48 --target-rate 0.15 --budget 1.0)

declare -A JOBS=(
  [density-surprisal]="--density-signal surprisal --dump-boundaries ckpt/density_surprisal_seglens.npz"
  [density-entropy]="--density-signal entropy"
  [density-cosdist]="--density-signal cosdist"
)
for name in density-surprisal density-entropy density-cosdist; do
  # shellcheck disable=SC2086
  jid=$(sbatch --job-name="$name" --parsable scripts/sh_slurm_run.sh \
          python -m lmr.mosc.train_mqar ${JOBS[$name]} "${COMMON[@]}")
  echo "submitted $name -> job $jid"
done
