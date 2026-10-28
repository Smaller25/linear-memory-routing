#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# Validate the supervised Dynamic-MoSC method on Selective Copying (the standard content-vs-position
# task) — the external analog of the 0015 irregular-MQAR adaptivity probe. Data tokens sit at random
# positions among noise (scatter-mult*M), reproduced in order; a fixed stride cannot align to them,
# so only content-adaptive boundaries can. Four conditions mirror the 0015 table:
#   vanilla (gdn2) | mosc fixed chunk=2 | mosc oracle | mosc learned (distilled from oracle)
# Same backbone/recipe as 0015 (d256/4L/4heads, train-kv 4..64, 8000 steps, batch 48, lr 3e-3).
#
#   bash scripts/sh_selcopy.sh        # submits all four Slurm jobs
# ---------------------------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

COMMON=(--task selcopy --vocab 256 --scatter-mult 8 \
        --d-model 256 --n-layers 4 --num-heads 4 \
        --train-kv 4 8 16 32 64 --eval-kv 16 32 64 128 \
        --steps 8000 --batch 48)

declare -A JOBS=(
  [selcopy-vanilla]="--model gdn2"
  [selcopy-fixed]="--model mosc --chunk-mode fixed --chunk 2"
  [selcopy-oracle]="--model mosc --chunk-mode oracle"
  [selcopy-learned]="--model mosc --chunk-mode learned --boundary-distill 1.0 \
                     --dump-boundaries ckpt/selcopy_learned_seglens.npz"
)

for name in selcopy-vanilla selcopy-fixed selcopy-oracle selcopy-learned; do
  # shellcheck disable=SC2086
  jid=$(sbatch --job-name="$name" --parsable scripts/sh_slurm_run.sh \
          python -m lmr.mosc.train_mqar ${JOBS[$name]} "${COMMON[@]}")
  echo "submitted $name -> job $jid"
done
echo "watch: squeue -u \$USER ; logs in logs/sh_<name>_<jid>.out"
