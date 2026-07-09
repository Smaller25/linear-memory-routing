#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# Salience-gated retention (constant-memory) end-to-end test. Sparse needle + LONG filler (irregular
# MQAR, ctx-filler 256): does gating filler's write into the FIXED recurrent state keep needles
# recall-able without any cache? Three conditions, GDN2 backbone (d256/4L, ~6M):
#   none      = vanilla (baseline; state diluted by filler over length)
#   oracle    = gate 1.0 at needle positions, eps elsewhere (UPPER BOUND — is the mechanism sound?)
#   surprisal = detached-surprisal gate (the real self-supervised signal — does it approximate oracle?)
# Methodology: oracle ceiling first, then surprisal gap. Calibration on MQAR; RULER transfer is v2.
#
#   bash scripts/sh_salience.sh
# ---------------------------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

COMMON=(--model gdn2 --ctx-filler 256 --d-model 256 --n-layers 4 --num-heads 4 \
        --train-kv 4 8 16 32 64 --eval-kv 16 64 128 256 --steps 8000 --batch 48)

for s in none oracle surprisal; do
  jid=$(sbatch --job-name="salience-$s" --parsable scripts/sh_slurm_run.sh \
          python -m lmr.mosc.train_mqar --salience "$s" "${COMMON[@]}")
  echo "submitted salience-$s -> job $jid"
done
echo "watch: squeue -u \$USER ; logs logs/sh_salience-<s>_<jid>.out"
