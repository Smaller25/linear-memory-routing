#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# Axis-2 (hierarchical re-compression) — the recall-vs-memory experiment that attacks 0011 (flat cache
# not constant-memory, degrades ∝ B/N). Uses ORACLE boundaries so the segments are clean and the only
# variable is the cache policy: at a fixed budget B, does compressing distant segments (hier) beat
# dropping them (capped)? full = the O(N) ceiling. Irregular MQAR, same recipe as 0015 (d256/4L/4heads).
# Eval kv 64/128/256 => N up to 256 segments, so budgets 32/64 genuinely bind for kv>=128.
#
#   bash scripts/sh_axis2.sh        # submits the five cache-policy jobs
# ---------------------------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

COMMON=(--model mosc --chunk-mode oracle --ctx-filler 128 \
        --d-model 256 --n-layers 4 --num-heads 4 \
        --train-kv 4 8 16 32 64 --eval-kv 64 128 256 \
        --steps 8000 --batch 48)

declare -A JOBS=(
  [axis2-full]="--cache-mode full"
  [axis2-capped-32]="--cache-mode capped --cache-budget 32"
  [axis2-capped-64]="--cache-mode capped --cache-budget 64"
  [axis2-hier-32]="--cache-mode hier --cache-budget 32"
  [axis2-hier-64]="--cache-mode hier --cache-budget 64"
)
for name in axis2-full axis2-capped-32 axis2-capped-64 axis2-hier-32 axis2-hier-64; do
  # shellcheck disable=SC2086
  jid=$(sbatch --job-name="$name" --parsable scripts/sh_slurm_run.sh \
          python -m lmr.mosc.train_mqar ${JOBS[$name]} "${COMMON[@]}")
  echo "submitted $name -> job $jid"
done
