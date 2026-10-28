#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------
# Axis-2 (hierarchical re-compression) — the recall-vs-memory experiment that attacks 0011 (flat cache
# not constant-memory, degrades ∝ B/N). At a fixed budget B, does compressing distant segments (hier)
# beat dropping them (capped)? full = the O(N) ceiling. Irregular MQAR, same recipe as 0015.
# Two boundary regimes:
#   ORACLE  — clean per-fact cuts, isolates the cache-policy question.
#   DENSITY — the realistic axis-1 boundaries (surprisal signal): does the hier win survive when the
#             cuts are unsupervised? (Composes axis-1 + axis-2.)
# Eval kv 64/128/256 => N up to 256 segments, so budgets 32/64 genuinely bind for kv>=128.
#
#   bash scripts/sh_axis2.sh        # submits all ten jobs (5 oracle x cache-policy, 5 density)
# ---------------------------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

COMMON=(--model mosc --ctx-filler 128 --d-model 256 --n-layers 4 --num-heads 4 \
        --train-kv 4 8 16 32 64 --eval-kv 64 128 256 --steps 8000 --batch 48)
ORACLE=(--chunk-mode oracle)
DENSITY=(--chunk-mode density --density-signal surprisal --target-rate 0.15 --budget 1.0)

# cache-policy suffix -> flags
declare -A POLICY=(
  [full]="--cache-mode full"
  [capped-32]="--cache-mode capped --cache-budget 32"
  [capped-64]="--cache-mode capped --cache-budget 64"
  [hier-32]="--cache-mode hier --cache-budget 32"
  [hier-64]="--cache-mode hier --cache-budget 64"
)
ORDER=(full capped-32 capped-64 hier-32 hier-64)

submit() {  # name, boundary-flags..., policy-flags...
  local name=$1; shift
  # shellcheck disable=SC2086
  jid=$(sbatch --job-name="$name" --parsable scripts/sh_slurm_run.sh \
          python -m lmr.mosc.train_mqar "$@" "${COMMON[@]}")
  echo "submitted $name -> job $jid"
}

for pol in "${ORDER[@]}"; do
  # shellcheck disable=SC2086
  submit "axis2-oracle-$pol"  "${ORACLE[@]}"  ${POLICY[$pol]}
done
for pol in "${ORDER[@]}"; do
  # shellcheck disable=SC2086
  submit "axis2-density-$pol" "${DENSITY[@]}" ${POLICY[$pol]}
done
