#!/usr/bin/env bash
# Runs all 4 (model x task) X4 grid chunks SEQUENTIALLY on the single VESSL
# GPU (avoids contention/-perf risk of running them concurrently; the
# per-length timing-probe budget in x4_run.py assumes exclusive GPU use).
# Cell-level skip-existing resumability (x4_raw.json) means re-running this
# script after an interruption (container restart etc.) picks up where it
# left off -- safe to just re-invoke.
set -uo pipefail   # NOT -e: one (model,task) job failing should not skip the rest

cd "$(dirname "$0")/.."   # lmr/analysis/260725_mc_niah_analysis
source vessl/env_vessl.sh

for MT in "mc-5B niah_single_1" "mc-5B niah_multikey_1" "mc-30B niah_single_1" "mc-30B niah_multikey_1"; do
  set -- $MT
  MODEL=$1; TASK=$2
  echo "[x4-grid] ===== starting $MODEL / $TASK ($(date -Is)) ====="
  $PY x4_run.py --model "$MODEL" --task "$TASK"
  rc=$?
  echo "[x4-grid] ===== finished $MODEL / $TASK rc=$rc ($(date -Is)) ====="
done

$PY x4_table.py
echo "ALL DONE"
