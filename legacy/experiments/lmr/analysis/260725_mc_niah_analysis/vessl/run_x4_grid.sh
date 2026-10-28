#!/usr/bin/env bash
# Runs all 4 (model x task) X4 grid chunks SEQUENTIALLY on the single VESSL
# GPU (avoids contention/-perf risk of running them concurrently; the
# per-length timing-probe budget in x4_run.py assumes exclusive GPU use).
# Cell-level skip-existing resumability (x4_raw.json) means re-running this
# script after an interruption (container restart etc.) picks up where it
# left off -- safe to just re-invoke.
#
# 2026-07-28: chunks are now RETRIED (X4_MAX_ATTEMPTS, default 3). Observed
# once: a chunk died with rc=137 (SIGKILL) mid-length with no OOM traceback
# and no memory pressure attributable to this job (RSS 1.7GB, our GPU share
# 2.7GB) -- i.e. an external kill, not a leak. Because each cell is saved to
# x4_raw.json immediately, a retry re-enters at the first missing cell and
# costs nothing for the work already done. A chunk that keeps dying makes no
# progress across attempts, which the log makes obvious (all-skip passes).
set -uo pipefail   # NOT -e: one (model,task) job failing should not skip the rest

cd "$(dirname "$0")/.."   # lmr/analysis/260725_mc_niah_analysis
source vessl/env_vessl.sh

MAX_ATTEMPTS=${X4_MAX_ATTEMPTS:-3}

for MT in "mc-5B niah_single_1" "mc-5B niah_multikey_1" "mc-30B niah_single_1" "mc-30B niah_multikey_1"; do
  set -- $MT
  MODEL=$1; TASK=$2
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "[x4-grid] ===== starting $MODEL / $TASK (attempt $attempt/$MAX_ATTEMPTS, $(date -Is)) ====="
    $PY x4_run.py --model "$MODEL" --task "$TASK"
    rc=$?
    echo "[x4-grid] ===== finished $MODEL / $TASK rc=$rc (attempt $attempt, $(date -Is)) ====="
    [ "$rc" -eq 0 ] && break
    echo "[x4-grid] rc=$rc -- retrying (cell-level skip-existing means completed cells are kept)"
    sleep 30
  done
done

$PY x4_table.py
echo "ALL DONE"
