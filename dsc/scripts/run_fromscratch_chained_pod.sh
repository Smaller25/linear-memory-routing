#!/usr/bin/env bash
# Does an unfragmented recurrence recover what caching costs? From scratch.
#
# The protocol table says vanilla 11.0 and MC-SSC 2.7 at 8K, and the code says
# why: every segment is scanned from initial_state=None, so a position sees
# only its own 256 tokens and the top-k read is the sole bridge. Chaining that
# state at inference told us nothing — the checkpoint had never seen a state
# cross a boundary and both arms scored 0.0. Only training answers it.
#
# Two runs, SEQUENTIAL, because the GPU is shared. The second waits on the
# first's log marker rather than pgrep, which matches its own ssh command line
# and once stalled a run for 27 minutes.
#
# Same recipe for both arms, so the only difference is the compressor mode:
# mc_50M, FineWeb-Edu, 1.53B tokens (20 per total parameter), global batch
# 128x4096, LR 4e-4, warmup 1%. micro batch 2 rather than the 4 used when this
# box was idle: peak allocation lands near 12 GB instead of 19.6 GB, leaving
# room for the co-tenant to grow without taking this down.
#
# Checkpoints go to the persistent mount and are uploaded the moment each run
# ends. The last pod took 22 GPU-hours of ladder training with it.
set -uo pipefail
export MICRO_BATCH_SIZE=2
export MAX_TOKENS=1530000000
export OUTPUT_ROOT=/root/smaller/mc/ladder
export SAVE_STEP_INTERVAL=250
export EVAL_STEP_INTERVAL=200
export WANDB_DIR=/root/cache/wandb

run_arm() {
    local MODE=$1
    echo "=== ARM ${MODE} START ($(date +%T)) ==="
    CKPT_MODE="$MODE" EXP_NAME="mc50m_${MODE}" \
        bash /root/work/lmr/dsc/scripts/pretrain_mc_50m_chinchilla.sh
    local rc=$?
    echo "=== ARM ${MODE} DONE ($(date +%T)) rc=${rc} ==="
    local D="${OUTPUT_ROOT}/outputs/tsz128x4k_chinchilla_mc50m_${MODE}"
    ls -la "$D"/*.pth 2>/dev/null | awk '{print "  ", $5, $9}'
    return $rc
}

run_arm independent
run_arm chained
echo "=== FROMSCRATCH BOTH DONE ($(date +%T)) ==="
