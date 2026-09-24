#!/usr/bin/env bash
# Rung 1 of the scale ladder: mc_50M baseline, Chinchilla-optimal 1.53B tokens.
#
# micro_batch_size 4 is forced, not chosen. The probe measured 19.6 GB peak
# allocated at 4 while the other job on this GPU held 45.9 GB of 80; at 8 the
# activations roughly double and the box OOMs. Throughput was 950 ms per
# iteration for 4x4096 under that contention, about 17k tokens/s, so 1.53B
# tokens is roughly 24 hours.
#
# pretrain.py resumes automatically when out_dir already exists, so if the
# other job grows and this one dies, relaunching the identical command picks
# up from the last save. save_step_interval 250 bounds a crash to about two
# hours of lost work.
set -uo pipefail
export MICRO_BATCH_SIZE=4
export MAX_TOKENS=1530000000
export OUTPUT_ROOT=/root/ladder
export SAVE_STEP_INTERVAL=250
export EVAL_STEP_INTERVAL=200
bash /root/pretrain_mc_50m.sh
rc=$?
echo "=== LADDER 50M DONE ($(date +%T)) rc=${rc} ==="
