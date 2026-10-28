#!/usr/bin/env bash
#SBATCH -J mc_e3
#SBATCH -p main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -t 02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -o /data2/sohyung/mc_niah/logs/e3_%x_%j.log
#SBATCH -e /data2/sohyung/mc_niah/logs/e3_%x_%j.log
set -eo pipefail
source /home/sohyung/linear-memory-routing/lmr/analysis/260725_mc_niah_analysis/env_common.sh
# MODEL is exported by the submitting command, e.g.:
#   sbatch --job-name=mc_e3_5B  --export=ALL,MODEL=mc-5B  sbatch/e3.sbatch
#   sbatch --job-name=mc_e3_30B --export=ALL,MODEL=mc-30B sbatch/e3.sbatch
# Runs one model's per-layer write/read fidelity profiling over S+D
# (single+multi, all 32 rows each) -- 1 forward + 16 layers x 2 scans per row.
: "${MODEL:?set MODEL=mc-5B or mc-30B via --export}"
$PY "$ANA/fidelity.py" --model "$MODEL"
