#!/usr/bin/env bash
#SBATCH -J mc_smoke
#SBATCH -p main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -t 00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -o /data2/sohyung/mc_niah/logs/smoke_%j.log
#SBATCH -e /data2/sohyung/mc_niah/logs/smoke_%j.log
set -eo pipefail
source /home/sohyung/linear-memory-routing/lmr/analysis/260725_mc_niah_analysis/env_common.sh
$PY "$ANA/smoke.py"
