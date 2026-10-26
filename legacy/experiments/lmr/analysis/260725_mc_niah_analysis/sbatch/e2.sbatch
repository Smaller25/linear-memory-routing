#!/usr/bin/env bash
#SBATCH -J mc_e2
#SBATCH -p main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -t 05:50:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -o /data2/sohyung/mc_niah/logs/e2_%x_%j.log
#SBATCH -e /data2/sohyung/mc_niah/logs/e2_%x_%j.log
set -eo pipefail
source /home/sohyung/linear-memory-routing/lmr/analysis/260725_mc_niah_analysis/env_common.sh
# MODEL is exported by the submitting command, e.g.:
#   sbatch --job-name=mc_e2_5B  --export=ALL,MODEL=mc-5B  sbatch/e2.sbatch
#   sbatch --job-name=mc_e2_30B --export=ALL,MODEL=mc-30B sbatch/e2.sbatch
# Runs one model's S+D oracle-vs-baseline generation (~2-3h @ n_gen=128).
: "${MODEL:?set MODEL=mc-5B or mc-30B via --export}"
: "${N_GEN:=128}"
$PY "$ANA/oracle.py" --model "$MODEL" --n-gen "$N_GEN"
