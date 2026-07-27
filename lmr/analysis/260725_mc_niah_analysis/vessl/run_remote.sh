#!/usr/bin/env bash
# 0025 VESSL job launcher — runs LOCALLY.
#
# Usage:
#   run_remote.sh <script.py> [args...]   # launch a job on VESSL (nohup, detached)
#   run_remote.sh --sync-results          # rsync remote results/ -> local mirror
#   run_remote.sh --tail <logfile>        # tail -f a remote log (name or abs path)
set -euo pipefail

SSH_KEY=/home/sohyung/sohyung2.pem
SSH_PORT=32273
SSH_HOST=root@betelgeuse.cloud.vessl.ai
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -i "$SSH_KEY" -p "$SSH_PORT")

REMOTE_ROOT=/root/smaller/mc_niah
REMOTE_LMR="$REMOTE_ROOT/code/linear-memory-routing"
REMOTE_ENV="$REMOTE_LMR/lmr/analysis/260725_mc_niah_analysis/vessl/env_vessl.sh"
REMOTE_ANA_PREFIX="lmr/analysis/260725_mc_niah_analysis"

LOCAL_SYNC_DIR=/data2/sohyung/mc_niah/results/vessl

usage() {
  cat <<'USAGE'
Usage:
  run_remote.sh <script.py> [args...]   # launch a job on VESSL (nohup, background)
  run_remote.sh --sync-results          # rsync /root/smaller/mc_niah/results/ -> local
  run_remote.sh --tail <logfile>        # tail -f a remote log (name under logs/, or absolute path)
USAGE
}

if [ "$#" -eq 0 ]; then
  usage
  exit 1
fi

case "$1" in
  --sync-results)
    mkdir -p "$LOCAL_SYNC_DIR"
    rsync -avz -e "ssh ${SSH_OPTS[*]}" "$SSH_HOST:$REMOTE_ROOT/results/" "$LOCAL_SYNC_DIR/"
    echo "[run_remote] synced $SSH_HOST:$REMOTE_ROOT/results/ -> $LOCAL_SYNC_DIR/"
    echo "[run_remote] merging into repo results/ is manual"
    exit 0
    ;;
  --tail)
    LOG="${2:?usage: run_remote.sh --tail <logfile>}"
    case "$LOG" in
      /*) REMOTE_LOG="$LOG" ;;
      *) REMOTE_LOG="$REMOTE_ROOT/logs/$LOG" ;;
    esac
    exec ssh "${SSH_OPTS[@]}" "$SSH_HOST" "tail -f $(printf '%q' "$REMOTE_LOG")"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
esac

SCRIPT="$1"; shift
case "$SCRIPT" in
  lmr/*) REMOTE_SCRIPT_PATH="$SCRIPT" ;;
  *) REMOTE_SCRIPT_PATH="$REMOTE_ANA_PREFIX/$SCRIPT" ;;
esac

NAME=$(basename "$SCRIPT" .py)
TS=$(date +%Y%m%d_%H%M%S)
REMOTE_LOG="$REMOTE_ROOT/logs/${NAME}_${TS}.log"

QUOTED_ARGS=""
for a in "$@"; do
  QUOTED_ARGS+=" $(printf '%q' "$a")"
done

remote_script=$(cat <<REMOTE
set -e
cd '$REMOTE_LMR'
source '$REMOTE_ENV'
mkdir -p '$REMOTE_ROOT/logs'
nohup \$PY '$REMOTE_SCRIPT_PATH'$QUOTED_ARGS < /dev/null > '$REMOTE_LOG' 2>&1 &
pid=\$!
disown
echo "PID:\$pid"
echo "LOG:$REMOTE_LOG"
REMOTE
)

ssh "${SSH_OPTS[@]}" "$SSH_HOST" bash -c "$remote_script"
