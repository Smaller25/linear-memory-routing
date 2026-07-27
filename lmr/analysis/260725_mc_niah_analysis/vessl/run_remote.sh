#!/usr/bin/env bash
# 0025 VESSL job launcher — runs LOCALLY.
#
# Usage:
#   run_remote.sh <script.py> [args...]   # launch a job on VESSL (nohup, detached)
#   run_remote.sh --sync-results          # rsync remote results/ -> local mirror
#   run_remote.sh --tail <logfile>        # tail -f a remote log (name or abs path)
#   run_remote.sh --poll <marker-or-log> [pattern]
#                                          # block until a remote job finishes:
#                                          #   *.log path  -> poll every ~120s, done when
#                                          #                  `pattern` (default: "ALL DONE")
#                                          #                  appears in the log's tail (not
#                                          #                  strictly the last line — trailing
#                                          #                  framework warnings can follow it)
#                                          #   other path  -> poll every ~120s, done when
#                                          #                  the remote file/marker exists
#                                          # prints a status/tail line on every check;
#                                          # transient ssh failures are retried, not fatal.
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

POLL_INTERVAL="${POLL_INTERVAL:-120}"   # seconds between checks
POLL_MAX_WAIT="${POLL_MAX_WAIT:-14400}"  # give up after this many seconds (default 4h)

usage() {
  cat <<'USAGE'
Usage:
  run_remote.sh <script.py> [args...]   # launch a job on VESSL (nohup, background)
  run_remote.sh --sync-results          # rsync /root/smaller/mc_niah/results/ -> local
  run_remote.sh --tail <logfile>        # tail -f a remote log (name under logs/, or absolute path)
  run_remote.sh --poll <marker-or-log> [pattern]
                                         # block until the remote job is done (see header)
USAGE
}

# Resolve a bare logs/ filename to the absolute remote path, same rule --tail uses.
_resolve_remote_path() {
  case "$1" in
    /*) echo "$1" ;;
    *) echo "$REMOTE_ROOT/logs/$1" ;;
  esac
}

# ssh wrapper that never lets a transient connection failure kill the poll loop.
# Prints a warning and returns 1 on failure instead of `set -e`-aborting.
_ssh_soft() {
  if ! ssh "${SSH_OPTS[@]}" -o ConnectTimeout=15 "$SSH_HOST" "$@" 2>&1; then
    echo "[run_remote][poll][warn] ssh check failed (transient?) - will retry" >&2
    return 1
  fi
}

do_poll() {
  local target="${1:?usage: run_remote.sh --poll <marker-or-log> [pattern]}"
  local pattern="${2:-ALL DONE}"
  local remote_path
  remote_path=$(_resolve_remote_path "$target")

  local is_log=0
  case "$remote_path" in
    *.log) is_log=1 ;;
  esac

  echo "[run_remote][poll] target=$remote_path mode=$([ "$is_log" -eq 1 ] && echo log/pattern || echo marker/exists) interval=${POLL_INTERVAL}s max_wait=${POLL_MAX_WAIT}s"

  local waited=0
  while true; do
    if [ "$is_log" -eq 1 ]; then
      # Log mode: print the tail every check; done when the DONE pattern
      # appears in the tail window. NOT restricted to the literal last
      # line: frameworks (e.g. PyTorch's CUDA-alloc-config deprecation
      # notice) can print a benign warning to stderr *after* the script's
      # own completion line, which would make a strict last-line match
      # never fire.
      out=$(_ssh_soft "test -f '$remote_path' && tail -n 10 '$remote_path' || echo '(log not created yet)'") || out=""
      if [ -n "$out" ]; then
        echo "--- $(date -Is) tail($remote_path) ---"
        echo "$out"
        if echo "$out" | grep -qE -- "$pattern"; then
          echo "[run_remote][poll] DONE - pattern '$pattern' found in tail"
          return 0
        fi
      fi
    else
      # Marker mode: done when the file exists on the remote.
      out=$(_ssh_soft "test -e '$remote_path' && echo EXISTS || echo MISSING") || out=""
      echo "$(date -Is) marker($remote_path): ${out:-<ssh failed>}"
      if [ "$out" = "EXISTS" ]; then
        echo "[run_remote][poll] DONE - marker exists"
        return 0
      fi
    fi

    if [ "$waited" -ge "$POLL_MAX_WAIT" ]; then
      echo "[run_remote][poll][err] gave up after ${waited}s (POLL_MAX_WAIT=${POLL_MAX_WAIT}s)" >&2
      return 1
    fi
    sleep "$POLL_INTERVAL"
    waited=$((waited + POLL_INTERVAL))
  done
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
  --poll)
    do_poll "${2:-}" "${3:-}"
    exit $?
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
