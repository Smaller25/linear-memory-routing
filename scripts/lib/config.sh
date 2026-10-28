# shellcheck shell=bash
# Config loader shared by every script. Mirrors mslm/envfile.py:resolve()
# (tests/test_config_parity.py checks they agree).
#
#   source scripts/lib/config.sh
#   mslm_load_config "$CONFIG"      # exports every variable declared in configs/base.env
#
# Priority per variable: value already in the environment > CONFIG file > base.env.
# Variables are evaluated in base.env order, so a CONFIG that changes SCRATCH_ROOT
# also moves DATA_ROOT/RUNS_ROOT/... that base.env derives from it.

MSLM_REPO="${MSLM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export MSLM_REPO

mslm_load_config() {
  local cfg="${1:-}"
  local base="$MSLM_REPO/configs/base.env"
  [[ -n "$cfg" && ! -f "$cfg" && -f "$MSLM_REPO/$cfg" ]] && cfg="$MSLM_REPO/$cfg"
  if [[ -n "$cfg" && ! -f "$cfg" ]]; then
    echo "[config] not found: $cfg" >&2; return 1
  fi
  local keys k
  keys=$(grep -oE '^(export[[:space:]]+)?[A-Z_][A-Z0-9_]*=' "$base" | sed -E 's/^export[[:space:]]+//; s/=$//')
  # 1) remember overrides (variables already set by the caller)
  MSLM_OVERRIDES=""
  for k in $keys; do
    if [[ -n "${!k+x}" ]]; then MSLM_OVERRIDES+="$k "; fi
  done
  export MSLM_OVERRIDES
  # 2) build the merged script: base.env order, config line replaces base line,
  #    overridden keys are dropped
  local merged
  merged=$(awk -v ov=" $MSLM_OVERRIDES" '
    function keyof(line,   s) { s=line; sub(/^[ \t]*(export[ \t]+)?/, "", s); if (s ~ /^[A-Z_][A-Z0-9_]*=/) { sub(/=.*/, "", s); return s } return "" }
    FNR==NR { k=keyof($0); if (k!="") cfg[k]=$0; next }
    { k=keyof($0); if (k=="") next
      if (index(ov, " " k " ")) next
      if (k in cfg) print cfg[k]; else print $0 }
  ' "${cfg:-/dev/null}" "$base")
  if [[ -n "$cfg" ]]; then
    local undeclared
    undeclared=$(grep -oE '^(export[[:space:]]+)?[A-Z_][A-Z0-9_]*=' "$cfg" | sed -E 's/^export[[:space:]]+//; s/=$//' | while read -r k; do grep -qE "^(export[[:space:]]+)?$k=" "$base" || echo "$k"; done)
    if [[ -n "$undeclared" ]]; then
      echo "[config] $cfg sets variables not declared in configs/base.env: $undeclared" >&2; return 1
    fi
  fi
  set -a
  eval "$merged"
  set +a
  export CONFIG="${cfg}"
}

# Print the resolved config as KEY=value lines (base.env order).
mslm_print_config() {
  local k
  for k in $(grep -oE '^(export[[:space:]]+)?[A-Z_][A-Z0-9_]*=' "$MSLM_REPO/configs/base.env" | sed -E 's/^export[[:space:]]+//; s/=$//'); do
    printf '%s=%s\n' "$k" "${!k}"
  done
}

# Activate modules and the conda env named by MODULES / CONDA_ENV.
mslm_activate_env() {
  if [[ -n "${MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    module load $MODULES
  fi
  if [[ -n "${CONDA_ENV:-}" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    if command -v conda >/dev/null 2>&1; then
      set +u
      eval "$(conda shell.bash hook)"
      conda activate "$CONDA_ENV"
      set -u
    else
      echo "[env] conda not found; set MODULES (e.g. miniforge) or CONDA_ENV=\"\"" >&2; return 1
    fi
  fi
  export HF_HOME="$HF_CACHE_DIR"
  export PYTHONPATH="$MSLM_REPO${PYTHONPATH:+:$PYTHONPATH}"
  export FLA_CONV_BACKEND=triton      # never needs causal-conv1d
  export TOKENIZERS_PARALLELISM=false
}

# Compute nodes are assumed to have no internet: read only from HF_CACHE_DIR.
mslm_offline() {
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
}

# Quote a value for the shell the same way mslm/tower.py:shell_quote does.
mslm_quote() {
  if [[ "$1" =~ ^[A-Za-z0-9_./:=,@+%-]+$ ]]; then printf '%s' "$1"
  else printf "'%s'" "${1//\'/\'\\\'\'}"; fi
}

# The full sbatch command for JOB (train | eval | prepare_data | prepare_eval),
# built only from the resolved config. Printed, never submitted here.
mslm_sbatch_cmd() {
  local job="$1" k cfg_abs env_part="" opts
  cfg_abs="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
  for k in $MSLM_OVERRIDES; do env_part+=" $k=$(mslm_quote "${!k}")"; done
  opts="--job-name=$(mslm_quote "${SB_JOB_NAME}-${job}") --partition=$(mslm_quote "$SB_PARTITION")"
  opts+=" --nodes=$SB_NODES --ntasks-per-node=1 $SB_GPU_OPT --cpus-per-task=$SB_CPUS_PER_TASK"
  opts+=" --time=$SB_TIME"
  [[ -n "$SB_COMMENT" ]] && opts+=" --comment=$(mslm_quote "$SB_COMMENT")"
  [[ "$SB_EXCLUSIVE" == "1" ]] && opts+=" --exclusive"
  [[ -n "$SB_EXTRA" ]] && opts+=" $SB_EXTRA"
  opts+=" --output=$(mslm_quote "$SB_LOG_DIR/%x_%j.out") --error=$(mslm_quote "$SB_LOG_DIR/%x_%j.err")"
  printf 'mkdir -p %s %s && cd %s && MSLM_REPO=%s CONFIG=%s%s sbatch %s %s\n' \
    "$(mslm_quote "$SB_SUBMIT_DIR")" "$(mslm_quote "$SB_LOG_DIR")" "$(mslm_quote "$SB_SUBMIT_DIR")" \
    "$(mslm_quote "$MSLM_REPO")" "$(mslm_quote "$cfg_abs")" "$env_part" "$opts" \
    "$(mslm_quote "$MSLM_REPO/slurm/$job.sbatch")"
}
