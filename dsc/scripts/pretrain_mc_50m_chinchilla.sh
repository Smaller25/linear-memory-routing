#!/usr/bin/env bash
# MC SSC + GDN-2 50M on FineWeb-Edu, Chinchilla-optimal (1.53B tokens).
#
# First rung of a scale ladder under the 370M anchor. The point is not the
# score -- an 8K diverse-key NIAH score will be zero at this size, and the
# 370M itself only manages 2-4 untouched. The point is whether the failure
# has the same shape: routing signal concentrated in the first layers,
# routing hit near chance, and an oracle far above the actual score. If the
# shape matches, small models are a valid proxy and the two interventions
# can be trained in rather than bolted on, which is the one thing the frozen
# backbone cannot test. If it does not match, that is worth knowing for the
# cost of one short run.
#
# Everything except size and token count is the anchor's recipe: global
# batch 128 seqs x 4096 = 0.5M tokens, LR 4e-4, warmup 1%, same tokenizer,
# same corpus. A recipe change would become a competing explanation for any
# mismatch in shape.
#
# Train and val shards are disjoint (000-005 against 013). The anchor's
# launcher pointed val at a subdirectory of train, which makes the reported
# ppl a training-set number, and ppl is the primary metric here.
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/root/work/lmr/dsc}"
cd "${ROOT_DIR}"

export PYTHONPATH="/root/work/lmr:${ROOT_DIR}:/root/work/lmr/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/root/cache/hf}"
# wandb. pretrain.py already builds a WandbLogger with project
# "llm_next_gen", run name = exp_name and group = exp_group, and honours
# WANDB_MODE, so nothing there needs changing -- only the key.
#
# The key is read from a file rather than the command line so it never
# reaches a process listing, a shell history, or this log. A missing key
# degrades to an offline run instead of killing a 24-hour job.
if [[ -z "${WANDB_API_KEY:-}" && -r "${WANDB_KEY_FILE:-/root/.wandb_key}" ]]; then
    WANDB_API_KEY="$(tr -d "[:space:]" < "${WANDB_KEY_FILE:-/root/.wandb_key}")"
    export WANDB_API_KEY
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE="${WANDB_MODE:-online}"
else
    export WANDB_MODE="${WANDB_MODE:-disabled}"
fi
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/cache/triton}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MC_KERNEL_VERSION="${MC_KERNEL_VERSION:-v2}"
export PYTHONUNBUFFERED=1
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}"

# The pod's system python has no torch. Always the venv, never bare python.
PY="${PY:-/opt/conda/bin/python}"

MODEL="${MODEL:-mc_50M}"
CKPT_MODE="${CKPT_MODE:-independent}"   # or "chained"
EXP_NAME="${EXP_NAME:-mc_50m_fineweb_edu_chinchilla}"
EXP_GROUP="${EXP_GROUP:-scale_ladder}"
TRAIN_CONFIG="${TRAIN_CONFIG:-tsz128x4k_chinchilla}"   # batch shape only; budget is explicit
MAX_TOKENS="${MAX_TOKENS:-1530000000}"                 # 20 tok/param on 76.7M total

TRAIN_DATA="${TRAIN_DATA:-/root/smaller/mc/fwedu/train}"          # glob {dir}/*/*.parquet
VAL_DATA="${VAL_DATA:-/root/smaller/mc/fwedu/val}"                # glob {dir}/*.parquet
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/smaller/mc/ladder}"

LR="${LR:-4e-4}"
# The GPU is shared. micro 2 keeps peak allocation near 12 GB instead of
# the 19.6 GB micro 4 measured, so a co-tenant that grows does not take
# this run down with it. Global batch is unchanged: grad accum absorbs it.
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
SAVE_STEP_INTERVAL="${SAVE_STEP_INTERVAL:-500}"
EVAL_STEP_INTERVAL="${EVAL_STEP_INTERVAL:-200}"
EVAL_ITERS="${EVAL_ITERS:-15}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"

export CUDA_VISIBLE_DEVICES="${DEVICES:-0}"

echo "===== MC-GDN2 ${MODEL} on FineWeb-Edu ====="
echo "  budget: ${MAX_TOKENS} tokens   global batch: 128 seqs x 4096 = 524288"
echo "  steps:  $((MAX_TOKENS / 524288))   micro batch: ${MICRO_BATCH_SIZE}"
echo "  python: ${PY}"
echo "  wandb:  ${WANDB_MODE} (project llm_next_gen, run ${EXP_NAME}, group ${EXP_GROUP})"
echo "  train:  ${TRAIN_DATA}    val: ${VAL_DATA} (disjoint shards)"
echo "  out:    ${OUTPUT_ROOT}/outputs/${TRAIN_CONFIG}_${EXP_NAME}"
echo "==========================================="

exec "${PY}" -u "${ROOT_DIR}/pretrain.py" \
    --output_root "${OUTPUT_ROOT}" \
    --train_data_dir "${TRAIN_DATA}" \
    --train_data_dir_raw "${TRAIN_DATA}" \
    --val_data_dir_raw "${VAL_DATA}" \
    --model_name "${MODEL}" \
    --exp_name "${EXP_NAME}" \
    --exp_group "${EXP_GROUP}" \
    --train_config "${TRAIN_CONFIG}" \
    --config_overrides "mc_checkpoint_mode=${CKPT_MODE}" \
    --max_tokens "${MAX_TOKENS}" \
    --global_batch_size 128 \
    --corpus_name fineweb-edu \
    --use_stream_tok \
    --tokenizer_name TinyLlama/TinyLlama_v1.1 \
    --tokenizer_path TinyLlama/TinyLlama_v1.1 \
    --val_type val_sampled \
    --learning_rate "${LR}" \
    --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --eval_iters "${EVAL_ITERS}" \
    --save_step_interval "${SAVE_STEP_INTERVAL}" \
    --eval_step_interval "${EVAL_STEP_INTERVAL}" \
    --train_num_workers "${TRAIN_NUM_WORKERS}" \
    --actual_train_time 0 \
    --no-hf_upload
