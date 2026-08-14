#!/usr/bin/env bash
# Qwen2.5-Math-7B GRPO baseline / Table-2 recovery runner.
#
# Normal mode fills the 7B GRPO control row. Set RECOVERY_MODE=true to use the
# paper's deterministic Recovery@50/@100 validation protocol in the same arm.
# This remains a pure GRPO baseline: TAFR/FEPO and failure-SFT are disabled.
#
# Examples:
#   GPUS=0,1 bash experiments/fepo_paper/s03_grpo_7b.sh
#   RECOVERY_MODE=true TRAIN_SEED=3407 GPUS=0,1 \
#       bash experiments/fepo_paper/s03_grpo_7b.sh
#   DRY_RUN=1 GPUS=0,1 bash experiments/fepo_paper/s03_grpo_7b.sh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

# Capture caller overrides before _common.sh installs its shared defaults.
_requested_model=${MODEL_PATH:-}
_requested_seed=${TRAIN_SEED:-}
_requested_project=${PROJECT_NAME:-}
_requested_gpus=${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}
_requested_tag=${TAG:-}
_requested_run_name=${RUN_NAME:-}
_requested_rollout_mem=${ROLLOUT_GPU_MEM_UTIL:-}
_requested_mini_batch=${PPO_MINI_BATCH_SIZE:-}
_requested_train_batch=${TRAIN_PROMPT_BATCH_SIZE:-}
_requested_actor_attn=${ACTOR_ATTENTION_IMPL:-}
_requested_vllm_attn=${VLLM_ATTENTION_BACKEND:-}
_requested_recovery_mode=${RECOVERY_MODE:-true}
source "${SCRIPT_DIR}/_common.sh"

# Resolve the exact local Qwen snapshot; never silently substitute another model.
resolve_model() {
  local candidates=(
    "${_requested_model:-}"
    "${QWEN_MODEL_PATH:-}"
    "/workspace/models/Qwen2.5-Math-7B-Instruct"
    "/workspace/.hf_home/hub/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots"/*
    "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots"/*
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    [[ -f "${candidate}/config.json" ]] && { printf '%s' "${candidate}"; return 0; }
  done
  echo "Qwen2.5-Math-7B snapshot not found. Set MODEL_PATH or QWEN_MODEL_PATH." >&2
  return 4
}

export MODEL_PATH="$(resolve_model)"
export PROJECT_NAME="${_requested_project:-tafr-repro-qwen-math7b}"
export NDEVICES_PER_NODE=2
export CUDA_VISIBLE_DEVICES="${_requested_gpus}"
export TRAIN_SEED="${_requested_seed:-3407}"
export TAG="${_requested_tag:-qwen7b-s3407-20260813}"
export RUN_NAME="${_requested_run_name:-s03_grpo_7b}"

# Qwen2.5-Math-7B uses the configured context window. Keep prompt and answer
# lengths within context and use LoRA because full Adam/FSDP plus vLLM is too large
# for the available 2x96-GiB GPUs.
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
export DRGRPO_USE_LORA=true
export LORA_RANK=${LORA_RANK:-64}
export LORA_ALPHA=${LORA_ALPHA:-128}
export PPO_MINI_BATCH_SIZE=${_requested_mini_batch:-16}
export TRAIN_PROMPT_BATCH_SIZE=${_requested_train_batch:-32}
export NUM_GENERATIONS=${NUM_GENERATIONS:-8}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-500}

# Attention/runtime settings: actor uses FlashAttention 2; vLLM uses FlashInfer
# and paged KV-cache management. Limit concurrency for 7B MHA memory usage.
export ACTOR_ATTENTION_IMPL=${_requested_actor_attn:-flash_attention_2}
export VLLM_ATTENTION_BACKEND=${_requested_vllm_attn:-FLASHINFER}
export ROLLOUT_GPU_MEM_UTIL=${_requested_rollout_mem:-0.60}
# Architectural rollout scaling: allow more concurrent requests and token prefill
# without changing response length or generation count.
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}
export ROLLOUT_CALCULATE_LOG_PROBS=true

# This is a real GRPO control, not a diagnostic FEPO run.
export TAFR_ENABLE=false

# Optional exact Table-2 protocol. The main mode retains _common.sh's benchmark
# validation; recovery mode uses greedy n=1 at t0, t0+50, t0+100.
export RECOVERY_MODE="${_requested_recovery_mode}"
if [[ "${RECOVERY_MODE}" == "true" ]]; then
  export VAL_FILES="[${EVAL_DATA_DIR}/amc23.parquet,${EVAL_DATA_DIR}/minervamath.parquet,${EVAL_DATA_DIR}/olympiadbench.parquet]"
  export BEST_CKPT_SOURCES='["amc23","minervamath","olympiadbench"]'
  export BEST_CKPT_METRICS='["pass@1"]'
  export VAL_ROLLOUT_N=1
  export VAL_DO_SAMPLE=False
  export VAL_TEMPERATURE=0
  export VAL_TOP_P=0.95
  export VAL_BEFORE_TRAIN=true
  export VALIDATION_SEEDS='[3407]'
  export TEST_FREQ=10
  export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
  export RUN_NAME="${_requested_run_name:-s03_grpo_7b_recovery}"
fi

# Fail before allocating Ray/GPU resources on an invalid host.
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "Missing model config: ${MODEL_PATH}" >&2; exit 4; }
[[ -f "${TRAIN_FILE}" ]] || { echo "Missing training file: ${TRAIN_FILE}" >&2; exit 4; }
IFS=',' read -r -a _gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
(( ${#_gpu_list[@]} >= 2 )) || { echo "Need two GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2; exit 4; }

launch "${RUN_NAME}" "Qwen2.5-Math-7B pure GRPO${RECOVERY_MODE:+ deterministic recovery}" \
  actor_rollout_ref.rollout.load_format=auto \
  actor_rollout_ref.rollout.calculate_log_probs=${ROLLOUT_CALCULATE_LOG_PROBS} \
  algorithm.rollout_correction.bypass_mode=true \
  algorithm.rollout_correction.loss_type=ppo_clip

unset _requested_model _requested_seed _requested_project _requested_gpus _requested_tag _requested_run_name _requested_rollout_mem _requested_mini_batch _requested_train_batch _requested_actor_attn _requested_vllm_attn _requested_recovery_mode
