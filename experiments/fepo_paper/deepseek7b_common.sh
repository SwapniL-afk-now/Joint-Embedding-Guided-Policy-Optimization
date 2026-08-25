#!/usr/bin/env bash
# Shared DeepSeek-Math-7B configuration for the compact FEPO/Z-REF study.
# Source this file from one of the d*.sh arms; it intentionally reuses the
# paper-program launch/validation safeguards in _common.sh.
set -euo pipefail

# Preserve an explicitly supplied path before _common.sh installs its Qwen default.
_requested_model=${MODEL_PATH:-}
_requested_project=${PROJECT_NAME:-}
_requested_train_file=${TRAIN_FILE:-}
_requested_train_seed=${TRAIN_SEED:-}
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

resolve_deepseek_model() {
  if [[ -n "${_requested_model}" ]]; then
    printf '%s' "${_requested_model}"
    return
  fi
  local candidates=(
    "${DEEPSEEK_MODEL_PATH:-}"
    "/workspace/models/DeepSeek-Math-7B-Instruct"
    "/workspace/models/deepseek-math-7b-instruct"
    "/workspace/.hf_home/hub/models--deepseek-ai--deepseek-math-7b-instruct/snapshots"/*
    "/root/.cache/huggingface/hub/models--deepseek-ai--deepseek-math-7b-instruct/snapshots"/*
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}/config.json" ]]; then
      printf '%s' "${candidate}"
      return
    fi
  done
  # Resolve an already-cached HF snapshot without downloading implicitly.
  "${PYTHON_BIN:-.venv/bin/python}" - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("deepseek-ai/deepseek-math-7b-instruct", local_files_only=True))
PY
}

export MODEL_PATH="$(resolve_deepseek_model)"
export MODEL_ID=deepseek-ai/deepseek-math-7b-instruct
if [[ -n "${_requested_train_file}" ]]; then
  export TRAIN_FILE="${_requested_train_file}"
elif [[ -f /workspace/jepa-grpo-cache/data/dsr_math345/train_minus_probe.parquet ]]; then
  # Keep the fixed recovery probe held out of optimization.
  export TRAIN_FILE=/workspace/jepa-grpo-cache/data/dsr_math345/train_minus_probe.parquet
fi
export PROJECT_NAME=${_requested_project:-tafr-deepseek-math7b-compact}
export NDEVICES_PER_NODE=2
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1}
# The 7B actor + rollout engine + anchor/failure references needs LoRA on a
# 2-GPU/96GB host. Override DRGRPO_USE_LORA=false only on a larger host.
export DRGRPO_USE_LORA=true
export LORA_RANK=64
export LORA_ALPHA=128
export ROLLOUT_GPU_MEM_UTIL=0.40
export PPO_MINI_BATCH_SIZE=8
export TRAIN_PROMPT_BATCH_SIZE=32
export NUM_GENERATIONS=8
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=3072
export PPO_MAX_TOKEN_LEN_PER_GPU=16384
export TAFR_SFT_MAX_TOKEN_LEN_PER_GPU=16384
# DeepSeek-Math has a 4096-token context window; keep prompt+response <= 4096.

# Allow callers to run one seed at a time. The default is deliberately cheap;
# set TRAIN_SEED/SEEDS externally for the final multi-seed study.
export TRAIN_SEED=${_requested_train_seed:-3407}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-500}
export TEST_FREQ=${TEST_FREQ:-10}

unset _requested_model _requested_project _requested_train_file _requested_train_seed
