#!/usr/bin/env bash
# Diagnostic probe: does raising TAFR_DTC_DOSE actually move kl_anchor/train pass@1,
# unlike beta (0.1/0.5/1.0) which didn't? Same repro config, loss switched to
# failure_token_adv + dense_token_credit, dose=2.0 (4x the existing dtc-d05 arm).
# Capped short (300 steps) -- this is a probe, not a full repro run.
set -euo pipefail
cd "$(dirname "$0")"

unset RAY_ADDRESS
export CUDA_VISIBLE_DEVICES=1
export PYTHON_BIN=.venv/bin/python
export MODEL_PATH=/workspace/models/Qwen2.5-Math-1.5B-Instruct
export DRGRPO_USE_LORA=false
export TRAIN_FILE=/workspace/jepa-grpo-cache/data/dsr_math345/train.parquet
EVAL_DATA_DIR=/workspace/jepa-grpo-cache/eval_data
export VAL_FILES="[${EVAL_DATA_DIR}/math500.parquet,${EVAL_DATA_DIR}/amc23.parquet,${EVAL_DATA_DIR}/aime24.parquet,${EVAL_DATA_DIR}/aime25.parquet,${EVAL_DATA_DIR}/aime26.parquet]"
export BEST_CKPT_SOURCES='["math500","amc23","aime24","aime25","aime26"]'
export BEST_CKPT_METRICS='["pass@1"]'
export PROJECT_NAME=tafr-repro-math15b
export ROLLOUT_GPU_MEM_UTIL=0.7
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=3072
export ACTOR_LR=1e-6
export PPO_MINI_BATCH_SIZE=16
export VAL_ROLLOUT_N=1
export VAL_DO_SAMPLE=False
export VAL_TEMPERATURE=0
export VALIDATION_SEEDS='[3407]'
export TRAIN_SEED=3407
export TAFR_VARIANT=full
export TAFR_EMA_GAMMA=0.9
export TAFR_MIX_ETA=0.5
export TAFR_SFT_UPDATE_INTERVAL=5
export TAFR_CHECKPOINT_INTERVAL=10
export TAFR_SFT_LR=5.0e-7
export TAFR_SFT_BATCH_SIZE=64
export TEST_FREQ=10
export SAVE_FREQ=50
export VAL_BEFORE_TRAIN=false
export NDEVICES_PER_NODE=1
export TOTAL_TRAINING_STEPS=600

# --- the variable under test ---
export TAFR_LOSS_VERSION=failure_token_adv
export TAFR_ADV_NORM=dense_token_credit
export TAFR_BETA=1.0            # loss code requires 1.0 for dtc (lambda_a/f apply the real scale)
export TAFR_LAMBDA_A=0.5
export TAFR_LAMBDA_F=0.5
export TAFR_DTC_DOSE=1.0        # 2x the existing dtc-d05 arm's 0.5; ratio~dose so targets ~1.0x GRPO rms

export EXPERIMENT_NAME=tafr-repro-dtc-d10-20260806
export CKPTS_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
if [[ -d "${CKPTS_DIR}" ]]; then
  echo "REFUSING: ${CKPTS_DIR} exists." >&2
  exit 3
fi

echo "=== ${EXPERIMENT_NAME} on GPU ${CUDA_VISIBLE_DEVICES} — dtc dose=1.0 probe (600 steps)"
bash examples/tafr_grpo/run_tafr_grpo_fsdp.sh 2>&1 | tee "logs_${EXPERIMENT_NAME}.log"
