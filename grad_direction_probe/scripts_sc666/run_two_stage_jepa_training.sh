#!/usr/bin/env bash
# =============================================================================
# run_two_stage_jepa_training.sh — TWO-STAGE JEPA TRAINING (JEPA-RL recipe)
# =============================================================================
# Stage 1 (NEW, offline): JEPA + SIGReg pretraining on SOLVED TRACES, with the
#   OFFICIAL repo loss — flat-mean pull + SIGReg only, NO repulsion term, no
#   advantage weighting:
#       L = mean(1 - cos(p, z)) + lambda*SIGReg(p)
#       p = Enc(problem + <|predictor_1|>)   z = stopgrad Enc(solved trace)
#   Data: TWO solved-trace sources, both filtered to <= 3072 tokens:
#     (1) amcaime32k (THE SAME dataset as stage 2) — its own solved traces
#         (amcaime32k/teacher.cot.responses.pt, {index: [trace]}, 1:1 with the
#         rows) -> amcaime32k_solved_traces.pt
#     (2) open-r1/OpenR1-Math-220k verified-correct traces
#         (build_openr1_solved_trace_cache.py) -> solved_traces_openr1_max3072.pt
#   Runs as a plain single-GPU PyTorch loop (no RL, no vLLM, no ray).
#   Output: merged HF weights + tokenizer (+ predictor_delta) in STAGE1_OUT,
#   with a `latest` symlink for stage 2.
# Stage 2: PLAIN GRPO (Dr.GRPO, no JEPA loss) on the amcaime32k dataset,
#   initialized from the stage-1 weights (MODEL_PATH=$STAGE1_OUT/latest) —
#   tests whether JEPA-pretrained init beats base-model init for GRPO.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo /workspace/Joint-Embedding-Guided-Policy-Optimization)"

PYTHON_BIN=${PYTHON_BIN:-/workspace/exploration/.venv/bin/python3}
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN=python3
fi

# HF cache convention (same as run_drgrpo_jepa.sh)
export HF_HOME=${HF_HOME:-/workspace/jepa-grpo-cache/hf}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/workspace/jepa-grpo-cache/hf/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/workspace/jepa-grpo-cache/hf/hub}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# --------------------------- user-adjustable -------------------------------
# Stage 1
STAGE1_MODEL=${STAGE1_MODEL:-/workspace/models/Qwen2.5-Math-1.5B-Instruct}
STAGE1_SHARDS=${STAGE1_SHARDS:-/workspace/jepa-grpo-cache/amcaime32k_solved_traces.pt,/workspace/jepa-grpo-cache/openr1_solved_traces_max3072/solved_traces_openr1_max3072.pt}
STAGE1_OUT=${STAGE1_OUT:-/workspace/checkpoints/jepa-pretrain/amcaime32k-openr1-2ep-lr1e-5}
STAGE1_LOG=${STAGE1_LOG:-/workspace/exploration/grad_direction_probe/log_jepa_pretrain_amcaime32k_openr1_2ep_lr1e-5.log}
STAGE1_METRICS=${STAGE1_METRICS:-/workspace/exploration/grad_direction_probe/metrics_jepa_pretrain_amcaime32k_openr1_2ep_lr1e-5.jsonl}
STAGE1_EPOCHS=${STAGE1_EPOCHS:-2}
STAGE1_BATCH_SIZE=${STAGE1_BATCH_SIZE:-64}
STAGE1_LR=${STAGE1_LR:-1e-5}
STAGE1_SIGREG_LAMBDA=${STAGE1_SIGREG_LAMBDA:-0.1}
STAGE1_PREDICTOR_K=${STAGE1_PREDICTOR_K:-1}
STAGE1_SEED=${STAGE1_SEED:-31415}
STAGE1_SAVE_INTERVAL=${STAGE1_SAVE_INTERVAL:-100}
STAGE1_MAX_STEPS=${STAGE1_MAX_STEPS:--1}
STAGE1_MAX_PAIRS=${STAGE1_MAX_PAIRS:--1}

# Stage 2 (plain GRPO on amcaime32k, from the stage-1 init)
STAGE2_TRAIN_FILE=${STAGE2_TRAIN_FILE:-/workspace/jepa-grpo-cache/data/amcaime32k/train.parquet}
STAGE2_LOG=${STAGE2_LOG:-/workspace/exploration/grad_direction_probe/log_grpo_after_jepa_pretrain_666.log}
STAGE2_METRICS=${STAGE2_METRICS:-/workspace/exploration/grad_direction_probe/metrics_grpo_after_jepa_pretrain_666.jsonl}
STAGE2_EXPERIMENT=${STAGE2_EXPERIMENT:-grpo-after-jepa-pretrain-666-s3407}
STAGE2_ACTOR_LR=${STAGE2_ACTOR_LR:-1e-6}
STAGE2_SEED=${STAGE2_SEED:-3407}
STAGE2_STEPS=${STAGE2_STEPS:-666}
STAGE2_N_COT=${STAGE2_N_COT:-8}
STAGE2_ROLLOUT_N=${STAGE2_ROLLOUT_N:-8}
STAGE2_VAL_SEEDS=${STAGE2_VAL_SEEDS:-'[3407]'}
STAGE2_VAL_FILES=${STAGE2_VAL_FILES:-'[/workspace/jepa-grpo-cache/eval_data/aime24.parquet,/workspace/jepa-grpo-cache/eval_data/aime25.parquet,/workspace/jepa-grpo-cache/eval_data/aime26.parquet,/workspace/jepa-grpo-cache/eval_data/amc23.parquet]'}
STAGE2_BEST_CKPT_SOURCES=${STAGE2_BEST_CKPT_SOURCES:-'["aime24","aime25","aime26","amc23"]'}
STAGE2_RAY_PORT=${STAGE2_RAY_PORT:-37379}
STAGE2_RAY_DASH_PORT=${STAGE2_RAY_DASH_PORT:-8265}

# Rerun toggles: skip a finished stage
SKIP_STAGE_1=${SKIP_STAGE_1:-false}
SKIP_STAGE_2=${SKIP_STAGE_2:-false}

# --------------------------- Stage 1: offline JEPA + SIGReg -----------------
if [[ "${SKIP_STAGE_1}" != "true" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
    echo "================ STAGE 1: offline JEPA+SIGReg pretraining on solved traces ================"
    echo "model:   ${STAGE1_MODEL}"
    echo "shards:  ${STAGE1_SHARDS}"
    echo "out:     ${STAGE1_OUT}"
    echo "recipe:  ${STAGE1_EPOCHS} epochs, bs ${STAGE1_BATCH_SIZE}, lr ${STAGE1_LR}, "
    echo "         SIGReg lambda ${STAGE1_SIGREG_LAMBDA}, predictor_k ${STAGE1_PREDICTOR_K}, "
    echo "         official loss (flat pull + SIGReg, NO repulsion term)"
    export MODEL="${STAGE1_MODEL}" \
           DATA_SHARDS="${STAGE1_SHARDS}" \
           OUT_DIR="${STAGE1_OUT}" \
           METRICS_PATH="${STAGE1_METRICS}" \
           LOG_FILE="${STAGE1_LOG}" \
           EPOCHS="${STAGE1_EPOCHS}" \
           BATCH_SIZE="${STAGE1_BATCH_SIZE}" \
           LR="${STAGE1_LR}" \
           SIGREG_LAMBDA="${STAGE1_SIGREG_LAMBDA}" \
           PREDICTOR_K="${STAGE1_PREDICTOR_K}" \
           SEED="${STAGE1_SEED}" \
           SAVE_INTERVAL="${STAGE1_SAVE_INTERVAL}" \
           MAX_STEPS="${STAGE1_MAX_STEPS}" \
           MAX_PAIRS="${STAGE1_MAX_PAIRS}"
    "${PYTHON_BIN}" -m verl.experimental.jepa_grpo.pretrain_offline 2>&1 | tee "${STAGE1_LOG}"
    echo "Stage 1 finished. Weights: ${STAGE1_OUT}/latest"
else
    echo "======== SKIP_STAGE_1=true: using existing weights ${STAGE1_OUT}/latest"
fi
STAGE1_LATEST="${STAGE1_OUT}/latest"
if [[ ! -d "${STAGE1_LATEST}" ]]; then
    echo "ERROR: stage-1 weights missing at ${STAGE1_LATEST}" >&2
    exit 1
fi

# --------------------------- Stage 2: plain GRPO on amcaime32k --------------
if [[ "${SKIP_STAGE_2}" != "true" ]]; then
    echo "================ STAGE 2: plain GRPO (no JEPA loss) on amcaime32k from stage-1 init ================"
    echo "init:     ${STAGE1_LATEST}"
    echo "train:    ${STAGE2_TRAIN_FILE}"
    echo "recipe:   ${STAGE2_N_COT} CoT, lr ${STAGE2_ACTOR_LR}, seed ${STAGE2_SEED}, ${STAGE2_STEPS} steps, full FT"

    unset RAY_ADDRESS 2>/dev/null || true
    RAY_BIN="${PYTHON_BIN%/python3}/ray"
    if ! (ss -tln 2>/dev/null | grep -q ":${STAGE2_RAY_PORT} "); then
        echo "starting ray head on 127.0.0.1:${STAGE2_RAY_PORT} (dashboard :${STAGE2_RAY_DASH_PORT})"
        CUDA_VISIBLE_DEVICES=0 "${RAY_BIN}" start --head --port="${STAGE2_RAY_PORT}" --dashboard-port="${STAGE2_RAY_DASH_PORT}"
    else
        echo "reusing existing ray head on 127.0.0.1:${STAGE2_RAY_PORT}"
    fi

    cd "${REPO_ROOT}/examples/jepa_grpo_trainer/qwen2.5-math-1.5b"
    CUDA_VISIBLE_DEVICES=0 \
    MODEL_PATH="${STAGE1_LATEST}" \
    ALLOW_MODEL_PATH_OVERRIDE=true \
    N_COT=${STAGE2_N_COT} \
    N_CODE=0 \
    ROLLOUT_N=${STAGE2_ROLLOUT_N} \
    JEPA_ENABLE=False \
    JEPA_LOSS_TYPE=llm-jepa \
    ALPHA=0.1 \
    AUTO_OFF_MIN_COS=0.5 \
    AUTO_OFF_ENABLE=False \
    DAPO_USE_LORA=false \
    ACTOR_LR=${STAGE2_ACTOR_LR} \
    TRAIN_FILE=${STAGE2_TRAIN_FILE} \
    TRAIN_SEED=${STAGE2_SEED} \
    VALIDATION_SEEDS=${STAGE2_VAL_SEEDS} \
    VAL_FILES=${STAGE2_VAL_FILES} \
    BEST_CKPT_SOURCES=${STAGE2_BEST_CKPT_SOURCES} \
    ENABLE_OVERLONG_BUFFER=true \
    NDEVICES_PER_NODE=1 \
    TOTAL_TRAINING_BATCHES=${STAGE2_STEPS} \
    MAX_OPTIMIZER_STEPS=${STAGE2_STEPS} \
    TOTAL_TRAINING_STEPS=${STAGE2_STEPS} \
    VAL_BATCH_SIZE=128 \
    PROJECT_NAME=grpo-qwen-3b \
    EXPERIMENT_NAME=${STAGE2_EXPERIMENT} \
    LOGGER='["console","wandb","file"]' \
    VERL_FILE_LOGGER_PATH=${STAGE2_METRICS} \
    ./run_drgrpo_jepa.sh \
      actor_rollout_ref.model.lora_rank=0 \
      actor_rollout_ref.model.lora_alpha=0 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
      actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
      actor_rollout_ref.rollout.max_num_seqs=2048 \
      jepa.self_code_targets=True \
      jepa.paper_sigreg_lambda=0.1 \
      jepa.fuse_optimizer_step=True \
      jepa.tcr_reward_beta=0 \
      jepa.teacher_cache_path="" \
      jepa.code_teacher_cache_path="" \
      +ray_kwargs.ray_init.address=127.0.0.1:${STAGE2_RAY_PORT} 2>&1 | tee "${STAGE2_LOG}"
else
    echo "======== SKIP_STAGE_2=true: stage 2 not run"
fi

echo "====================================================================="
echo "DONE. Stage-1 weights: ${STAGE1_OUT} | Stage-2 log: ${STAGE2_LOG}"
echo "====================================================================="
