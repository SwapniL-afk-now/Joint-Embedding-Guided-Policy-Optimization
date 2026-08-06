#!/usr/bin/env bash
set -euo pipefail

# Example TAFR-GRPO launch. Override paths and model/data settings from the
# environment to keep the script usable for baselines.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "$REPO_ROOT"

# Reduce CUDA fragmentation on the colocated (vLLM + FSDP) single GPU.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}

if [[ -f "${REPO_ROOT}/.env" ]]; then
    _XTRACE_WAS_ON=0
    case $- in
        *x*) _XTRACE_WAS_ON=1; set +x ;;
    esac
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
    if [[ ${_XTRACE_WAS_ON} -eq 1 ]]; then
        set -x
    fi
    unset _XTRACE_WAS_ON
fi

PROJECT_NAME=${PROJECT_NAME:-verl_drgrpo_deepscaler}
export WANDB_PROJECT=${WANDB_PROJECT:-${PROJECT_NAME}}
export WANDB_SILENT=${WANDB_SILENT:-true}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}
# Default model/data/benchmark settings mirror
# examples/jepa_grpo_trainer/run_qwen_1_5b_ray.sh so TAFR-GRPO
# is an apples-to-apples comparison against the JEPA-GRPO runs.
TAFR_VARIANT=${TAFR_VARIANT:-full}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-tafr_grpo_${TAFR_VARIANT}_qwen25math_1_5b-${RUN_TIMESTAMP}}
MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen2.5-Math-1.5B-Instruct}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-1}

# Match the wesserstein trainer's exact training and testing datasets.
TRAIN_DATASET=${TRAIN_DATASET:-zhuzilin/dapo-math-17k}
TRAIN_DATASET_CONFIG=${TRAIN_DATASET_CONFIG:-default}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
TRAIN_FILE=${TRAIN_FILE:-/workspace/jepa-grpo-cache/data/amcaime32k/train.parquet}
PREPARE_TRAIN_DATA=${PREPARE_TRAIN_DATA:-false}

EVAL_DATA_DIR=${EVAL_DATA_DIR:-/workspace/jepa-grpo-cache/eval_data}
PREPARE_EVAL_DATA=${PREPARE_EVAL_DATA:-true}
if [[ -z "${VAL_FILES:-}" ]]; then
    # Same in-training core-math subset as the JEPA-GRPO run.
    VAL_FILES="[${EVAL_DATA_DIR}/aime24.parquet,${EVAL_DATA_DIR}/aime25.parquet,${EVAL_DATA_DIR}/aime26.parquet,${EVAL_DATA_DIR}/amc23.parquet]"
fi

if [[ "${PREPARE_TRAIN_DATA}" == "true" && ! -f "${TRAIN_FILE}" ]]; then
    mkdir -p "$(dirname "${TRAIN_FILE}")"
    "$PYTHON_BIN" -m verl.experimental.fepo.data \
        --dataset "${TRAIN_DATASET}" \
        --config "${TRAIN_DATASET_CONFIG}" \
        --split "${TRAIN_SPLIT}" \
        --max-samples "${TRAIN_MAX_SAMPLES}" \
        --output "${TRAIN_FILE}"
fi

if [[ "${PREPARE_EVAL_DATA}" == "true" ]]; then
    mkdir -p "${EVAL_DATA_DIR}"
    if [[ ! -f "${EVAL_DATA_DIR}/amc23.parquet" ]]; then
        "$PYTHON_BIN" -m verl.experimental.fepo.data --dataset math-ai/amc23 --split test --output "${EVAL_DATA_DIR}/amc23.parquet"
    fi
    if [[ ! -f "${EVAL_DATA_DIR}/aime24.parquet" ]]; then
        "$PYTHON_BIN" -m verl.experimental.fepo.data --dataset math-ai/aime24 --split test --output "${EVAL_DATA_DIR}/aime24.parquet"
    fi
    if [[ ! -f "${EVAL_DATA_DIR}/aime25.parquet" ]]; then
        "$PYTHON_BIN" -m verl.experimental.fepo.data --dataset math-ai/aime25 --split test --output "${EVAL_DATA_DIR}/aime25.parquet"
    fi
fi

# ── TAFR-GRPO hyperparameters ────────────────────────────────────────────────
# false => plain GRPO with every other setting untouched (the control arm).
TAFR_ENABLE=${TAFR_ENABLE:-true}
# TAFR_VARIANT defaulted near the top (used in EXPERIMENT_NAME): full | anchor_only | replay_only
TAFR_LOSS_VERSION=${TAFR_LOSS_VERSION:-failure_token_adv}   # failure_token_adv | k3_legacy
TAFR_LOGPROB_BACKEND=${TAFR_LOGPROB_BACKEND:-} # resolved after the LoRA/full-finetune mode is known
TAFR_VLLM_SCORE_MICRO_BATCH_SIZE=${TAFR_VLLM_SCORE_MICRO_BATCH_SIZE:-16}
# Paper A_reg: anchor stability plus difficulty-gated failure avoidance.
# The complete detached advantage is RMS-matched to GRPO before clipping.
TAFR_BETA=${TAFR_BETA:-0.5}                                # target regularization/GRPO advantage RMS ratio
# k3_legacy only: independent weights for the two opposing halves of the TAFR loss
# (+b_a*anchor_kl - b_r*replay_kl). Empty = use TAFR_BETA for both (unchanged behaviour).
TAFR_BETA_ANCHOR=${TAFR_BETA_ANCHOR:-}
TAFR_BETA_REPLAY=${TAFR_BETA_REPLAY:-}
TAFR_FAILURE_ADVANTAGE_CLIP=${TAFR_FAILURE_ADVANTAGE_CLIP:-5.0}
# A_reg normalization: p99 (legacy global RMS/quantile match) | dense_token_credit
# (zero-mean per sequence, scaled by each rollout's own |A_GRPO|). Under
# dense_token_credit, LAMBDA_A/LAMBDA_F are the dense weights and TAFR_BETA must be
# 1.0 -- losses.py multiplies the returned advantage by beta on top of them.
TAFR_ADV_NORM=${TAFR_ADV_NORM:-p99}
TAFR_LAMBDA_A=${TAFR_LAMBDA_A:-1.0}                         # -> custom_tafr_grpo.anchor_weight
TAFR_LAMBDA_F=${TAFR_LAMBDA_F:-1.0}                         # -> custom_tafr_grpo.failure_weight
TAFR_DTC_DOSE=${TAFR_DTC_DOSE:-0.5}
TAFR_DTC_TAU=${TAFR_DTC_TAU:-5.0}
TAFR_DTC_EPS_FLOOR=${TAFR_DTC_EPS_FLOOR:-0.01}
TAFR_DTC_EMA_DECAY=${TAFR_DTC_EMA_DECAY:-0.99}
TAFR_FAILURE_CLIP_RATIO=${TAFR_FAILURE_CLIP_RATIO:-}        # empty = use actor clip_ratio
TAFR_REUSE_ROLLOUT_LOGPROBS=${TAFR_REUSE_ROLLOUT_LOGPROBS:-}   # old_log_probs = rollout log-probs (no old-policy forward)
TAFR_VERIFY_ROLLOUT_LOGPROBS=${TAFR_VERIFY_ROLLOUT_LOGPROBS:-false} # log parity vs HF recomputation before trusting reuse
TAFR_SCORE_ONLY_ACTIVE_GROUPS=${TAFR_SCORE_ONLY_ACTIVE_GROUPS:-true} # skip all-correct groups in failure scoring
TAFR_FAILURE_MIN_UPDATES_BEFORE_USE=${TAFR_FAILURE_MIN_UPDATES_BEFORE_USE:-1} # failure-SFT updates before the advantage activates
TAFR_EMA_GAMMA=${TAFR_EMA_GAMMA:-0.9}            # EMA decay for the failure EMA tracker
TAFR_MIX_ETA=${TAFR_MIX_ETA:-0.5}               # mix weight for anchor/failure EMA with the fixed reference

# Failure-SFT schedule
TAFR_SFT_UPDATE_INTERVAL=${TAFR_SFT_UPDATE_INTERVAL:-5}       # run SFT every N GRPO steps
TAFR_CHECKPOINT_INTERVAL=${TAFR_CHECKPOINT_INTERVAL:-10}      # save + refresh EMA every N GRPO steps

# Failure-SFT optimizer
TAFR_SFT_LR=${TAFR_SFT_LR:-1.0e-5}                           # failure-SFT learning rate
# Chunk size when iterating over the interval's buffered failures.
# The buffer is cleared after every SFT update, so this controls
# how many examples go into each optimizer step within the interval.
TAFR_SFT_BATCH_SIZE=${TAFR_SFT_BATCH_SIZE:-8}
# Token-budgeted failure-SFT micro-batching. >0 packs records greedily so each
# step's padded tokens (rows*max_len) stay under this budget => higher GPU util
# than the fixed batch size. Defaults to PPO_MAX_TOKEN_LEN_PER_GPU below (mirrors the
# actor-update budget); set 0 to disable and fall back to TAFR_SFT_BATCH_SIZE.
# (Real default applied after PPO_MAX_TOKEN_LEN_PER_GPU is defined; only a user
# override is honored here.)
TAFR_SFT_MAX_TOKEN_LEN_PER_GPU=${TAFR_SFT_MAX_TOKEN_LEN_PER_GPU:-}

# TAFR disk-save throttling (disk shortage): the EMA refresh + vLLM adapter export
# still run every checkpoint interval, but the failure model / optimizer / tafr_state
# are only written when global_step % this == 0 (0 = every refresh). Old dirs rotate,
# so the latest save replaces the previous. Set the optimizer flag false to drop the
# largest file (loses failure-SFT momentum across a resume).
TAFR_SAVE_TO_DISK_INTERVAL=${TAFR_SAVE_TO_DISK_INTERVAL:-0}
TAFR_SAVE_FAILURE_OPTIMIZER=${TAFR_SAVE_FAILURE_OPTIMIZER:-true}
# Hard cap on optimizer steps per interval (9999 = effectively unlimited).
TAFR_SFT_MAX_UPDATES=${TAFR_SFT_MAX_UPDATES:-9999}

# Failure data collector — cleared automatically after each SFT update,
# so this is just a safety cap in case of an unusually large interval.
TAFR_FAILURE_DATA_MAX_SIZE=${TAFR_FAILURE_DATA_MAX_SIZE:-null} # null = unlimited
TAFR_FAILURE_DATA_SAMPLING=${TAFR_FAILURE_DATA_SAMPLING:-recent} # recent | uniform

TRAIN_PROMPT_BATCH_SIZE=${TRAIN_PROMPT_BATCH_SIZE:-64}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${TRAIN_PROMPT_BATCH_SIZE}}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
# Failure-SFT token budget mirrors the actor-update budget unless overridden above.
TAFR_SFT_MAX_TOKEN_LEN_PER_GPU=${TAFR_SFT_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
ACTOR_ATTENTION_IMPL=${ACTOR_ATTENTION_IMPL:-flash_attention_2}
DRGRPO_USE_LORA=${DRGRPO_USE_LORA:-false}
LORA_RANK=${LORA_RANK:-512}
LORA_ALPHA=${LORA_ALPHA:-1024}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}

if [[ -z "${TAFR_LOGPROB_BACKEND}" ]]; then
    if [[ "${DRGRPO_USE_LORA}" == "true" ]]; then
        TAFR_LOGPROB_BACKEND=vllm
    else
        TAFR_LOGPROB_BACKEND=hf
    fi
fi
if [[ "${TAFR_LOGPROB_BACKEND}" == "vllm" && "${DRGRPO_USE_LORA}" != "true" ]]; then
    echo "TAFR_LOGPROB_BACKEND=vllm requires DRGRPO_USE_LORA=true; use hf for full fine-tuning." >&2
    exit 2
fi

if [[ -z "${TAFR_REUSE_ROLLOUT_LOGPROBS}" ]]; then
    if [[ "${TAFR_LOGPROB_BACKEND}" == "vllm" ]]; then
        TAFR_REUSE_ROLLOUT_LOGPROBS=true
    else
        TAFR_REUSE_ROLLOUT_LOGPROBS=false
    fi
fi
if [[ "${TAFR_LOGPROB_BACKEND}" == "hf" && "${TAFR_REUSE_ROLLOUT_LOGPROBS}" == "true" ]]; then
    echo "HF TAFR scoring requires TAFR_REUSE_ROLLOUT_LOGPROBS=false to avoid mixed-backend A_reg noise." >&2
    exit 2
fi
if [[ "${TAFR_LOGPROB_BACKEND}" == "vllm" && "${TAFR_REUSE_ROLLOUT_LOGPROBS}" != "true" ]]; then
    echo "vLLM TAFR scoring requires TAFR_REUSE_ROLLOUT_LOGPROBS=true so all A_reg policies use vLLM log-probs." >&2
    exit 2
fi

ACTOR_LR=${ACTOR_LR:-1e-6}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
PPO_LOSS_COEF=${PPO_LOSS_COEF:-1}
CLIP_RATIO=${CLIP_RATIO:-0.2}
CLIP_RATIO_C=${CLIP_RATIO_C:-3.0}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
ROLLOUT_N=${ROLLOUT_N:-${NUM_GENERATIONS}}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-2048}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-131072}
if [[ -z "${ROLLOUT_MAX_LORAS:-}" ]]; then
    # The optimized path needs the rollout adapter plus one failure adapter;
    # legacy k3 also needs its anchor adapter.
    if [[ "${TAFR_LOSS_VERSION}" == "failure_token_adv" ]]; then
        ROLLOUT_MAX_LORAS=2
    else
        ROLLOUT_MAX_LORAS=3
    fi
fi
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-8}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}
VALIDATION_SEEDS=${VALIDATION_SEEDS:-'[31415,271828,14142,16180,299792]'}
TRAIN_SEED=${TRAIN_SEED:-31415}
TRAIN_TEMPERATURE=${TRAIN_TEMPERATURE:-1.0}
TRAIN_TOP_P=${TRAIN_TOP_P:-1.0}
TRAIN_TOP_K=${TRAIN_TOP_K:--1}

# Wasserstein guidance
WG_ENABLE=${WG_ENABLE:-false}
WG_LAMBDA=${WG_LAMBDA:-0.01}
WG_ALPHA=${WG_ALPHA:-0.2}
WG_EMBED_MODEL=${WG_EMBED_MODEL:-BAAI/bge-small-en-v1.5}
WG_DECODE_MODEL=${WG_DECODE_MODEL:-${MODEL_PATH}}  # use actor tokenizer to decode response token IDs

MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-666}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
# best-checkpoint selection sources (matches the JEPA-GRPO run)
BEST_CKPT_SOURCES=${BEST_CKPT_SOURCES:-'["aime24","aime25","aime26","amc23"]'}
# Selection metric(s), e.g. '["pass@1"]'. Unset keeps the trainer default ("combined",
# = 0.5*(avg@k + pass@k)), which needs BOTH keys -- so a greedy n=1 run must set this
# to pass@1 or no best/ checkpoint is ever written.
BEST_CKPT_METRICS=${BEST_CKPT_METRICS:-}
LOGGER=${LOGGER:-'["console","wandb"]'}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-${MAX_OPTIMIZER_STEPS}}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-null}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-null}
CKPTS_DIR=${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=False
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILE"
    data.val_files=${VAL_FILES}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.dataloader_num_workers=${DATALOADER_NUM_WORKERS}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=${ACTOR_ATTENTION_IMPL}
)

if [[ "${DRGRPO_USE_LORA}" == "true" ]]; then
    MODEL+=(
        actor_rollout_ref.model.lora_rank=${LORA_RANK}
        actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
        actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}
    )
fi

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO}
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_loss_coef=${PPO_LOSS_COEF}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.data_loader_seed=${TRAIN_SEED}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=${TRAIN_TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${TRAIN_TOP_P}
    actor_rollout_ref.rollout.top_k=${TRAIN_TOP_K}
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_AGENT_NUM_WORKERS}
    actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE}
    +actor_rollout_ref.rollout.enable_sleep_mode=${ROLLOUT_FREE_CACHE_ENGINE}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_loras=${ROLLOUT_MAX_LORAS}
)
if [[ "${TAFR_LOSS_VERSION}" == "failure_token_adv" && "${TAFR_REUSE_ROLLOUT_LOGPROBS}" == "true" ]]; then
    # Capture per-chosen-token log-probs during generation so the old-policy
    # forward pass can be skipped in the optimized TAFR path.
    ROLLOUT+=(
        actor_rollout_ref.rollout.calculate_log_probs=True
    )
fi

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.resume_mode=auto
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger=${LOGGER}
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.default_local_dir=${CKPTS_DIR}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.max_actor_ckpt_to_keep=1
    trainer.test_freq=${TEST_FREQ}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.log_val_generations=${LOG_VAL_GENERATIONS}
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}
    trainer.validation_data_dir=${VALIDATION_DATA_DIR}
    +trainer.best_ckpt_sources=${BEST_CKPT_SOURCES}
    +trainer.validation_seeds=${VALIDATION_SEEDS}
)
if [[ -n "${BEST_CKPT_METRICS}" ]]; then
    TRAINER+=( "+trainer.best_ckpt_metrics=${BEST_CKPT_METRICS}" )
fi

TAFR=(
    # false runs plain GRPO: ray_trainer gates every TAFR path on this
    # (self.tafr_enabled), so the rest of this array is inert and the run is a
    # single-variable control for whatever TAFR arm shares its other settings.
    custom_tafr_grpo.enable="${TAFR_ENABLE}"
    custom_tafr_grpo.variant="${TAFR_VARIANT}"
    +custom_tafr_grpo.loss_version="${TAFR_LOSS_VERSION}"
    custom_tafr_grpo.logprob_backend="${TAFR_LOGPROB_BACKEND}"
    custom_tafr_grpo.vllm_score_micro_batch_size="${TAFR_VLLM_SCORE_MICRO_BATCH_SIZE}"
    # Failure-token advantage (optimized path)
    custom_tafr_grpo.beta="${TAFR_BETA}"
    ${TAFR_BETA_ANCHOR:+ +custom_tafr_grpo.beta_anchor="${TAFR_BETA_ANCHOR}"}
    ${TAFR_BETA_REPLAY:+ +custom_tafr_grpo.beta_replay="${TAFR_BETA_REPLAY}"}
    +custom_tafr_grpo.failure_advantage_clip="${TAFR_FAILURE_ADVANTAGE_CLIP}"
    +custom_tafr_grpo.adv_norm="${TAFR_ADV_NORM}"
    +custom_tafr_grpo.anchor_weight="${TAFR_LAMBDA_A}"
    +custom_tafr_grpo.failure_weight="${TAFR_LAMBDA_F}"
    +custom_tafr_grpo.dtc_dose="${TAFR_DTC_DOSE}"
    +custom_tafr_grpo.dtc_tau="${TAFR_DTC_TAU}"
    +custom_tafr_grpo.dtc_eps_floor="${TAFR_DTC_EPS_FLOOR}"
    +custom_tafr_grpo.dtc_ema_decay="${TAFR_DTC_EMA_DECAY}"
    +custom_tafr_grpo.reuse_rollout_log_probs="${TAFR_REUSE_ROLLOUT_LOGPROBS}"
    +custom_tafr_grpo.verify_rollout_log_probs="${TAFR_VERIFY_ROLLOUT_LOGPROBS}"
    +custom_tafr_grpo.score_only_active_groups="${TAFR_SCORE_ONLY_ACTIVE_GROUPS}"
    +custom_tafr_grpo.failure_min_updates_before_use="${TAFR_FAILURE_MIN_UPDATES_BEFORE_USE}"
)
if [[ -n "${TAFR_FAILURE_CLIP_RATIO:-}" ]]; then
    TAFR+=(
        custom_tafr_grpo.failure_clip_ratio="${TAFR_FAILURE_CLIP_RATIO}"
    )
fi

TAFR+=(
    # Failure-model EMA (optimized path) / anchor mix (k3_legacy)
    custom_tafr_grpo.ema_gamma="${TAFR_EMA_GAMMA}"
    custom_tafr_grpo.mix_eta="${TAFR_MIX_ETA}"
    # Disable verl built-in KL (TAFR manages its own)
    custom_tafr_grpo.disable_builtin_kl=true
    # Failure-SFT schedule
    custom_tafr_grpo.sft_update_interval_grpo_steps="${TAFR_SFT_UPDATE_INTERVAL}"
    custom_tafr_grpo.checkpoint_interval_grpo_steps="${TAFR_CHECKPOINT_INTERVAL}"
    # Failure-SFT optimizer
    custom_tafr_grpo.failure_sft_lr="${TAFR_SFT_LR}"
    custom_tafr_grpo.failure_sft_batch_size="${TAFR_SFT_BATCH_SIZE}"
    custom_tafr_grpo.failure_sft_max_token_len_per_gpu="${TAFR_SFT_MAX_TOKEN_LEN_PER_GPU}"
    custom_tafr_grpo.failure_sft_max_updates_per_interval="${TAFR_SFT_MAX_UPDATES}"
    # Disk-save throttling
    custom_tafr_grpo.save_to_disk_interval_grpo_steps="${TAFR_SAVE_TO_DISK_INTERVAL}"
    custom_tafr_grpo.save_failure_sft_optimizer="${TAFR_SAVE_FAILURE_OPTIMIZER}"
    # Failure data collector
    custom_tafr_grpo.failure_data_max_size="${TAFR_FAILURE_DATA_MAX_SIZE}"
    custom_tafr_grpo.failure_data_sampling="${TAFR_FAILURE_DATA_SAMPLING}"
)

WASSERSTEIN=(
    actor_rollout_ref.actor.wasserstein_guidance.enable=${WG_ENABLE}
    actor_rollout_ref.actor.wasserstein_guidance.lambda_wg=${WG_LAMBDA}
    actor_rollout_ref.actor.wasserstein_guidance.alpha_transport=${WG_ALPHA}
    actor_rollout_ref.actor.wasserstein_guidance.embed_model=${WG_EMBED_MODEL}
    actor_rollout_ref.actor.wasserstein_guidance.decode_model="${WG_DECODE_MODEL}"
)

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${TAFR[@]}" \
    "${WASSERSTEIN[@]}" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.ref.strategy=fsdp \
    critic.enable=false \
    "$@"
