#!/usr/bin/env bash
set -euo pipefail

# vLLM imports matplotlib through its CLI module. Training is non-interactive;
# force a headless backend so inherited Jupyter MPLBACKEND values cannot abort
# worker startup.
export MPLBACKEND=Agg

# Example SDC-GRPO launch. Override paths and model/data settings from the
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
# examples/jepa_grpo_trainer/run_qwen_1_5b_ray.sh so SDC-GRPO
# is an apples-to-apples comparison against the JEPA-GRPO runs.
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sdc_grpo_qwen25math_1_5b-${RUN_TIMESTAMP}}
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

# ── SDC-GRPO hyperparameters ────────────────────────────────────────────────
SDC_ENABLE=${SDC_ENABLE:-true}
SDC_BETA=${SDC_BETA:-0.5}
SDC_LOGPROB_BACKEND=${SDC_LOGPROB_BACKEND:-}
SDC_VLLM_SCORE_MICRO_BATCH_SIZE=${SDC_VLLM_SCORE_MICRO_BATCH_SIZE:-16}
SDC_REUSE_ROLLOUT_LOGPROBS=${SDC_REUSE_ROLLOUT_LOGPROBS:-}
SDC_VERIFY_ROLLOUT_LOGPROBS=${SDC_VERIFY_ROLLOUT_LOGPROBS:-false}
SDC_USE_IMPORTANCE_WEIGHT=${SDC_USE_IMPORTANCE_WEIGHT:-true}
SDC_IMPORTANCE_WEIGHT_CLIP=${SDC_IMPORTANCE_WEIGHT_CLIP:-10.0}
SDC_BASE_MIX_GAMMA=${SDC_BASE_MIX_GAMMA:-0.9}
SDC_MIN_SFT_UPDATES_BEFORE_USE=${SDC_MIN_SFT_UPDATES_BEFORE_USE:-1}
SDC_SFT_UPDATE_INTERVAL=${SDC_SFT_UPDATE_INTERVAL:-5}       # run SFT every N GRPO steps
SDC_CHECKPOINT_INTERVAL=${SDC_CHECKPOINT_INTERVAL:-10}
SDC_SFT_LR=${SDC_SFT_LR:-1.0e-6}
SDC_SFT_BATCH_SIZE=${SDC_SFT_BATCH_SIZE:-8}
SDC_SFT_MAX_TOKEN_LEN_PER_GPU=${SDC_SFT_MAX_TOKEN_LEN_PER_GPU:-}
SDC_SFT_MAX_UPDATES=${SDC_SFT_MAX_UPDATES:-1}
SDC_SAVE_TO_DISK_INTERVAL=${SDC_SAVE_TO_DISK_INTERVAL:-0}
SDC_DATA_MAX_SIZE=${SDC_DATA_MAX_SIZE:-null}
SDC_DATA_SAMPLING=${SDC_DATA_SAMPLING:-recent}

TRAIN_PROMPT_BATCH_SIZE=${TRAIN_PROMPT_BATCH_SIZE:-64}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${TRAIN_PROMPT_BATCH_SIZE}}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
# SFT token budget mirrors the actor-update budget unless overridden above.
SDC_SFT_MAX_TOKEN_LEN_PER_GPU=${SDC_SFT_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
ACTOR_ATTENTION_IMPL=${ACTOR_ATTENTION_IMPL:-flash_attention_2}
# Load HF actors in bf16; fp32 is incompatible with FlashAttention 2 on Llama/DeepSeek.
MODEL_DTYPE=${MODEL_DTYPE:-bf16}
DRGRPO_USE_LORA=${DRGRPO_USE_LORA:-false}
LORA_RANK=${LORA_RANK:-512}
LORA_ALPHA=${LORA_ALPHA:-1024}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}

if [[ -z "${SDC_LOGPROB_BACKEND}" ]]; then
    if [[ "${DRGRPO_USE_LORA}" == "true" ]]; then
        SDC_LOGPROB_BACKEND=vllm
    else
        SDC_LOGPROB_BACKEND=hf
    fi
fi
if [[ "${SDC_LOGPROB_BACKEND}" == "vllm" && "${DRGRPO_USE_LORA}" != "true" ]]; then
    echo "SDC_LOGPROB_BACKEND=vllm requires DRGRPO_USE_LORA=true; use hf for full fine-tuning." >&2
    exit 2
fi

if [[ -z "${SDC_REUSE_ROLLOUT_LOGPROBS}" ]]; then
    if [[ "${SDC_LOGPROB_BACKEND}" == "vllm" ]]; then
        SDC_REUSE_ROLLOUT_LOGPROBS=true
    else
        SDC_REUSE_ROLLOUT_LOGPROBS=false
    fi
fi
if [[ "${SDC_LOGPROB_BACKEND}" == "hf" && "${SDC_REUSE_ROLLOUT_LOGPROBS}" == "true" ]]; then
    echo "HF SDC scoring requires SDC_REUSE_ROLLOUT_LOGPROBS=false so all SDC references use HF log-probs." >&2
    exit 2
fi
if [[ "${SDC_LOGPROB_BACKEND}" == "vllm" && "${SDC_REUSE_ROLLOUT_LOGPROBS}" != "true" ]]; then
    echo "vLLM SDC scoring requires SDC_REUSE_ROLLOUT_LOGPROBS=true so the PPO reference uses rollout log-probs." >&2
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
    ROLLOUT_MAX_LORAS=3
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
    actor_rollout_ref.actor.fsdp_config.model_dtype=${MODEL_DTYPE}
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
if [[ "${SDC_LOGPROB_BACKEND}" == "vllm" && "${SDC_REUSE_ROLLOUT_LOGPROBS}" == "true" ]]; then
    # Capture per-chosen-token log-probs during generation so the old-policy
    # forward pass can be skipped in the optimized SDC path.
    ROLLOUT+=(
        actor_rollout_ref.rollout.calculate_log_probs=True
    )
fi

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.model_dtype=${MODEL_DTYPE}
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

SDC=(
    custom_sdc_grpo.enable="${SDC_ENABLE}"
    custom_sdc_grpo.logprob_backend="${SDC_LOGPROB_BACKEND}"
    custom_sdc_grpo.vllm_score_micro_batch_size="${SDC_VLLM_SCORE_MICRO_BATCH_SIZE}"
    custom_sdc_grpo.beta="${SDC_BETA}"
    custom_sdc_grpo.reuse_rollout_log_probs="${SDC_REUSE_ROLLOUT_LOGPROBS}"
    custom_sdc_grpo.verify_rollout_log_probs="${SDC_VERIFY_ROLLOUT_LOGPROBS}"
    custom_sdc_grpo.use_importance_weight="${SDC_USE_IMPORTANCE_WEIGHT}"
    custom_sdc_grpo.importance_weight_clip="${SDC_IMPORTANCE_WEIGHT_CLIP}"
    custom_sdc_grpo.base_mix_gamma="${SDC_BASE_MIX_GAMMA}"
    custom_sdc_grpo.min_sft_updates_before_use="${SDC_MIN_SFT_UPDATES_BEFORE_USE}"
    custom_sdc_grpo.sft_update_interval_grpo_steps="${SDC_SFT_UPDATE_INTERVAL}"
    custom_sdc_grpo.checkpoint_interval_grpo_steps="${SDC_CHECKPOINT_INTERVAL}"
    custom_sdc_grpo.sft_lr="${SDC_SFT_LR}"
    custom_sdc_grpo.sft_batch_size="${SDC_SFT_BATCH_SIZE}"
    custom_sdc_grpo.sft_max_token_len_per_gpu="${SDC_SFT_MAX_TOKEN_LEN_PER_GPU}"
    custom_sdc_grpo.sft_max_updates_per_interval="${SDC_SFT_MAX_UPDATES}"
    custom_sdc_grpo.save_to_disk_interval_grpo_steps="${SDC_SAVE_TO_DISK_INTERVAL}"
    custom_sdc_grpo.data_max_size="${SDC_DATA_MAX_SIZE}"
    custom_sdc_grpo.data_sampling="${SDC_DATA_SAMPLING}"
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
    "${SDC[@]}" \
    "${WASSERSTEIN[@]}" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.ref.strategy=fsdp \
    critic.enable=false \
    "$@"
