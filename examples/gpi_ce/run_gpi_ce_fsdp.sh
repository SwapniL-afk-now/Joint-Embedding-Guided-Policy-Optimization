#!/usr/bin/env bash
set -euo pipefail

# Group Policy-Improvement Cross Entropy (GPI-CE), mixed online/offline candidates.
#
#   bash examples/gpi_ce/run_gpi_ce_fsdp.sh
#   GPI_ONLINE=8 GPI_OFFLINE=0 bash examples/gpi_ce/run_gpi_ce_fsdp.sh   # online-only arm
#
# Structure mirrors examples/tafr_grpo/run_tafr_grpo_fsdp.sh so the runs are
# directly comparable; the deltas are the custom_gpi_ce block and the actor
# settings that keep candidate groups atomic and force one optimizer step.
#
# The loop is the stock synchronous verl one, which is already exactly what the
# method needs:
#   vLLM generate (K_on) -> sleep_replicas -> [append K_off offline] -> actor
#   score (theta_n) -> q* -> ONE optimizer step -> update_weights (sync + wake).
#
# The training data must carry per-prompt verified offline solutions:
#   offline_responses: list[str], offline_rewards: list[float]
# See verl/experimental/gpi_ce/dataset.py for the schema.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "$REPO_ROOT"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

PROJECT_NAME=${PROJECT_NAME:-grpo-qwen-3b}
export WANDB_PROJECT=${WANDB_PROJECT:-${PROJECT_NAME}}
export WANDB_SILENT=${WANDB_SILENT:-true}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}

# --- the algorithm's own knobs -------------------------------------------------
GPI_ONLINE=${GPI_ONLINE:-4}          # K_on: what vLLM generates == rollout.n
GPI_OFFLINE=${GPI_OFFLINE:-4}        # K_off: appended from the dataset after sleep
GPI_TEMPERATURE=${GPI_TEMPERATURE:-0.5}   # tau in q* = softmax(s_old + r/tau)
GPI_VERIFY_OFFLINE=${GPI_VERIFY_OFFLINE:-true}
GPI_ALLOW_SHORT_OFFLINE=${GPI_ALLOW_SHORT_OFFLINE:-false}
# Skip the separate old-policy forward; q* is built inside the training forward.
GPI_FUSE_OLD_LOG_PROB=${GPI_FUSE_OLD_LOG_PROB:-true}
GPI_FILTER_DEGENERATE=${GPI_FILTER_DEGENERATE:-false}
GROUP_SIZE=$(( GPI_ONLINE + GPI_OFFLINE ))

EXPERIMENT_NAME=${EXPERIMENT_NAME:-gpi-ce-on${GPI_ONLINE}-off${GPI_OFFLINE}-tau${GPI_TEMPERATURE}-${RUN_TIMESTAMP}}
MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen2.5-Math-1.5B-Instruct}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-1}

TRAIN_FILE=${TRAIN_FILE:-/workspace/jepa-grpo-cache/data/amcaime32k_gpi/train.parquet}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-/workspace/jepa-grpo-cache/eval_data}
if [[ -z "${VAL_FILES:-}" ]]; then
    VAL_FILES="[${EVAL_DATA_DIR}/aime24.parquet,${EVAL_DATA_DIR}/aime25.parquet,${EVAL_DATA_DIR}/aime26.parquet,${EVAL_DATA_DIR}/amc23.parquet]"
fi

# TRAIN_FILE may be a single path or a hydra list "[a.parquet,b.parquet,...]".
if [[ "${TRAIN_FILE}" != \[* ]] && [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "GPI-CE train file not found: ${TRAIN_FILE}" >&2
    echo "It must contain offline_responses/offline_rewards columns; see verl/experimental/gpi_ce/dataset.py." >&2
    exit 2
fi

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}          # B, in prompts
# False => SequentialSampler. Required if the parquet encodes a curriculum.
DATA_SHUFFLE=${DATA_SHUFFLE:-True}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}  # stock RoPE: 1024+3072 <= 4096

# One optimizer step per mixed batch: the mini-batch IS the whole batch (in prompts),
# and accumulate_minibatch_grads makes any residual split a gradient-accumulation unit.
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
# Only used when USE_DYNAMIC_BSZ=False. With force_group_size=K this counts GROUPS
# per micro-batch, not rows: the actor sees GPI_GROUPS_PER_MICRO_BATCH * K sequences.
GPI_GROUPS_PER_MICRO_BATCH=${GPI_GROUPS_PER_MICRO_BATCH:-1}
# Token-budget batching (preferred): bounds micro-batch memory by TOKENS rather than by
# a fixed group count, which is what makes long offline traces safe. Both budgets must
# be >= max_prompt_length + max_response_length -- rearrange_micro_batches asserts the
# budget can hold the longest single sequence.
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
_MAX_SEQ_LEN=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-${_MAX_SEQ_LEN}}
LOGPROB_MAX_TOKEN_LEN_PER_GPU=${LOGPROB_MAX_TOKEN_LEN_PER_GPU:-${_MAX_SEQ_LEN}}

ACTOR_LR=${ACTOR_LR:-1e-6}
ACTOR_ATTENTION_IMPL=${ACTOR_ATTENTION_IMPL:-flash_attention_2}
# Chunked vocab projection: computes per-token log-probs WITHOUT materializing the
# [tokens, vocab] logits. At Qwen's 151936 vocab a 32k-token micro-batch needs 17.6 GB
# for the logits gradient alone, which OOMs the backward. GPI-CE never needs logits
# (no entropy term, no sum-pi-squared), so this is pure savings.
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-True}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}   # leave the actor room for long-sequence backward
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-2048}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-131072}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}
TRAIN_TEMPERATURE=${TRAIN_TEMPERATURE:-1.0}
TRAIN_TOP_P=${TRAIN_TOP_P:-1.0}
TRAIN_TOP_K=${TRAIN_TOP_K:--1}
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-8}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}
VALIDATION_SEEDS=${VALIDATION_SEEDS:-'[31415]'}
TRAIN_SEED=${TRAIN_SEED:-31415}

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-400}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
BEST_CKPT_SOURCES=${BEST_CKPT_SOURCES:-'["aime24","aime25","aime26","amc23"]'}
LOGGER=${LOGGER:-'["console","wandb"]'}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
CKPTS_DIR=${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}

echo "=== ${EXPERIMENT_NAME}"
echo "    K = ${GPI_ONLINE} online + ${GPI_OFFLINE} offline = ${GROUP_SIZE} | tau=${GPI_TEMPERATURE}"
echo "    B = ${TRAIN_BATCH_SIZE} prompts -> $(( TRAIN_BATCH_SIZE * GROUP_SIZE )) trained sequences,"
echo "        $(( TRAIN_BATCH_SIZE * GPI_ONLINE )) generated by vLLM"

DATA=(
    # adv_estimator is unused under GPI-CE (compute_advantage is skipped entirely);
    # it stays set because the config schema requires a valid value.
    algorithm.adv_estimator=grpo
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
    # false => SequentialSampler, which is what preserves a difficulty curriculum
    data.shuffle=${DATA_SHUFFLE}
    data.custom_cls.path=verl/experimental/gpi_ce/dataset.py
    data.custom_cls.name=GPICEDataset
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=${ACTOR_ATTENTION_IMPL}
    actor_rollout_ref.model.use_fused_kernels=${USE_FUSED_KERNELS}
)

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=gpi_ce
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${GPI_GROUPS_PER_MICRO_BATCH}
    # Token-budget micro-batching. Safe here because force_group_size=K makes
    # rearrange_micro_batches partition whole GROUPS (seqlen_balancing.py:414), so a
    # group is never cut apart; one carrying a long offline trace simply gets a
    # micro-batch to itself while short groups pack together.
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.shuffle=False
    actor_rollout_ref.actor.accumulate_minibatch_grads=True
    # q* already carries the KL trust region (the tau-weighted KL to p_old).
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
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
    # K_on, NOT the mixed group size: the offline half is appended after generation.
    actor_rollout_ref.rollout.n=${GPI_ONLINE}
    actor_rollout_ref.rollout.temperature=${TRAIN_TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${TRAIN_TOP_P}
    actor_rollout_ref.rollout.top_k=${TRAIN_TOP_K}
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    # Scoring is per-sequence with no group coupling, so dynamic batching is safe
    # here; only the ACTOR update needs group-atomic micro-batches.
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOGPROB_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_AGENT_NUM_WORKERS}
    actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE}
    +actor_rollout_ref.rollout.enable_sleep_mode=${ROLLOUT_FREE_CACHE_ENGINE}
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
    +trainer.best_ckpt_sources=${BEST_CKPT_SOURCES}
    +trainer.validation_seeds=${VALIDATION_SEEDS}
)

# Drop groups whose rewards all tie. q* == p_old exactly there, so they contribute a
# provably ZERO gradient while still costing a full forward+backward. Opt-in because
# verl's filter shrinks the batch without resampling, and the GPI-CE denominator is the
# surviving group count -- so the surviving groups get reweighted, i.e. it acts as an
# effective LR increase rather than a pure compute saving. Turn on once that is intended.
if [[ "${GPI_FILTER_DEGENERATE}" == "true" ]]; then
    GPI_FILTER=(
        +algorithm.filter_groups.enable=true
        +algorithm.filter_groups.metric=acc
    )
else
    GPI_FILTER=()
fi

GPI=(
    custom_gpi_ce.enable=true
    custom_gpi_ce.fuse_old_log_prob=${GPI_FUSE_OLD_LOG_PROB}
    custom_gpi_ce.online_per_prompt=${GPI_ONLINE}
    custom_gpi_ce.offline_per_prompt=${GPI_OFFLINE}
    custom_gpi_ce.temperature=${GPI_TEMPERATURE}
    custom_gpi_ce.score_normalization=token_mean
    custom_gpi_ce.verify_offline=${GPI_VERIFY_OFFLINE}
    custom_gpi_ce.allow_short_offline=${GPI_ALLOW_SHORT_OFFLINE}
)

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${GPI[@]}" \
    "${GPI_FILTER[@]}" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.ref.strategy=fsdp \
    critic.enable=false \
    "$@"
