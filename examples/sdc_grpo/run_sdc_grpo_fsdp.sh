#!/usr/bin/env bash
set -euo pipefail

# Match the GXPO reference run: this SDC entrypoint is fixed to physical GPUs
# 0 and 1, not whichever devices happen to be inherited by the shell.
export CUDA_VISIBLE_DEVICES=0,1

# vLLM imports matplotlib through its CLI module. Training is non-interactive;
# force a headless backend so inherited Jupyter MPLBACKEND values cannot abort
# worker startup.
export MPLBACKEND=Agg

# Example SDC-GRPO launch. Override paths and model/data settings from the
# environment to keep the script usable for baselines.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)
if [[ -z "${PYTHON_BIN:-}" && -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
    PYTHON_BIN=${PYTHON_BIN:-python3}
fi

cd "$REPO_ROOT"

# Reduce CUDA fragmentation on the colocated (vLLM + FSDP) GPUs.
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

# Keep .env settings from silently moving this reference run to another GPU
# pair.
export CUDA_VISIBLE_DEVICES=0,1

# ══════════════════════════════════════════════════════════════════════════
# RUN CONFIGURATION — edit this block to change the campaign.
# Everything the run needs is set here (env/.env still win if you export
# a variable before launching). Values below = the frozen 1.5B reference
# campaign: batch 256 x 8 rollouts, minibatch 64, 2x Blackwell 6000,
# full-parameter SDC with HF scoring. Do not add new knobs; edit values.
# ══════════════════════════════════════════════════════════════════════════
: "${MODEL_PATH:=/models/Qwen2.5-Math-1.5B-Instruct}"          # ← EDIT: local model dir
: "${GXPO_DATA_ROOT:=/data/sdc}"                                # ← EDIT: parquet root
: "${TRAIN_DAPO_FILE:=${GXPO_DATA_ROOT}/dapo_math/train.parquet}"
: "${TRAIN_LIGHTEVAL_FILE:=${GXPO_DATA_ROOT}/lighteval-math/train.parquet}"
: "${NNODES:=1}"
: "${NDEVICES_PER_NODE:=2}"

: "${TRAIN_PROMPT_BATCH_SIZE:=256}"      # prompts per step
: "${NUM_GENERATIONS:=8}"                # rollouts per prompt -> 2048 rows/step
: "${PPO_MINI_BATCH_SIZE:=64}"           # 4 optimizer slices; see note below
: "${MAX_PROMPT_LENGTH:=1024}"
: "${MAX_RESPONSE_LENGTH:=3072}"
: "${MODEL_DTYPE:=fp32}"                 # fp32 master weights, BF16 compute
: "${ACTOR_ATTENTION_IMPL:=flash_attention_2}"
: "${ROLLOUT_MAX_NUM_SEQS:=1024}"
: "${SAVE_FREQ:=20}"
: "${TEST_FREQ:=5}"

# SDC (single recipe, no variant flags). beta is the only algorithm knob.
: "${SDC_BETA:=0.5}"
: "${SDC_BASE_MIX_GAMMA:=0.9}"
: "${SDC_MIN_SFT_UPDATES_BEFORE_USE:=2}"
: "${SDC_SFT_UPDATE_INTERVAL:=5}"
: "${SDC_CHECKPOINT_INTERVAL:=20}"
: "${SDC_SFT_LR:=1.0e-5}"
: "${SDC_DATA_MAX_SIZE:=8192}"
: "${SDC_SCORING_BACKEND:=hf}"           # hf = full-parameter teachers
: "${SDC_DISTRIBUTED_METRICS:=false}"
# Performance-only toggles; flip after confirming VRAM headroom:
: "${SDC_SIDECAR_RESIDENCY:=auto_offload}"
: "${SDC_SIDECAR_GRADIENT_CHECKPOINTING:=}"
# ══════════════════════════════════════════════════════════════════════════

# Use the same local model and prepared parquet assets as the GXPO reference
# launcher. These paths remain overridable for another machine.
GXPO_REFERENCE_ROOT=${GXPO_REFERENCE_ROOT:-${WORKSPACE_ROOT}/gradient-extrapolation-based-policy-optimization-gxpo-speed-audit}
GXPO_DATA_ROOT=${GXPO_DATA_ROOT:-${GXPO_REFERENCE_ROOT}/Code/SFPO/data}

# Reuse the verified GXPO CUDA/FlashAttention2/Liger runtime. The SDC venv
# contains the matching CUDA 13 libraries, while the GXPO venv contains the
# tested FlashAttention2 and Liger Python packages. Keep SDC's own site-packages
# first so its PyTorch/Transformers versions remain authoritative.
LOCAL_CUDA_LIB_ROOT=${LOCAL_CUDA_LIB_ROOT:-${REPO_ROOT}/.venv/lib/python3.12/site-packages/nvidia}
for LOCAL_CUDA_LIB_DIR in \
    "${LOCAL_CUDA_LIB_ROOT}/cu13/lib" \
    "${LOCAL_CUDA_LIB_ROOT}/cublas/lib" \
    "${LOCAL_CUDA_LIB_ROOT}/cuda_cupti/lib" \
    "${LOCAL_CUDA_LIB_ROOT}/cuda_nvrtc/lib" \
    "${LOCAL_CUDA_LIB_ROOT}/cuda_runtime/lib" \
    "${LOCAL_CUDA_LIB_ROOT}/cudnn/lib" \
    "${LOCAL_CUDA_LIB_ROOT}/cufft/lib" \
    "${LOCAL_CUDA_LIB_ROOT}/curand/lib" \
    "${LOCAL_CUDA_LIB_ROOT}/cusolver/lib" \
    "${LOCAL_CUDA_LIB_ROOT}/cusparse/lib" \
    "${LOCAL_CUDA_LIB_ROOT}/nccl/lib" \
    "${LOCAL_CUDA_LIB_ROOT}/nvjitlink/lib" \
    "${LOCAL_CUDA_LIB_ROOT}/nvshmem/lib" \
    "${LOCAL_CUDA_LIB_ROOT}/nvtx/lib"; do
    if [[ -d "${LOCAL_CUDA_LIB_DIR}" ]]; then
        export LD_LIBRARY_PATH="${LOCAL_CUDA_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
done
GXPO_SITE_PACKAGES=${GXPO_SITE_PACKAGES:-${GXPO_REFERENCE_ROOT}/.venv/lib/python3.12/site-packages}
SDC_SITE_PACKAGES=${SDC_SITE_PACKAGES:-${REPO_ROOT}/.venv/lib/python3.12/site-packages}

# Resolve FlashAttention2 + Liger: prefer the verified GXPO bundle, fall back
# to the SDC venv's own packages (e.g. a self-contained Blackwell install),
# and only fail when NEITHER location provides both packages.
_EXTRA_PYTHONPATH=""
if [[ -d "${GXPO_SITE_PACKAGES}/flash_attn" && -d "${GXPO_SITE_PACKAGES}/liger_kernel" ]]; then
    _EXTRA_PYTHONPATH="${GXPO_SITE_PACKAGES}"
elif [[ -d "${SDC_SITE_PACKAGES}/flash_attn" && -d "${SDC_SITE_PACKAGES}/liger_kernel" ]]; then
    echo "Using FlashAttention2/Liger from the SDC venv (${SDC_SITE_PACKAGES}); GXPO bundle not found."
elif [[ -d "${GXPO_SITE_PACKAGES}/flash_attn" || -d "${SDC_SITE_PACKAGES}/flash_attn" ]] && [[ ! -d "${GXPO_SITE_PACKAGES}/liger_kernel" && ! -d "${SDC_SITE_PACKAGES}/liger_kernel" ]]; then
    echo "WARNING: flash_attn found but liger_kernel missing; disabling Liger kernels." >&2
    USE_LIGER=false
else
    echo "ERROR: no FlashAttention2 package found in either of:" >&2
    echo "  - ${GXPO_SITE_PACKAGES}" >&2
    echo "  - ${SDC_SITE_PACKAGES}" >&2
    echo "Build FA2 for sm_120 (see BLACKWELL_SETUP.md) or point GXPO_SITE_PACKAGES at a bundle that has it." >&2
    exit 2
fi
export VERL_EXTRA_PYTHONPATH="${_EXTRA_PYTHONPATH}"
if [[ -n "${_EXTRA_PYTHONPATH}" ]]; then
    export PYTHONPATH="${SDC_SITE_PACKAGES}:${_EXTRA_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
fi
TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${REPO_ROOT}/.cache/triton}
export TRITON_CACHE_DIR
mkdir -p "${TRITON_CACHE_DIR}"

GXPO_CUDA_HOME=${GXPO_CUDA_HOME:-${GXPO_REFERENCE_ROOT}/.cuda-toolkit}
if [[ -x "${GXPO_CUDA_HOME}/bin/nvcc" ]]; then
    export CUDA_HOME="${GXPO_CUDA_HOME}"
    export CUDA_PATH="${GXPO_CUDA_HOME}"
    export CUDACXX="${GXPO_CUDA_HOME}/bin/nvcc"
    export PATH="${GXPO_CUDA_HOME}/bin:${PATH}"
fi

# Triton JIT-compiles a small CUDA driver helper when vLLM starts. The host
# Python 3.12 runtime omits system development headers, so reuse the matching
# local header bundle already staged with the GXPO environment.
PYTHON_DEV_INCLUDE_ROOT=${PYTHON_DEV_INCLUDE_ROOT:-${GXPO_REFERENCE_ROOT}/.python-dev/usr/include}
if [[ -f "${PYTHON_DEV_INCLUDE_ROOT}/python3.12/Python.h" ]]; then
    export CPATH="${PYTHON_DEV_INCLUDE_ROOT}/python3.12:${PYTHON_DEV_INCLUDE_ROOT}${CPATH:+:${CPATH}}"
fi

PROJECT_NAME=${PROJECT_NAME:-verl_sdc_deepscaler}
export WANDB_PROJECT=${WANDB_PROJECT:-${PROJECT_NAME}}
export WANDB_SILENT=${WANDB_SILENT:-true}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}
# Defaults for anything the RUN CONFIGURATION block above left unset.
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sdc_${BASE_ALGO:-drgrpo}_qwen25math_1_5b-${RUN_TIMESTAMP}}
MODEL_PATH=${MODEL_PATH:-${GXPO_REFERENCE_ROOT}/models/Qwen2.5-Math-1.5B-Instruct}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-1}

# Match GXPO's exact prepared training and validation assets.
TRAIN_DATASET=${TRAIN_DATASET:-haizhongzheng/DAPO-Math-17K-cleaned}
TRAIN_DATASET_CONFIG=${TRAIN_DATASET_CONFIG:-default}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
TRAIN_DAPO_FILE=${TRAIN_DAPO_FILE:-${GXPO_DATA_ROOT}/dapo_math/train.parquet}
TRAIN_LIGHTEVAL_FILE=${TRAIN_LIGHTEVAL_FILE:-${GXPO_DATA_ROOT}/lighteval-math/train.parquet}
TRAIN_FILE=${TRAIN_FILE:-${TRAIN_DAPO_FILE}}
TRAIN_FILES=${TRAIN_FILES:-"[${TRAIN_DAPO_FILE},${TRAIN_LIGHTEVAL_FILE}]"}
PREPARE_TRAIN_DATA=${PREPARE_TRAIN_DATA:-false}

# The GXPO benchmark parquets disagree on nested ground-truth types and some
# shared auxiliary columns, which Hugging Face Datasets cannot concatenate.
# These cached copies retain the RL fields while normalizing the schema and
# canonical data_source names used by validation and best-checkpoint selection.
SDC_VAL_CACHE_ROOT=${SDC_VAL_CACHE_ROOT:-${REPO_ROOT}/data/sdc_validation_normalized}
MATH500_FILE=${MATH500_FILE:-${SDC_VAL_CACHE_ROOT}/math500.parquet}
AIME24_FILE=${AIME24_FILE:-${SDC_VAL_CACHE_ROOT}/aime2024.parquet}
AIME25_FILE=${AIME25_FILE:-${SDC_VAL_CACHE_ROOT}/aime2025.parquet}
AMC23_FILE=${AMC23_FILE:-${SDC_VAL_CACHE_ROOT}/amc23.parquet}
MINERVA_FILE=${MINERVA_FILE:-${SDC_VAL_CACHE_ROOT}/minervamath.parquet}
OLYMPIADBENCH_FILE=${OLYMPIADBENCH_FILE:-${SDC_VAL_CACHE_ROOT}/olympiadbench.parquet}
if [[ -z "${VAL_FILES:-}" ]]; then
    VAL_FILES="[${MATH500_FILE},${AIME24_FILE},${AIME25_FILE},${AMC23_FILE},${MINERVA_FILE},${OLYMPIADBENCH_FILE}]"
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

for required_file in \
    "${TRAIN_DAPO_FILE}" "${TRAIN_LIGHTEVAL_FILE}" \
    "${MATH500_FILE}" "${AIME24_FILE}" "${AIME25_FILE}" \
    "${AMC23_FILE}" "${MINERVA_FILE}" "${OLYMPIADBENCH_FILE}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: missing dataset file: ${required_file}" >&2
        echo "Override the defaults instead of editing the launcher, e.g.:" >&2
        echo "  TRAIN_DAPO_FILE=/path/train.parquet TRAIN_LIGHTEVAL_FILE=/path/train.parquet \\" >&2
        echo "  SDC_VAL_CACHE_ROOT=/dir/with-math500-aime24-aime25-amc23-minervamath-olympiadbench-parquets \\" >&2
        echo "  bash examples/sdc_grpo/run_sdc_grpo_fsdp.sh" >&2
        exit 2
    fi
done

# Model preflight: fail with the fix in the message, not deep inside HF hub code.
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    echo "Override it, e.g.: MODEL_PATH=/models/Qwen2.5-Math-1.5B-Instruct bash examples/sdc_grpo/run_sdc_grpo_fsdp.sh" >&2
    exit 2
fi

# ── SDC-GRPO hyperparameters ────────────────────────────────────────────────
SDC_ENABLE=${SDC_ENABLE:-true}
SDC_BETA=${SDC_BETA:-0.5}
SDC_SCORING_BACKEND=${SDC_SCORING_BACKEND:-}
SDC_VLLM_SCORE_MICRO_BATCH_SIZE=${SDC_VLLM_SCORE_MICRO_BATCH_SIZE:-16}
SDC_REUSE_ROLLOUT_LOGPROBS=${SDC_REUSE_ROLLOUT_LOGPROBS:-}
SDC_VERIFY_ROLLOUT_LOGPROBS=${SDC_VERIFY_ROLLOUT_LOGPROBS:-false}
SDC_USE_IMPORTANCE_WEIGHT=${SDC_USE_IMPORTANCE_WEIGHT:-true}
SDC_IMPORTANCE_WEIGHT_CLIP=${SDC_IMPORTANCE_WEIGHT_CLIP:-5.0}
SDC_BASE_MIX_GAMMA=${SDC_BASE_MIX_GAMMA:-0.9}
# Let both outcome models see two balanced refreshes before their log-prob
# contrast contributes to the actor loss.
SDC_MIN_SFT_UPDATES_BEFORE_USE=${SDC_MIN_SFT_UPDATES_BEFORE_USE:-2}
SDC_SFT_UPDATE_INTERVAL=${SDC_SFT_UPDATE_INTERVAL:-5}       # run SFT every N policy steps
SDC_CHECKPOINT_INTERVAL=${SDC_CHECKPOINT_INTERVAL:-20}
SDC_SFT_LR=${SDC_SFT_LR:-1.0e-5}
SDC_SFT_BATCH_SIZE=${SDC_SFT_BATCH_SIZE:-8}
# 8 rows * (1024 prompt + 3072 response) tokens. The implementation uses the
# worst-case padded length to derive the effective SFT batch size.
SDC_SFT_MAX_TOKEN_LEN_PER_GPU=${SDC_SFT_MAX_TOKEN_LEN_PER_GPU:-32768}
SDC_SFT_MAX_UPDATES=${SDC_SFT_MAX_UPDATES:-1}
SDC_SAVE_TO_DISK_INTERVAL=${SDC_SAVE_TO_DISK_INTERVAL:-0}
SDC_DATA_MAX_SIZE=${SDC_DATA_MAX_SIZE:-8192}
SDC_DATA_SAMPLING=${SDC_DATA_SAMPLING:-recent}
SDC_FUSED_ADAMW=${SDC_FUSED_ADAMW:-true}
# Sidecar residency: '' (unset) -> library default 'auto_offload' (per-pass CPU
# offload, historical behavior). 'resident' keeps the SDC clones on the actor
# GPU across scoring/SFT; only use it with a trainer that calls
# sdc_release_residency() before vLLM wake-up.
SDC_SIDECAR_RESIDENCY=${SDC_SIDECAR_RESIDENCY:-}
# Sidecar gradient checkpointing: '' (unset) -> library default null (inherit
# the actor setting). 'true'/'false' force it on/off for the SDC clones only.
SDC_SIDECAR_GRADIENT_CHECKPOINTING=${SDC_SIDECAR_GRADIENT_CHECKPOINTING:-}


# Match the GXPO actor optimization stack. These are explicit launcher
# settings rather than relying on model/config defaults.
USE_LIGER=${USE_LIGER:-true}
# The measured SDC run was slower with the custom Triton linear-CE path than
# the standard BF16 logits path. Keep it available as an explicit A/B option,
# but use the verified standard path by default for throughput and stability.
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-false}
FUSED_KERNEL_BACKEND=${FUSED_KERNEL_BACKEND:-triton}
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-true}
OPTIM_FUSED=${OPTIM_FUSED:-true}
# Exact cross-rank metric reductions inside every actor microbatch are useful
# for debugging but add synchronization to the hot path.
SDC_DISTRIBUTED_METRICS=${SDC_DISTRIBUTED_METRICS:-false}

TRAIN_PROMPT_BATCH_SIZE=${TRAIN_PROMPT_BATCH_SIZE:-256}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${TRAIN_PROMPT_BATCH_SIZE}}
# Minibatch 64 = 4 optimizer slices over the 2048-row logical batch.
# NOTE: this disables the fused old-log-prob fast path (it requires
# ppo_mini_batch_size * rollout.n >= batch rows; 64*8=512 < 2048), so each
# step pays one separate old-policy forward. Set 256 to restore the fusion.
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
FUSE_OLD_LOG_PROB=${FUSE_OLD_LOG_PROB:-true}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
# Larger per-GPU token budget reduces dynamic microbatching and FSDP synchronization overhead on 98 GiB GPUs.
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-49152}
# Use the same verified FlashAttention2 path as GXPO. The CUDA 13 runtime
# libraries and package path are configured above before Ray workers start.
ACTOR_ATTENTION_IMPL=${ACTOR_ATTENTION_IMPL:-flash_attention_2}
# Match GXPO's FSDP1 setup: keep master parameters in FP32 while FSDP's
# mixed-precision policy performs BF16 forward/reduction work.
MODEL_DTYPE=${MODEL_DTYPE:-fp32}
DRGRPO_USE_LORA=${DRGRPO_USE_LORA:-false}
LORA_RANK=${LORA_RANK:-512}
LORA_ALPHA=${LORA_ALPHA:-1024}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}

if [[ -z "${SDC_SCORING_BACKEND}" ]]; then
    if [[ "${DRGRPO_USE_LORA}" == "true" ]]; then
        SDC_SCORING_BACKEND=vllm
    else
        SDC_SCORING_BACKEND=hf
    fi
fi
if [[ "${SDC_SCORING_BACKEND}" == "vllm" && "${DRGRPO_USE_LORA}" != "true" ]]; then
    echo "SDC_SCORING_BACKEND=vllm requires DRGRPO_USE_LORA=true; use hf for full fine-tuning." >&2
    exit 2
fi

if [[ -z "${SDC_REUSE_ROLLOUT_LOGPROBS}" ]]; then
    if [[ "${SDC_SCORING_BACKEND}" == "vllm" ]]; then
        SDC_REUSE_ROLLOUT_LOGPROBS=true
    else
        SDC_REUSE_ROLLOUT_LOGPROBS=false
    fi
fi
if [[ "${SDC_SCORING_BACKEND}" == "hf" && "${SDC_REUSE_ROLLOUT_LOGPROBS}" == "true" ]]; then
    echo "HF SDC scoring requires SDC_REUSE_ROLLOUT_LOGPROBS=false so all SDC references use HF log-probs." >&2
    exit 2
fi
if [[ "${SDC_SCORING_BACKEND}" == "vllm" && "${SDC_REUSE_ROLLOUT_LOGPROBS}" != "true" ]]; then
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
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-98304}
if [[ -z "${ROLLOUT_MAX_LORAS:-}" ]]; then
    ROLLOUT_MAX_LORAS=3
fi
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-1}
# Validation is sent as one logical dataset batch by the trainer; numeric
# val_batch_size is deprecated and does not split these benchmark files.
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-null}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-False}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0}
VAL_TOP_P=${VAL_TOP_P:-1.0}
VALIDATION_SEEDS=${VALIDATION_SEEDS:-'[0]'}
TRAIN_SEED=${TRAIN_SEED:-3407}
TRAIN_TEMPERATURE=${TRAIN_TEMPERATURE:-1.0}
TRAIN_TOP_P=${TRAIN_TOP_P:-1.0}
TRAIN_TOP_K=${TRAIN_TOP_K:--1}

# Wasserstein guidance
WG_ENABLE=${WG_ENABLE:-false}
WG_LAMBDA=${WG_LAMBDA:-0.01}
WG_ALPHA=${WG_ALPHA:-0.2}
WG_EMBED_MODEL=${WG_EMBED_MODEL:-BAAI/bge-small-en-v1.5}
WG_DECODE_MODEL=${WG_DECODE_MODEL:-${MODEL_PATH}}  # use actor tokenizer to decode response token IDs

MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-400}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
# Select checkpoints over the same six validation benchmarks as GXPO.
BEST_CKPT_SOURCES=${BEST_CKPT_SOURCES:-'["math500","aime24","aime25","amc23","minervamath","olympiadbench"]'}
# Selection metric(s), e.g. '["pass@1"]'. Unset keeps the trainer default ("combined",
# = 0.5*(avg@k + pass@k)), which needs BOTH keys -- so a greedy n=1 run must set this
# to pass@1 or no best/ checkpoint is ever written.
# Greedy n=1 validation has a well-defined pass@1 metric.
BEST_CKPT_METRICS=${BEST_CKPT_METRICS:-'["pass@1"]'}
LOGGER=${LOGGER:-'["console","wandb"]'}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-${MAX_OPTIMIZER_STEPS}}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-null}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-null}
CKPTS_DIR=${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}

BASE_ALGO=${BASE_ALGO:-drgrpo}
case "${BASE_ALGO}" in
    grpo)   BASE_ADV_ESTIMATOR=grpo;  BASE_NORM=true;  BASE_LOSS_MODE=vanilla; BASE_CLIP_LOW=${CLIP_RATIO}; BASE_CLIP_HIGH=${CLIP_RATIO} ;;
    drgrpo) BASE_ADV_ESTIMATOR=grpo;  BASE_NORM=false; BASE_LOSS_MODE=vanilla; BASE_CLIP_LOW=${CLIP_RATIO}; BASE_CLIP_HIGH=${CLIP_RATIO} ;;
    dapo)   BASE_ADV_ESTIMATOR=grpo;  BASE_NORM=true;  BASE_LOSS_MODE=vanilla; BASE_CLIP_LOW=0.2; BASE_CLIP_HIGH=0.28 ;;
    ngrpo)  BASE_ADV_ESTIMATOR=ngrpo; BASE_NORM=true;  BASE_LOSS_MODE=ngrpo;  BASE_CLIP_LOW=0.16; BASE_CLIP_HIGH=0.24 ;;
    avspo)  BASE_ADV_ESTIMATOR=avspo; BASE_NORM=true;  BASE_LOSS_MODE=vanilla; BASE_CLIP_LOW=${CLIP_RATIO}; BASE_CLIP_HIGH=${CLIP_RATIO} ;;
    *) echo "BASE_ALGO must be grpo, drgrpo, dapo, ngrpo, or avspo" >&2; exit 2 ;;
esac

DATA=(
    algorithm.base_rl_name=${BASE_ALGO}
    algorithm.adv_estimator=${BASE_ADV_ESTIMATOR}
    algorithm.norm_adv_by_std_in_grpo=${BASE_NORM}
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILES"
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
    actor_rollout_ref.model.use_liger=${USE_LIGER}
    actor_rollout_ref.model.use_fused_kernels=${USE_FUSED_KERNELS}
    actor_rollout_ref.model.fused_kernel_options.impl_backend=${FUSED_KERNEL_BACKEND}
    +actor_rollout_ref.model.override_config.attn_implementation=${ACTOR_ATTENTION_IMPL}
)

if [[ "${DRGRPO_USE_LORA}" == "true" ]]; then
    MODEL+=(
        actor_rollout_ref.model.lora_rank=${LORA_RANK}
        actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
        actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}
    )
else
    # Keep the full-finetuning mode explicit instead of relying on the model
    # config's current default. The actor uses FSDP1 below; SDC clones and
    # updates every success/failure parameter when lora_rank is zero.
    MODEL+=(actor_rollout_ref.model.lora_rank=0)
fi

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=${BASE_LOSS_MODE}
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO}
    actor_rollout_ref.actor.clip_ratio_low=${BASE_CLIP_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${BASE_CLIP_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.fused=${OPTIM_FUSED}
    actor_rollout_ref.actor.ppo_loss_coef=${PPO_LOSS_COEF}
    actor_rollout_ref.actor.use_torch_compile=${USE_TORCH_COMPILE}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    +actor_rollout_ref.actor.fuse_old_log_prob=${FUSE_OLD_LOG_PROB}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.model_dtype=${MODEL_DTYPE}
    actor_rollout_ref.actor.data_loader_seed=${TRAIN_SEED}
    algorithm.ngrpo_r_max=1.0
    algorithm.ngrpo_epsilon_pos=0.24
    algorithm.ngrpo_epsilon_neg=0.16
    algorithm.avspo_collapse_tau=1e-6
    algorithm.avspo_alpha=0.5
    algorithm.avspo_anchor_reward=0.1
    algorithm.avspo_adapt_tau_init=0.5
    algorithm.avspo_adapt_eta=0.01
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
if [[ "${SDC_SCORING_BACKEND}" == "vllm" && "${SDC_REUSE_ROLLOUT_LOGPROBS}" == "true" ]]; then
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
    custom_sdc.enable="${SDC_ENABLE}"
    custom_sdc.scoring_backend="${SDC_SCORING_BACKEND}"
    custom_sdc.vllm_score_micro_batch_size="${SDC_VLLM_SCORE_MICRO_BATCH_SIZE}"
    custom_sdc.beta="${SDC_BETA}"
    custom_sdc.reuse_rollout_log_probs="${SDC_REUSE_ROLLOUT_LOGPROBS}"
    custom_sdc.verify_rollout_log_probs="${SDC_VERIFY_ROLLOUT_LOGPROBS}"
    custom_sdc.use_importance_weight="${SDC_USE_IMPORTANCE_WEIGHT}"
    custom_sdc.importance_weight_clip="${SDC_IMPORTANCE_WEIGHT_CLIP}"
    custom_sdc.base_mix_gamma="${SDC_BASE_MIX_GAMMA}"
    custom_sdc.min_sft_updates_before_use="${SDC_MIN_SFT_UPDATES_BEFORE_USE}"
    custom_sdc.sft_update_interval_policy_steps="${SDC_SFT_UPDATE_INTERVAL}"
    custom_sdc.checkpoint_interval_policy_steps="${SDC_CHECKPOINT_INTERVAL}"
    custom_sdc.sft_lr="${SDC_SFT_LR}"
    custom_sdc.sft_batch_size="${SDC_SFT_BATCH_SIZE}"
    custom_sdc.sft_max_token_len_per_gpu="${SDC_SFT_MAX_TOKEN_LEN_PER_GPU}"
    custom_sdc.sft_max_updates_per_interval="${SDC_SFT_MAX_UPDATES}"
    custom_sdc.save_to_disk_interval_policy_steps="${SDC_SAVE_TO_DISK_INTERVAL}"
    custom_sdc.data_max_size="${SDC_DATA_MAX_SIZE}"
    custom_sdc.data_sampling="${SDC_DATA_SAMPLING}"
    custom_sdc.fused_adamw="${SDC_FUSED_ADAMW}"
    custom_sdc.distributed_metrics="${SDC_DISTRIBUTED_METRICS}"
)

if [[ -n "${SDC_SIDECAR_RESIDENCY}" ]]; then
    SDC+=(
        custom_sdc.sidecar_residency="${SDC_SIDECAR_RESIDENCY}"
    )
fi
if [[ -n "${SDC_SIDECAR_GRADIENT_CHECKPOINTING}" ]]; then
    SDC+=(
        custom_sdc.sidecar_gradient_checkpointing="${SDC_SIDECAR_GRADIENT_CHECKPOINTING}"
    )
fi


WASSERSTEIN=(
    actor_rollout_ref.actor.wasserstein_guidance.enable=${WG_ENABLE}
    actor_rollout_ref.actor.wasserstein_guidance.lambda_wg=${WG_LAMBDA}
    actor_rollout_ref.actor.wasserstein_guidance.alpha_transport=${WG_ALPHA}
    actor_rollout_ref.actor.wasserstein_guidance.embed_model=${WG_EMBED_MODEL}
    actor_rollout_ref.actor.wasserstein_guidance.decode_model="${WG_DECODE_MODEL}"
)

if [[ "${BASE_ALGO}" == "dapo" ]]; then
    DATA+=(
        algorithm.filter_groups.enable=true
        algorithm.filter_groups.metric=acc
        algorithm.filter_groups.max_num_gen_batches=0
    )
fi

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
