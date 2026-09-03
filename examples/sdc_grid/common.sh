#!/usr/bin/env bash
#
# examples/sdc_grid/common.sh
#
# Shared launcher for the SDC / SDC-TR comparison grid.
#
#   3 bases  x  3 modes  =  9 cells   (sdc_tr additionally sweeps 4 alphas)
#
#     BASE_ALGO : drgrpo | dapo | avspo     (advantage estimator + clip range)
#     SDC_MODE  : base   | sdc  | sdc_tr    (no SDC / additive SDC / trust-region SDC)
#
# ALL fairness-critical settings live HERE. The entrypoints under base/, sdc/, and
# sdc_tr/ select only BASE_ALGO, SDC_MODE, and (for sdc_tr) SDC_TR_ALPHA. Anything
# exported before the launcher still wins, so sweeps do not require editing files.
#
# Every learning-relevant hyperparameter is inherited from the GXPO reference run
#   /workspace/final-gxpo/Code/SFPO/experiments/gxpo_efficiency/
#       qwen25_math_1p5b_gxpo_b256_mb64_gate_v6.sh  ->  common.sh
# so that the SDC term is the only thing that varies. The handful of settings that
# CANNOT be identical (the two repos are different verl versions) are enumerated in
# the SDC_GRID_DIVERGENCES block below and asserted by
# tests/trainer/test_sdc_grid_launchers_on_cpu.py.
#
# Usage (always via an entrypoint, never directly):
#   bash examples/sdc_grid/base/drgrpo.sh                # launch
#   bash examples/sdc_grid/base/drgrpo.sh --dry-run      # print resolved config, no launch
#   SDC_BETA=1.0 bash examples/sdc_grid/sdc/dapo.sh      # override anything
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && { DRY_RUN=1; shift; }

cd "$REPO_ROOT"

# ═══════════════════════════════════════════════════════════════════════════════
# ARM SELECTION (set by the entrypoint)
# ═══════════════════════════════════════════════════════════════════════════════
BASE_ALGO="${BASE_ALGO:-}"
SDC_MODE="${SDC_MODE:-}"

if [[ -z "$BASE_ALGO" || -z "$SDC_MODE" ]]; then
    echo "ERROR: common.sh is not meant to be run directly." >&2
    echo "  Use an entrypoint, e.g. bash examples/sdc_grid/base/drgrpo.sh" >&2
    exit 2
fi

case "$BASE_ALGO" in
    drgrpo|dapo|avspo) ;;
    ngrpo)
        # NGRPO needs policy_loss.loss_mode=ngrpo, which SDC-TR replaces. It is
        # excluded from this grid rather than silently producing a cell whose base
        # objective is not the one its name claims.
        echo "ERROR: BASE_ALGO=ngrpo is not part of this grid." >&2
        echo "  NGRPO requires policy_loss.loss_mode=ngrpo, which SDC_MODE=sdc_tr" >&2
        echo "  would overwrite. Use examples/sdc/presets/ngrpo.sh for NGRPO work." >&2
        exit 2
        ;;
    *) echo "ERROR: BASE_ALGO must be drgrpo, dapo, or avspo (got '$BASE_ALGO')" >&2; exit 2 ;;
esac

case "$SDC_MODE" in
    base|sdc|sdc_tr) ;;
    *) echo "ERROR: SDC_MODE must be base, sdc, or sdc_tr (got '$SDC_MODE')" >&2; exit 2 ;;
esac

# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME: interpreter, secrets, caches
# ═══════════════════════════════════════════════════════════════════════════════
# The GXPO H200 venv holds the validated CUDA 13 / FlashAttention-3 / FlashInfer /
# Liger stack. verl itself is resolved from THIS repo's cwd (see the guard below),
# not from that venv's editable install.
H200_REPO_ROOT="${H200_REPO_ROOT:-${WORKSPACE_ROOT}/gradient-extrapolation-based-policy-optimization}"
H200_VENV_ROOT="${H200_VENV_ROOT:-${H200_REPO_ROOT}/.venv-h200}"
if [[ -z "${PYTHON_BIN:-}" && -x "${H200_VENV_ROOT}/bin/python" ]]; then
    PYTHON_BIN="${H200_VENV_ROOT}/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

# A stray ~/.local scipy/numpy shadows the venv and dies with
# "module 'numpy' has no attribute 'long'". Never inherit user site-packages.
export PYTHONNOUSERSITE=1

# Checkout-local secrets (WANDB_API_KEY). Sourced quietly; never printed.
if [[ -f "${WORKSPACE_ROOT}/.env" || -f "${REPO_ROOT}/.env" ]]; then
    set -a
    for _ENV_FILE in "${WORKSPACE_ROOT}/.env" "${REPO_ROOT}/.env"; do
        [[ -f "${_ENV_FILE}" ]] && source "${_ENV_FILE}"
    done
    set +a
    unset _ENV_FILE
fi

# The image exports an invalid bare host (RAY_ADDRESS=127.0.0.1 with no port), which
# makes ray.init() die with "Invalid address format". Let Ray create a local head.
unset RAY_ADDRESS

# .env must not be able to silently move this run onto another GPU set.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MPLBACKEND=Agg
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
# Expandable segments cut allocator fragmentation across the alternating
# actor / vLLM / SDC-sidecar phases.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"

# CUDA 13 runtime libraries staged with the H200 environment.
LOCAL_CUDA_LIB_ROOT="${LOCAL_CUDA_LIB_ROOT:-/usr/local/lib/python3.12/dist-packages/nvidia}"
for _lib in cu13 cublas cuda_cupti cuda_nvrtc cuda_runtime cudnn cufft curand \
            cusolver cusparse nccl nvjitlink nvshmem nvtx; do
    if [[ -d "${LOCAL_CUDA_LIB_ROOT}/${_lib}/lib" ]]; then
        export LD_LIBRARY_PATH="${LOCAL_CUDA_LIB_ROOT}/${_lib}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
done
unset _lib

H200_SITE_PACKAGES="${H200_SITE_PACKAGES:-${H200_VENV_ROOT}/lib/python3.12/site-packages}"

GXPO_CUDA_HOME="${GXPO_CUDA_HOME:-${H200_REPO_ROOT}/.cuda-toolkit}"
if [[ -x "${GXPO_CUDA_HOME}/bin/nvcc" ]]; then
    export CUDA_HOME="${GXPO_CUDA_HOME}"
    export CUDA_PATH="${GXPO_CUDA_HOME}"
    export CUDACXX="${GXPO_CUDA_HOME}/bin/nvcc"
    export PATH="${GXPO_CUDA_HOME}/bin:${PATH}"
fi

# Triton JIT-compiles a CUDA driver helper at vLLM startup and needs Python headers,
# which the host 3.12 runtime omits.
PYTHON_DEV_INCLUDE_ROOT="${PYTHON_DEV_INCLUDE_ROOT:-${H200_REPO_ROOT}/.python-dev/usr/include}"
if [[ -f "${PYTHON_DEV_INCLUDE_ROOT}/python3.12/Python.h" ]]; then
    export CPATH="${PYTHON_DEV_INCLUDE_ROOT}/python3.12:${PYTHON_DEV_INCLUDE_ROOT}${CPATH:+:${CPATH}}"
fi

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${REPO_ROOT}/.cache/triton}"
mkdir -p "${TRITON_CACHE_DIR}"

# ═══════════════════════════════════════════════════════════════════════════════
# ASSETS: model and prepared parquets (identical to the GXPO reference run)
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen2.5-Math-1.5B-Instruct}"
DATA_ROOT="${GXPO_DATA_ROOT:-/workspace/data}"

TRAIN_DAPO_FILE="${TRAIN_DAPO_FILE:-${DATA_ROOT}/dapo_math/train.parquet}"
TRAIN_LIGHTEVAL_FILE="${TRAIN_LIGHTEVAL_FILE:-${DATA_ROOT}/lighteval-math/train.parquet}"

# The raw GXPO benchmark parquets disagree on nested ground-truth types and some
# auxiliary columns, which HF Datasets cannot concatenate. These cached copies keep
# the RL fields while normalizing the schema and the canonical data_source names used
# by validation and best-checkpoint selection.
SDC_VAL_CACHE_ROOT="${SDC_VAL_CACHE_ROOT:-${DATA_ROOT}/sdc_validation_normalized}"
MATH500_FILE="${MATH500_FILE:-${SDC_VAL_CACHE_ROOT}/math500.parquet}"
AIME24_FILE="${AIME24_FILE:-${SDC_VAL_CACHE_ROOT}/aime2024.parquet}"
AIME25_FILE="${AIME25_FILE:-${SDC_VAL_CACHE_ROOT}/aime2025.parquet}"
AMC23_FILE="${AMC23_FILE:-${SDC_VAL_CACHE_ROOT}/amc23.parquet}"
MINERVA_FILE="${MINERVA_FILE:-${SDC_VAL_CACHE_ROOT}/minervamath.parquet}"
OLYMPIADBENCH_FILE="${OLYMPIADBENCH_FILE:-${SDC_VAL_CACHE_ROOT}/olympiadbench.parquet}"

TRAIN_FILES="${TRAIN_FILES:-[${TRAIN_DAPO_FILE},${TRAIN_LIGHTEVAL_FILE}]}"
VAL_FILES="${VAL_FILES:-[${MATH500_FILE},${AIME24_FILE},${AIME25_FILE},${AMC23_FILE},${MINERVA_FILE},${OLYMPIADBENCH_FILE}]}"

# ═══════════════════════════════════════════════════════════════════════════════
# INHERITED FROM GXPO — identical across all 18 arms.
#
# Source of truth: the resolved command line of the live GXPO run
# (tmux qwen25-1p5b-throughput0902). Do not change a value here without changing it
# in the GXPO launcher too, or the comparison stops being apples-to-apples.
# ═══════════════════════════════════════════════════════════════════════════════
# -- data ----------------------------------------------------------------------
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
TRAIN_SEED="${TRAIN_SEED:-3407}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"

# -- model ---------------------------------------------------------------------
ACTOR_ATTENTION_IMPL="${ACTOR_ATTENTION_IMPL:-flash_attention_3}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"     # H200-native; FA3 requires BF16/FP16
USE_LIGER="${USE_LIGER:-true}"

# -- optimizer -----------------------------------------------------------------
ACTOR_LR="${ACTOR_LR:-1e-6}"
OPTIM_FUSED="${OPTIM_FUSED:-true}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-true}"

# -- actor ---------------------------------------------------------------------
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-49152}"
CLIP_RATIO="${CLIP_RATIO:-0.2}"
CLIP_RATIO_C="${CLIP_RATIO_C:-3.0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"

# -- KL (GXPO runs with the actor KL loss ON) ----------------------------------
ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-True}"
ACTOR_KL_LOSS_COEF="${ACTOR_KL_LOSS_COEF:-0.01}"
ACTOR_KL_LOSS_TYPE="${ACTOR_KL_LOSS_TYPE:-low_var_kl}"

# -- parallelism ---------------------------------------------------------------
NNODES="${NNODES:-1}"
NDEVICES_PER_NODE="${NDEVICES_PER_NODE:-4}"
FSDP_SIZE="${FSDP_SIZE:-4}"

# -- rollout (training) --------------------------------------------------------
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-1.0}"
TRAIN_TOP_P="${TRAIN_TOP_P:-1.0}"
TRAIN_TOP_K="${TRAIN_TOP_K:--1}"

# -- rollout (validation): greedy, matching GXPO -------------------------------
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-False}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0}"
VAL_TOP_P="${VAL_TOP_P:-1.0}"
VAL_TOP_K="${VAL_TOP_K:--1}"
VALIDATION_SEEDS="${VALIDATION_SEEDS:-[0]}"

# -- trainer -------------------------------------------------------------------
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
SAVE_FREQ="${SAVE_FREQ:-25}"
TEST_FREQ="${TEST_FREQ:-5}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"

# ═══════════════════════════════════════════════════════════════════════════════
# SDC_GRID_DIVERGENCES — settings that deliberately DIFFER from GXPO.
#
# Format: gxpo_key|gxpo_value|sdc_value|reason
# Parsed verbatim by tests/trainer/test_sdc_grid_launchers_on_cpu.py, which fails if
# a divergence appears that is not listed here (or if a listed one disappears).
# ═══════════════════════════════════════════════════════════════════════════════
read -r -d '' SDC_GRID_DIVERGENCES <<'DIVERGENCES' || true
data.system_prompt|(empty)|omitted|key does not exist in this verl; GXPO passed it empty, so it was already a no-op
trainer.max_steps|400|trainer.total_training_steps=400|+trainer.max_steps is a GXPO-fork-only key
trainer.keep_last_ckpts|1|trainer.max_actor_ckpt_to_keep=1|renamed upstream
trainer.keep_all_ckpts|False|omitted|GXPO-fork-only key; max_actor_ckpt_to_keep covers it
trainer.keep_last_validations|1|omitted|GXPO-fork-only key
rollout.attention_backend|FLASHINFER|env VLLM_ATTENTION_BACKEND=FLASHINFER|not a hydra key in this verl
model.attn_implementation|flash_attention_3|+model.override_config.attn_implementation|key moved upstream
actor.optim.name|adamw|omitted|AdamW is the only optimizer in this verl
rollout.free_cache_engine|False|True (+enable_sleep_mode)|the KV cache MUST be released so the two SDC teachers fit on the same GPUs
rollout.gpu_memory_utilization|0.70|0.80|safe only BECAUSE sleep mode frees it during the training phase
rollout.max_num_seqs|1536|2048|tuned for this verl's async server rollout
rollout.max_num_batched_tokens|131072|262144|tuned for this verl's async server rollout
rollout.log_prob_micro_batch_size_per_gpu|16|log_prob_use_dynamic_bsz=True|like-for-like with the actor's dynamic-bsz path
rollout.mode|sync colocated|async|this verl has no sync colocated rollout mode
trainer.resume_mode|disable|auto|a long sequential queue must survive a restart; override with TRAINER_RESUME_MODE
DIVERGENCES

# Test-only escape hatch: dump the divergence table and exit, before preflight (so
# it needs no model/dataset checkout). Used by
# tests/trainer/test_sdc_grid_launchers_on_cpu.py::test_divergences_are_exhaustive.
if [[ "${SDC_GRID_DUMP_DIVERGENCES:-0}" == "1" ]]; then
    printf '%s\n' "${SDC_GRID_DIVERGENCES}"
    exit 0
fi

# Values for the divergent settings.
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.80}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-2048}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-262144}"
# Keep CUDA graphs enabled by default. Set ROLLOUT_ENFORCE_EAGER=True only for
# debugging; eager execution costs rollout throughput.
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}"
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-True}"
TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-auto}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"

# ═══════════════════════════════════════════════════════════════════════════════
# SDC hyperparameters — identical across every SDC arm (sdc and sdc_tr alike), so
# the additive-vs-trust-region comparison isolates the loss formulation only.
# ═══════════════════════════════════════════════════════════════════════════════
SDC_BETA="${SDC_BETA:-0.5}"
SDC_SCORING_BACKEND="${SDC_SCORING_BACKEND:-hf}"
SDC_TEACHER_MODE="${SDC_TEACHER_MODE:-full_fsdp}"
SDC_REUSE_ROLLOUT_LOGPROBS="${SDC_REUSE_ROLLOUT_LOGPROBS:-false}"
SDC_VERIFY_ROLLOUT_LOGPROBS="${SDC_VERIFY_ROLLOUT_LOGPROBS:-false}"
SDC_USE_IMPORTANCE_WEIGHT="${SDC_USE_IMPORTANCE_WEIGHT:-true}"
SDC_IMPORTANCE_WEIGHT_CLIP="${SDC_IMPORTANCE_WEIGHT_CLIP:-5.0}"
SDC_BASE_MIX_GAMMA="${SDC_BASE_MIX_GAMMA:-0.9}"
# Let both teachers see two balanced refreshes before their contrast reaches the actor.
SDC_MIN_SFT_UPDATES_BEFORE_USE="${SDC_MIN_SFT_UPDATES_BEFORE_USE:-2}"
SDC_SFT_UPDATE_INTERVAL="${SDC_SFT_UPDATE_INTERVAL:-5}"
SDC_CHECKPOINT_INTERVAL="${SDC_CHECKPOINT_INTERVAL:-50}"
SDC_SFT_LR="${SDC_SFT_LR:-1.0e-5}"
SDC_SFT_BATCH_SIZE="${SDC_SFT_BATCH_SIZE:-8}"
# 8 rows x (1024 prompt + 3072 response); the implementation sizes from the padded worst case.
SDC_SFT_MAX_TOKEN_LEN_PER_GPU="${SDC_SFT_MAX_TOKEN_LEN_PER_GPU:-32768}"
SDC_SFT_MAX_UPDATES="${SDC_SFT_MAX_UPDATES:-1}"
SDC_SAVE_TO_DISK_INTERVAL="${SDC_SAVE_TO_DISK_INTERVAL:-0}"
SDC_DATA_MAX_SIZE="${SDC_DATA_MAX_SIZE:-32768}"
SDC_DATA_SAMPLING="${SDC_DATA_SAMPLING:-recent}"
SDC_FUSED_ADAMW="${SDC_FUSED_ADAMW:-true}"
SDC_SIDECAR_RESIDENCY="${SDC_SIDECAR_RESIDENCY:-auto_offload}"
SDC_SIDECAR_GRADIENT_CHECKPOINTING="${SDC_SIDECAR_GRADIENT_CHECKPOINTING:-false}"
# Exact cross-rank metric reductions inside every actor micro-batch are useful for
# debugging but add synchronization to the hot path.
SDC_DISTRIBUTED_METRICS="${SDC_DISTRIBUTED_METRICS:-false}"

# SDC-TR: alpha is the mixing coefficient. 1.0 == exact vanilla-PPO equivalence
# (the on-GPU sanity cell); lower values tilt the PPO ratio harder.
SDC_TR_ALPHA="${SDC_TR_ALPHA:-0.9}"
SDC_TR_TILT_CLIP="${SDC_TR_TILT_CLIP:-10.0}"

# SDC-TR zero-advantage repair. Dr.GRPO-style advantages are group-centered, so a
# group in which EVERY response is wrong has A == 0 on every token and contributes
# no gradient at all. With a non-zero coefficient the loss injects a pseudo-advantage
# on those active failed tokens, built from the (detached, normalized) contrast
#   c = log pi_failure - log pi_success
# and scaled by the SDC reliability gate:
#   mu_eff = SDC_TR_DEGENERATE_COEF * reliability_gate,  A_eff = A - mu_eff * c
# 0.0 (the default) is bit-for-bit identical to the previous behaviour.
#
# The hydra override is emitted ONLY when this variable is set EXPLICITLY in the
# environment. That keeps the 12 pre-existing sdc_tr arms byte-identical in their
# override list (see tests/trainer/test_sdc_grid_launchers_on_cpu.py::
# test_sdc_tr_alpha_one_differs_from_sdc_sibling_only_in_loss_mode).
if [[ -n "${SDC_TR_DEGENERATE_COEF+x}" ]]; then
    SDC_TR_DEGENERATE_COEF_EXPLICIT=1
else
    SDC_TR_DEGENERATE_COEF_EXPLICIT=0
fi
SDC_TR_DEGENERATE_COEF="${SDC_TR_DEGENERATE_COEF:-0.0}"

# ═══════════════════════════════════════════════════════════════════════════════
# BASE ALGORITHM
#
# All three bases use the VANILLA policy loss and differ only in the advantage
# estimator, its normalization, and the clip range. That is precisely why SDC-TR
# composes with all of them: it replaces the vanilla clip block with a tilted one
# that still reads clip_ratio_low/high, so DAPO's asymmetric clipping survives.
# ═══════════════════════════════════════════════════════════════════════════════
case "$BASE_ALGO" in
    drgrpo) BASE_ADV_ESTIMATOR=grpo;  BASE_NORM=False; BASE_LOSS_MODE=vanilla; BASE_CLIP_LOW="${CLIP_RATIO}"; BASE_CLIP_HIGH="${CLIP_RATIO}" ;;
    dapo)   BASE_ADV_ESTIMATOR=grpo;  BASE_NORM=True;  BASE_LOSS_MODE=vanilla; BASE_CLIP_LOW=0.2;             BASE_CLIP_HIGH=0.28 ;;
    avspo)  BASE_ADV_ESTIMATOR=avspo; BASE_NORM=True;  BASE_LOSS_MODE=vanilla; BASE_CLIP_LOW="${CLIP_RATIO}"; BASE_CLIP_HIGH="${CLIP_RATIO}" ;;
esac

# SDC-TR replaces the policy loss; the base still contributes via its estimator.
case "$SDC_MODE" in
    base)   SDC_ENABLE=false; EFFECTIVE_LOSS_MODE="${BASE_LOSS_MODE}" ;;
    sdc)    SDC_ENABLE=true;  EFFECTIVE_LOSS_MODE="${BASE_LOSS_MODE}" ;;
    sdc_tr) SDC_ENABLE=true;  EFFECTIVE_LOSS_MODE=sdc_tr ;;
esac

# ═══════════════════════════════════════════════════════════════════════════════
# RUN IDENTITY
# ═══════════════════════════════════════════════════════════════════════════════
ARM_SUFFIX="${SDC_MODE}"
if [[ "$SDC_MODE" == "sdc_tr" ]]; then
    ARM_SUFFIX="sdc_tr_a${SDC_TR_ALPHA}"
    # Keep the zero-advantage-repair arms in their own run/wandb namespace so they
    # never overwrite the plain sdc_tr_a<alpha> checkpoints.
    if [[ "${SDC_TR_DEGENERATE_COEF_EXPLICIT}" == "1" && "${SDC_TR_DEGENERATE_COEF}" != "0.0" \
          && "${SDC_TR_DEGENERATE_COEF}" != "0" ]]; then
        ARM_SUFFIX="${ARM_SUFFIX}_dg${SDC_TR_DEGENERATE_COEF}"
    fi
fi
ARM_NAME="${BASE_ALGO}_${ARM_SUFFIX}"

PROJECT_NAME="${PROJECT_NAME:-sdc-grid-final}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_math_1p5b_${ARM_NAME}_b${TRAIN_BATCH_SIZE}_mb${PPO_MINI_BATCH_SIZE}_s${TRAIN_SEED}}"
RESULTS_ROOT="${SDC_GRID_RESULTS_ROOT:-${REPO_ROOT}/results/sdc_grid}"
RUN_DIR="${CKPTS_DIR:-${RESULTS_ROOT}/${EXPERIMENT_NAME}}"

export WANDB_PROJECT="${WANDB_PROJECT:-${PROJECT_NAME}}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
[[ -n "${WANDB_GROUP:-}" ]] || export WANDB_GROUP="${BASE_ALGO}"
[[ -n "${WANDB_TAGS:-}" ]] || export WANDB_TAGS="base:${BASE_ALGO},mode:${SDC_MODE},experiment:sdc-grid"
LOGGER="${LOGGER:-[console,wandb]}"

# Keep vLLM / FlashInfer autotune artifacts writable and isolated per run.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${RUN_DIR}/vllm_cache}"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-${RUN_DIR}/flashinfer_autotune_cache}"

# Select checkpoints over the same six benchmarks GXPO uses. The trainer default
# metric ("combined" = 0.5*(avg@k + pass@k)) needs BOTH keys, so a greedy n=1 run
# must ask for pass@1 or no best/ checkpoint is ever written.
BEST_CKPT_SOURCES="${BEST_CKPT_SOURCES:-[math500,aime2024,aime2025,amc23,minervamath,olympiadbench]}"
BEST_CKPT_METRICS="${BEST_CKPT_METRICS:-[pass@1]}"

# ═══════════════════════════════════════════════════════════════════════════════
# PREFLIGHT — fail with the fix in the message, not deep inside Ray
# ═══════════════════════════════════════════════════════════════════════════════
MISSING=0

# The GXPO venv carries an editable install of the *GXPO* verl. If cwd is wrong,
# that one is imported and this run silently trains the wrong codebase. Assert the
# verl actually importable here belongs to this repo.
RESOLVED_VERL="$("$PYTHON_BIN" -c 'import verl, os; print(os.path.realpath(verl.__file__))' 2>/dev/null || true)"
if [[ -z "$RESOLVED_VERL" ]]; then
    echo "PREFLIGHT FAIL: could not import verl with $PYTHON_BIN" >&2
    MISSING=1
elif [[ "$RESOLVED_VERL" != "$(realpath "$REPO_ROOT")"/* ]]; then
    echo "PREFLIGHT FAIL: 'import verl' resolves OUTSIDE this repo." >&2
    echo "  resolved : $RESOLVED_VERL" >&2
    echo "  expected : ${REPO_ROOT}/verl/..." >&2
    echo "  The GXPO venv ships an editable verl install; this run would train the" >&2
    echo "  wrong codebase. Launch from ${REPO_ROOT} (the entrypoints already cd there)." >&2
    MISSING=1
fi

if [[ "$SDC_MODE" == "sdc_tr" ]]; then
    if ! "$PYTHON_BIN" -c "
from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY
raise SystemExit(0 if 'sdc_tr' in POLICY_LOSS_REGISTRY else 1)
" 2>/dev/null; then
        echo "PREFLIGHT FAIL: policy loss 'sdc_tr' is not registered in this checkout." >&2
        echo "  Expected @register_policy_loss(\"sdc_tr\") in verl/trainer/ppo/core_algos.py" >&2
        MISSING=1
    fi
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "PREFLIGHT FAIL: model weights not found at ${MODEL_PATH}" >&2
    echo "  hf download Qwen/Qwen2.5-Math-1.5B-Instruct --local-dir ${MODEL_PATH}" >&2
    echo "  (or point MODEL_PATH at an existing local copy)" >&2
    MISSING=1
fi

for required_file in "${TRAIN_DAPO_FILE}" "${TRAIN_LIGHTEVAL_FILE}" \
                     "${MATH500_FILE}" "${AIME24_FILE}" "${AIME25_FILE}" \
                     "${AMC23_FILE}" "${MINERVA_FILE}" "${OLYMPIADBENCH_FILE}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "PREFLIGHT FAIL: missing prepared dataset: ${required_file}" >&2
        echo "  Override GXPO_DATA_ROOT / SDC_VAL_CACHE_ROOT instead of editing this file." >&2
        MISSING=1
    fi
done

# FlashAttention-3 and Liger must be importable, or the actor silently falls back.
if ! "$PYTHON_BIN" - "$ACTOR_ATTENTION_IMPL" <<'PY' 2>/dev/null
import importlib.util, sys
attn = sys.argv[1]
required = ["pandas", "wandb", "tensordict", "liger_kernel"]
required += ["flash_attn"] if attn == "flash_attention_2" else ["flash_attn_3", "flash_attn_interface"]
missing = [n for n in required if importlib.util.find_spec(n) is None]
if missing:
    print("missing: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
then
    echo "PREFLIGHT FAIL: runtime dependencies for ${ACTOR_ATTENTION_IMPL} are unavailable." >&2
    echo "  Expected them in ${H200_SITE_PACKAGES}" >&2
    MISSING=1
fi

if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE}" != "offline" ]]; then
    echo "PREFLIGHT WARN: WANDB_API_KEY unset and WANDB_MODE!=offline;" >&2
    echo "                metrics will fail to upload (training continues)." >&2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    BUSY="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', *' '$2 > 4096 {printf "GPU%s:%sMiB ", $1, $2}')"
    if [[ -n "$BUSY" ]]; then
        echo "PREFLIGHT WARN: GPUs already in use: ${BUSY}" >&2
        echo "                Another job is probably running; expect OOM." >&2
    fi
fi

if [[ "$MISSING" -ne 0 ]]; then
    echo "Preflight failed - fix the items above and re-run." >&2
    exit 2
fi

# ═══════════════════════════════════════════════════════════════════════════════
# HYDRA OVERRIDES
# ═══════════════════════════════════════════════════════════════════════════════
ALGORITHM=(
    algorithm.base_rl_name="${BASE_ALGO}"
    algorithm.adv_estimator="${BASE_ADV_ESTIMATOR}"
    algorithm.norm_adv_by_std_in_grpo="${BASE_NORM}"
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.000
)

DATA=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.val_batch_size="${VAL_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.seed="${TRAIN_SEED}"
    data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}"
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_liger="${USE_LIGER}"
    actor_rollout_ref.model.lora_rank=0
    +actor_rollout_ref.model.override_config.attn_implementation="${ACTOR_ATTENTION_IMPL}"
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.policy_loss.loss_mode="${EFFECTIVE_LOSS_MODE}"
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}"
    actor_rollout_ref.actor.clip_ratio="${CLIP_RATIO}"
    actor_rollout_ref.actor.clip_ratio_low="${BASE_CLIP_LOW}"
    actor_rollout_ref.actor.clip_ratio_high="${BASE_CLIP_HIGH}"
    actor_rollout_ref.actor.clip_ratio_c="${CLIP_RATIO_C}"
    actor_rollout_ref.actor.grad_clip="${GRAD_CLIP}"
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}"
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
    actor_rollout_ref.actor.optim.fused="${OPTIM_FUSED}"
    actor_rollout_ref.actor.use_torch_compile="${USE_TORCH_COMPILE}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss="${ACTOR_USE_KL_LOSS}"
    actor_rollout_ref.actor.kl_loss_coef="${ACTOR_KL_LOSS_COEF}"
    actor_rollout_ref.actor.kl_loss_type="${ACTOR_KL_LOSS_TYPE}"
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}"
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.model_dtype="${MODEL_DTYPE}"
    actor_rollout_ref.actor.data_loader_seed="${TRAIN_SEED}"
)

# NGRPO/AVSPO estimator constants. Inert unless the matching estimator is selected;
# kept unconditional so every arm resolves an identical algorithm block.
ALGORITHM+=(
    algorithm.ngrpo_r_max=1.0
    algorithm.ngrpo_epsilon_pos=0.24
    algorithm.ngrpo_epsilon_neg=0.16
    algorithm.avspo_collapse_tau=1e-6
    algorithm.avspo_alpha=0.5
    algorithm.avspo_anchor_reward=0.1
    algorithm.avspo_adapt_tau_init=0.5
    algorithm.avspo_adapt_eta=0.01
)

if [[ "$SDC_MODE" == "sdc_tr" ]]; then
    ACTOR+=(
        actor_rollout_ref.actor.policy_loss.sdc_tr_alpha="${SDC_TR_ALPHA}"
        actor_rollout_ref.actor.policy_loss.sdc_tr_tilt_clip="${SDC_TR_TILT_CLIP}"
    )
    if [[ "${SDC_TR_DEGENERATE_COEF_EXPLICIT}" == "1" ]]; then
        ACTOR+=(
            actor_rollout_ref.actor.policy_loss.sdc_tr_degenerate_coef="${SDC_TR_DEGENERATE_COEF}"
        )
    fi
fi

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.temperature="${TRAIN_TEMPERATURE}"
    actor_rollout_ref.rollout.top_p="${TRAIN_TOP_P}"
    actor_rollout_ref.rollout.top_k="${TRAIN_TOP_K}"
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_ROLLOUT_N}"
    actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE}"
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}"
    actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}"
    actor_rollout_ref.rollout.val_kwargs.top_k="${VAL_TOP_K}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}"
    +actor_rollout_ref.rollout.enable_sleep_mode="${ROLLOUT_FREE_CACHE_ENGINE}"
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.model_dtype="${MODEL_DTYPE}"
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.balance_batch=True
    trainer.logger="${LOGGER}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.default_local_dir="${RUN_DIR}"
    trainer.n_gpus_per_node="${NDEVICES_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}"
    trainer.resume_mode="${TRAINER_RESUME_MODE}"
    trainer.val_before_train="${VAL_BEFORE_TRAIN}"
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
    +trainer.validation_seeds="${VALIDATION_SEEDS}"
    +trainer.best_ckpt_sources="${BEST_CKPT_SOURCES}"
    +trainer.best_ckpt_metrics="${BEST_CKPT_METRICS}"
)

SDC=(
    custom_sdc.enable="${SDC_ENABLE}"
)
if [[ "${SDC_ENABLE}" == "true" ]]; then
    SDC+=(
        custom_sdc.beta="${SDC_BETA}"
        custom_sdc.scoring_backend="${SDC_SCORING_BACKEND}"
        custom_sdc.teacher_mode="${SDC_TEACHER_MODE}"
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
        custom_sdc.sidecar_residency="${SDC_SIDECAR_RESIDENCY}"
        custom_sdc.sidecar_gradient_checkpointing="${SDC_SIDECAR_GRADIENT_CHECKPOINTING}"
        custom_sdc.distributed_metrics="${SDC_DISTRIBUTED_METRICS}"
    )
fi

# DAPO's dynamic sampling: drop groups whose accuracy carries no learning signal.
if [[ "$BASE_ALGO" == "dapo" ]]; then
    # filter_groups is optional and defaults to null, so add the complete
    # structured value in one Hydra override rather than overriding children
    # beneath a null node.
    ALGORITHM+=(
        +algorithm.filter_groups='{enable:true,metric:acc,max_num_gen_batches:0}'
    )
fi

OVERRIDES=(
    "${ALGORITHM[@]}"
    "${DATA[@]}"
    "${MODEL[@]}"
    "${ACTOR[@]}"
    "${ROLLOUT[@]}"
    "${REF[@]}"
    "${TRAINER[@]}"
    "${SDC[@]}"
    critic.enable=false
)

# Test-only escape hatch: dump the exact hydra override list, one per line, and
# exit. Used by tests/trainer/test_sdc_grid_launchers_on_cpu.py to feed each arm's
# real overrides into an in-process hydra.compose() without needing a GPU, a model
# checkout, or launching main_ppo. Not part of the normal launch path.
if [[ "${SDC_GRID_DUMP_OVERRIDES:-0}" == "1" ]]; then
    printf '%s\n' "${OVERRIDES[@]}"
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
# DRY RUN / LAUNCH
# ═══════════════════════════════════════════════════════════════════════════════
cat <<EOT
[sdc-grid] resolved launch configuration
  arm                : ${ARM_NAME}
  base_algo          : ${BASE_ALGO}  (adv_estimator=${BASE_ADV_ESTIMATOR}, norm_adv_by_std=${BASE_NORM})
  sdc_mode           : ${SDC_MODE}   (custom_sdc.enable=${SDC_ENABLE})
  policy_loss        : ${EFFECTIVE_LOSS_MODE}
  clip low/high/c    : ${BASE_CLIP_LOW} / ${BASE_CLIP_HIGH} / ${CLIP_RATIO_C}
EOT
if [[ "$SDC_MODE" == "sdc_tr" ]]; then
cat <<EOT
  sdc_tr_alpha       : ${SDC_TR_ALPHA}   (1.0 == exact vanilla-PPO equivalence)
  sdc_tr_tilt_clip   : ${SDC_TR_TILT_CLIP}
  sdc_tr_degen_coef  : ${SDC_TR_DEGENERATE_COEF}   (0.0 == off; zero-advantage repair for all-wrong groups)
EOT
fi
if [[ "${SDC_ENABLE}" == "true" ]]; then
cat <<EOT
  sdc beta / gamma   : ${SDC_BETA} / ${SDC_BASE_MIX_GAMMA}
  sdc sft            : lr=${SDC_SFT_LR} every ${SDC_SFT_UPDATE_INTERVAL} steps, warmup ${SDC_MIN_SFT_UPDATES_BEFORE_USE} updates
  sdc teachers       : backend=${SDC_SCORING_BACKEND} mode=${SDC_TEACHER_MODE} residency=${SDC_SIDECAR_RESIDENCY}
EOT
fi
cat <<EOT
  --- inherited from GXPO (identical across all arms) ---
  model              : ${MODEL_PATH}
  batch / minibatch  : ${TRAIN_BATCH_SIZE} / ${PPO_MINI_BATCH_SIZE}   (rollout.n=${ROLLOUT_N})
  seq len            : prompt ${MAX_PROMPT_LENGTH} / response ${MAX_RESPONSE_LENGTH}
  lr / entropy       : ${ACTOR_LR} / ${ENTROPY_COEFF}
  kl loss            : use=${ACTOR_USE_KL_LOSS} coef=${ACTOR_KL_LOSS_COEF} type=${ACTOR_KL_LOSS_TYPE}
  grad_clip          : ${GRAD_CLIP}
  token budget/gpu   : ${PPO_MAX_TOKEN_LEN_PER_GPU}
  gpus               : ${NDEVICES_PER_NODE} (FSDP_SIZE=${FSDP_SIZE}, ids ${CUDA_VISIBLE_DEVICES})
  attention          : train ${ACTOR_ATTENTION_IMPL} | vllm ${VLLM_ATTENTION_BACKEND}
  train sampling     : temp ${TRAIN_TEMPERATURE} top_p ${TRAIN_TOP_P} top_k ${TRAIN_TOP_K}
  val sampling       : greedy n=${VAL_ROLLOUT_N} do_sample=${VAL_DO_SAMPLE} temp=${VAL_TEMPERATURE}
  steps / seed       : ${TOTAL_TRAINING_STEPS} / ${TRAIN_SEED}
  save / test freq   : ${SAVE_FREQ} / ${TEST_FREQ}   val_before_train=${VAL_BEFORE_TRAIN}
  --- deliberate divergences from GXPO ---
$(printf '%s\n' "${SDC_GRID_DIVERGENCES}" | awk -F'|' 'NF>=4 {printf "  %-38s %s -> %s\n      (%s)\n", $1, $2, $3, $4}')
  --- output ---
  python             : ${PYTHON_BIN}
  verl               : ${RESOLVED_VERL}
  run_dir            : ${RUN_DIR}
  wandb              : ${WANDB_PROJECT} / ${EXPERIMENT_NAME} (mode ${WANDB_MODE})
EOT

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[sdc-grid] preflight OK - would launch now."
    exit 0
fi

mkdir -p "${RUN_DIR}" "${VLLM_CACHE_ROOT}" "${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR}"

# Not `exec`: this stage of the pipeline still needs to survive to feed `tee`.
"$PYTHON_BIN" -u -m verl.trainer.main_ppo \
    "${OVERRIDES[@]}" \
    "$@" \
    2>&1 | tee -a "${RUN_DIR}/train.log"
