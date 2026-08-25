#!/usr/bin/env bash
# GPI-CE on DeepMath-103K with Qwen2.5-1.5B-Instruct.
#
#   ARM_GPU=0 bash examples/gpi_ce/run_deepmath_qwen15b.sh
#
# Group = 5 online rollouts + 1 offline R1 solution (K=6). The offline candidate
# is a verified-correct trace from the dataset; the 5 online ones will be a mix of
# correct and wrong, which is what gives q* something to tilt.
#
# Qwen2.5-1.5B-**Instruct**, not the Math variant: 32k context, so a 6144-token R1
# solution fits in the group alongside the prompt.
#
# Curriculum: the parquet is pre-sorted by DeepMath's difficulty column (easiest
# first, 2.0 -> 9.5, mass concentrated at 5-8), and data.shuffle=false makes verl
# use a SequentialSampler, so training walks mid -> hard. This is the ONLY thing
# holding the curriculum: turning shuffle on silently destroys it.
#
# Read at step 10 (the metric that decides whether mixing works at all):
#   gpi/target_mass_offline   mass q* puts on the R1 trace. Near zero means the
#                             actor prior suppresses it before reward weighting can
#                             help -- see the README's failure modes.
#   gpi/old_mass_offline      the same under p_old, i.e. before the reward tilt.
#   gpi/target_reward_improvement   E_q*[r] - E_p_old[r]; must be >= 0 every step.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source .venv/bin/activate   # repo venv (torch 2.9.1), NOT /venv/main
# The container presets RAY_ADDRESS=127.0.0.1 (no port -> ray rejects it) and
# RAY_ARGS pinning port 6379.
unset RAY_ADDRESS RAY_ARGS
export RAY_TMPDIR=/tmp/ray_gpi_ce

export CUDA_VISIBLE_DEVICES=${ARM_GPU:-0}
export PROJECT_NAME=grpo-qwen-3b
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-gpi-ce-deepmath-on${GPI_ONLINE:-5}off${GPI_OFFLINE:-3}-$(date -u +%Y%m%d)}
export CKPTS_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

# run_gpi_ce_fsdp.sh hardcodes trainer.resume_mode=auto, so an existing dir is
# silently continued. Refuse rather than produce a mislabelled run.
[[ "${RESUME:-0}" == 1 ]] || { [[ -d "${CKPTS_DIR}" ]] && { echo "REFUSING: ${CKPTS_DIR} exists (resume_mode=auto would continue it). Set RESUME=1 to continue it deliberately." >&2; exit 3; }; }

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
# The parquet is sharded: with three full R1 solutions per row a single file grows
# big enough that pyarrow chunks its nested columns, and reading it back then fails
# with "Nested data conversions not implemented for chunked array outputs".
GPI_DATA_DIR=${GPI_DATA_DIR:-/workspace/jepa-grpo-cache/data/deepmath_gpi}
if [[ -z "${TRAIN_FILE:-}" ]]; then
    shopt -s nullglob
    _shards=("${GPI_DATA_DIR}"/train-*-of-*.parquet)
    shopt -u nullglob
    if (( ${#_shards[@]} )); then
        TRAIN_FILE="[$(IFS=,; echo "${_shards[*]}")]"
    else
        TRAIN_FILE="${GPI_DATA_DIR}/train.parquet"
    fi
fi
export TRAIN_FILE

# K = 8 = 5 online + 3 offline (r1_solution_1/2/3). rollout.n is K_on only.
export GPI_ONLINE=${GPI_ONLINE:-5}
export GPI_OFFLINE=${GPI_OFFLINE:-3}
export GPI_TEMPERATURE=${GPI_TEMPERATURE:-0.5}

export DATA_SHUFFLE=False               # <-- the curriculum depends on this
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
# Must be >= the prep script's --max-response-tokens, or a stored offline solution
# gets truncated on the way in and turns into an unscoreable candidate.
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-31744}
export GPI_GROUPS_PER_MICRO_BATCH=${GPI_GROUPS_PER_MICRO_BATCH:-1}

export TRAIN_SEED=31415
export VALIDATION_SEEDS='[31415]'
export TEST_FREQ=${TEST_FREQ:-10}
export SAVE_FREQ=${SAVE_FREQ:-10}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-400}
export NDEVICES_PER_NODE=1

exec bash examples/gpi_ce/run_gpi_ce_fsdp.sh "$@" 2>&1 | tee -a "logs_${EXPERIMENT_NAME}.log"
