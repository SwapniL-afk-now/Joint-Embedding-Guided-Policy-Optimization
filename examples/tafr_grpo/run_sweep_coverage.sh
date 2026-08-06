#!/usr/bin/env bash
# TAFR sweep arm A — COVERAGE.
#
#   ARM_GPU=0 bash examples/tafr_grpo/run_sweep_coverage.sh
#
# Question: can more tokens be given a nonzero A_reg?
# In the prod run (wandb 5j6rgx1l) |A_reg| has p50 = 1e-4 — ~62% of tokens sit at or
# below 1e-3, because the anchor, the failure model and the policy all agree there.
# No rescaling reaches them (0 x anything = 0), so this arm changes the MODELS:
#   - eta 0.5 -> 1.0     anchor is the pure EMA; the frozen base component is dropped,
#                        so the anchor is a trained model rather than half-untrained
#   - interval 10 -> 20  anchor lag = interval/(1-gamma) = 200 steps, was 100
#   - sft_lr 1e-6 -> 1e-5  failure model moves further from the policy
# beta stays at the prod 0.5 on purpose: per-token magnitude must stay comparable to
# the prod run, or this arm is confounded with the beta2 arm.
#
# Read at step 20 (~40 min), before committing the full run:
#   advantage_abs_p50   1e-4 now. -> ~1e-2 means coverage widened; still ~1e-4 means the
#                       models fundamentally agree on easy tokens and 37.8% is the ceiling.
#   raw_anchor_rms      0.0286 now. eta=1.0 removes the base component; if lag-200 does
#                       not compensate this drops and the anchor term went quiet.
#   train/exploration_collapse_rate  6% now, GRPO reaches 21%. If it climbs, the frozen
#                       base component was the anti-collapse mechanism, not the attribution.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source .venv/bin/activate   # repo venv (torch 2.9.1), NOT /venv/main
# The container presets RAY_ADDRESS=127.0.0.1 (no port -> ray rejects it) and
# RAY_ARGS pinning port 6379. Both arms need their own local cluster.
unset RAY_ADDRESS RAY_ARGS
export RAY_TMPDIR=/tmp/ray_tafr_b1

export CUDA_VISIBLE_DEVICES=${ARM_GPU:-0}
export PROJECT_NAME=grpo-qwen-3b
export EXPERIMENT_NAME=tafr-sweep-b1.0-$(date -u +%Y%m%d)
export WANDB_RUN_ID=g44s0lem
export CKPTS_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

# The launcher hardcodes trainer.resume_mode=auto, so an existing dir is silently
# resumed instead of started fresh. Refuse rather than produce a mislabelled run.
[[ "${RESUME:-0}" == 1 ]] || { [[ -d "${CKPTS_DIR}" ]] && { echo "REFUSING: ${CKPTS_DIR} exists (resume_mode=auto would continue it). Set RESUME=1 to continue it deliberately." >&2; exit 3; }; }

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-400}
export TEST_FREQ=${TEST_FREQ:-10}
export SAVE_FREQ=${SAVE_FREQ:-10}
export VAL_BEFORE_TRAIN=false
export VALIDATION_SEEDS='[31415]'   # was 5 seeds; validation cost was dominating
export NDEVICES_PER_NODE=1              # RL arms stay on 1 GPU: the DP gradient deficit
export TRAIN_SEED=31415

# --- prod-run values, except the three coverage knobs ---
export TAFR_BETA=1.0                    # <-- the only difference between the two arms
export TAFR_EMA_GAMMA=0.9               # unchanged: higher gamma re-weights the EMA seed
export TAFR_FAILURE_DATA_MAX_SIZE=null  # unchanged
export TAFR_FAILURE_DATA_SAMPLING=recent
export TAFR_MIX_ETA=0.9                 # anchor = 90% own EMA, 10% frozen base
export TAFR_SAVE_TO_DISK_INTERVAL=40   # disk-write throttle ONLY; EMA/anchor still refresh every 10 (transformer_impl.py:2186)
export TAFR_CHECKPOINT_INTERVAL=10      # prod value
export TAFR_SFT_UPDATE_INTERVAL=10      # same clock as the EMA refresh
export TAFR_SFT_LR=1.0e-6               # prod value, held: beta is the only delta

echo "=== ${EXPERIMENT_NAME} on GPU ${CUDA_VISIBLE_DEVICES}"
echo "    COVERAGE arm: mix_eta=1.0 ckpt_interval=20 sft_interval=20 sft_lr=1e-5 | beta=0.5 (held)"
exec bash examples/tafr_grpo/run_tafr_grpo_fsdp.sh "$@" 2>&1 | tee -a "logs_${EXPERIMENT_NAME}.log"
