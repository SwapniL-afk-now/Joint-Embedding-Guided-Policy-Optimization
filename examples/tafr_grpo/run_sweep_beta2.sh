#!/usr/bin/env bash
# TAFR sweep arm B — COVERAGE, SHORT LAG.
#
#   ARM_GPU=1 bash examples/tafr_grpo/run_sweep_beta2.sh
#
# Identical to run_sweep_coverage.sh except the anchor refresh interval: 20 -> 5.
# Anchor lag is interval/(1-gamma), so this arm anchors to the policy 50 steps back
# instead of 200, and the failure model is re-fit 4x more often.
#
# Why this is the right second arm: with eta=1.0 the frozen base component is gone, so
# ALL of the anchor's distance from pi_old comes from lag. Arm A (lag 200) and this arm
# (lag 50) bracket the axis 4x apart, so the pair measures how much distance lag actually
# buys — with the prod run's lag 100 as a third point (though at eta=0.5, so not exactly
# comparable). A short lag also keeps the anchor FRESH: it tracks the current policy
# rather than a stale one, which is the opposite trade to arm A.
#
# Prediction from the prod-run data: shorter lag -> smaller raw_anchor_rms -> narrower
# coverage. If this arm wins anyway, the lag model is wrong and freshness beats distance.
#
# Read at step 20 (~40 min):
#   advantage_abs_p50   1e-4 now. The number that decides whether coverage moved at all.
#   raw_anchor_rms      0.0286 now. Should be LOWER here than in arm A. If the two arms
#                       match despite a 4x lag difference, lag does not control anchor
#                       distance and the whole anchor-tuning direction is a dead end.
#   train/exploration_collapse_rate  6% now, GRPO reaches 21%.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source .venv/bin/activate   # repo venv (torch 2.9.1), NOT /venv/main
# The container presets RAY_ADDRESS=127.0.0.1 (no port -> ray rejects it) and
# RAY_ARGS pinning port 6379. Both arms need their own local cluster.
unset RAY_ADDRESS RAY_ARGS
export RAY_TMPDIR=/tmp/ray_tafr_b2

export CUDA_VISIBLE_DEVICES=${ARM_GPU:-1}
export PROJECT_NAME=grpo-qwen-3b
export EXPERIMENT_NAME=tafr-sweep-b2.0-$(date -u +%Y%m%d)
export WANDB_RUN_ID=84lrmbck
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
export TAFR_BETA=2.0                    # <-- the only difference between the two arms
export TAFR_EMA_GAMMA=0.9               # unchanged: higher gamma re-weights the EMA seed
export TAFR_FAILURE_DATA_MAX_SIZE=null  # unchanged
export TAFR_FAILURE_DATA_SAMPLING=recent
export TAFR_MIX_ETA=0.9                 # anchor = 90% own EMA, 10% frozen base
export TAFR_SAVE_TO_DISK_INTERVAL=40   # disk-write throttle ONLY; EMA/anchor still refresh every 10 (transformer_impl.py:2186)
export TAFR_CHECKPOINT_INTERVAL=10      # prod value
export TAFR_SFT_UPDATE_INTERVAL=10      # same clock as the EMA refresh
export TAFR_SFT_LR=1.0e-6               # prod value, held: beta is the only delta

echo "=== ${EXPERIMENT_NAME} on GPU ${CUDA_VISIBLE_DEVICES}"
echo "    COVERAGE arm, lag 50: mix_eta=1.0 ckpt_interval=5 sft_interval=5 sft_lr=1e-5 | beta=0.5 (held)"
exec bash examples/tafr_grpo/run_tafr_grpo_fsdp.sh "$@" 2>&1 | tee -a "logs_${EXPERIMENT_NAME}.log"
