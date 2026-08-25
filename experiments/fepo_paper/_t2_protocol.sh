#!/usr/bin/env bash
# Shared Table 2 greedy benchmark-recovery protocol. Sourced by t2_*.sh.
# Not runnable on its own.
#
# Recovery is evaluated on fixed benchmark prompts with one deterministic greedy
# completion per prompt. There is deliberately no sampled success-rate p or
# difficulty bucket: with n=1, p is binary. The five validation seeds are retained
# because this repository's validation path can produce different outputs for them;
# recovery must be averaged over those evaluation seeds, while training remains at
# the single TRAIN_SEED from _common.sh.

# --- deterministic benchmark validation ---------------------------------------
export VAL_ROLLOUT_N=1
export VAL_DO_SAMPLE=False
export VAL_TEMPERATURE=0
export VAL_TOP_P=0.95
export VALIDATION_SEEDS='[3407,31415,27182,16180,42]'

# Table 2 benchmark pool: 500 + 272 + 674 = 1,446 fixed prompts.
export VAL_FILES="[${EVAL_DATA_DIR}/math500.parquet,${EVAL_DATA_DIR}/minervamath.parquet,${EVAL_DATA_DIR}/olympiadbench.parquet]"
export BEST_CKPT_SOURCES='["math500","minervamath","olympiadbench"]'
export BEST_CKPT_METRICS='["pass@1"]'

# Keep training identical to the main arm. Recovery is not a held-out training
# probe; it measures deterministic transitions on benchmark evaluation prompts.
export TRAIN_FILE=/workspace/jepa-grpo-cache/data/dsr_math345/train.parquet

# The reference must be genuinely pre-training; otherwise Rec.@50 is measured from
# an already-trained step-10 model.
export VAL_BEFORE_TRAIN=true
export TEST_FREQ=10               # step-0, 50, and 100 are all emitted
