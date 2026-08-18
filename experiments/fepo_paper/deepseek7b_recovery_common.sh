#!/usr/bin/env bash
# DeepSeek-Math-7B deterministic Table-2 recovery protocol.
# Matches the paper: greedy n=1 at t0, t0+50, t0+100 on AMC23,
# MinervaMath, and OlympiadBench. Each training seed gets one top-level dump.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/deepseek7b_common.sh"
export VAL_FILES="[${EVAL_DATA_DIR}/amc23.parquet,${EVAL_DATA_DIR}/minervamath.parquet,${EVAL_DATA_DIR}/olympiadbench.parquet]"
export BEST_CKPT_SOURCES='["amc23","minervamath","olympiadbench"]'
export BEST_CKPT_METRICS='["pass@1"]'
export VAL_ROLLOUT_N=1
export VAL_DO_SAMPLE=False
export VAL_TEMPERATURE=0
export VAL_TOP_P=0.95
export VAL_BEFORE_TRAIN=true
export TEST_FREQ=10
export VALIDATION_SEEDS='[3407]'
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
