#!/usr/bin/env bash
# Table 5 | Llama-3.2-3B GRPO baseline on rule-verifiable instruction following.
# Blocked on C8 (IFEval/FollowBench parquets + reward manager + MMLU regression check),
# which is the only genuinely large piece of work in the program. If the schedule
# slips, this is the table to cut -- the tex frames it as a narrow sanity check.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export MODEL_PATH=/workspace/models/Llama-3.2-3B-Instruct
export TRAIN_FILE=/workspace/jepa-grpo-cache/data/ifeval/train.parquet
export VAL_FILES="[${EVAL_DATA_DIR}/ifeval.parquet,${EVAL_DATA_DIR}/followbench.parquet,${EVAL_DATA_DIR}/mmlu.parquet]"
export BEST_CKPT_SOURCES='["ifeval","followbench"]'
# MMLU is a regression check only -- it must never drive checkpoint selection.

launch x01_llama_grpo "Table 5: Llama-3.2-3B GRPO baseline" "${diagnostic_only[@]}"
