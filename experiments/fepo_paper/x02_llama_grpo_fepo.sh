#!/usr/bin/env bash
# Table 5 | Llama-3.2-3B GRPO + FEPO. Pairs with x01. Blocked on C8.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export MODEL_PATH=/workspace/models/Llama-3.2-3B-Instruct
export TRAIN_FILE=/workspace/jepa-grpo-cache/data/ifeval/train.parquet
export VAL_FILES="[${EVAL_DATA_DIR}/ifeval.parquet,${EVAL_DATA_DIR}/followbench.parquet,${EVAL_DATA_DIR}/mmlu.parquet]"
export BEST_CKPT_SOURCES='["ifeval","followbench"]'

launch x02_llama_grpo_fepo "Table 5: Llama-3.2-3B GRPO + FEPO"
