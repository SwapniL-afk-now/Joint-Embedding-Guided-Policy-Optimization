#!/usr/bin/env bash
# Table 1 (7B panel) | DAPO + FEPO at scale. Pairs with s01.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export MODEL_PATH=/workspace/models/Qwen2.5-Math-7B-Instruct
export ROLLOUT_GPU_MEM_UTIL=0.6
export PPO_MINI_BATCH_SIZE=8

launch s02_dapo_fepo_7b "Table 1: DAPO + FEPO, 7B" "${dapo[@]}"
