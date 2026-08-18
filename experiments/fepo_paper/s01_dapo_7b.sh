#!/usr/bin/env bash
# Table 1 (7B panel) | DAPO baseline at scale.
# ~45 GPU-h. Uses the FEPO hyperparameters selected on 1.5B, unchanged -- that
# transfer-without-retuning is the actual claim being tested here, so do not
# re-tune beta/gamma/K for 7B even if the first run underperforms.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export MODEL_PATH=/workspace/models/Qwen2.5-Math-7B-Instruct
# 7B holds far more optimizer state next to a resident vLLM engine.
export ROLLOUT_GPU_MEM_UTIL=0.6
export PPO_MINI_BATCH_SIZE=8

launch s01_dapo_7b "Table 1: DAPO baseline, 7B" \
  "${dapo[@]}" "${diagnostic_only[@]}"
