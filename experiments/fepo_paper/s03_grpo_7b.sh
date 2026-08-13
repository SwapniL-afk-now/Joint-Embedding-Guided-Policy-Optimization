#!/usr/bin/env bash
# Table 1 (7B panel) | GRPO baseline, using the same protocol/configuration as
# the 1.5B GRPO arm. The only intentional change is MODEL_PATH.
# Usage: ARM_GPU=0 ./experiments/fepo_paper/s03_grpo_7b.sh
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# Intentional model-only and world-size changes: preserve dataset, seed, lengths,
# optimizer, rollout count, validation protocol, and 500-step budget from the 1.5B arm.
export MODEL_PATH=/workspace/models/Qwen2.5-Math-7B-Instruct
export PROJECT_NAME=tafr-repro-math7b
export NDEVICES_PER_NODE=2
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1}"
export ROLLOUT_GPU_MEM_UTIL=0.55

launch s03_grpo_7b "Table 1: GRPO baseline, 7B (1.5B protocol matched)" \
  "${diagnostic_only[@]}"
