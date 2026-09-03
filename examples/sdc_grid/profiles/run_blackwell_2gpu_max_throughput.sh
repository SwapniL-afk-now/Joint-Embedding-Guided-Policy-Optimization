#!/usr/bin/env bash
# Maximum-throughput starting profile for the 2x RTX PRO 6000 Blackwell run.
# Usage: bash "$0" <arm-path> [--dry-run] [extra Hydra args...]
#
# This wraps the validated Blackwell runner. It changes runtime capacity only;
# learning-related batch, rollout count, loss, and optimizer settings are kept
# unchanged. The values are intentionally aggressive but leave memory headroom
# for the actor and both SDC teacher sidecars.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Keep caller overrides if supplied. These are the throughput knobs:
#   vLLM: larger KV-cache budget, with concurrency matched to 3072-token responses
#   actor: larger dynamic token budget -> fewer forward/backward microbatches
# Observed at 0.70 with 2048 sequences: ~94 GiB/GPU during generation. 0.72
# adds a small cache budget. 2048 matches the actual request (batch 256 x n 8)
# and max batched tokens is held to 131072 to limit transient prefill memory.
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.72}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-2048}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-131072}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-65536}"

# CUDA graphs are required for the fast vLLM path. The common launcher now
# consumes this variable and passes an explicit False Hydra override.
export ROLLOUT_ENFORCE_EAGER=False
# vLLM 0.16 memory pools are incompatible with expandable_segments.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-true}"

exec bash "${SCRIPT_DIR}/run_blackwell_2gpu.sh" "$@"
