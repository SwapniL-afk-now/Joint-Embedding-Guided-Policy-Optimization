#!/usr/bin/env bash
# Source this file before running JEPA/verl training:
#   source ./activate_jepa.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this file instead: source ./activate_jepa.sh" >&2
    exit 1
fi

export JEPA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${JEPA_ROOT}/.venv/bin/activate"

# FlashAttention 2.8.3 is the cu13 wheel recommended by REPRODUCE.md. Keep
# its bundled CUDA 13 libraries visible alongside torch's CUDA 12.8 runtime.
export LD_LIBRARY_PATH="${JEPA_ROOT}/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${JEPA_ROOT}/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

# Make Ray and shell launchers use this repository's interpreter and source.
export PYTHON_BIN="${JEPA_ROOT}/.venv/bin/python"
export PYTHONPATH="${JEPA_ROOT}:${PYTHONPATH:-}"

# Use the verified inference/training attention implementations.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export ACTOR_ATTENTION_IMPL="${ACTOR_ATTENTION_IMPL:-flash_attention_2}"

# Keep caches local to this repository unless explicitly overridden.
export HF_HOME="${HF_HOME:-${JEPA_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${JEPA_ROOT}/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${JEPA_ROOT}/.cache/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${JEPA_ROOT}/.cache/vllm}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" \
    "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}"

# Blackwell dual-GPU workaround documented by this repository.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-0}"
export PYTHONINTMAXSTRDIGITS="${PYTHONINTMAXSTRDIGITS:-0}"

echo "JEPA environment active: ${JEPA_ROOT}"
echo "Python: ${PYTHON_BIN}"
