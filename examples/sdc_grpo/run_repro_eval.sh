#!/usr/bin/env bash
# Greedy (n=1, temp 0) multi-seed eval of a merged best checkpoint.
# Usage: ARM=repro|repro-grpo GPU=0 bash examples/sdc_grpo/run_repro_eval.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

ARM=${ARM:?set ARM=repro or repro-grpo}
GPU=${GPU:?set GPU}
MODEL=/workspace/merged_ckpt/${ARM}
OUT=/workspace/eval_results/${ARM}
mkdir -p "${OUT}"

# 3407 already measured in-training for the 5 core math benchmarks, so those get
# the 4 remaining seeds only; everything else gets all 5.
SEEDS_ALL=3407,31415,27182,16180,42
SEEDS_4=31415,27182,16180,42

export CUDA_VISIBLE_DEVICES=${GPU}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=.venv/bin/python
COMMON="--n 1 --temperature 0 --top-p 1 --max-tokens 3072 --gpu-memory-utilization 0.9"

echo "### [${ARM}] math core, 4 seeds"
${PY} eval_math_multiseed_batched.py --model "${MODEL}" ${COMMON} \
  --benchmarks math500,amc23,aime24,aime25,aime26 --seeds "${SEEDS_4}" \
  --out "${OUT}/math_core.json"

echo "### [${ARM}] math extra, 5 seeds"
${PY} eval_math_multiseed_batched.py --model "${MODEL}" ${COMMON} \
  --benchmarks olympiadbench,minervamath --seeds "${SEEDS_ALL}" \
  --out "${OUT}/math_extra.json"

echo "### [${ARM}] code, 5 seeds"
${PY} eval_code_multiseed_batched.py --model "${MODEL}" ${COMMON} \
  --benchmarks humanevalplus,mbppplus,livecodebench --seeds "${SEEDS_ALL}" \
  --out "${OUT}/code.json"

echo "### [${ARM}] done -> ${OUT}"
