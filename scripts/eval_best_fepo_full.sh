#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/Joint-Embedding-Guided-Policy-Optimization
NAME="$1"; GPU="$2"
MODEL="$ROOT/eval_models/$NAME-global_step_500"
OUT="$ROOT/eval_results/fepo_${NAME}_full"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU" VLLM_WORKER_MULTIPROC_METHOD=spawn
PY="$ROOT/.venv/bin/python"
COMMON="--n 1 --temperature 0 --top-p 1 --max-tokens 3072 --gpu-memory-utilization 0.9"
SEEDS_ALL=3407,31415,27182,16180,42
SEEDS_4=31415,27182,16180,42
$PY "$ROOT/eval_math_multiseed_batched.py" --model "$MODEL" $COMMON --benchmarks math500,amc23,aime24,aime25,aime26 --seeds "$SEEDS_4" --out "$OUT/math_core.json"
$PY "$ROOT/eval_math_multiseed_batched.py" --model "$MODEL" $COMMON --benchmarks olympiadbench,minervamath --seeds "$SEEDS_ALL" --out "$OUT/math_extra.json"
$PY "$ROOT/eval_code_multiseed_batched.py" --model "$MODEL" $COMMON --benchmarks humanevalplus,mbppplus,livecodebench --seeds "$SEEDS_ALL" --out "$OUT/code.json"
echo "DONE $OUT"
