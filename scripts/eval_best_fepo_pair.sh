#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/Joint-Embedding-Guided-Policy-Optimization
NAME="$1"; GPU="$2"
case "$NAME" in
  ngrpo) RUN=fepo-r1_ngrpo_fepo_beta1-20260813; OUT=fepo_r1_ngrpo_best_eval.json ;;
  avspo) RUN=fepo-r2_avspo_fepo_beta1-20260813; OUT=fepo_r2_avspo_best_eval.json ;;
  *) echo "usage: $0 ngrpo|avspo GPU" >&2; exit 2;;
esac
ACTOR="$ROOT/checkpoints/tafr-repro-math15b/$RUN/global_step_500/actor"
MODEL="$ROOT/eval_models/$NAME-global_step_500"
mkdir -p "$ROOT/eval_models" "$ROOT/eval_results"
if [[ ! -f "$MODEL/config.json" ]]; then
  "$ROOT/.venv/bin/python" "$ROOT/scripts/legacy_model_merger.py" merge --backend fsdp --local_dir "$ACTOR" --target_dir "$MODEL"
fi
CUDA_VISIBLE_DEVICES="$GPU" VLLM_WORKER_MULTIPROC_METHOD=spawn "$ROOT/.venv/bin/python" "$ROOT/eval_math_multiseed_batched.py" \
 --model "$MODEL" --out "$ROOT/eval_results/$OUT" --benchmarks math500,amc23,aime24,aime25,aime26 --seeds 3407 --n 1 --temperature 0 --top-p 1 --max-tokens 3072 --gpu-memory-utilization 0.9
