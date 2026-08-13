#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/Joint-Embedding-Guided-Policy-Optimization
NAME="$1"; GPU="$2"
case "$NAME" in
 m09) RUN=fepo-m09_ngrpo_fepo-20260812_avspo_ngrpo_token_beta05; OUT=fepo_m09_last_eval.json;;
 m10) RUN=fepo-m10_avspo_fepo-20260812_avspo_ngrpo_token_beta05; OUT=fepo_m10_last_eval.json;;
 *) echo bad name; exit 2;;
esac
ACTOR="$ROOT/checkpoints/tafr-repro-math15b/$RUN/global_step_500/actor"
MODEL="$ROOT/eval_models/$NAME-global_step_500"
mkdir -p "$ROOT/eval_models" "$ROOT/eval_results"
if [[ ! -f "$MODEL/config.json" ]]; then
  "$ROOT/.venv/bin/python" "$ROOT/scripts/legacy_model_merger.py" merge --backend fsdp --local_dir "$ACTOR" --target_dir "$MODEL"
fi
CUDA_VISIBLE_DEVICES="$GPU" "$ROOT/.venv/bin/python" "$ROOT/eval_math_multiseed_batched.py" --model "$MODEL" --out "$ROOT/eval_results/$OUT" --benchmarks math500,amc23,aime24,aime25,aime26 --seeds 3407 --n 1
