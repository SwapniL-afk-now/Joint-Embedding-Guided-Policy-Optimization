#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=fepo-m10_avspo_fepo-20260812_avspo_ngrpo_token_beta05
PID=1545437
REPO=ismamNur/fepo-m10-avspo-full
ROOT=/workspace/Joint-Embedding-Guided-Policy-Optimization
cd "$ROOT"
LOG="$ROOT/logs/hf_upload_${REPO##*/}.log"
exec >>"$LOG" 2>&1
echo "[$(date -u)] queued upload: pid=$PID run=$RUN_DIR repo=$REPO"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "[$(date -u)] training PID ended; waiting for checkpoint flush"
sleep 60
RUN="$ROOT/checkpoints/tafr-repro-math15b/$RUN_DIR"
[[ -d "$RUN" ]] || { echo "missing run dir: $RUN"; exit 1; }
ITER=$(cat "$RUN/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
if [[ "$ITER" -lt 500 ]] || [[ ! -d "$RUN/global_step_500" ]]; then
  echo "[$(date -u)] REFUSING upload: incomplete run (iteration=$ITER, global_step_500 missing)"
  exit 2
fi
# Use the full run directory: best_pass_at_1, global_step_500, and metadata.
HF_REPO="$REPO" HF_PRIVATE=true "$ROOT/scripts/push_full_checkpoint_repo.sh" "$RUN"
echo "[$(date -u)] upload finished"
