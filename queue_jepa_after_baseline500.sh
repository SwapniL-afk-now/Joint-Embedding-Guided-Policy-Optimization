#!/bin/bash
# Watch the no-JEPA baseline; at step 500 stop it and hand GPU0 to the LLM-JEPA
# CoT<->Code alignment run. Runs in its own tmux session so it survives a shell exit.
set -u
cd /workspace/Joint-Embedding-Guided-Policy-Optimization

BASE_SESSION=baseline_nojepa
BASE_LOG=logs_baseline_nojepa_20260725.log
JEPA_LOG=logs_llmjepa_cotcode_20260726.log
JEPA_SCRIPT=examples/jepa_grpo_trainer/long-context/run_llmjepa_cotcode_math15b.sh
TARGET_STEP=500

echo "[queue] $(date) watching $BASE_LOG for step $TARGET_STEP"
# "step:500 - " matches the metrics line exactly; a bare "step:500" would also hit
# a substring of step:5000 (unreachable at 800 total, but the guard is free).
until grep -q "step:${TARGET_STEP} - " "$BASE_LOG"; do
    if ! tmux has-session -t "$BASE_SESSION" 2>/dev/null; then
        echo "[queue] ABORT $(date): baseline session '$BASE_SESSION' disappeared before"
        echo "[queue] step $TARGET_STEP (last: $(grep -o 'step:[0-9]*' "$BASE_LOG" | tail -1))."
        echo "[queue] NOT launching the JEPA run -- a crashed baseline is a decision for a human."
        exit 1
    fi
    sleep 60
done

echo "[queue] $(date) baseline reached step $TARGET_STEP; stopping it"
tmux send-keys -t "$BASE_SESSION" C-c 2>/dev/null
sleep 20
tmux kill-session -t "$BASE_SESSION" 2>/dev/null
pkill -f "baseline-nojepa-full-Qwen2.5-Math-1.5B" 2>/dev/null
sleep 20

# Wait for GPU0 to actually release before starting a run that needs ~90GB on it.
for _ in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    [ "${used:-99999}" -lt 2000 ] && break
    sleep 10
done
echo "[queue] $(date) GPU0 at ${used} MiB; launching LLM-JEPA CoT<->Code"

: > "$JEPA_LOG"
exec bash "$JEPA_SCRIPT" 2>&1 | tee "$JEPA_LOG"
