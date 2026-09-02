#!/usr/bin/env bash
#
# examples/sdc_grid/queue.sh
#
# Run one or more sdc_grid arms sequentially inside a named tmux session, per this
# repo's CLAUDE.md ("Every script runs in tmux. No exceptions.").
#
# Usage:
#   bash examples/sdc_grid/queue.sh <session-name> <arm> [arm...]
#   bash examples/sdc_grid/queue.sh <session-name> --force <arm> [arm...]   # skip the GPU-idle check
#
# <arm> is a path relative to examples/sdc_grid/, e.g.:
#   base/drgrpo  sdc/dapo  sdc_tr/avspo_a0.9
#
# Examples:
#   bash examples/sdc_grid/queue.sh sdc-grid-1 base/drgrpo sdc/drgrpo sdc_tr/drgrpo_a0.9
#   bash examples/sdc_grid/queue.sh sdc-grid-full \
#       base/drgrpo base/dapo base/avspo \
#       sdc/drgrpo sdc/dapo sdc/avspo \
#       sdc_tr/drgrpo_a1.0 sdc_tr/drgrpo_a0.9 sdc_tr/drgrpo_a0.7 sdc_tr/drgrpo_a0.5 \
#       sdc_tr/dapo_a1.0   sdc_tr/dapo_a0.9   sdc_tr/dapo_a0.7   sdc_tr/dapo_a0.5 \
#       sdc_tr/avspo_a1.0  sdc_tr/avspo_a0.9  sdc_tr/avspo_a0.7  sdc_tr/avspo_a0.5
#
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <session-name> [--force] <arm> [arm...]" >&2
    exit 2
fi

SESSION="$1"; shift

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
    shift
fi

if [[ $# -eq 0 ]]; then
    echo "ERROR: no arms given." >&2
    exit 2
fi

ARMS=()
for arm in "$@"; do
    script="${SCRIPT_DIR}/${arm}.sh"
    if [[ ! -f "$script" ]]; then
        echo "ERROR: no such arm '${arm}' (expected ${script})" >&2
        exit 2
    fi
    ARMS+=("$arm")
done

if [[ "$FORCE" -ne 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
    BUSY="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', *' '$2 > 4096 {printf "GPU%s:%sMiB ", $1, $2}')"
    if [[ -n "$BUSY" ]]; then
        echo "ERROR: GPUs already busy: ${BUSY}" >&2
        echo "  Pass --force to queue anyway (e.g. once the current job finishes)." >&2
        exit 2
    fi
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '${SESSION}' already exists." >&2
    echo "  Attach with: tmux attach -t ${SESSION}" >&2
    echo "  Or pick a different session name." >&2
    exit 2
fi

echo "Queueing ${#ARMS[@]} arm(s) in tmux session '${SESSION}':"
printf '  %s\n' "${ARMS[@]}"

tmux new-session -d -s "$SESSION" -c "$SCRIPT_DIR"
# A just-created pane eats the first send-keys during shell startup; send a cd
# first and confirm cwd before queuing real work.
tmux send-keys -t "$SESSION" "cd '${SCRIPT_DIR}' && pwd" C-m
sleep 1

QUEUE_CMD="set -e"
for arm in "${ARMS[@]}"; do
    QUEUE_CMD+=$'\n'"echo '=== sdc_grid arm: ${arm} ==='; bash '${SCRIPT_DIR}/${arm}.sh'"
done
QUEUE_CMD+=$'\n'"echo '=== sdc_grid queue finished ==='"

QUEUE_SCRIPT="$(mktemp "${SCRIPT_DIR}/.queue.${SESSION}.XXXXXX.sh")"
printf '%s\n' "$QUEUE_CMD" > "$QUEUE_SCRIPT"
tmux send-keys -t "$SESSION" "bash '${QUEUE_SCRIPT}'; rm -f '${QUEUE_SCRIPT}'" C-m

echo "Launched. Attach with: tmux attach -t ${SESSION}"
