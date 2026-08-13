#!/usr/bin/env bash
# Upload a complete trainer checkpoint directory (best + final/global_step + metadata)
# to a Hugging Face model repository, preserving the local directory layout.
#
# Usage (after training):
#   HF_REPO=USER/REPO ./train-scripts/push_full_checkpoint_repo.sh RUN_DIR
#
# Queue safely while training is running (waits for both PIDs, then uploads):
#   HF_REPO=USER/REPO ./train-scripts/push_full_checkpoint_repo.sh RUN_DIR \
#       --wait-pid PID1 --wait-pid PID2
#
# RUN_DIR may be absolute or relative to checkpoints/tafr-repro-math15b.
# The script uploads raw checkpoint files exactly as saved; it does not delete or
# modify local files and does not convert them to Hugging Face model format.
set -euo pipefail

ROOT=${CHECKPOINT_ROOT:-/workspace/Joint-Embedding-Guided-Policy-Optimization/checkpoints/tafr-repro-math15b}
RUN_DIR=${1:?"usage: $0 RUN_DIR [--wait-pid PID] [--private]"}
shift
[[ "$RUN_DIR" = /* ]] || RUN_DIR="$ROOT/$RUN_DIR"

REPO=${HF_REPO:-}
PRIVATE=${HF_PRIVATE:-true}
PIDS=()
while (($#)); do
  case "$1" in
    --wait-pid) PIDS+=("${2:?missing PID}"); shift 2 ;;
    --private) PRIVATE=true; shift ;;
    --public) PRIVATE=false; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$REPO" ]] || { echo 'Set HF_REPO=USER/REPO' >&2; exit 2; }
[[ -d "$RUN_DIR" ]] || { echo "Missing run directory: $RUN_DIR" >&2; exit 1; }

# Load the token without printing it. Prefer the project .env.
if [[ -f /workspace/Joint-Embedding-Guided-Policy-Optimization/.env ]]; then
  set -a; source /workspace/Joint-Embedding-Guided-Policy-Optimization/.env; set +a
elif [[ -f /workspace/gradient-extrapolation-based-policy-optimization/.env ]]; then
  set -a; source /workspace/gradient-extrapolation-based-policy-optimization/.env; set +a
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
[[ -n "$HUGGING_FACE_HUB_TOKEN" ]] || { echo 'HF_TOKEN/HUGGING_FACE_HUB_TOKEN is not set' >&2; exit 1; }

for pid in "${PIDS[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "Waiting for training PID $pid to finish..."
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
  else
    echo "PID $pid is not running; continuing (verify its exit status yourself)." >&2
  fi
done

# Never upload an incomplete run accidentally: require a checkpoint and the
# trainer's completion marker (override with ALLOW_INCOMPLETE=1 for inspection).
shopt -s nullglob
steps=("$RUN_DIR"/global_step_*)
(( ${#steps[@]} > 0 )) || { echo "No global_step_* checkpoint found" >&2; exit 1; }
if [[ "${ALLOW_INCOMPLETE:-0}" != 1 && ! -f "$RUN_DIR/latest_checkpointed_iteration.txt" ]]; then
  echo "Missing latest_checkpointed_iteration.txt; refusing upload" >&2; exit 1
fi

private_arg=''
[[ "$PRIVATE" == true ]] && private_arg='--private'
# upload_folder preserves best_pass_at_1/, global_step_*/, and all metadata.
exec env RUN_DIR="$RUN_DIR" REPO="$REPO" PRIVATE="$PRIVATE" \
  /workspace/Joint-Embedding-Guided-Policy-Optimization/.venv/bin/python - <<'PY'
import os
from huggingface_hub import HfApi
run_dir=os.environ['RUN_DIR']; repo=os.environ['REPO']
private=os.environ['PRIVATE'].lower() == 'true'
api=HfApi(token=os.environ['HUGGING_FACE_HUB_TOKEN'])
api.create_repo(repo_id=repo, repo_type='model', private=private, exist_ok=True)
print(f'Uploading {run_dir} -> https://huggingface.co/{repo}')
api.upload_folder(repo_id=repo, repo_type='model', folder_path=run_dir,
                   path_in_repo='', commit_message=f'Upload full checkpoint: {os.path.basename(run_dir)}',
                   allow_patterns=['**'])
print('Upload complete.')
PY
