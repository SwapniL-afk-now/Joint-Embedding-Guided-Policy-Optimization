#!/usr/bin/env bash
# Run the compact DeepSeek-Math-7B FEPO/Z-REF study.
# This script is intentionally confirmation-gated because each arm is expensive.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/../.."

usage() {
  cat <<'EOF'
Usage: run_deepseek7b_compact.sh [--dry-run] [--confirm] [--phase PHASE]

PHASE is one of: main, recovery, ablations, all (default: all).
Environment:
  MAIN_SEEDS=3407              seeds for GRPO and GRPO+FEPO
  AUX_SEEDS=3407               seeds for ablations/control
  GPUS=0,1                     two visible GPUs
  DEEPSEEK_MODEL_PATH=...      local model snapshot (auto-resolved if omitted)
  TOTAL_TRAINING_STEPS=500

Examples:
  ... --dry-run
  MAIN_SEEDS=3407,31415,27182,16180,42 AUX_SEEDS=3407,31415,27182 \
    GPUS=0,1 ... --confirm --phase all
EOF
}
phase=all; dry_run=0; confirm=0
while (($#)); do
  case "$1" in
    --dry-run) dry_run=1; shift;;
    --confirm) confirm=1; shift;;
    --phase) phase=${2:?missing phase}; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done
[[ "$phase" == main || "$phase" == recovery || "$phase" == ablations || "$phase" == all ]] || { echo "invalid phase: $phase" >&2; exit 2; }
if (( ! dry_run && ! confirm )); then
  echo "Refusing to launch expensive 7B training without --confirm (or use --dry-run)." >&2
  exit 2
fi

IFS=',' read -r -a main_seeds <<< "${MAIN_SEEDS:-3407}"
IFS=',' read -r -a aux_seeds <<< "${AUX_SEEDS:-3407}"
IFS=',' read -r -a recovery_seeds <<< "${RECOVERY_SEEDS:-${MAIN_SEEDS:-3407}}"
main_arms=(d01_grpo_7b.sh d02_grpo_fepo_7b.sh)
recovery_arms=(d07_recovery_grpo_7b.sh d08_recovery_fepo_7b.sh)
aux_arms=(d03_anchor_only_7b.sh d04_replay_only_7b.sh d05_nogate_7b.sh d06_negce_7b.sh)
run_arm() {
  local arm=$1 seed=$2
  local tag="deepseek7b-${arm%.sh}-s${seed}"
  echo "=== ${arm} seed=${seed} gpu=${GPUS:-0,1} tag=${tag} ==="
  # MODEL_PATH is intentionally unset so the DeepSeek resolver cannot inherit a
  # stale Qwen path from the parent shell.
  env -u MODEL_PATH TRAIN_SEED="${seed}" TAG="${tag}" \
    GPUS="${GPUS:-0,1}" DEEPSEEK_MODEL_PATH="${DEEPSEEK_MODEL_PATH:-}" \
    DRY_RUN="${dry_run}" bash "${SCRIPT_DIR}/${arm}"
}
if [[ "$phase" == main || "$phase" == all ]]; then
  for seed in "${main_seeds[@]}"; do for arm in "${main_arms[@]}"; do run_arm "$arm" "$seed"; done; done
fi
if [[ "$phase" == recovery || "$phase" == all ]]; then
  # Recovery is a separate two-arm run because the main arms use greedy n=1
  # validation and cannot yield per-prompt sampled success rates.
  for seed in "${recovery_seeds[@]}"; do for arm in "${recovery_arms[@]}"; do run_arm "$arm" "$seed"; done; done
fi
if [[ "$phase" == ablations || "$phase" == all ]]; then
  for seed in "${aux_seeds[@]}"; do for arm in "${aux_arms[@]}"; do run_arm "$arm" "$seed"; done; done
fi
echo "DeepSeek-Math-7B compact study complete. Preserve val_dumps and eval JSON; delete checkpoints only after evaluation."
