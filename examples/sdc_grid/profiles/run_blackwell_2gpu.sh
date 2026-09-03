#!/usr/bin/env bash
#
# examples/sdc_grid/profiles/run_blackwell_2gpu.sh
#
# Launch one examples/sdc_grid arm on the local 4x RTX PRO 6000 Blackwell box,
# using GPU 0 and 1 only, with the working sm_120 stack.
#
#   bash examples/sdc_grid/profiles/run_blackwell_2gpu.sh <arm-path> [--dry-run] [hydra args...]
#
# <arm-path> may be
#     sdc_tr/drgrpo_a0.5.sh                        (relative to examples/sdc_grid)
#     sdc_tr/drgrpo_a0.5                           (the .sh is optional)
#     examples/sdc_grid/sdc_tr/drgrpo_a0.5.sh      (relative to the repo root)
#     /abs/path/to/arm.sh
#
# Examples:
#   bash examples/sdc_grid/profiles/run_blackwell_2gpu.sh sdc_tr/drgrpo_a0.5.sh --dry-run
#   bash examples/sdc_grid/profiles/run_blackwell_2gpu.sh sdc_tr/drgrpo_a0.5.sh \
#        actor_rollout_ref.actor.policy_loss.sdc_tr_degenerate_coef=0.05
#
# Anything exported before this script still wins (the profile uses ${VAR:-...}).
set -euo pipefail

PROFILE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GRID_DIR="$(cd -- "${PROFILE_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${GRID_DIR}/../.." && pwd)"

usage() {
    cat >&2 <<EOT
usage: bash $0 <arm-path> [--dry-run] [extra hydra args...]

available arms:
EOT
    (cd "$GRID_DIR" && find base sdc sdc_tr sdc_tr_zeroadv -name '*.sh' | sort | sed 's/^/  /') >&2
    exit 2
}

[[ $# -ge 1 ]] || usage
case "${1:-}" in -h|--help|"") usage ;; esac

ARM_ARG="$1"; shift

# ── resolve the arm script ───────────────────────────────────────────────────
ARM=""
for cand in "$ARM_ARG" "${ARM_ARG}.sh" \
            "${GRID_DIR}/${ARM_ARG}" "${GRID_DIR}/${ARM_ARG}.sh" \
            "${REPO_ROOT}/${ARM_ARG}" "${REPO_ROOT}/${ARM_ARG}.sh"; do
    if [[ -f "$cand" ]]; then ARM="$(cd -- "$(dirname -- "$cand")" && pwd)/$(basename -- "$cand")"; break; fi
done
if [[ -z "$ARM" ]]; then
    echo "ERROR: no such arm: ${ARM_ARG}" >&2
    usage
fi
case "$ARM" in
    "${GRID_DIR}"/*) ;;
    *) echo "ERROR: ${ARM} is outside ${GRID_DIR}" >&2; exit 2 ;;
esac
if [[ "$(basename -- "$ARM")" == "common.sh" ]]; then
    echo "ERROR: common.sh is not an arm; pass e.g. sdc_tr/drgrpo_a0.5.sh" >&2
    exit 2
fi

# ── common.sh only accepts --dry-run as its FIRST argument ───────────────────
DRY_RUN=0
PASSTHRU=()
for a in "$@"; do
    if [[ "$a" == "--dry-run" ]]; then DRY_RUN=1; else PASSTHRU+=("$a"); fi
done

# ── environment ──────────────────────────────────────────────────────────────
# shellcheck source=./blackwell_2gpu.env
source "${PROFILE_DIR}/blackwell_2gpu.env"

# Refuse to start a real run on GPUs somebody else is using. GPU 2 and 3 on this
# box normally carry another job; a dry run is still allowed so you can inspect
# the resolved config at any time.
if [[ "$DRY_RUN" -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1; then
    BUSY=""
    IFS=',' read -r -a _WANTED <<< "${CUDA_VISIBLE_DEVICES}"
    for g in "${_WANTED[@]}"; do
        used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 0)"
        (( used > 4096 )) && BUSY="${BUSY} GPU${g}:${used}MiB"
    done
    if [[ -n "$BUSY" ]]; then
        echo "ERROR: requested GPUs are already in use:${BUSY}" >&2
        echo "  Wait for that job, or pick free devices:" >&2
        echo "    CUDA_VISIBLE_DEVICES=<free,ids> bash $0 ${ARM_ARG}" >&2
        echo "  or re-run with --dry-run to only print the resolved config." >&2
        exit 3
    fi
fi

cat <<EOT
[blackwell-2gpu] profile   : ${PROFILE_DIR}/blackwell_2gpu.env
[blackwell-2gpu] arm       : ${ARM}
[blackwell-2gpu] python    : ${PYTHON_BIN}
[blackwell-2gpu] gpus      : ${CUDA_VISIBLE_DEVICES} (n_gpus_per_node=${NDEVICES_PER_NODE}, fsdp_size=${FSDP_SIZE})
[blackwell-2gpu] attention : actor=${ACTOR_ATTENTION_IMPL} vllm=${VLLM_ATTENTION_BACKEND}
[blackwell-2gpu] dry_run   : ${DRY_RUN}
EOT

cd "$REPO_ROOT"
if [[ "$DRY_RUN" -eq 1 ]]; then
    exec bash "$ARM" --dry-run ${PASSTHRU[@]+"${PASSTHRU[@]}"}
else
    exec bash "$ARM" ${PASSTHRU[@]+"${PASSTHRU[@]}"}
fi
