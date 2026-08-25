#!/usr/bin/env bash
set -euo pipefail

# Small SDC-GRPO parameter sweep. Each arm uses the same launcher and changes
# only the direct contrastive coefficient or the SFT schedule.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAUNCHER="${SCRIPT_DIR}/run_sdc_grpo_fsdp.sh"

case "${1:-all}" in
  beta0.25) SDC_BETA=0.25 exec bash "${LAUNCHER}" ;;
  beta0.5)  SDC_BETA=0.5 exec bash "${LAUNCHER}" ;;
  beta1.0)  SDC_BETA=1.0 exec bash "${LAUNCHER}" ;;
  sft10)    SDC_SFT_UPDATE_INTERVAL=10 exec bash "${LAUNCHER}" ;;
  all)
    for arm in beta0.25 beta0.5 beta1.0 sft10; do
      bash "${BASH_SOURCE[0]}" "${arm}"
    done
    ;;
  *)
    echo "usage: $0 {beta0.25|beta0.5|beta1.0|sft10|all}" >&2
    exit 2
    ;;
esac
