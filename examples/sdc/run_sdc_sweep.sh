#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_ALGO=${BASE_ALGO:-drgrpo} exec bash "${SCRIPT_DIR}/../sdc_grpo/run_sdc_sweep.sh" "$@"
