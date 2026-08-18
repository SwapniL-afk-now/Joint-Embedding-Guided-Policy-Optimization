#!/usr/bin/env bash
set -euo pipefail

# Canonical modular SDC launcher.  BASE_ALGO selects the independent base
# recipe; SDC remains the same additive auxiliary in every preset.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_ALGO=${BASE_ALGO:-drgrpo} exec bash "${SCRIPT_DIR}/../sdc_grpo/run_sdc_grpo_fsdp.sh" "$@"
