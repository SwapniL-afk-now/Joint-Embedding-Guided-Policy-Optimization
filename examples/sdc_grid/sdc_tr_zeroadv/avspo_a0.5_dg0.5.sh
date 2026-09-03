#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_ALGO=avspo SDC_MODE=sdc_tr SDC_TR_ALPHA=0.5 SDC_TR_DEGENERATE_COEF=0.5 exec bash "${SCRIPT_DIR}/../common.sh" "$@"
