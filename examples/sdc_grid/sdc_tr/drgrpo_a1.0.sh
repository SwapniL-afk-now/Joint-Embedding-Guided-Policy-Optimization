#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_ALGO=drgrpo SDC_MODE=sdc_tr SDC_TR_ALPHA=1.0 exec bash "${SCRIPT_DIR}/../common.sh" "$@"
