#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_ALGO=drgrpo SDC_MODE=sdc exec bash "${SCRIPT_DIR}/../common.sh" "$@"
