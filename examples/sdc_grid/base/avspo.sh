#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_ALGO=avspo SDC_MODE=base exec bash "${SCRIPT_DIR}/../common.sh" "$@"
