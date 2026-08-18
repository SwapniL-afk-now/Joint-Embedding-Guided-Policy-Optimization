#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_ALGO=ngrpo exec bash "${SCRIPT_DIR}/../run_sdc_fsdp.sh" "$@"
