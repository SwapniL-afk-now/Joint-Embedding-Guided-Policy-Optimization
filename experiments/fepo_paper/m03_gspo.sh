#!/usr/bin/env bash
# Table 1 | GSPO baseline.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# Requested fully disabled, not diagnostic_only: TAFR_ENABLE=false skips building
# the anchor/failure models entirely (no SFT updates, no anchor/replay log-prob
# scoring), unlike m01/diagnostic_only which keeps them alive just to log KL.
# validate_tafr_config() short-circuits on `not custom.enable` (config.py:200), so
# diagnostic_only is moot here -- dropped it rather than pass a no-op flag.
# Consequence: this row has no tafr_grpo/kl_anchor|kl_replay to fill Tables 2/3's
# GSPO column -- a deliberate trade for a plain, TAFR-compute-free GSPO baseline.
export TAFR_ENABLE=false

launch m03_gspo "Table 1: GSPO baseline (TAFR disabled)" \
  "${gspo[@]}"
