#!/usr/bin/env bash
# Table 3 | + anchor + failure, no gate. Tests whether conditioning the push on
# empirical failure frequency is necessary. Needs gate_mode (Phase 1, C3): the gate
# 1-r_bar is hardcoded in tafr_loss.py:25 and is not configurable today.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

launch a03_nogate "Table 3: GRPO + anchor + failure, constant gate" \
  +custom_tafr_grpo.gate_mode=constant
