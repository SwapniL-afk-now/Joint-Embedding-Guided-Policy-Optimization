#!/usr/bin/env bash
# Negative-CE control using the identical failed replay stream and cadence.
source "$(dirname -- "${BASH_SOURCE[0]}")/deepseek7b_common.sh"
launch deepseek_negative_ce_7b "DeepSeek-Math-7B negative-CE control" \
  +custom_tafr_grpo.negative_ce=true
