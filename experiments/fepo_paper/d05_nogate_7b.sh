#!/usr/bin/env bash
# No-gate ablation: applies the replay push to every group.
source "$(dirname -- "${BASH_SOURCE[0]}")/deepseek7b_common.sh"
launch deepseek_nogate_7b "DeepSeek-Math-7B full reference without gate" \
  +custom_tafr_grpo.gate_mode=constant
