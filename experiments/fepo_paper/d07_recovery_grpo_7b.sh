#!/usr/bin/env bash
# Table 2: deterministic greedy recovery, DeepSeek GRPO control.
source "$(dirname -- "${BASH_SOURCE[0]}")/deepseek7b_recovery_common.sh"
launch deepseek_recovery_grpo_7b "DeepSeek-Math-7B GRPO deterministic recovery" "${diagnostic_only[@]}"
