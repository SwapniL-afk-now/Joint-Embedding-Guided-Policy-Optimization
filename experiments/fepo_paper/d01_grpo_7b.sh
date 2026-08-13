#!/usr/bin/env bash
# DeepSeek-Math-7B compact study: GRPO control.
source "$(dirname -- "${BASH_SOURCE[0]}")/deepseek7b_common.sh"
launch deepseek_grpo_7b "DeepSeek-Math-7B GRPO baseline" "${diagnostic_only[@]}"
