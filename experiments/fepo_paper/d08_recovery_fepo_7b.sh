#!/usr/bin/env bash
# Table 2: deterministic greedy recovery, DeepSeek GRPO + FEPO/Z-REF.
source "$(dirname -- "${BASH_SOURCE[0]}")/deepseek7b_recovery_common.sh"
launch deepseek_recovery_fepo_7b "DeepSeek-Math-7B FEPO deterministic recovery"
