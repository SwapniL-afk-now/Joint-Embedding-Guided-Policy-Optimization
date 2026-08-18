#!/usr/bin/env bash
# DeepSeek-Math-7B compact study: full FEPO/Z-REF plug-in.
source "$(dirname -- "${BASH_SOURCE[0]}")/deepseek7b_common.sh"
launch deepseek_grpo_fepo_7b "DeepSeek-Math-7B GRPO + full reference" 
