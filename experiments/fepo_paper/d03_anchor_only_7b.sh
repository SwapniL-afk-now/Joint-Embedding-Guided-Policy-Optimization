#!/usr/bin/env bash
# Anchor-only ablation: tests whether EMA stabilization explains the gain.
source "$(dirname -- "${BASH_SOURCE[0]}")/deepseek7b_common.sh"
export TAFR_VARIANT=anchor_only
launch deepseek_anchor_only_7b "DeepSeek-Math-7B anchor-only ablation"
