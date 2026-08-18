#!/usr/bin/env bash
# Replay/failure-reference-only ablation. Instability is an informative result;
# do not rescue it by retuning the arm-specific hyperparameters.
source "$(dirname -- "${BASH_SOURCE[0]}")/deepseek7b_common.sh"
export TAFR_VARIANT=replay_only
launch deepseek_replay_only_7b "DeepSeek-Math-7B replay-only ablation"
