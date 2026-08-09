#!/usr/bin/env bash
# Table 1 | GRPO baseline. diagnostic_only keeps the anchor/failure models alive
# so tafr_grpo/kl_anchor is logged, without letting them touch the loss -- that is
# what gives Tables 2 and 3 their KL column for a row that has no regularizer.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

launch m01_grpo "Table 1: GRPO baseline (KL measured, not applied)" \
  "${diagnostic_only[@]}"
