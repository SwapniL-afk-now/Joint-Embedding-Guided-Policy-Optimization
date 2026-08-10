#!/usr/bin/env bash
# Table 2 | GRPO base (no FEPO). Pairs with t2_grpo_fepo via `analyze.py compare`.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname -- "${BASH_SOURCE[0]}")/_t2_protocol.sh"

# diagnostic_only, matching m01: anchor/failure models stay alive so kl_anchor is
# logged, but contribute nothing to the loss. Table 2's base column is a run with
# no regularizer applied, not one with no regularizer measured.
launch t2_grpo "Table 2: GRPO base, sampled probe (n=8, temp 0.6)" \
  "${diagnostic_only[@]}"
