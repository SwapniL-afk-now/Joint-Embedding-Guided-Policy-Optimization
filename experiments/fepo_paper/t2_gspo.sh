#!/usr/bin/env bash
# Table 2 | GSPO base (no FEPO). Pairs with t2_gspo_fepo.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname -- "${BASH_SOURCE[0]}")/_t2_protocol.sh"

launch t2_gspo "Table 2: GSPO base, sampled probe (n=8, temp 0.6)" \
  "${diagnostic_only[@]}" "${gspo[@]}"
