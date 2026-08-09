#!/usr/bin/env bash
# Table 8 | beta = 2 * beta0.
# Note the divergence risk: wandb run oyul6igu ran anchor 0.5 / replay 1.0 and blew
# up after step ~144. This arm's replay weight (2.0) exceeds that absolute value
# outright, even though the ratio here (1:1) is less replay-heavy than oyul6igu's
# (1:2) -- watch grad_norm closely from the start, not just past step ~130.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export TAFR_BETA_ANCHOR=2.0
export TAFR_BETA_REPLAY=2.0

launch p02_beta_dbl "Table 8: beta = 2*beta0 (anchor 2.0 / replay 2.0)"
