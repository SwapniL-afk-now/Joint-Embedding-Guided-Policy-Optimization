#!/usr/bin/env bash
# Table 8 | beta = 0.5 * beta0.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export TAFR_BETA_ANCHOR=0.5
export TAFR_BETA_REPLAY=0.5

launch p01_beta_half "Table 8: beta = 0.5*beta0 (anchor 0.5 / replay 0.5)"
