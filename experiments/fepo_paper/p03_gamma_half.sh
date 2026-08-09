#!/usr/bin/env bash
# Table 8 | anchor EMA gamma = 0.5 * gamma0. Confirm the rate-vs-decay convention
# in the code before reporting this number in the tex (see plan B2).
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export TAFR_EMA_GAMMA=0.45

launch p03_gamma_half "Table 8: anchor EMA gamma 0.9 -> 0.45"
