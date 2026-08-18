#!/usr/bin/env bash
# Table 8 | anchor EMA gamma = 2 * gamma0, clipped to 0.99 (2*0.9 is not a valid
# decay). Report it as 0.99, not '2*gamma0' -- the doubling is not representable here.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export TAFR_EMA_GAMMA=0.99

launch p04_gamma_dbl "Table 8: anchor EMA gamma 0.9 -> 0.99"
