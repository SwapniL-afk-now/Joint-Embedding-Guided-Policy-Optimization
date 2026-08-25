#!/usr/bin/env bash
# Table 2 | GSPO + FEPO. The +FEPO half of the GSPO pair.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname -- "${BASH_SOURCE[0]}")/_t2_protocol.sh"

export TAFR_BETA_ANCHOR=0.5
export TAFR_BETA_REPLAY=0.5
export TAFR_SFT_UPDATE_INTERVAL=5
export TAFR_CHECKPOINT_INTERVAL=5

launch t2_gspo_fepo "Table 2: GSPO + FEPO, sampled probe (beta 0.5/0.5)" \
  "${gspo[@]}"
