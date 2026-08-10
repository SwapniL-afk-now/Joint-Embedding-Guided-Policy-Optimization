#!/usr/bin/env bash
# Table 2 | GRPO + FEPO. The +FEPO half of the paired mechanism comparison.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname -- "${BASH_SOURCE[0]}")/_t2_protocol.sh"

# Same FEPO strength as the Table 1 arm this explains (m02/m04 = 0.5/0.5, K=5).
# If Table 1's betas change, change them here too or Table 2 explains a run that
# does not exist.
export TAFR_BETA_ANCHOR=0.5
export TAFR_BETA_REPLAY=0.5
export TAFR_SFT_UPDATE_INTERVAL=5
export TAFR_CHECKPOINT_INTERVAL=5

launch t2_grpo_fepo "Table 2: GRPO + FEPO, sampled probe (beta 0.5/0.5)"
