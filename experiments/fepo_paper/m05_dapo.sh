#!/usr/bin/env bash
# Table 1 | DAPO baseline. Also the Table 4 systems-cost reference row.
# TAFR is fully OFF (not diagnostic_only) -- Table 3's ablation and Table 2's
# mechanism study both moved to a GRPO base earlier, so nothing left needs this
# arm's KL column. diagnostic_only still builds and fits the anchor/failure clone
# every sft_update_interval (real optimizer steps, real GPU time) just to zero its
# OWN gradient contribution -- paying that cost here would make Table 4's "DAPO"
# row already include FEPO's reference-policy overhead, understating what FEPO
# actually costs on top of it. A true bare-DAPO row needs custom_tafr_grpo.enable=false.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export TAFR_ENABLE=false

launch m05_dapo "Table 1/4: DAPO baseline (TAFR fully disabled, true cost baseline)" \
  "${dapo[@]}"
