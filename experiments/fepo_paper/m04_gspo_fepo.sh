#!/usr/bin/env bash
# Table 1 | GSPO + FEPO. The plug-in pair for m03.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# Matches the m02 probe point: symmetric 0.5/0.5, below the paper's beta0=1.0 centre.
# Kept identical to m02 on purpose -- m02/m04 are the GRPO and GSPO halves of the same
# plug-in comparison, so their FEPO strength has to be the same number.
export TAFR_BETA_ANCHOR=0.5
export TAFR_BETA_REPLAY=0.5
export TAFR_SFT_UPDATE_INTERVAL=5   # _common.sh default is already 5; pinned so a
                                    # change to the shared default cannot silently
                                    # break the m02/m04 pairing.
export TAFR_CHECKPOINT_INTERVAL=5   # 5, NOT _common.sh's paper default of 10.

# Betas in the name: it drives the checkpoint dir, val_dumps AND the wandb run id.
launch m04_gspo_fepo_ba0.5_br0.5 "Table 1: GSPO + FEPO (beta_anchor=0.5, beta_replay=0.5)" "${gspo[@]}"
