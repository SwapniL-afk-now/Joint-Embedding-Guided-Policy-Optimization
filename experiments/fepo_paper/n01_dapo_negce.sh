#!/usr/bin/env bash
# Table 7 (appendix) | Negative-CE control on the SAME failed replay buffer.
# This is what stops the paper being reduced to 'negative training on failed tokens'.
# Needs C7. Must use the identical buffer as m02 (GRPO + FEPO) or the control proves
# nothing -- base objective is GRPO here, not DAPO, to match m02 exactly.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

launch n01_dapo_negce "Table 7: GRPO + negative CE on failed trajectories" \
  +custom_tafr_grpo.negative_ce=true
