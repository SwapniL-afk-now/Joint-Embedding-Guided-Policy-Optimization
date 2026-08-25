#!/usr/bin/env bash
# Table 1 | NGRPO + FEPO. The plug-in pair for m07.
#
# k3_legacy's gate and KL terms come from _tafr_group_reward_mean (raw per-uid
# reward) and log-probs only -- nothing in the loss reads the advantage tensor --
# so composing FEPO with NGRPO's repaired advantage is a real composition, not a
# diagnostic no-op. diagnostic_only is NOT set here: FEPO actually contributes.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

launch m09_ngrpo_fepo "Table 1: NGRPO + FEPO" \
  algorithm.adv_estimator=ngrpo
