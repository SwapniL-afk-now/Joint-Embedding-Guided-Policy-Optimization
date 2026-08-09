#!/usr/bin/env bash
# Table 1 | AVSPO + FEPO. The plug-in pair for m08 -- the sharpest mechanism
# contrast in the paper: AVSPO repairs the scalar advantage on homogeneous groups,
# FEPO leaves the scalar advantage untouched and supplies a learned direction. This
# arm tests whether the two are complementary or redundant when combined.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

launch m10_avspo_fepo "Table 1: AVSPO + FEPO" \
  algorithm.adv_estimator=avspo
