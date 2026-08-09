#!/usr/bin/env bash
# Table 1 | AVSPO baseline. The tex names this "our closest scalar-collapse
# baseline", so it is the row a reviewer is most likely to demand.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

launch m08_avspo "Table 1: AVSPO baseline" \
  algorithm.adv_estimator=avspo "${diagnostic_only[@]}"
