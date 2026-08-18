#!/usr/bin/env bash
# Table 1 | NGRPO baseline.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

launch m07_ngrpo "Table 1: NGRPO baseline" \
  algorithm.adv_estimator=ngrpo "${diagnostic_only[@]}"
