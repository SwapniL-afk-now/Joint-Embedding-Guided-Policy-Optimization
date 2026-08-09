#!/usr/bin/env bash
# Table 8 | failure-policy refresh cadence K = 0.5 * K0 (every 5 -> 3 steps).
# 2.5 is not an integer; 3 is the nearest. Say so in the caption.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export TAFR_SFT_UPDATE_INTERVAL=3

launch p05_k_half "Table 8: failure refresh every 3 steps (K0=5)"
