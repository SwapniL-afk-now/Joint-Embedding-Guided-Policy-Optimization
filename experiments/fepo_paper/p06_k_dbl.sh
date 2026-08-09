#!/usr/bin/env bash
# Table 8 | failure-policy refresh cadence K = 2 * K0 (every 5 -> 10 steps).
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export TAFR_SFT_UPDATE_INTERVAL=10

launch p06_k_dbl "Table 8: failure refresh every 10 steps (K0=5)"
