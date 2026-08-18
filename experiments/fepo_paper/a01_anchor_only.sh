#!/usr/bin/env bash
# Table 3 | + anchor only. Tests whether any gain is just EMA stabilization.
# Base objective is GRPO, not DAPO -- cheaper and DAPO is Table 1's own object of
# study, not needed as scaffolding here.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export TAFR_VARIANT=anchor_only

launch a01_anchor_only "Table 3: GRPO + anchor only"
