#!/usr/bin/env bash
# Table 3 | + failure repulsion only. Tests whether unconstrained repulsion suffices.
# Expect this to be the unstable row: without the anchor pull there is nothing
# opposing the replay push. Watch grad_norm closely (see RUNBOOK health check).
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export TAFR_VARIANT=replay_only

launch a02_failure_only "Table 3: GRPO + failure repulsion only"
