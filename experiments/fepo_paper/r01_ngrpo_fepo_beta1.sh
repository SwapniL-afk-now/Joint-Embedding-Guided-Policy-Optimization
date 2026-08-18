#!/usr/bin/env bash
# Re-run NGRPO + FEPO with explicit beta_anchor=1.0 and beta_failure/replay=1.0.
# All other settings are inherited unchanged from _common.sh.
# Usage: ARM_GPU=0 bash experiments/fepo_paper/r01_ngrpo_fepo_beta1.sh
set -euo pipefail
export MPLBACKEND=Agg
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
export TAFR_BETA_ANCHOR=1.0
export TAFR_BETA_REPLAY=1.0
export TAFR_SFT_UPDATE_INTERVAL=5
export TAFR_CHECKPOINT_INTERVAL=10
launch r1_ngrpo_fepo_beta1 "Table 1 rerun: NGRPO + FEPO (beta_anchor=1.0, beta_replay=1.0)" \
  algorithm.adv_estimator=ngrpo
