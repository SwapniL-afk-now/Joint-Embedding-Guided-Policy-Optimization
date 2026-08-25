#!/usr/bin/env bash
# Table 1 | DAPO + FEPO -- the paper's headline configuration.
# Doubles as: Table 3 'full FEPO' row, Table 4 FEPO row, Table 8 sensitivity centre.
# This is the canary run: launch it first and check the Phase-0 gates before
# committing the other 21 arms.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# Arm-local probe, not the paper's beta0=1.0 centre used by Tables 3/4/8 -- do not
# read this run as replacing the headline row those tables cite. 10x beta on both
# terms (still gating only replay_loss/failure KL; anchor_loss stays ungated, the
# known-stable asymmetric config from before the symmetric-gate experiment) and a
# tighter 5-step SFT/checkpoint-sync/save cadence (paper defaults: K0=5, checkpoint
# interval=10 -- only the checkpoint interval changes here, SFT was already 5).
export TAFR_BETA_ANCHOR=0.5
export TAFR_BETA_REPLAY=0.5
export TAFR_SFT_UPDATE_INTERVAL=5
export TAFR_CHECKPOINT_INTERVAL=5

launch m06_dapo_fepo "Table 1/3/4/8: DAPO + FEPO (probe: beta=10, tighter cadence)" \
  "${dapo[@]}"
