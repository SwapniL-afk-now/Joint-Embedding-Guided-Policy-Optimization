#!/usr/bin/env bash
# Table 1 | GRPO + FEPO. The plug-in pair for m01.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# Arm-local probe. Was 10/10 (matching the m06_dapo_fepo probe); now 0.5/0.5 --
# symmetric and BELOW the paper's beta0=1.0 centre, i.e. a weaker pull/push on both
# KL terms. Keeps the tighter 5-step checkpoint-sync cadence (paper default: 10).
# Untested point: _common.sh's divergence note covers 0.5/1.0 (replay-heavier, which
# blew up in wandb oyul6igu) and 1.0/1.0; equal-and-halved is neither. Watch
# kl_replay -- the risk at low beta is the opposite failure to oyul6igu's, the clone
# never separating from pi_theta, which leaves the escape mechanism inert.
export TAFR_BETA_ANCHOR=0.5
export TAFR_BETA_REPLAY=0.5
export TAFR_SFT_UPDATE_INTERVAL=5
export TAFR_CHECKPOINT_INTERVAL=5

# Name carries the betas: EXPERIMENT_NAME drives the checkpoint dir, the val_dumps
# dir AND (via _common.sh) the wandb run id, so a differently-configured m02 can
# never land in the 10/10 run's history.
launch m02_grpo_fepo_ba0.5_br0.5 "Table 1: GRPO + FEPO (probe: beta_anchor=0.5, beta_replay=0.5, tighter cadence)"
