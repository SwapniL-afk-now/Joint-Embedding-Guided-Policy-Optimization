# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Group Policy-Improvement Cross Entropy (GPI-CE).

A finite-group policy-improvement objective, not a GRPO policy-loss variant: the
K candidates of a prompt form a categorical distribution, reward tilts it into an
improved target q*, and the actor is fit to q* by cross entropy. No PPO ratio, no
clipping, no advantage, no critic.
"""

from .config import GPICEConfig, validate_gpi_ce_config
from .loss import compute_gpi_ce_loss
from .mixed_candidate_builder import MixedCandidateBuilder
from .target import compute_gpi_targets, masked_mean_sequence_score, target_metrics

__all__ = [
    "GPICEConfig",
    "MixedCandidateBuilder",
    "compute_gpi_ce_loss",
    "compute_gpi_targets",
    "masked_mean_sequence_score",
    "target_metrics",
    "validate_gpi_ce_config",
]
