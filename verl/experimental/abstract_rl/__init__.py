# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""On-policy abstract-reasoning compression with GRPO.

Each rollout emits a reasoning trace + ``\\boxed{}`` answer followed by a short
``<abstract>...</abstract>`` summary of the *general* method. The abstraction is

  * rewarded through the utility term ``U_i = r_i (1 + clip(Delta_i, -1, 1))``,
    where ``Delta_i`` is the per-token likelihood gain the abstraction gives the
    frozen old policy when re-predicting the reasoning trace, and
  * penalised by a compression KL toward a question-only prior, so it only keeps
    trace-specific information that pays for itself.

No second generation stage, no critic, no extra trainable parameters: the
"teacher" is the pre-update actor snapshot that already produces ``old_log_probs``.
"""

from .config import AbstractRLConfig, abstract_rl_enabled, validate_abstract_rl_config

__all__ = ["AbstractRLConfig", "abstract_rl_enabled", "validate_abstract_rl_config"]
