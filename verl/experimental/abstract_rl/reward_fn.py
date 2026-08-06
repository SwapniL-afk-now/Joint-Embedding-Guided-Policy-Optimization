# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Task verifier that never sees the abstraction.

Wired through ``reward.custom_reward_function`` so no reward-manager code has to
change. Without it, a ``\\boxed{}`` written inside ``<abstract>...</abstract>``
would become the graded answer, since the math parser takes the *last* boxed span.

The cut is at the **first** opening tag -- strictly more conservative than the
span parser (which resolves the last well-formed pair), and identical to it
except in the rare repeated-block case.
"""

from __future__ import annotations

from verl.utils.reward_score import default_compute_score

DEFAULT_OPEN_TAG = "<abstract>"


def strip_abstract(solution_str: str, open_tag: str = DEFAULT_OPEN_TAG) -> str:
    """Return the reasoning-and-answer part Y of a response."""
    cut = solution_str.find(open_tag)
    return solution_str if cut < 0 else solution_str[:cut]


def compute_score_ignoring_abstract(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    open_tag: str = DEFAULT_OPEN_TAG,
    **kwargs,
):
    return default_compute_score(
        data_source=data_source,
        solution_str=strip_abstract(solution_str, open_tag),
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
