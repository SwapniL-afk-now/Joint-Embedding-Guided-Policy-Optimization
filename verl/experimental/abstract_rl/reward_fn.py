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

Removes the abstract **block** (open tag through close tag, inclusive) wherever
it sits, rather than truncating at the open tag -- that used to be equivalent
under the trailing layout (nothing followed the close tag anyway), but
``bootstrap_mode=fill_gap`` always reassembles abstract-first, and truncating at
the first open tag there would zero out the entire response. Block removal is
correct for both layouts. Only the first well-formed pair is removed; a missing
close tag falls back to the old truncate-at-open behavior (nothing to remove Y
from safely).
"""

from __future__ import annotations

from verl.utils.reward_score import default_compute_score

DEFAULT_OPEN_TAG = "<abstract>"
DEFAULT_CLOSE_TAG = "</abstract>"


def strip_abstract(solution_str: str, open_tag: str = DEFAULT_OPEN_TAG, close_tag: str = DEFAULT_CLOSE_TAG) -> str:
    """Return the reasoning-and-answer part Y of a response."""
    start = solution_str.find(open_tag)
    if start < 0:
        return solution_str
    end = solution_str.find(close_tag, start + len(open_tag))
    if end < 0:
        return solution_str[:start]
    return (solution_str[:start] + solution_str[end + len(close_tag) :]).strip()


def compute_score_ignoring_abstract(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    open_tag: str = DEFAULT_OPEN_TAG,
    close_tag: str = DEFAULT_CLOSE_TAG,
    **kwargs,
):
    return default_compute_score(
        data_source=data_source,
        solution_str=strip_abstract(solution_str, open_tag, close_tag),
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
