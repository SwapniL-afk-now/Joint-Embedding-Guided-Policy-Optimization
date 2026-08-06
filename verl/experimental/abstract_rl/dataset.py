# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dataset that carries the abstraction instruction through length filtering.

Injecting the instruction later -- in ``_get_gen_batch`` -- silently breaks the
prompt-length contract: ``maybe_filter_out_long_prompts`` measures the *raw*
``prompt`` column at load time, so a prompt that passes at 1010 tokens becomes
~1086 once the instruction is appended, overflows ``max_prompt_length``, and the
rollout dies in ``agent_loop._postprocess`` on

    RuntimeError: Sizes of tensors must match except in dimension 0.
                  Expected size 1024 but got size 1051

Only prompts near the ceiling overflow, so this surfaces tens of steps in, not
immediately. Appending here -- before the parent filters -- means the filter sees
the true final prompt and drops what genuinely does not fit.

Note this cannot be fixed by overriding ``_build_messages``: that runs per item in
``__getitem__``, long after the dataframe has been filtered.

Wired via ``data.custom_cls`` and used for both the train and val datasets:

    data.custom_cls.path=verl/experimental/abstract_rl/dataset.py
    data.custom_cls.name=AbstractRLDataset
    +data.abstract_rl_instruction=strong
"""

from __future__ import annotations

import datasets

# Absolute imports: data.custom_cls loads this file by path as a standalone module,
# so a relative import raises "attempted relative import with no known parent package".
from verl.experimental.abstract_rl.config import INSTRUCTIONS, ROLLOUT_INSTRUCTION
from verl.utils.dataset.rl_dataset import RLHFDataset


def append_instruction(messages: list[dict], instruction: str) -> list[dict]:
    """Append the instruction to the last user turn, idempotently."""
    out = [dict(m) for m in messages]
    for msg in reversed(out):
        if msg.get("role") == "user":
            if instruction not in msg["content"]:
                msg["content"] = f"{msg['content']}\n\n{instruction}"
            break
    return out


class AbstractRLDataset(RLHFDataset):
    """RLHFDataset that appends the abstraction instruction before filtering.

    The variant comes from ``data.abstract_rl_instruction`` (``spec`` or
    ``strong``), or ``data.abstract_rl_instruction_override`` for free text. It
    must match ``custom_abstract_rl.instruction_variant`` so the trainer-side
    injection stays a no-op; a mismatch would append a second instruction.
    """

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        dataframe = dataframe if dataframe is not None else self.dataframe
        override = self.config.get("abstract_rl_instruction_override", "")
        variant = self.config.get("abstract_rl_instruction", "spec")
        instruction = override or INSTRUCTIONS.get(variant, ROLLOUT_INSTRUCTION)
        prompt_key = self.prompt_key

        dataframe = dataframe.map(
            lambda doc: {prompt_key: append_instruction(doc[prompt_key], instruction)},
            num_proc=self.num_workers,
            desc="Appending the abstraction instruction",
        )
        return super().maybe_filter_out_long_prompts(dataframe)
