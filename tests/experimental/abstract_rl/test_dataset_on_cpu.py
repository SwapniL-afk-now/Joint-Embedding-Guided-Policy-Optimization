# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The instruction must be inside the prompt *before* length filtering.

Regression test for the crash that killed abstractrl-full-math15b at step 52:
injecting after the filter let a 1010-token prompt become 1086 and blow up
agent_loop._postprocess on mismatched pad widths.
"""

import os

import datasets
import pytest
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.experimental.abstract_rl.config import INSTRUCTIONS
from verl.experimental.abstract_rl.dataset import AbstractRLDataset, append_instruction

MODEL = os.environ.get("ABSTRACT_RL_TEST_MODEL", "/workspace/models/Qwen2.5-Math-1.5B-Instruct")


@pytest.fixture(scope="module")
def tokenizer():
    if not os.path.isdir(MODEL):
        pytest.skip(f"tokenizer not available at {MODEL}")
    return AutoTokenizer.from_pretrained(MODEL)


def test_append_is_idempotent():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "solve"}]
    once = append_instruction(msgs, "INSTR")
    twice = append_instruction(once, "INSTR")
    assert once[-1]["content"].endswith("INSTR")
    assert once == twice, "re-injection must not duplicate the instruction"
    assert msgs[-1]["content"] == "solve", "the input must not be mutated"


def test_append_targets_the_last_user_turn():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    out = append_instruction(msgs, "INSTR")
    assert out[0]["content"] == "first"
    assert out[2]["content"].endswith("INSTR")


def _dataset(tokenizer, tmp_path, rows, max_prompt_length, variant="spec"):
    path = str(tmp_path / "d.parquet")
    datasets.Dataset.from_list(rows).to_parquet(path)
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": max_prompt_length,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 1,
            "abstract_rl_instruction": variant,
            "truncation": "error",
        }
    )
    return AbstractRLDataset(data_files=path, tokenizer=tokenizer, config=config)


def _row(text):
    return {
        "prompt": [{"role": "user", "content": text}],
        "data_source": "math_dapo",
        "reward_model": {"style": "rule", "ground_truth": "1"},
        "extra_info": {"problem": text},
    }


def test_filter_measures_the_prompt_with_the_instruction(tokenizer, tmp_path):
    """A prompt that fits only *without* the instruction must be filtered out."""
    instruction_len = len(tokenizer(INSTRUCTIONS["spec"], add_special_tokens=False)["input_ids"])
    body = "word " * 200
    body_len = len(tokenizer(body, add_special_tokens=False)["input_ids"])
    # budget admits the body alone, but not the body plus the instruction
    budget = body_len + instruction_len // 2

    ds = _dataset(tokenizer, tmp_path, [_row(body)], max_prompt_length=budget)
    assert len(ds) == 0, "the overlong combined prompt must not survive filtering"

    roomy = _dataset(tokenizer, tmp_path, [_row(body)], max_prompt_length=budget + instruction_len * 2)
    assert len(roomy) == 1


def test_surviving_rows_fit_the_budget_with_the_instruction(tokenizer, tmp_path):
    rows = [_row("word " * n) for n in (10, 80, 200, 400)]
    budget = 400
    ds = _dataset(tokenizer, tmp_path, rows, max_prompt_length=budget)
    assert len(ds) > 0
    for i in range(len(ds)):
        messages = ds.dataframe[i]["prompt"]
        assert INSTRUCTIONS["spec"] in messages[-1]["content"], "instruction must be carried into the row"
        n = len(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True))
        assert n <= budget, f"row {i} is {n} tokens, over the {budget} budget the rollout pads to"


def test_strong_variant_is_honoured(tokenizer, tmp_path):
    ds = _dataset(tokenizer, tmp_path, [_row("short question")], max_prompt_length=2048, variant="strong")
    content = ds.dataframe[0]["prompt"][-1]["content"]
    assert INSTRUCTIONS["strong"] in content and INSTRUCTIONS["spec"] not in content


def test_trainer_hook_does_not_double_inject(tokenizer, tmp_path):
    """The dataset injects; the trainer-side hook must then be a no-op."""
    import numpy as np

    from verl.experimental.abstract_rl import trainer_hooks
    from verl.experimental.abstract_rl.config import AbstractRLConfig

    ds = _dataset(tokenizer, tmp_path, [_row("short question")], max_prompt_length=2048)
    raw = np.empty(1, dtype=object)
    raw[0] = list(ds.dataframe[0]["prompt"])

    class _G:
        non_tensor_batch = {"raw_prompt": raw}

    gen = _G()
    before = gen.non_tensor_batch["raw_prompt"][0][-1]["content"]
    trainer_hooks.inject_instruction(AbstractRLConfig(enable=True), gen)
    after = gen.non_tensor_batch["raw_prompt"][0][-1]["content"]
    assert before == after
    assert after.count(INSTRUCTIONS["spec"]) == 1


def test_loads_via_the_custom_cls_path():
    """data.custom_cls imports this file standalone -- relative imports break it."""
    from verl.utils.import_utils import load_extern_object

    cls = load_extern_object("verl/experimental/abstract_rl/dataset.py", "AbstractRLDataset")
    assert cls.__name__ == "AbstractRLDataset"
    assert issubclass(cls, __import__("torch").utils.data.Dataset)
