# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""bootstrap_mode=fill_gap: order-agnostic pass-1 check, PRIOR_TEMPLATE pass-2
prompt, whole-response reassembly, instruction placement, and the reward-side
abstract-first strip fix."""

import os

import pytest
import torch
from test_scoring_on_cpu import make_batch  # noqa: F401  (pytest prepends this dir)

from verl.experimental.abstract_rl.bootstrap import build_prior_batch, find_abstract_text, reposition_responses
from verl.experimental.abstract_rl.config import AbstractRLConfig, apply_instruction
from verl.experimental.abstract_rl.parsing import parse_batch
from verl.experimental.abstract_rl.reward_fn import strip_abstract

MODEL = os.environ.get("ABSTRACT_RL_TEST_MODEL", "/workspace/models/Qwen2.5-Math-1.5B-Instruct")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    if not os.path.isdir(MODEL):
        pytest.skip(f"tokenizer not available at {MODEL}")
    return AutoTokenizer.from_pretrained(MODEL)


def _cfg(**kw):
    base = dict(enable=True, bootstrap_mode="fill_gap", bootstrap_steps=10, bootstrap_graft_fraction=0.5)
    base.update(kw)
    return AbstractRLConfig(**base)


# --- find_abstract_text: order-agnostic pass-1 check --------------------------


def test_finds_a_trailing_abstract():
    found = find_abstract_text("Reasoning. \\boxed{7}.\n<abstract>Use symmetry.</abstract>", "<abstract>", "</abstract>")
    assert found == ("Use symmetry.", "Reasoning. \\boxed{7}.")


def test_finds_a_leading_abstract():
    found = find_abstract_text("<abstract>Use symmetry.</abstract>\nReasoning. \\boxed{7}.", "<abstract>", "</abstract>")
    assert found == ("Use symmetry.", "Reasoning. \\boxed{7}.")


def test_finds_a_mid_response_abstract():
    found = find_abstract_text(
        "Step one. <abstract>Use symmetry.</abstract> Therefore \\boxed{7}.", "<abstract>", "</abstract>"
    )
    assert found == ("Use symmetry.", "Step one.  Therefore \\boxed{7}.")


def test_missing_tags_returns_none():
    assert find_abstract_text("Reasoning. \\boxed{7}.", "<abstract>", "</abstract>") is None


def test_missing_close_tag_returns_none():
    assert find_abstract_text("Reasoning. <abstract>Use symmetry.", "<abstract>", "</abstract>") is None


# --- reposition_responses: whole-row replace -----------------------------------


def test_reposition_produces_abstract_first(tokenizer):
    cfg = _cfg()
    batch = make_batch(tokenizer, ["Reasoning here. Therefore \\boxed{7}."], [True], width=512)
    n, rejected = reposition_responses(tokenizer, batch, {0: ("Use symmetry to reduce the number of cases that need to be checked by hand.", "Reasoning here. Therefore \\boxed{7}.")}, cfg)
    assert (n, rejected) == (1, 0)

    spans = parse_batch(
        tokenizer, batch.batch["responses"], batch.batch["response_mask"], cfg.open_tag, cfg.close_tag, 512, leading=True
    )
    assert spans[0].valid
    assert spans[0].a_start == 0 or spans[0].a_start < spans[0].y_start, "abstract must come before Y"
    y = tokenizer.decode(batch.batch["responses"][0, spans[0].y_start : spans[0].y_end])
    assert "\\boxed{7}" in y


def test_reposition_zeroes_the_tail_when_shrinking(tokenizer):
    """Old content may be longer than the reassembled sequence -- the leftover
    tail must not survive as stale tokens."""
    cfg = _cfg()
    long_text = "Reasoning. " * 200 + "Therefore \\boxed{7}."
    batch = make_batch(tokenizer, [long_text], [True], width=1024)
    old_len = int(batch.batch["response_mask"][0].sum())

    n, rejected = reposition_responses(
        tokenizer, batch, {0: ("Set up an equation from the totals, then solve it and check the result.", "Short Y.")}, cfg
    )
    assert (n, rejected) == (1, 0)
    new_len = int(batch.batch["response_mask"][0].sum())
    assert new_len < old_len
    assert not batch.batch["response_mask"][0, new_len:].any()
    assert not batch.batch["responses"][0, new_len:].any()


def test_reposition_rejects_degenerate_abstract(tokenizer):
    cfg = _cfg()
    batch = make_batch(tokenizer, ["Reasoning here. Therefore \\boxed{7}."], [True], width=512)
    n, rejected = reposition_responses(tokenizer, batch, {0: ("`{`text field={text`", "Reasoning here.")}, cfg)
    assert (n, rejected) == (0, 1)


def test_reposition_rejects_oversize_sequence(tokenizer):
    cfg = _cfg()
    batch = make_batch(tokenizer, ["short"], [True], width=32)
    long_y = "word " * 500
    n, rejected = reposition_responses(tokenizer, batch, {0: ("A real strategy statement here.", long_y)}, cfg)
    assert (n, rejected) == (0, 1)


def test_reposition_drops_stale_rollout_log_probs(tokenizer):
    cfg = _cfg()
    batch = make_batch(tokenizer, ["Reasoning here. Therefore \\boxed{7}."], [True], width=512)
    batch.batch["rollout_log_probs"] = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
    reposition_responses(
        tokenizer, batch, {0: ("Use symmetry to reduce the number of cases that need checking.", "Reasoning here. Therefore \\boxed{7}.")}, cfg
    )
    assert "rollout_log_probs" not in batch.batch


def test_reposition_empty_assignments_is_a_noop(tokenizer):
    cfg = _cfg()
    batch = make_batch(tokenizer, ["Reasoning here."], [True], width=512)
    assert reposition_responses(tokenizer, batch, {}, cfg) == (0, 0)


# --- build_prior_batch: question-only pass-2 prompt ----------------------------


def test_prior_batch_carries_question_but_not_a_solution(tokenizer):
    batch = make_batch(tokenizer, ["Reasoning here. Therefore \\boxed{7}."], [True], width=512)
    gen, kept = build_prior_batch(tokenizer, batch, [0], max_prompt_length=1024)
    assert kept == [0]
    content = gen.non_tensor_batch["raw_prompt"][0][0]["content"]
    assert "Q0" in content
    assert "\\boxed{7}" not in content, "pass-2 must not see the pass-1 solution"
    assert "abstract" in content.lower()
    for key in ("data_source", "reward_model", "extra_info"):
        assert key in gen.non_tensor_batch


def test_prior_batch_drops_rows_that_cannot_fit(tokenizer):
    batch = make_batch(tokenizer, ["Reasoning here."], [True], width=512)
    gen, kept = build_prior_batch(tokenizer, batch, [0], max_prompt_length=1)
    assert gen is None and kept == []


# --- instruction placement -----------------------------------------------------


def test_user_turn_placement_appends_to_last_user_message():
    messages = [{"role": "user", "content": "Solve this."}]
    out = apply_instruction(messages, "CUE", "user_turn")
    assert out[0]["role"] == "user"
    assert "CUE" in out[0]["content"] and "Solve this." in out[0]["content"]


def test_system_placement_prepends_a_system_message():
    messages = [{"role": "user", "content": "Solve this."}]
    out = apply_instruction(messages, "CUE", "system")
    assert out[0] == {"role": "system", "content": "CUE"}
    assert out[1] == messages[0]


def test_system_placement_extends_an_existing_system_message():
    messages = [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Solve this."}]
    out = apply_instruction(messages, "CUE", "system")
    assert len(out) == 2
    assert "Be terse." in out[0]["content"] and "CUE" in out[0]["content"]


def test_instruction_insertion_is_idempotent():
    messages = [{"role": "user", "content": "Solve this."}]
    once = apply_instruction(messages, "CUE", "user_turn")
    twice = apply_instruction(once, "CUE", "user_turn")
    assert once == twice
    system_once = apply_instruction(messages, "CUE", "system")
    system_twice = apply_instruction(system_once, "CUE", "system")
    assert system_once == system_twice


# --- reward-side strip: abstract-first layout ----------------------------------


def test_strip_abstract_handles_leading_layout():
    text = "<abstract>Use symmetry.</abstract>\nReasoning. \\boxed{7}."
    assert strip_abstract(text) == "Reasoning. \\boxed{7}."


def test_strip_abstract_handles_trailing_layout_unchanged():
    text = "Reasoning. \\boxed{7}.\n<abstract>Use symmetry.</abstract>"
    assert strip_abstract(text) == "Reasoning. \\boxed{7}."


def test_strip_abstract_handles_mid_response_layout():
    text = "Step one. <abstract>Use symmetry.</abstract> Therefore \\boxed{7}."
    assert strip_abstract(text) == "Step one.  Therefore \\boxed{7}."


def test_strip_abstract_falls_back_to_truncate_when_close_tag_missing():
    text = "Reasoning. <abstract>Use symmetry, no close tag."
    assert strip_abstract(text) == "Reasoning. "


def test_strip_abstract_is_a_noop_without_tags():
    assert strip_abstract("Reasoning. \\boxed{7}.") == "Reasoning. \\boxed{7}."


def test_boxed_inside_abstract_never_reaches_the_verifier():
    text = "<abstract>Use \\boxed{99} as a decoy.</abstract>\nReasoning. \\boxed{7}."
    stripped = strip_abstract(text)
    assert "\\boxed{99}" not in stripped
    assert "\\boxed{7}" in stripped


# --- config validation ----------------------------------------------------------


def test_config_rejects_bad_instruction_placement():
    from omegaconf import OmegaConf

    from verl.experimental.abstract_rl.config import validate_abstract_rl_config

    cfg = {
        "algorithm": {"adv_estimator": "grpo"},
        "reward": {"custom_reward_function": {"path": "x.py"}},
        "data": {"custom_cls": {"name": "AbstractRLDataset"}, "abstract_rl_instruction": "spec"},
        "custom_abstract_rl": {"enable": True, "instruction_placement": "nowhere"},
    }
    with pytest.raises(ValueError, match="instruction_placement"):
        validate_abstract_rl_config(OmegaConf.create(cfg))


def test_config_rejects_bad_bootstrap_mode():
    from omegaconf import OmegaConf

    from verl.experimental.abstract_rl.config import validate_abstract_rl_config

    cfg = {
        "algorithm": {"adv_estimator": "grpo"},
        "reward": {"custom_reward_function": {"path": "x.py"}},
        "data": {"custom_cls": {"name": "AbstractRLDataset"}, "abstract_rl_instruction": "spec"},
        "custom_abstract_rl": {"enable": True, "bootstrap_mode": "sideways"},
    }
    with pytest.raises(ValueError, match="bootstrap_mode"):
        validate_abstract_rl_config(OmegaConf.create(cfg))
