# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Two-phase format bootstrap: selection, splicing, and switch-off."""

import os

import pytest
import torch
from test_scoring_on_cpu import BAD, GOOD, make_batch  # noqa: F401  (pytest prepends this dir)

from verl import DataProto
from verl.experimental.abstract_rl.bootstrap import (
    BootstrapState,
    build_phase2_batch,
    clean_abstract,
    fit_phase2_prompt,
    graft,
    is_degenerate,
    select_rows,
)
from verl.experimental.abstract_rl.config import AbstractRLConfig
from verl.experimental.abstract_rl.parsing import parse_batch

MODEL = os.environ.get("ABSTRACT_RL_TEST_MODEL", "/workspace/models/Qwen2.5-Math-1.5B-Instruct")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    if not os.path.isdir(MODEL):
        pytest.skip(f"tokenizer not available at {MODEL}")
    return AutoTokenizer.from_pretrained(MODEL)


def _cfg(**kw):
    base = dict(enable=True, bootstrap_steps=10, bootstrap_graft_fraction=0.5, bootstrap_max_abstract_tokens=64)
    base.update(kw)
    return AbstractRLConfig(**base)


def _spans(tokenizer, batch, cfg):
    return parse_batch(
        tokenizer, batch.batch["responses"], batch.batch["response_mask"], cfg.open_tag, cfg.close_tag, 512
    )


def _phase2(tokenizer, texts, width=64):
    pad = tokenizer.pad_token_id or 0
    responses = torch.full((len(texts), width), pad, dtype=torch.long)
    mask = torch.zeros((len(texts), width), dtype=torch.long)
    for i, t in enumerate(texts):
        ids = tokenizer(t, add_special_tokens=False)["input_ids"][:width]
        responses[i, : len(ids)] = torch.tensor(ids)
        mask[i, : len(ids)] = 1
    return DataProto.from_single_dict({"responses": responses, "response_mask": mask})


# --- selection ---------------------------------------------------------------


def test_only_rows_without_an_abstraction_are_eligible(tokenizer):
    cfg = _cfg(bootstrap_graft_fraction=0.99)
    batch = make_batch(tokenizer, [GOOD, BAD, BAD, GOOD], [True] * 4, width=512)
    spans = _spans(tokenizer, batch, cfg)
    rows = select_rows(spans, 512, cfg, seed=0)
    assert all(not spans[i].valid for i in rows)
    assert 0 not in rows and 3 not in rows


def test_graft_fraction_leaves_untagged_siblings(tokenizer):
    """The contrast is the point: a fully grafted group teaches no format signal."""
    cfg = _cfg(bootstrap_graft_fraction=0.5)
    batch = make_batch(tokenizer, [BAD] * 8, [True] * 8, width=512)
    spans = _spans(tokenizer, batch, cfg)
    rows = select_rows(spans, 512, cfg, seed=1)
    assert 0 < len(rows) < 8, "some rollouts must stay untagged"
    assert len(rows) == 4


def test_selection_is_deterministic_per_step(tokenizer):
    cfg = _cfg()
    batch = make_batch(tokenizer, [BAD] * 8, [True] * 8, width=512)
    spans = _spans(tokenizer, batch, cfg)
    assert select_rows(spans, 512, cfg, seed=7) == select_rows(spans, 512, cfg, seed=7)
    assert select_rows(spans, 512, cfg, seed=7) != select_rows(spans, 512, cfg, seed=8)


def test_rows_without_room_are_skipped(tokenizer):
    cfg = _cfg(bootstrap_max_abstract_tokens=64, bootstrap_graft_fraction=0.99)
    batch = make_batch(tokenizer, [BAD], [True], width=512)
    spans = _spans(tokenizer, batch, cfg)
    # response budget barely larger than the used length -> no room for the block
    assert select_rows(spans, spans[0].response_len + 4, cfg, seed=0) == []
    assert select_rows(spans, spans[0].response_len + 200, cfg, seed=0) == [0]


# --- splicing ----------------------------------------------------------------


def test_graft_produces_a_parseable_abstraction(tokenizer):
    cfg = _cfg()
    batch = make_batch(tokenizer, [BAD, BAD], [True, True], width=512)
    spans = _spans(tokenizer, batch, cfg)
    assert not any(s.valid for s in spans)

    # Long enough to clear the degeneracy floors, which sit above the length a
    # collapsing self-distillation loop drives abstracts down to.
    phase2 = _phase2(
        tokenizer,
        [
            "Count the selections with the binomial coefficient, then simplify the resulting ratio.",
            "Factor the expression as a difference of cubes, then solve each resulting factor for zero.",
        ],
    )
    n, rejected = graft(tokenizer, batch, spans, [0, 1], phase2, cfg)
    assert (n, rejected) == (2, 0)

    after = _spans(tokenizer, batch, cfg)
    for i, span in enumerate(after):
        assert span.valid, f"row {i} must now parse as a valid abstraction"
        assert span.n_abstract > 0
        assert "binomial" in span.abstract_text or "cubes" in span.abstract_text
    # the reasoning span still holds the original answer
    y = tokenizer.decode(batch.batch["responses"][0, : after[0].y_end])
    assert "\\boxed{7}" in y


def test_graft_keeps_the_actor_tensors_consistent(tokenizer):
    cfg = _cfg()
    batch = make_batch(tokenizer, [BAD], [True], width=512)
    spans = _spans(tokenizer, batch, cfg)
    before_len = spans[0].response_len

    graft(tokenizer, batch, spans, [0], _phase2(tokenizer, ["Use symmetry to reduce the number of cases."]), cfg)

    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"]
    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch["attention_mask"]
    position_ids = batch.batch["position_ids"]
    resp_len = responses.shape[1]
    prompt_len = input_ids.shape[1] - resp_len

    new_len = int(response_mask[0].sum())
    assert new_len > before_len, "the response must have grown"
    # response_mask stays a contiguous prefix
    assert response_mask[0, :new_len].all() and not response_mask[0, new_len:].any()
    # input_ids mirrors responses on the response half
    assert torch.equal(input_ids[0, prompt_len : prompt_len + new_len], responses[0, :new_len])
    # attention_mask agrees with the mask
    assert int(attention_mask[0, prompt_len:].sum()) == new_len
    # position_ids stay strictly incremental across the seam
    pos = position_ids[0, prompt_len : prompt_len + new_len]
    assert torch.equal(pos.diff(), torch.ones_like(pos.diff()))


def test_graft_respects_the_response_budget(tokenizer):
    cfg = _cfg(bootstrap_max_abstract_tokens=64)
    batch = make_batch(tokenizer, [BAD], [True], width=512)
    spans = _spans(tokenizer, batch, cfg)
    # long but varied: a repetitive filler would be rejected as a repetition loop
    long_text = " ".join(f"lemma{i} bounds coefficient{i} via identity{i}," for i in range(150))
    graft(tokenizer, batch, spans, [0], _phase2(tokenizer, [long_text], width=512), cfg)

    mask = batch.batch["response_mask"]
    assert int(mask[0].sum()) <= 512
    span = _spans(tokenizer, batch, cfg)[0]
    assert span.valid, "a capped block must still be closed and parseable"
    assert span.n_abstract <= cfg.bootstrap_max_abstract_tokens


def test_graft_skips_empty_phase2_output(tokenizer):
    cfg = _cfg()
    batch = make_batch(tokenizer, [BAD], [True], width=512)
    spans = _spans(tokenizer, batch, cfg)
    assert graft(tokenizer, batch, spans, [0], _phase2(tokenizer, ["   "]), cfg) == (0, 1)
    assert not _spans(tokenizer, batch, cfg)[0].valid


def test_phase2_prompt_carries_question_and_solution(tokenizer):
    cfg = _cfg()
    batch = make_batch(tokenizer, [BAD], [True], width=512)
    spans = _spans(tokenizer, batch, cfg)
    gen, kept = build_phase2_batch(tokenizer, batch, spans, [0], max_prompt_length=1024)
    assert kept == [0]
    content = gen.non_tensor_batch["raw_prompt"][0][0]["content"]
    assert "Q0" in content, "the question must be in the phase-2 prompt"
    assert "\\boxed{7}" in content, "the model's own solution must be in the phase-2 prompt"
    assert "general method" in content.lower()
    # the agent loop scores every generation unconditionally, so the verifier
    # columns must ride along or it dies with KeyError: 'data_source'
    for key in ("data_source", "reward_model", "extra_info"):
        assert key in gen.non_tensor_batch, f"phase-2 batch is missing {key}"
    assert len(gen.non_tensor_batch["data_source"]) == 1


# --- switch-off --------------------------------------------------------------


def test_disabled_when_bootstrap_steps_is_zero():
    state = BootstrapState(AbstractRLConfig(enable=True, bootstrap_steps=0))
    assert not state.active(1)


def test_active_only_inside_the_window():
    state = BootstrapState(_cfg(bootstrap_steps=10))
    assert state.active(1) and state.active(10)
    assert not state.active(11)


def test_latches_off_after_sustained_native_compliance():
    cfg = _cfg(bootstrap_steps=1000, bootstrap_native_threshold=0.3, bootstrap_patience=3)
    state = BootstrapState(cfg)
    for _ in range(2):
        state.observe(0.35)
    assert state.active(5), "must not latch off before patience is met"
    state.observe(0.35)
    assert not state.active(5), "sustained native compliance must switch grafting off"


def test_a_dip_resets_the_patience_counter():
    cfg = _cfg(bootstrap_steps=1000, bootstrap_native_threshold=0.3, bootstrap_patience=3)
    state = BootstrapState(cfg)
    state.observe(0.35)
    state.observe(0.35)
    state.observe(0.05)
    state.observe(0.35)
    assert state.active(5)


def test_latch_is_permanent():
    cfg = _cfg(bootstrap_steps=1000, bootstrap_native_threshold=0.3, bootstrap_patience=1)
    state = BootstrapState(cfg)
    state.observe(0.9)
    assert not state.active(5)
    state.observe(0.0)
    assert not state.active(6), "grafting must not resume after latching off"


def test_config_rejects_full_grafting():
    from verl.experimental.abstract_rl.config import validate_abstract_rl_config

    cfg = {
        "algorithm": {"adv_estimator": "grpo"},
        "reward": {"custom_reward_function": {"path": "x.py"}},
        "data": {"custom_cls": {"name": "AbstractRLDataset"}, "abstract_rl_instruction": "spec"},
        "custom_abstract_rl": {"enable": True, "bootstrap_steps": 10, "bootstrap_graft_fraction": 1.0},
    }
    from omegaconf import OmegaConf

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        validate_abstract_rl_config(OmegaConf.create(cfg))


def test_selection_handles_an_all_valid_batch(tokenizer):
    cfg = _cfg()
    batch = make_batch(tokenizer, [GOOD] * 4, [True] * 4, width=512)
    spans = _spans(tokenizer, batch, cfg)
    assert select_rows(spans, 512, cfg, seed=0) == []


def test_graft_drops_stale_rollout_log_probs(tokenizer):
    cfg = _cfg()
    batch = make_batch(tokenizer, [BAD], [True], width=512)
    batch.batch["rollout_log_probs"] = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
    spans = _spans(tokenizer, batch, cfg)
    graft(tokenizer, batch, spans, [0], _phase2(tokenizer, ["Use symmetry to reduce the number of cases."]), cfg)
    assert "rollout_log_probs" not in batch.batch, "grafted rows were never sampled token-by-token"


# --- prompt budget ------------------------------------------------------------


def _n_prompt_tokens(tokenizer, content):
    return len(
        tokenizer.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True)
    )


def test_phase2_prompt_is_trimmed_to_the_budget(tokenizer):
    """The rollout pads every prompt to max_prompt_length and errors above it."""
    question = "What is the sum?"
    solution = "step " * 4000
    budget = 1024
    content = fit_phase2_prompt(tokenizer, question, solution, budget)
    assert content is not None
    assert _n_prompt_tokens(tokenizer, content) <= budget
    assert question in content, "the question must survive trimming"
    assert "general method" in content.lower(), "the ask must survive trimming"


def test_phase2_prompt_untrimmed_when_it_already_fits(tokenizer):
    content = fit_phase2_prompt(tokenizer, "Q", "short solution", 1024)
    assert "short solution" in content


def test_phase2_prompt_gives_up_when_the_question_alone_overflows(tokenizer):
    assert fit_phase2_prompt(tokenizer, "word " * 5000, "sol", 256) is None


def test_build_phase2_batch_drops_rows_that_cannot_fit(tokenizer):
    cfg = _cfg()
    batch = make_batch(tokenizer, [BAD, BAD], [True, True], width=512)
    spans = _spans(tokenizer, batch, cfg)
    gen, kept = build_phase2_batch(tokenizer, batch, spans, [0, 1], max_prompt_length=8)
    assert gen is None and kept == []


def test_every_built_phase2_prompt_fits(tokenizer):
    cfg = _cfg()
    long_solution = "Reasoning. " * 900 + "Therefore \\boxed{7}."
    batch = make_batch(tokenizer, [long_solution], [True], width=4096)
    spans = _spans(tokenizer, batch, cfg)
    gen, kept = build_phase2_batch(tokenizer, batch, spans, [0], max_prompt_length=1024)
    assert kept == [0]
    for messages in gen.non_tensor_batch["raw_prompt"]:
        assert _n_prompt_tokens(tokenizer, messages[0]["content"]) <= 1024


# --- content quality ----------------------------------------------------------


def test_boxed_answers_are_stripped_from_abstractions():
    assert "boxed" not in clean_abstract("Use symmetry \\boxed{30} throughout.")
    assert clean_abstract("Use symmetry.") == "Use symmetry."


def test_token_soup_is_rejected():
    # observed verbatim from Qwen2.5-Math-1.5B at temperature 1.0
    assert is_degenerate("`. abstract ({`text field={text`{`text field={text`{`text")
    assert is_degenerate("a a a a a a a a a a a a a a a")
    assert is_degenerate("short")
    assert is_degenerate("")
    assert is_degenerate("Solve <|im_start|> the thing carefully now")


def test_real_abstractions_are_kept():
    assert not is_degenerate(
        "Apply the principle of inclusion-exclusion for three sets: add the sizes, "
        "subtract the pairwise intersections, then add back the triple intersection."
    )
    assert not is_degenerate(
        "Factor the common multiple out of an arithmetic series, then use the closed "
        "form n(n+1)/2 for the sum of the first n integers."
    )


def test_degenerate_graft_leaves_the_row_untagged(tokenizer):
    cfg = _cfg()
    batch = make_batch(tokenizer, [BAD], [True], width=512)
    spans = _spans(tokenizer, batch, cfg)
    grafted, rejected = graft(
        tokenizer, batch, spans, [0], _phase2(tokenizer, ["`{`text field={text`{`text field={text`"]), cfg
    )
    assert (grafted, rejected) == (0, 1)
    assert not _spans(tokenizer, batch, cfg)[0].valid, "garbage must not be reinforced"


def test_abstract_cache_grafts_frozen_targets(tokenizer, tmp_path):
    """The offline cache path must produce the same parseable result as phase-2."""
    import pandas as pd

    from verl.experimental.abstract_rl.bootstrap import AbstractCache, graft_texts

    path = tmp_path / "abstracts.parquet"
    pd.DataFrame(
        [{"question": "Q1", "abstract": "Set up an equation from the totals, then solve it and check the result."}]
    ).to_parquet(path)

    cache = AbstractCache(str(path))
    assert cache.get("Q1")
    assert cache.get("missing") is None
    assert cache.hit_rate == 0.5  # one hit, one miss

    cfg = _cfg()
    batch = make_batch(tokenizer, [BAD, BAD], [True, True], width=512)
    spans = _spans(tokenizer, batch, cfg)
    # row 0 hits the cache, row 1 misses -> empty text -> rejected, never grafted
    grafted, rejected = graft_texts(tokenizer, batch, spans, [0, 1], [cache.get("Q1"), ""], cfg)
    assert (grafted, rejected) == (1, 1)

    after = _spans(tokenizer, batch, cfg)
    assert after[0].valid and "equation" in after[0].abstract_text
    assert not after[1].valid
