# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Scoring-batch construction, Delta/U arithmetic, and reward shaping."""

import os
import types

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from verl import DataProto
from verl.experimental.abstract_rl import trainer_hooks
from verl.experimental.abstract_rl.config import AbstractRLConfig
from verl.experimental.abstract_rl.parsing import build_masks, parse_batch
from verl.experimental.abstract_rl.scoring import (
    CLASS_BASE,
    CLASS_COND,
    CLASS_PRIOR,
    build_scoring_batch,
    compute_delta_utility,
)

MODEL = os.environ.get("ABSTRACT_RL_TEST_MODEL", "/workspace/models/Qwen2.5-Math-1.5B-Instruct")
GOOD = "Reasoning here. Therefore \\boxed{7}.\n<abstract>Use symmetry.</abstract>"
BAD = "Reasoning here. Therefore \\boxed{7}."


@pytest.fixture(scope="module")
def tokenizer():
    if not os.path.isdir(MODEL):
        pytest.skip(f"tokenizer not available at {MODEL}")
    return AutoTokenizer.from_pretrained(MODEL)


def make_batch(tokenizer, texts, rewards, width=128, prompt_width=16):
    n = len(texts)
    pad = tokenizer.pad_token_id or 0
    responses = torch.full((n, width), pad, dtype=torch.long)
    response_mask = torch.zeros((n, width), dtype=torch.long)
    for i, text in enumerate(texts):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        responses[i, : len(ids)] = torch.tensor(ids)
        response_mask[i, : len(ids)] = 1
    prompts = torch.full((n, prompt_width), pad, dtype=torch.long)
    attention_mask = torch.cat([torch.ones((n, prompt_width), dtype=torch.long), response_mask], dim=1)
    input_ids = torch.cat([prompts, responses], dim=1)
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0) * attention_mask

    scores = torch.zeros((n, width))
    last = response_mask.sum(dim=1) - 1
    for i in range(n):
        scores[i, last[i]] = 1.0 if rewards[i] else -1.0

    batch = DataProto.from_single_dict(
        {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "token_level_scores": scores,
            "token_level_rewards": scores,  # aliased, exactly as the trainer sets it
            "old_log_probs": torch.full((n, width), -0.5),
            "uid": np.array(["g0"] * n, dtype=object),
            "acc": np.array([float(bool(r)) for r in rewards], dtype=object),
            "extra_info": np.array([{"problem": f"Q{i}"} for i in range(n)], dtype=object),
        }
    )
    return batch


def _spans(tokenizer, batch, cfg):
    return parse_batch(
        tokenizer,
        batch.batch["responses"],
        batch.batch["response_mask"],
        cfg.open_tag,
        cfg.close_tag,
        cfg.max_scan_tokens,
    )


def test_scoring_rows_score_only_the_target(tokenizer):
    cfg = AbstractRLConfig(enable=True)
    batch = make_batch(tokenizer, [GOOD], [True])
    spans = _spans(tokenizer, batch, cfg)
    td, meta = build_scoring_batch(tokenizer, batch, spans, [0], cfg)

    assert [m[1] for m in meta] == [CLASS_COND, CLASS_BASE, CLASS_PRIOR]
    prompts = td["prompts"]
    resp_mask, attn = td["response_mask"], td["attention_mask"]
    p_width = prompts.shape[1]

    for k, (_, _, tgt_len) in enumerate(meta):
        # the loss mask covers exactly the target tokens, never the prompt
        assert resp_mask[k].sum() == tgt_len
        assert resp_mask[k, :tgt_len].all() and not resp_mask[k, tgt_len:].any()
        # prompts are left padded, targets right padded, and the attention mask agrees
        prompt_len = int(attn[k, :p_width].sum())
        assert attn[k, p_width - prompt_len : p_width].all()
        assert not attn[k, : p_width - prompt_len].any()
        assert int(attn[k, p_width:].sum()) == tgt_len


def test_targets_are_slices_of_the_rollout_ids(tokenizer):
    cfg = AbstractRLConfig(enable=True)
    batch = make_batch(tokenizer, [GOOD], [True])
    spans = _spans(tokenizer, batch, cfg)
    span = spans[0]
    td, meta = build_scoring_batch(tokenizer, batch, spans, [0], cfg)
    responses = batch.batch["responses"]

    for k, (_, cls, tgt_len) in enumerate(meta):
        target = td["responses"][k, :tgt_len]
        if cls == CLASS_PRIOR:
            expected = responses[0, span.a_start : span.a_end]
        else:
            expected = responses[0, : span.y_end]
        assert torch.equal(target, expected), "targets must be exact rollout id slices"


def test_baseline_row_differs_from_conditioned_only_by_the_abstract(tokenizer):
    cfg = AbstractRLConfig(enable=True)
    batch = make_batch(tokenizer, [GOOD], [True])
    spans = _spans(tokenizer, batch, cfg)
    td, meta = build_scoring_batch(tokenizer, batch, spans, [0], cfg)
    cond = td["prompts"][0][td["attention_mask"][0, : td["prompts"].shape[1]].bool()]
    base = td["prompts"][1][td["attention_mask"][1, : td["prompts"].shape[1]].bool()]
    assert not torch.equal(cond, base)
    assert "Use symmetry." in tokenizer.decode(cond)
    assert "Use symmetry." not in tokenizer.decode(base)


def test_stored_baseline_drops_the_matched_rows(tokenizer):
    cfg = AbstractRLConfig(enable=True, delta_baseline="stored")
    batch = make_batch(tokenizer, [GOOD], [True])
    spans = _spans(tokenizer, batch, cfg)
    _, meta = build_scoring_batch(tokenizer, batch, spans, [0], cfg)
    assert [m[1] for m in meta] == [CLASS_COND, CLASS_PRIOR]


def test_invalid_rows_are_never_scored(tokenizer):
    cfg = AbstractRLConfig(enable=True)
    batch = make_batch(tokenizer, [BAD], [True])
    spans = _spans(tokenizer, batch, cfg)
    td, meta = build_scoring_batch(tokenizer, batch, spans, [], cfg)
    assert td is None and meta == []


def _fake_result(spans, reward, cond, base, prior_value, cfg, resp_len, stored=None):
    meta, rows = [], []
    max_len = max(max(s.n_reasoning, s.n_abstract) for s in spans)
    for i, span in enumerate(spans):
        if not span.valid:
            continue
        meta.append((i, CLASS_COND, span.n_reasoning))
        rows.append(torch.full((max_len,), cond[i] / max(span.n_reasoning, 1)))
        if cfg.delta_baseline == "matched":
            meta.append((i, CLASS_BASE, span.n_reasoning))
            rows.append(torch.full((max_len,), base[i] / max(span.n_reasoning, 1)))
        meta.append((i, CLASS_PRIOR, span.n_abstract))
        rows.append(torch.full((max_len,), prior_value))
    log_probs = torch.stack(rows) if rows else None
    if stored is None:
        stored = torch.zeros(len(spans))
    return compute_delta_utility(spans, reward, log_probs, meta, stored, cfg, resp_len)


def test_delta_utility_arithmetic(tokenizer):
    cfg = AbstractRLConfig(enable=True)
    batch = make_batch(tokenizer, [GOOD, GOOD, BAD], [True, False, True])
    spans = _spans(tokenizer, batch, cfg)
    n_y = spans[0].n_reasoning
    # row 0: +0.25/token gain; row 1: same gain but wrong answer; row 2: no abstraction
    cond = {0: 0.25 * n_y, 1: 0.25 * n_y}
    base = {0: 0.0, 1: 0.0}
    reward = torch.tensor([1.0, 0.0, 1.0])
    res = _fake_result(spans, reward, cond, base, -0.1, cfg, batch.batch["responses"].shape[1])

    assert res.delta[0] == pytest.approx(0.25, abs=1e-5)
    # valid_bonus (0.2 default) is added on top of Delta, then reclamped to [-1, 1]
    assert res.delta_tilde[0] == pytest.approx(0.45, abs=1e-5)
    assert res.utility[0] == pytest.approx(1.45, abs=1e-5)
    assert res.utility[1] == 0.0, "incorrect reasoning must give U = 0"
    assert res.delta_tilde[2] == 0.0, "invalid_delta='zero': no abstraction means no bonus, not a penalty"
    assert res.utility[2] == 1.0, "a correct answer must keep its task reward with no abstraction"
    assert torch.isfinite(res.utility).all()
    assert (res.utility >= 0).all() and (res.utility <= 2).all()


def test_invalid_delta_penalty_restores_the_spec(tokenizer):
    """invalid_delta='penalty' is the spec: no abstraction zeroes the whole reward.

    Off by default because at a ~10% valid rate it erases correctness from the
    reward entirely -- measured 81-91% flat groups and accuracy 0.29 -> 0.07.
    """
    cfg = AbstractRLConfig(enable=True, invalid_delta="penalty")
    batch = make_batch(tokenizer, [GOOD, BAD], [True, True])
    spans = _spans(tokenizer, batch, cfg)
    n_y = spans[0].n_reasoning
    res = _fake_result(spans, torch.ones(2), {0: 0.0}, {0: 0.0}, -0.1, cfg, batch.batch["responses"].shape[1])
    assert n_y > 0
    assert res.delta_tilde[1] == -1.0 and res.utility[1] == 0.0
    # and the default keeps the same correct row at its full task reward
    zero = _fake_result(
        _spans(tokenizer, batch, AbstractRLConfig(enable=True)),
        torch.ones(2),
        {0: 0.0},
        {0: 0.0},
        -0.1,
        AbstractRLConfig(enable=True),
        batch.batch["responses"].shape[1],
    )
    assert zero.utility[1] == 1.0


def test_delta_is_clipped(tokenizer):
    cfg = AbstractRLConfig(enable=True)
    batch = make_batch(tokenizer, [GOOD, GOOD], [True, True])
    spans = _spans(tokenizer, batch, cfg)
    n_y = spans[0].n_reasoning
    res = _fake_result(
        spans,
        torch.ones(2),
        {0: 5.0 * n_y, 1: -5.0 * n_y},
        {0: 0.0, 1: 0.0},
        -0.1,
        cfg,
        batch.batch["responses"].shape[1],
    )
    # +bonus pulls the top clip in (already at ceiling, no room) and the bottom
    # clip up (0.2 headroom before it re-hits -1)
    assert res.delta_tilde[0] == 1.0 and res.delta_tilde[1] == pytest.approx(-0.8, abs=1e-5)
    assert res.utility[0] == 2.0 and res.utility[1] == pytest.approx(0.2, abs=1e-5)


def test_use_utility_false_reduces_to_task_reward(tokenizer):
    cfg = AbstractRLConfig(enable=True, use_utility=False)
    batch = make_batch(tokenizer, [GOOD, BAD], [True, False])
    spans = _spans(tokenizer, batch, cfg)
    td, meta = build_scoring_batch(tokenizer, batch, spans, [0], cfg)
    assert [m[1] for m in meta] == [CLASS_PRIOR], "no sufficiency rows without the utility term"
    res = compute_delta_utility(
        spans, torch.tensor([1.0, 0.0]), None, [], torch.zeros(2), cfg, batch.batch["responses"].shape[1]
    )
    assert torch.equal(res.utility, torch.tensor([1.0, 0.0]))


def test_length_penalty_decreases_utility_monotonically_with_abstract_length(tokenizer):
    """plan section 15: U_i must strictly decrease in |A_i| at fixed Delta, or the
    marginal abstraction token is still free."""
    short = "Reasoning here. Therefore \\boxed{7}.\n<abstract>Use symmetry.</abstract>"
    long_ = "Reasoning here. Therefore \\boxed{7}.\n<abstract>" + ("Use symmetry. " * 20) + "</abstract>"
    cfg = AbstractRLConfig(enable=True, length_penalty_coef=0.3, max_abstract_tokens=1000, valid_bonus=0.0)
    batch = make_batch(tokenizer, [short, long_], [True, True])
    spans = _spans(tokenizer, batch, cfg)
    assert spans[1].n_abstract > spans[0].n_abstract
    n_y = spans[0].n_reasoning
    res = _fake_result(spans, torch.ones(2), {0: 0.0, 1: 0.0}, {0: 0.0, 1: 0.0}, -0.1, cfg, batch.batch["responses"].shape[1])
    assert res.utility[1] < res.utility[0], "the longer abstract must score strictly lower at equal Delta"
    expected_gap = cfg.length_penalty_coef * (spans[1].n_abstract - spans[0].n_abstract) / cfg.max_abstract_tokens
    assert float(res.utility[0] - res.utility[1]) == pytest.approx(expected_gap, abs=1e-4)


def test_length_penalty_coef_zero_reproduces_pre_section15_utility(tokenizer):
    """beta=0 is a regression guard: identical U_i to before the length penalty existed."""
    cfg_off = AbstractRLConfig(enable=True, length_penalty_coef=0.0)
    cfg_on = AbstractRLConfig(enable=True, length_penalty_coef=0.3, max_abstract_tokens=1000)
    batch = make_batch(tokenizer, [GOOD, GOOD], [True, True])
    spans_off = _spans(tokenizer, batch, cfg_off)
    spans_on = _spans(tokenizer, batch, cfg_on)
    res_off = _fake_result(spans_off, torch.ones(2), {0: 0.1, 1: 0.1}, {0: 0.0, 1: 0.0}, -0.1, cfg_off, batch.batch["responses"].shape[1])
    res_on = _fake_result(spans_on, torch.ones(2), {0: 0.1, 1: 0.1}, {0: 0.0, 1: 0.0}, -0.1, cfg_on, batch.batch["responses"].shape[1])
    assert not torch.equal(res_off.utility, res_on.utility), "sanity: the two configs must actually differ"
    assert res_off.utility[0] > res_on.utility[0], "coef=0 must be the unpenalized (higher) utility"


def test_stored_baseline_uses_the_masked_rollout_logprobs(tokenizer):
    """log pi(Y|X) is the stored old_log_probs summed over the Y mask."""
    cfg = AbstractRLConfig(enable=True, delta_baseline="stored")
    batch = make_batch(tokenizer, [GOOD], [True])
    spans = _spans(tokenizer, batch, cfg)
    reasoning_mask, _ = build_masks(spans, batch.batch["response_mask"])
    stored = (batch.batch["old_log_probs"] * reasoning_mask.float()).sum(dim=-1)
    assert stored[0] == pytest.approx(-0.5 * spans[0].n_reasoning, abs=1e-4)

    res = _fake_result(spans, torch.ones(1), {0: 0.0}, {}, -0.1, cfg, 128, stored=stored)
    assert res.delta[0] == pytest.approx(0.5, abs=1e-5)  # (0 - (-0.5*n_y)) / n_y


# --- end-to-end reward shaping through the trainer hook -------------------------


class _FakeTrainer:
    def __init__(self, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.abstract_rl_config = cfg
        self.actor_rollout_wg = types.SimpleNamespace(world_size=1)
        rollout = types.SimpleNamespace(temperature=1.0)
        self.config = types.SimpleNamespace(actor_rollout_ref=types.SimpleNamespace(rollout=rollout))


def _run_hook(monkeypatch, tokenizer, texts, rewards, cfg, gain=0.25):
    batch = make_batch(tokenizer, texts, rewards)

    def fake_run_scoring(worker_group, td, cfg_, temperature):
        # constant per-token log-prob; the conditioned rows get `gain` more
        rows = []
        for _, cls, _ in fake_run_scoring.meta:
            rows.append(torch.full((td["responses"].shape[1],), gain if cls == CLASS_COND else 0.0))
        return torch.stack(rows)

    real_build = trainer_hooks.build_scoring_batch

    def spy_build(*args, **kwargs):
        td, meta = real_build(*args, **kwargs)
        fake_run_scoring.meta = meta
        return td, meta

    monkeypatch.setattr(trainer_hooks, "build_scoring_batch", spy_build)
    monkeypatch.setattr(trainer_hooks, "run_scoring", fake_run_scoring)

    trainer = _FakeTrainer(tokenizer, cfg)
    metrics = trainer_hooks.score_and_shape(trainer, batch, {})
    return batch, metrics


def test_score_and_shape_writes_utility_and_leaves_scores_intact(monkeypatch, tokenizer):
    cfg = AbstractRLConfig(enable=True)
    batch, metrics = _run_hook(monkeypatch, tokenizer, [GOOD, BAD], [True, False], cfg)

    scores = batch.batch["token_level_scores"].sum(dim=-1)
    assert torch.equal(scores, torch.tensor([1.0, -1.0])), "the raw task reward must not be clobbered"

    shaped = batch.batch["token_level_rewards"].sum(dim=-1)
    assert shaped[0] > 1.0, "a useful abstraction lifts the reward above the correct baseline"
    assert shaped[1] == pytest.approx(-1.0), "incorrect rollout keeps the affine-matched -1"

    for key in ("abstract_mask", "abstract_reasoning_mask", "abstract_prior_log_probs", "abstract_kl_weight"):
        assert key in batch.batch
    assert batch.batch["abstract_kl_weight"].tolist() == [1.0, 0.0]
    assert metrics["abstract_rl/valid_abstract_rate"] == 0.5
    assert metrics["abstract_rl/pass_rate"] == 0.5
    assert all(np.isfinite(v) for k, v in metrics.items() if k.endswith("_rate"))


def test_spec_exact_scale_when_affine_disabled(monkeypatch, tokenizer):
    # valid_bonus=0 isolates the affine/clip arithmetic this test targets
    cfg = AbstractRLConfig(enable=True, utility_affine=False, valid_bonus=0.0)
    batch, _ = _run_hook(monkeypatch, tokenizer, [GOOD, BAD], [True, False], cfg, gain=0.0)
    shaped = batch.batch["token_level_rewards"].sum(dim=-1)
    assert shaped.tolist() == [1.0, 0.0], "U in [0,2] with Delta = 0"


def test_affine_matches_the_baseline_reward_when_delta_is_zero(monkeypatch, tokenizer):
    cfg = AbstractRLConfig(enable=True, valid_bonus=0.0)
    batch, _ = _run_hook(monkeypatch, tokenizer, [GOOD, GOOD], [True, False], cfg, gain=0.0)
    shaped = batch.batch["token_level_rewards"].sum(dim=-1)
    assert shaped.tolist() == [1.0, -1.0], "2U-1 reproduces the +-1 task reward exactly"


@pytest.mark.parametrize(
    "texts,rewards",
    [
        ([GOOD, GOOD], [False, False]),  # no correct sample in the group
        ([GOOD, GOOD], [True, False]),  # exactly one correct
        ([GOOD, GOOD], [True, True]),  # all correct
        ([BAD, BAD], [True, True]),  # no valid abstractions
        ([GOOD, BAD], [False, False]),  # neither correct nor valid
    ],
)
def test_edge_case_groups_stay_finite(monkeypatch, tokenizer, texts, rewards):
    cfg = AbstractRLConfig(enable=True)
    batch, metrics = _run_hook(monkeypatch, tokenizer, texts, rewards, cfg)
    assert torch.isfinite(batch.batch["token_level_rewards"]).all()
    assert torch.isfinite(batch.batch["abstract_prior_log_probs"]).all()
    assert torch.isfinite(batch.batch["abstract_kl_weight"]).all()
    for key, value in metrics.items():
        assert not isinstance(value, float) or not np.isinf(value), key


def test_extremely_long_abstraction(monkeypatch, tokenizer):
    cfg = AbstractRLConfig(enable=True)
    long_text = "R. \\boxed{1}.\n<abstract>" + ("compress the recurrence " * 20) + "</abstract>"
    batch, metrics = _run_hook(monkeypatch, tokenizer, [long_text], [True], cfg)
    assert metrics["abstract_rl/compression_ratio"] > 1.0
    assert torch.isfinite(batch.batch["token_level_rewards"]).all()
