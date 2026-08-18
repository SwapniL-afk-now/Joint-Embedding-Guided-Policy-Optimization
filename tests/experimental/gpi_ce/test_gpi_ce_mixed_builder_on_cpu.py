# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Mixed online/offline candidate assembly (plan sections 5 and 12)."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.experimental.gpi_ce import MixedCandidateBuilder
from verl.protocol import DataProto

PROMPT_LEN, RESP_LEN, PAD = 4, 6, 0
K_ON, K_OFF = 2, 2
K = K_ON + K_OFF


def _online_batch(num_prompts=3, offline_per_prompt=K_OFF, offline_len=3):
    rows = num_prompts * K_ON
    prompts = torch.arange(1, num_prompts + 1).repeat_interleave(K_ON).view(-1, 1) * 100
    prompt_ids = prompts + torch.arange(PROMPT_LEN).view(1, -1)
    responses = torch.arange(rows).view(-1, 1) * 10 + torch.arange(1, RESP_LEN + 1).view(1, -1)

    attention_mask = torch.ones(rows, PROMPT_LEN + RESP_LEN, dtype=torch.long)
    attention_mask[:, -1] = 0  # every online response is one token short of full
    input_ids = torch.cat([prompt_ids, responses], dim=1)
    position_ids = torch.arange(PROMPT_LEN + RESP_LEN).expand(rows, -1).clone()

    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": attention_mask[:, -RESP_LEN:].clone(),
        },
        batch_size=(rows,),
    )
    extra = []
    for b in range(num_prompts):
        offline = [[900 + b * 10 + j] * offline_len for j in range(offline_per_prompt)]
        extra.extend([{"gpi_offline_input_ids": offline}] * K_ON)
    non_tensor = {
        "uid": np.array([f"p{b}" for b in range(num_prompts) for _ in range(K_ON)], dtype=object),
        "extra_info": np.array(extra, dtype=object),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor, meta_info={})


def test_group_layout_and_invariants():
    builder = MixedCandidateBuilder(pad_token_id=PAD)
    mixed = builder.build(_online_batch(), K_ON, K_OFF)

    assert len(mixed.batch["input_ids"]) == 3 * K
    np.testing.assert_array_equal(mixed.non_tensor_batch["gpi_candidate_id"], np.tile(np.arange(K), 3))
    np.testing.assert_array_equal(
        mixed.non_tensor_batch["gpi_source"],
        np.array(["online", "online", "offline", "offline"] * 3, dtype=object),
    )
    torch.testing.assert_close(
        mixed.batch["gpi_group_id"], torch.arange(3, dtype=torch.long).repeat_interleave(K)
    )
    uids = np.asarray(mixed.non_tensor_batch["uid"]).reshape(-1, K)
    assert all(len(set(row)) == 1 for row in uids), "groups must be contiguous and single-prompt"


def test_offline_rows_share_the_prompt_and_carry_the_offline_response():
    builder = MixedCandidateBuilder(pad_token_id=PAD)
    online = _online_batch()
    mixed = builder.build(online, K_ON, K_OFF)

    for b in range(3):
        base = b * K
        prompt = mixed.batch["input_ids"][base, :PROMPT_LEN]
        for j in range(K):
            # identical conditioning context is what makes the group softmax meaningful
            torch.testing.assert_close(mixed.batch["input_ids"][base + j, :PROMPT_LEN], prompt)
        for j in range(K_OFF):
            row = base + K_ON + j
            expected_token = 900 + b * 10 + j
            assert torch.all(mixed.batch["responses"][row, :3] == expected_token)
            assert torch.all(mixed.batch["responses"][row, 3:] == PAD)
            assert torch.all(mixed.batch["response_mask"][row, :3] == 1)
            assert torch.all(mixed.batch["response_mask"][row, 3:] == 0)
            torch.testing.assert_close(
                mixed.batch["input_ids"][row, PROMPT_LEN:], mixed.batch["responses"][row]
            )


def test_online_rows_are_untouched():
    builder = MixedCandidateBuilder(pad_token_id=PAD)
    online = _online_batch()
    before = online.batch["responses"].clone()
    mixed = builder.build(online, K_ON, K_OFF)

    for b in range(3):
        for j in range(K_ON):
            torch.testing.assert_close(mixed.batch["responses"][b * K + j], before[b * K_ON + j])
    torch.testing.assert_close(online.batch["responses"], before, msg="builder mutated the input batch")


def test_offline_response_is_truncated_to_max_response_length():
    builder = MixedCandidateBuilder(pad_token_id=PAD)
    mixed = builder.build(_online_batch(offline_len=RESP_LEN + 5), K_ON, K_OFF)
    assert mixed.batch["responses"].shape[1] == RESP_LEN
    assert torch.all(mixed.batch["response_mask"][K_ON:K] == 1)


def test_too_few_offline_responses_raises_unless_allowed():
    online = _online_batch(offline_per_prompt=1)
    with pytest.raises(ValueError, match="offline_per_prompt"):
        MixedCandidateBuilder(pad_token_id=PAD).build(online, K_ON, K_OFF)

    mixed = MixedCandidateBuilder(pad_token_id=PAD, allow_short_offline=True).build(
        _online_batch(offline_per_prompt=1), K_ON, K_OFF
    )
    assert len(mixed.batch["input_ids"]) == 3 * K


def test_offline_zero_is_a_passthrough():
    online = _online_batch()
    mixed = MixedCandidateBuilder(pad_token_id=PAD).build(online, K_ON, 0)
    assert len(mixed.batch["input_ids"]) == 3 * K_ON
    np.testing.assert_array_equal(mixed.non_tensor_batch["gpi_source"], np.array(["online"] * 6, dtype=object))


def test_offline_rows_do_not_keep_the_template_rows_reward():
    """Regression: offline rows are CLONES of an online row.

    Rewards come from the agent loop during generation, so rm_scores already exists
    when the splice happens. Without explicit rescoring each offline candidate silently
    inherits the reward of its group's first online rollout -- which made verified-
    correct R1 solutions score like the model's own attempts and inverted q*.
    """
    builder = MixedCandidateBuilder(pad_token_id=PAD)
    online = _online_batch()
    # distinctive per-row reward on the last response token
    rm = torch.zeros(len(online.batch["input_ids"]), RESP_LEN)
    for i in range(rm.shape[0]):
        rm[i, -1] = float(i + 1)
    online.batch["rm_scores"] = rm

    mixed = builder.build(online, K_ON, K_OFF)
    scores = mixed.batch["rm_scores"].sum(-1)
    offline = mixed.batch["gpi_is_offline"].bool()

    # The clone DOES carry the template's reward -- that is exactly the trap, so the
    # trainer must overwrite it. Pin the behaviour so the hook stays load-bearing.
    for b in range(3):
        template_score = float(scores[b * K])
        for j in range(K_OFF):
            assert float(scores[b * K + K_ON + j]) == template_score, (
                "offline row no longer clones the template reward; "
                "score_offline_rows may no longer be necessary -- re-check the trainer hook"
            )
    assert offline.sum() == 3 * K_OFF


def test_non_contiguous_online_groups_raise():
    online = _online_batch()
    online.non_tensor_batch["uid"] = np.array(["p0", "p1", "p0", "p1", "p2", "p2"], dtype=object)
    with pytest.raises(ValueError, match="group-contiguous"):
        MixedCandidateBuilder(pad_token_id=PAD).build(online, K_ON, K_OFF)
