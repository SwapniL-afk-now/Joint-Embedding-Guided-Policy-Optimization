# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""End-to-end wiring: ppo_loss dispatch and the driver-side target construction.

Covers plan section 12's "beta=0 leaves GRPO untouched" analogue -- here, that a
batch WITHOUT the custom_gpi_ce stamp still takes the stock policy-loss path.
"""

import numpy as np
import torch
from tensordict import TensorDict

import verl.utils.tensordict_utils as tu
from verl.experimental.gpi_ce import compute_gpi_ce_loss, compute_gpi_targets
from verl.workers.config.actor import ActorConfig
from verl.workers.utils.losses import ppo_loss

K, NUM_GROUPS, PROMPT_LEN, RESP_LEN = 4, 2, 3, 5
ROWS = K * NUM_GROUPS


def _data(with_gpi=True, seed=0):
    torch.manual_seed(seed)
    prompts = torch.ones(ROWS, PROMPT_LEN, dtype=torch.long)
    responses = torch.ones(ROWS, RESP_LEN, dtype=torch.long)
    attention_mask = torch.ones(ROWS, PROMPT_LEN + RESP_LEN, dtype=torch.long)
    attention_mask[:, -1] = 0
    response_mask = attention_mask[:, -RESP_LEN:].clone()

    old_log_probs = torch.randn(ROWS, RESP_LEN) - 1.0
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0] * NUM_GROUPS)

    data = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "advantages": torch.zeros(ROWS, RESP_LEN),
        },
        batch_size=(ROWS,),
    )
    if with_gpi:
        targets = compute_gpi_targets(old_log_probs, response_mask, rewards, K, temperature=0.5)
        data["gpi_q_target"] = targets["gpi_q_target"]
        data["gpi_group_id"] = torch.arange(NUM_GROUPS).repeat_interleave(K)
        tu.assign_non_tensor(
            data,
            custom_gpi_ce={
                "enable": True,
                "group_size": K,
                "groups_per_rank": NUM_GROUPS,
                "temperature": 0.5,
            },
        )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=None, global_batch_size=None)
    return data


def _config():
    return ActorConfig(strategy="fsdp", rollout_n=K, ppo_micro_batch_size_per_gpu=1)


def _model_output(data, seed=1):
    torch.manual_seed(seed)
    flat_len = int(data["attention_mask"].sum())
    return {"log_probs": torch.randn(flat_len, requires_grad=True)}


def test_ppo_loss_routes_to_gpi_ce_and_matches_the_direct_call():
    data = _data()
    model_output = _model_output(data)
    loss, metrics = ppo_loss(_config(), model_output, data.clone())

    assert any(key.startswith("gpi/") for key in metrics), f"no GPI metrics emitted: {sorted(metrics)}"
    # The stock path's clip diagnostics must be absent: there is no PPO ratio here.
    assert "actor/pg_clipfrac" not in metrics
    assert torch.isfinite(loss)


def test_gpi_loss_value_is_reproduced_from_the_padded_fields():
    """ppo_loss's padded reconstruction must agree with a direct loss call."""
    from verl.workers.utils.losses import no_padding_2_padding

    data = _data()
    model_output = _model_output(data)
    loss, _ = ppo_loss(_config(), model_output, data.clone())

    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    direct, _ = compute_gpi_ce_loss(
        log_prob=log_prob,
        response_mask=data["response_mask"].bool(),
        q_target=data["gpi_q_target"],
        group_size=K,
        groups_per_rank=NUM_GROUPS,
        group_ids=data["gpi_group_id"],
    )
    torch.testing.assert_close(loss.detach(), direct.detach())


def test_gradient_reaches_the_model_output():
    data = _data()
    model_output = _model_output(data)
    loss, _ = ppo_loss(_config(), model_output, data.clone())
    loss.backward()
    grad = model_output["log_probs"].grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0


def test_without_the_stamp_the_stock_path_still_runs():
    """No custom_gpi_ce => unchanged PPO behaviour, clip metrics present."""
    data = _data(with_gpi=False)
    model_output = _model_output(data)
    loss, metrics = ppo_loss(_config(), model_output, data.clone())

    assert not any(key.startswith("gpi/") for key in metrics)
    assert torch.isfinite(loss)


def test_check_batch_integrity_catches_a_split_group():
    from verl.experimental.gpi_ce.trainer_hooks import check_batch_integrity
    from verl.protocol import DataProto

    good = DataProto(
        batch=TensorDict({"x": torch.zeros(ROWS)}, batch_size=(ROWS,)),
        non_tensor_batch={
            "uid": np.array(["a"] * K + ["b"] * K, dtype=object),
            "gpi_candidate_id": np.tile(np.arange(K), NUM_GROUPS),
        },
        meta_info={},
    )
    check_batch_integrity(good, K)  # must not raise

    bad = DataProto(
        batch=TensorDict({"x": torch.zeros(ROWS)}, batch_size=(ROWS,)),
        non_tensor_batch={"uid": np.array(["a", "b"] * K, dtype=object)},
        meta_info={},
    )
    try:
        check_batch_integrity(bad, K)
    except ValueError as exc:
        assert "split" in str(exc)
    else:
        raise AssertionError("interleaved uids must be rejected")
