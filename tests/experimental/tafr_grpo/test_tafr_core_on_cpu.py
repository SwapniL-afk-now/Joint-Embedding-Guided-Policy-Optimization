# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.experimental.tafr_grpo.config import (
    TAFRGRPOConfig,
    should_checkpoint_and_refresh,
    should_run_failure_sft,
    validate_tafr_config,
)
from verl.experimental.tafr_grpo.ema_mixer import assert_frozen, assert_optimizer_excludes_module, freeze_module
from verl.experimental.tafr_grpo.failure_data_collector import FailureDataCollector
from verl.experimental.tafr_grpo.failure_sft_trainer import failure_sft_loss
from verl.experimental.tafr_grpo.tafr_loss import (
    compute_regularization_policy_loss,
    compute_regularization_token_advantage,
    compute_tafr_grpo_auxiliary_loss,
    replay_gate,
    response_length_normalized_mean,
)
from verl.experimental.tafr_grpo.vllm_scoring import (
    TAFR_VLLM_ADAPTERS,
    extract_response_logprobs_from_prompt_logprobs,
)
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.losses import ppo_loss


def test_validate_rejects_builtin_kl_when_enabled():
    cfg = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": True},
            "actor_rollout_ref": {"actor": {"use_kl_loss": False}},
            "custom_tafr_grpo": {"enable": True, "disable_builtin_kl": True},
        }
    )
    with pytest.raises(ValueError, match="use_kl_in_reward=false"):
        validate_tafr_config(cfg)


def test_failure_collector_accepts_only_reward_zero():
    collector = FailureDataCollector(max_size=10)
    assert collector.add(prompt="p", response="wrong", reward=0)
    assert not collector.add(prompt="p", response="correct", reward=1)
    assert len(collector) == 1
    assert collector.to_records()[0]["wrong_response"] == "wrong"


def test_replay_gate_boundaries():
    rewards = torch.tensor([1.0, 0.0, 0.5])
    gates = replay_gate(rewards)
    assert gates[0].item() == 0.0
    assert gates[1].item() == 1.0
    assert gates[2].item() == 0.5


def test_tafr_replay_loss_zero_for_all_correct_group():
    log_prob = torch.tensor([[-3.0, -4.0]], requires_grad=True)
    replay_log_prob = torch.tensor([[-1.0, -1.0]])
    out = compute_tafr_grpo_auxiliary_loss(
        log_prob=log_prob,
        response_mask=torch.ones_like(log_prob, dtype=torch.bool),
        group_reward_mean=torch.tensor([1.0]),
        beta=0.7,
        variant="replay_only",
        replay_log_prob=replay_log_prob,
    )
    assert out.loss.item() == 0.0


def test_tafr_replay_loss_full_for_all_wrong_group():
    log_prob = torch.tensor([[-3.0, -4.0]], requires_grad=True)
    replay_log_prob = torch.tensor([[-1.0, -1.0]])
    out = compute_tafr_grpo_auxiliary_loss(
        log_prob=log_prob,
        response_mask=torch.ones_like(log_prob, dtype=torch.bool),
        group_reward_mean=torch.tensor([0.0]),
        beta=0.5,
        variant="replay_only",
        replay_log_prob=replay_log_prob,
    )
    # k3 estimator of D_KL(pi_replay || pi_theta): (r - 1) - log_ratio, clamped, then masked-mean.
    log_ratio = (replay_log_prob - log_prob).clamp(min=-20, max=20)
    r = torch.exp(log_ratio)
    expected_kl = (((r - 1) - log_ratio).clamp(min=-10, max=10).mean()).detach()
    assert torch.allclose(out.loss, -0.5 * expected_kl)


def test_length_normalization_uses_response_mask_only():
    values = torch.tensor([[10.0, 2.0, 4.0]])
    mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)
    seq_mean = response_length_normalized_mean(values, mask)
    assert seq_mean.item() == 3.0


def test_failure_sft_masks_prompt_tokens():
    log_prob = torch.tensor([[-100.0, -2.0, -4.0]])
    mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)
    loss = failure_sft_loss(log_prob, mask)
    assert loss.item() == 3.0


def test_freeze_and_optimizer_exclusion():
    actor = torch.nn.Linear(2, 2)
    frozen = freeze_module(torch.nn.Linear(2, 2))
    opt = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    assert_frozen(frozen)
    assert_optimizer_excludes_module(opt, frozen)


def test_optimizer_exclusion_catches_overlap():
    model = freeze_module(torch.nn.Linear(2, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(AssertionError, match="Optimizer contains"):
        assert_optimizer_excludes_module(opt, model)


def test_global_grpo_clock_controls_sft_and_checkpoint_schedule():
    cfg = TAFRGRPOConfig(enable=True, sft_update_interval_grpo_steps=5, checkpoint_interval_grpo_steps=10)
    assert not should_run_failure_sft(1, cfg)
    assert should_run_failure_sft(5, cfg)
    assert should_run_failure_sft(10, cfg)
    assert not should_checkpoint_and_refresh(5, cfg)
    assert should_checkpoint_and_refresh(10, cfg)


def test_vllm_prompt_logprob_slice_returns_response_only():
    prompt_logprobs = [[-0.1], [-0.2], [-1.0], [-1.5], [-2.0], [0.0]]
    response = extract_response_logprobs_from_prompt_logprobs(
        prompt_logprobs=prompt_logprobs,
        prompt_len=3,
        response_len=3,
    )
    assert response == [-1.0, -1.5, -2.0]


def test_tafr_vllm_adapter_ids_do_not_collide_with_rollout_adapter():
    rollout_lora_id = 123
    ids = {spec.int_id for spec in TAFR_VLLM_ADAPTERS.values()}
    assert rollout_lora_id not in ids
    assert len(ids) == len(TAFR_VLLM_ADAPTERS)


def test_tafr_vllm_has_failure_slot():
    assert "failure" in TAFR_VLLM_ADAPTERS
    spec = TAFR_VLLM_ADAPTERS["failure"]
    assert spec.name == "tafr_failure"
    assert spec.path == "tafr_failure_lora_path"
    assert spec.int_id not in {TAFR_VLLM_ADAPTERS[k].int_id for k in TAFR_VLLM_ADAPTERS if k != "failure"}


# ── Plan §14: advantage algebra ──────────────────────────────────────────────

def _regularization_advantage(
    *,
    old_log_prob,
    failure_log_prob,
    group_reward_mean,
    response_mask,
    advantage_clip,
    failure_model_ready,
    anchor_log_prob=None,
    grpo_advantage=None,
):
    anchor_log_prob = old_log_prob if anchor_log_prob is None else anchor_log_prob
    grpo_advantage = torch.ones_like(old_log_prob) if grpo_advantage is None else grpo_advantage
    return compute_regularization_token_advantage(
        anchor_log_prob=anchor_log_prob,
        old_log_prob=old_log_prob,
        failure_log_prob=failure_log_prob,
        grpo_advantage=grpo_advantage,
        group_reward_mean=group_reward_mean,
        response_mask=response_mask,
        advantage_clip=advantage_clip,
        failure_model_ready=failure_model_ready,
    )



def test_failure_advantage_zero_when_models_equal():
    old = torch.full((2, 3), -1.0)
    failure = old.clone()
    rmean = torch.tensor([0.0, 1.0])
    mask = torch.ones(2, 3, dtype=torch.bool)
    adv, stats = _regularization_advantage(
        old_log_prob=old,
        failure_log_prob=failure,
        group_reward_mean=rmean,
        response_mask=mask,
        advantage_clip=5.0,
        failure_model_ready=True,
    )
    torch.testing.assert_close(adv, torch.zeros_like(adv))
    torch.testing.assert_close(stats["difficulty"], torch.tensor([[1.0], [0.0]]))
    loss, _ = compute_regularization_policy_loss(
        current_log_prob=old.clone().requires_grad_(),
        old_log_prob=old,
        regularization_advantage=adv,
        response_mask=mask,
        clip_ratio=0.2,
        loss_agg_mode="token-mean",
    )
    assert loss.item() == 0.0


def test_failure_advantage_zero_before_ready():
    old = torch.full((2, 3), -1.0)
    failure = torch.full((2, 3), -0.5)
    rmean = torch.tensor([0.0, 0.0])
    mask = torch.ones(2, 3, dtype=torch.bool)
    adv, stats = _regularization_advantage(
        old_log_prob=old,
        failure_log_prob=failure,
        group_reward_mean=rmean,
        response_mask=mask,
        advantage_clip=5.0,
        failure_model_ready=False,
    )
    torch.testing.assert_close(adv, torch.zeros_like(adv))
    torch.testing.assert_close(stats["difficulty"], torch.zeros_like(stats["difficulty"]))


def test_anchor_stability_remains_active_before_failure_model_ready():
    old = torch.tensor([[-1.0, -2.0]])
    anchor = torch.tensor([[-0.5, -2.5]])
    mask = torch.ones_like(old, dtype=torch.bool)
    adv, stats = _regularization_advantage(
        anchor_log_prob=anchor,
        old_log_prob=old,
        failure_log_prob=torch.full_like(old, 99.0),
        grpo_advantage=torch.tensor([[2.0, -2.0]]),
        group_reward_mean=torch.tensor([0.0]),
        response_mask=mask,
        advantage_clip=5.0,
        failure_model_ready=False,
    )
    torch.testing.assert_close(stats["raw_advantage"], anchor - old)
    torch.testing.assert_close(stats["failure_component"], torch.zeros_like(old))
    assert adv.abs().sum().item() > 0.0


def test_failure_advantage_zero_all_correct_group():
    old = torch.tensor([[-1.0, -2.0]])
    failure = torch.tensor([[-5.0, -6.0]])  # very different -> must not matter
    rmean = torch.tensor([1.0])
    mask = torch.ones_like(old, dtype=torch.bool)
    adv, stats = _regularization_advantage(
        old_log_prob=old,
        failure_log_prob=failure,
        group_reward_mean=rmean,
        response_mask=mask,
        advantage_clip=5.0,
        failure_model_ready=True,
    )
    torch.testing.assert_close(adv, torch.zeros_like(adv))
    assert stats["difficulty"].item() == 0.0


def test_all_correct_group_keeps_anchor_stability_but_zeroes_failure_term():
    old = torch.tensor([[-1.0, -2.0]])
    anchor = torch.tensor([[-0.75, -2.5]])
    mask = torch.ones_like(old, dtype=torch.bool)
    adv, stats = _regularization_advantage(
        anchor_log_prob=anchor,
        old_log_prob=old,
        failure_log_prob=torch.tensor([[-20.0, 20.0]]),
        grpo_advantage=torch.ones_like(old),
        group_reward_mean=torch.tensor([1.0]),
        response_mask=mask,
        advantage_clip=5.0,
        failure_model_ready=True,
    )
    torch.testing.assert_close(stats["raw_advantage"], anchor - old)
    torch.testing.assert_close(stats["failure_component"], torch.zeros_like(old))
    assert adv.abs().sum().item() > 0.0


def test_regularization_raw_advantage_matches_paper_equation_and_is_detached():
    anchor = torch.tensor([[-0.8, -2.2]], requires_grad=True)
    old = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    failure = torch.tensor([[-1.5, -1.5]], requires_grad=True)
    mask = torch.ones_like(old, dtype=torch.bool)
    _, stats = _regularization_advantage(
        anchor_log_prob=anchor,
        old_log_prob=old,
        failure_log_prob=failure,
        grpo_advantage=torch.ones_like(old),
        group_reward_mean=torch.tensor([0.25]),
        response_mask=mask,
        advantage_clip=5.0,
        failure_model_ready=True,
    )
    expected_stability = anchor.detach() - old.detach()
    expected_failure = 0.75 * (old.detach() - failure.detach())
    torch.testing.assert_close(stats["stability_signal"], expected_stability)
    torch.testing.assert_close(stats["failure_component"], expected_failure)
    torch.testing.assert_close(stats["raw_advantage"], expected_stability + expected_failure)
    assert stats["advantage"].requires_grad is False


def test_beta_scales_areg_tail_to_the_grpo_advantage():
    """Contract change: A_reg is matched to GRPO at the TAIL (p99), not on RMS.

    RMS-matching a heavy-tailed signal against a flat one left the top 1% of tokens
    at ~4.5x GRPO, overriding the outcome signal exactly where A_reg fires. beta now
    scales a tail-matched advantage, so beta=0.5 caps per-token influence at ~half
    GRPO rather than merely averaging to half.
    """
    old = torch.zeros(2, 2)
    mask = torch.ones_like(old, dtype=torch.bool)
    grpo = torch.tensor([[2.0, -2.0], [1.0, -1.0]])
    adv, stats = _regularization_advantage(
        anchor_log_prob=torch.tensor([[0.1, -0.2], [0.3, -0.4]]),
        old_log_prob=old,
        failure_log_prob=torch.tensor([[-0.5, 0.5], [-0.25, 0.25]]),
        grpo_advantage=grpo,
        group_reward_mean=torch.tensor([0.0, 0.5]),
        response_mask=mask,
        advantage_clip=100.0,
        failure_model_ready=True,
    )
    grpo_rms = grpo.square().mean().sqrt()
    torch.testing.assert_close(stats["grpo_advantage_rms"], grpo_rms)
    # tail sits at GRPO's scale; nothing exceeds it by more than the clip allows
    assert adv.abs().max().item() <= float(grpo_rms) * 1.05 + 1e-6


def test_failure_advantage_all_wrong_group_signs():
    old = torch.tensor([[-1.0, -1.0]])
    failure = torch.tensor([[-0.5, -2.0]])
    rmean = torch.tensor([0.0])
    mask = torch.ones_like(old, dtype=torch.bool)
    adv, stats = _regularization_advantage(
        old_log_prob=old,
        failure_log_prob=failure,
        group_reward_mean=rmean,
        response_mask=mask,
        advantage_clip=5.0,
        failure_model_ready=True,
    )
    assert stats["difficulty"].item() == 1.0
    assert adv[0, 0].item() < 0.0  # failure_log_prob > old_log_prob -> negative
    assert adv[0, 1].item() > 0.0  # old_log_prob > failure_log_prob -> positive


def test_failure_advantage_partial_group_difficulty():
    old = torch.full((1, 2), -1.0)
    failure = torch.full((1, 2), -2.0)
    rmean = torch.tensor([0.25])
    mask = torch.ones_like(old, dtype=torch.bool)
    adv, stats = _regularization_advantage(
        old_log_prob=old,
        failure_log_prob=failure,
        group_reward_mean=rmean,
        response_mask=mask,
        advantage_clip=5.0,
        failure_model_ready=True,
    )
    assert stats["difficulty"].item() == pytest.approx(0.75)
    assert stats["raw_advantage"][0, 0].item() == pytest.approx(0.75)
    assert adv[0, 0].item() == pytest.approx(1.0)


def test_failure_advantage_clip_applied():
    mask = torch.ones(1, 1, dtype=torch.bool)
    rmean = torch.tensor([0.0])
    adv, _ = _regularization_advantage(
        old_log_prob=torch.tensor([[10.0]]),
        failure_log_prob=torch.tensor([[-10.0]]),
        group_reward_mean=rmean,
        response_mask=mask,
        advantage_clip=5.0,
        failure_model_ready=True,
        grpo_advantage=torch.tensor([[10.0]]),
    )
    assert adv[0, 0].item() == 5.0
    adv, _ = _regularization_advantage(
        old_log_prob=torch.tensor([[-10.0]]),
        failure_log_prob=torch.tensor([[10.0]]),
        group_reward_mean=rmean,
        response_mask=mask,
        advantage_clip=5.0,
        failure_model_ready=True,
        grpo_advantage=torch.tensor([[10.0]]),
    )
    assert adv[0, 0].item() == -5.0


def test_failure_advantage_respects_mask():
    old = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
    failure = torch.tensor([[-4.0, -3.0, -2.0, -1.0]])
    rmean = torch.tensor([0.0])
    mask = torch.tensor([[True, True, False, False]])
    adv, _ = _regularization_advantage(
        old_log_prob=old,
        failure_log_prob=failure,
        group_reward_mean=rmean,
        response_mask=mask,
        advantage_clip=5.0,
        failure_model_ready=True,
    )
    assert adv[0, 2:].abs().sum().item() == 0.0
    assert adv[0, 0].item() != 0.0


def test_failure_loss_zero_when_advantage_zero():
    current = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    old = torch.tensor([[-1.0, -2.0]])
    mask = torch.ones_like(current, dtype=torch.bool)
    loss, stats = compute_regularization_policy_loss(
        current_log_prob=current,
        old_log_prob=old,
        regularization_advantage=torch.zeros_like(current),
        response_mask=mask,
        clip_ratio=0.2,
        loss_agg_mode="token-mean",
    )
    assert loss.item() == 0.0


# ── Plan §14: gradient tests ─────────────────────────────────────────────────

def test_failure_loss_gradient_routing():
    current = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    old = torch.tensor([[-1.0, -2.0]])
    failure_adv = torch.tensor([[1.0, 1.0]])
    mask = torch.ones_like(current, dtype=torch.bool)
    assert old.requires_grad is False
    assert failure_adv.requires_grad is False
    loss, _ = compute_regularization_policy_loss(
        current_log_prob=current,
        old_log_prob=old,
        regularization_advantage=failure_adv,
        response_mask=mask,
        clip_ratio=0.2,
        loss_agg_mode="token-mean",
    )
    loss.backward()
    assert current.grad is not None
    # token-mean: d(loss)/d(current[t]) = -(1/n) * ratio * adv = -0.5
    assert current.grad[0, 0].item() == pytest.approx(-0.5)


def test_failure_loss_gradient_descent_direction():
    current = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    old = torch.tensor([[-1.0, -2.0]])
    mask = torch.ones_like(current, dtype=torch.bool)
    adv = torch.tensor([[1.0, -1.0]])
    loss, _ = compute_regularization_policy_loss(
        current_log_prob=current,
        old_log_prob=old,
        regularization_advantage=adv,
        response_mask=mask,
        clip_ratio=0.2,
        loss_agg_mode="token-mean",
    )
    loss.backward()
    g = current.grad.clone()
    updated = current.detach() - 0.1 * g
    # positive advantage -> gradient descent increases the token log-probability
    assert updated[0, 0].item() > current.detach()[0, 0].item()
    # negative advantage -> gradient descent decreases it
    assert updated[0, 1].item() < current.detach()[0, 1].item()


# ── Plan §14: mask tests ─────────────────────────────────────────────────────

def test_failure_loss_mask_invariance():
    mask = torch.tensor([[True, True, False, False]])
    base_loss, base_stats = compute_regularization_policy_loss(
        current_log_prob=torch.tensor([[-1.0, -2.0, -3.0, -4.0]], requires_grad=True),
        old_log_prob=torch.tensor([[-1.0, -2.0, -3.0, -4.0]]),
        regularization_advantage=torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        response_mask=mask,
        clip_ratio=0.2,
        loss_agg_mode="token-mean",
    )
    current2 = torch.tensor([[-1.0, -2.0, 99.0, 99.0]], requires_grad=True)
    old2 = torch.tensor([[-1.0, -2.0, 88.0, 88.0]])
    adv2 = torch.tensor([[1.0, 1.0, 77.0, 77.0]])
    changed_loss, changed_stats = compute_regularization_policy_loss(
        current_log_prob=current2,
        old_log_prob=old2,
        regularization_advantage=adv2,
        response_mask=mask,
        clip_ratio=0.2,
        loss_agg_mode="token-mean",
    )
    torch.testing.assert_close(base_loss, changed_loss)
    torch.testing.assert_close(base_stats["ppo_ratio_clip_fraction"], changed_stats["ppo_ratio_clip_fraction"])
    base_loss.backward()
    changed_loss.backward()
    torch.testing.assert_close(current2.grad[0, :2], current2.grad[0, :2])
    assert current2.grad[0, 2:].abs().sum().item() == 0.0


# ── Plan §14: PPO clipping tests ─────────────────────────────────────────────

def test_failure_loss_clipping_branches():
    clip = 0.2
    mask = torch.ones(1, 1, dtype=torch.bool)

    # positive advantage, ratio above 1 + epsilon -> clipped branch (1+clip)*adv
    current = torch.tensor([[0.0]], requires_grad=True)
    old = torch.tensor([[-0.5]])  # ratio = exp(0.5) ~ 1.6487 > 1.2
    adv = torch.tensor([[1.0]])
    loss, _ = compute_regularization_policy_loss(
        current_log_prob=current,
        old_log_prob=old,
        regularization_advantage=adv,
        response_mask=mask,
        clip_ratio=clip,
        loss_agg_mode="token-mean",
    )
    torch.testing.assert_close(loss, torch.tensor(-(1.0 + clip)))

    # negative advantage, ratio below 1 - epsilon -> clipped branch (1-clip)*adv
    current = torch.tensor([[-0.5]], requires_grad=True)
    old = torch.tensor([[0.0]])  # ratio = exp(-0.5) ~ 0.6065 < 0.8
    adv = torch.tensor([[-1.0]])
    loss, _ = compute_regularization_policy_loss(
        current_log_prob=current,
        old_log_prob=old,
        regularization_advantage=adv,
        response_mask=mask,
        clip_ratio=clip,
        loss_agg_mode="token-mean",
    )
    torch.testing.assert_close(loss, torch.tensor(-(1.0 - clip) * (-1.0)))


# ── Plan §14: config validation ──────────────────────────────────────────────

def test_config_validation_new_fields():
    base = {
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "actor_rollout_ref": {"actor": {"use_kl_loss": False}},
        "custom_tafr_grpo": {"enable": True},
    }
    with pytest.raises(ValueError, match="loss_version"):
        validate_tafr_config(OmegaConf.create({**base, "custom_tafr_grpo": {"enable": True, "loss_version": "bogus"}}))
    with pytest.raises(ValueError, match="failure_advantage_clip"):
        validate_tafr_config(OmegaConf.create({**base, "custom_tafr_grpo": {"enable": True, "failure_advantage_clip": 0.0}}))
    with pytest.raises(ValueError, match="failure_min_updates_before_use"):
        validate_tafr_config(
            OmegaConf.create({**base, "custom_tafr_grpo": {"enable": True, "failure_min_updates_before_use": 0}})
        )
    with pytest.raises(ValueError, match="failure_clip_ratio"):
        validate_tafr_config(OmegaConf.create({**base, "custom_tafr_grpo": {"enable": True, "failure_clip_ratio": -1.0}}))


# ── Plan §14: loss integration (new mode) ────────────────────────────────────

def _attach_regularization_fields(data, *, ready, anchor_offset=0.0):
    old = data["old_log_probs"]
    anchor = old + float(anchor_offset)
    advantage, stats = compute_regularization_token_advantage(
        anchor_log_prob=anchor,
        old_log_prob=old,
        failure_log_prob=data["tafr_failure_log_prob"],
        grpo_advantage=data["advantages"],
        group_reward_mean=data["tafr_group_reward_mean"],
        response_mask=data["response_mask"],
        advantage_clip=5.0,
        failure_model_ready=ready,
    )
    data["tafr_regularization_advantage"] = advantage
    data["tafr_stability_signal"] = stats["stability_signal"]
    data["tafr_failure_signal"] = stats["failure_signal"]
    data["tafr_failure_component"] = stats["failure_component"]
    data["tafr_raw_regularization_advantage"] = stats["raw_advantage"]
    data["tafr_scaled_regularization_advantage"] = stats["scaled_advantage"]
    data["tafr_regularization_scale"] = stats["normalization_scale"].reshape(1, 1).expand(len(data), 1)
    return data


def _make_failure_batch(
    group_reward_mean,
    ready=True,
    update_count=1,
    loss_version="failure_token_adv",
    anchor_offset=0.0,
):
    bsz = len(group_reward_mean)
    resp_len = 4
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]] * bsz),
            "responses": torch.zeros(bsz, resp_len, dtype=torch.long),
            "attention_mask": torch.ones(bsz, 1 + resp_len, dtype=torch.long),
            "response_mask": torch.ones(bsz, resp_len, dtype=torch.bool),
            "old_log_probs": torch.zeros(bsz, resp_len),
            "advantages": torch.zeros(bsz, resp_len),
            "tafr_group_reward_mean": torch.tensor(group_reward_mean, dtype=torch.float),
            "tafr_failure_log_prob": torch.zeros(bsz, resp_len),
        },
        batch_size=[bsz],
    )
    _attach_regularization_fields(data, ready=ready, anchor_offset=anchor_offset)
    tu.assign_non_tensor(
        data,
        dp_size=1,
        batch_num_tokens=None,
        global_batch_size=None,
        custom_tafr_grpo={
            "enable": True,
            "loss_version": loss_version,
            "beta": 0.5,
            "failure_advantage_clip": 5.0,
            "failure_clip_ratio": None,
            "reuse_rollout_log_probs": True,
            "verify_rollout_log_probs": False,
            "score_only_active_groups": True,
            "failure_min_updates_before_use": 1,
            "tafr_failure_model_ready": ready,
            "tafr_failure_model_update_count": update_count,
        },
    )
    return data


def _make_actor_config():
    return ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1)


def test_ppo_loss_failure_token_adv_metrics():
    data = _make_failure_batch([0.0, 0.0, 0.25, 1.0], ready=True, update_count=3)
    config = _make_actor_config()
    model_output = {"log_probs": torch.zeros(4 * 5, requires_grad=True)}
    loss, metrics = ppo_loss(config=config, model_output=model_output, data=data.clone())
    for key in (
        "tafr_adv/failure_loss",
        "tafr_adv/weighted_failure_loss",
        "tafr_adv/total_policy_loss",
        "tafr_adv/group_difficulty_mean",
        "tafr_adv/active_group_fraction",
        "tafr_adv/scored_token_fraction",
        "tafr_adv/failure_signal_mean",
        "tafr_adv/failure_signal_std",
        "tafr_adv/failure_signal_positive_fraction",
        "tafr_adv/advantage_mean",
        "tafr_adv/advantage_std",
        "tafr_adv/advantage_positive_fraction",
        "tafr_adv/advantage_clip_fraction",
        "tafr_adv/ppo_ratio_mean",
        "tafr_adv/ppo_ratio_clip_fraction",
        "tafr_adv/all_wrong/advantage_mean",
        "tafr_adv/regularization_loss",
        "tafr_adv/combined_policy_loss",
        "tafr_adv/loss_delta_vs_grpo",
        "tafr_adv/weighted_regularization_loss",
        "tafr_adv/stability_signal_mean",
        "tafr_adv/regularization_to_grpo_rms_ratio",
        "tafr_adv/all_correct/failure_component_abs_max",
        "tafr_adv/mixed/advantage_mean",
        "tafr_adv/all_correct/advantage_abs_max",
    ):
        assert key in metrics, key
    assert metrics["tafr_adv/active_group_fraction"].aggregate() == pytest.approx(0.75)
    assert metrics["tafr_adv/failure_model_ready"].aggregate() == 1.0
    assert metrics["tafr_adv/failure_model_update_count"].aggregate() == 3.0
    assert "tafr_adv/loss_grpo" in metrics
    assert "tafr_grpo/loss_grpo" not in metrics
    assert loss.requires_grad
    loss.backward()
    assert model_output["log_probs"].grad is not None


def test_ppo_loss_all_correct_group_keeps_anchor_and_zeroes_failure_component():
    data = _make_failure_batch([0.0, 0.0, 0.25, 1.0], ready=True, update_count=1, anchor_offset=0.25)
    # all-wrong and mixed rows: failure model disagrees with the old policy -> nonzero signal
    data["tafr_failure_log_prob"][0] = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
    data["tafr_failure_log_prob"][1] = torch.tensor([[-2.0, -2.0, -2.0, -2.0]])
    # all-correct row: failure model disagrees too, but difficulty zeroes the advantage
    data["tafr_failure_log_prob"][3] = torch.tensor([[-5.0, -6.0, -7.0, -8.0]])
    _attach_regularization_fields(data, ready=True, anchor_offset=0.25)
    config = _make_actor_config()
    model_output = {"log_probs": torch.zeros(4 * 5, requires_grad=True)}
    loss, metrics = ppo_loss(config=config, model_output=model_output, data=data.clone())
    assert metrics["tafr_adv/all_correct/advantage_abs_max"].aggregate() > 0.0
    assert metrics["tafr_adv/all_correct/failure_component_abs_max"].aggregate() == 0.0
    assert metrics["tafr_adv/all_wrong/advantage_mean"].aggregate() != 0.0
    assert metrics["tafr_adv/failure_loss"].aggregate() != 0.0
    assert metrics["tafr_adv/combined_policy_loss"].aggregate() != metrics["tafr_adv/loss_grpo"].aggregate()


def test_ppo_loss_new_mode_requires_precomputed_regularization_advantage():
    data = _make_failure_batch([0.0, 0.25, 1.0], ready=True)
    del data["tafr_regularization_advantage"]
    config = _make_actor_config()
    model_output = {"log_probs": torch.zeros(3 * 5, requires_grad=True)}
    with pytest.raises(ValueError, match="tafr_regularization_advantage"):
        ppo_loss(config=config, model_output=model_output, data=data.clone())


def test_ppo_loss_warm_start_zero_failure_contribution():
    data = _make_failure_batch([0.0, 0.0, 0.0], ready=False, update_count=0)
    data["tafr_failure_log_prob"] = torch.full((3, 4), -9.0)
    _attach_regularization_fields(data, ready=False)
    config = _make_actor_config()
    model_output = {"log_probs": torch.zeros(3 * 5, requires_grad=True)}
    loss, metrics = ppo_loss(config=config, model_output=model_output, data=data.clone())
    assert metrics["tafr_adv/failure_loss"].aggregate() == 0.0
    assert metrics["tafr_adv/failure_model_ready"].aggregate() == 0.0
    assert metrics["tafr_adv/failure_model_update_count"].aggregate() == 0.0
    assert metrics["tafr_adv/active_group_fraction"].aggregate() == 0.0
    assert metrics["tafr_adv/scored_token_fraction"].aggregate() == 0.0
    assert metrics["tafr_adv/ppo_ratio_mean"].aggregate() == 1.0
    assert metrics["tafr_adv/ppo_ratio_clip_fraction"].aggregate() == 0.0
    assert metrics["tafr_adv/advantage_mean"].aggregate() == 0.0


def test_ppo_loss_legacy_k3_mode():
    data = _make_failure_batch([0.0, 0.25, 1.0], ready=True, loss_version="k3_legacy")
    data["tafr_anchor_log_probs"] = torch.zeros(3, 4)
    data["tafr_replay_log_probs"] = torch.zeros(3, 4)
    config = _make_actor_config()
    model_output = {"log_probs": torch.zeros(3 * 5, requires_grad=True)}
    loss, metrics = ppo_loss(config=config, model_output=model_output, data=data.clone())
    assert "tafr_grpo/loss_grpo" in metrics
    assert "tafr_adv/failure_loss" not in metrics
    assert "tafr_grpo/kl_anchor" in metrics or "tafr_grpo/kl_replay" in metrics


def test_anchor_scoring_before_old_logprob_reuse_uses_response_device():
    batch = DataProto(
        batch=TensorDict(
            {
                "responses": torch.zeros(2, 3, dtype=torch.long),
                "response_mask": torch.ones(2, 3, dtype=torch.bool),
            },
            batch_size=[2],
        )
    )
    trainer = SimpleNamespace(
        tafr_enabled=True,
        tafr_config=SimpleNamespace(variant="full", beta=0.5),
        _tafr_can_score_with_vllm=lambda: False,
        _tafr_config_dict=lambda: {},
        actor_rollout_wg=SimpleNamespace(
            tafr_compute_anchor_log_prob=lambda data: TensorDict(
                {"tafr_anchor_log_probs": torch.full((2, 3), -1.0)}, batch_size=[2]
            )
        ),
    )

    RayPPOTrainer._tafr_compute_anchor_log_probs(trainer, batch)

    assert "old_log_probs" not in batch.batch
    torch.testing.assert_close(batch.batch["tafr_anchor_log_probs"], torch.full((2, 3), -1.0))


def test_areg_prefers_same_kernel_path_old_log_prob():
    """A_reg must use the pi_old scored on the anchor/failure path when present.

    Mixing the actor's varlen remove-padding old_log_probs with the anchor's dense
    forward injects a ~0.03 nats/token path difference that RMS matching amplifies.
    """
    trainer = SimpleNamespace()
    batch = DataProto(
        batch=TensorDict(
            {
                "old_log_probs": torch.full((2, 3), -1.0),
                "tafr_old_log_probs": torch.full((2, 3), -2.0),
            },
            batch_size=[2],
        )
    )
    chosen = RayPPOTrainer._tafr_reg_old_log_prob(trainer, batch)
    torch.testing.assert_close(chosen, torch.full((2, 3), -2.0))

    # vLLM path: no separate copy, so fall back to the canonical old_log_probs.
    del batch.batch["tafr_old_log_probs"]
    chosen = RayPPOTrainer._tafr_reg_old_log_prob(trainer, batch)
    torch.testing.assert_close(chosen, torch.full((2, 3), -1.0))


def test_anchor_worker_old_log_prob_is_propagated():
    batch = DataProto(
        batch=TensorDict(
            {
                "responses": torch.zeros(2, 3, dtype=torch.long),
                "response_mask": torch.ones(2, 3, dtype=torch.bool),
            },
            batch_size=[2],
        )
    )
    trainer = SimpleNamespace(
        tafr_enabled=True,
        tafr_config=SimpleNamespace(variant="full", beta=0.5),
        _tafr_can_score_with_vllm=lambda: False,
        _tafr_config_dict=lambda: {},
        actor_rollout_wg=SimpleNamespace(
            tafr_compute_anchor_log_prob=lambda data: TensorDict(
                {
                    "tafr_anchor_log_probs": torch.full((2, 3), -1.0),
                    "tafr_old_log_probs": torch.full((2, 3), -1.5),
                },
                batch_size=[2],
            )
        ),
    )

    RayPPOTrainer._tafr_compute_anchor_log_probs(trainer, batch)

    torch.testing.assert_close(batch.batch["tafr_old_log_probs"], torch.full((2, 3), -1.5))



# ── Plan §14: selective scoring ───────────────────────────────────────────────

def _make_selective_batch():
    """8 rows: group0 all-wrong x4, group1 mixed x2, group2 all-correct x2."""
    td = TensorDict(
        {
            "response_mask": torch.ones(8, 4, dtype=torch.bool),
            "tafr_group_reward_mean": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 1.0, 1.0]),
            "old_log_probs": torch.zeros(8, 4),
            "tafr_failure_log_prob": torch.full((8, 4), -9.0),
        },
        batch_size=[8],
    )
    return DataProto(batch=td)


def test_selective_active_group_mask():
    batch = _make_selective_batch()
    trainer = object.__new__(RayPPOTrainer)
    trainer.tafr_enabled = True
    trainer.tafr_config = TAFRGRPOConfig(enable=True, loss_version="failure_token_adv", score_only_active_groups=True)
    mask = RayPPOTrainer._tafr_active_group_mask(trainer, batch)
    assert mask.dtype == torch.bool
    assert mask[:6].all() and not mask[6:].any()


def test_selective_inactive_rows_filled_from_old_log_probs():
    batch = _make_selective_batch()
    trainer = object.__new__(RayPPOTrainer)
    trainer.tafr_enabled = True
    trainer.tafr_config = TAFRGRPOConfig(enable=True, loss_version="failure_token_adv")
    trainer.tafr_failure_model_ready = True
    RayPPOTrainer._tafr_sync_inactive_failure_log_probs(trainer, batch)
    lp = batch.batch["tafr_failure_log_prob"]
    # active rows keep their failure-model scores
    assert (lp[:6] == -9.0).all()
    # all-correct rows are filled from old_log_probs (zero), not the scorer
    torch.testing.assert_close(lp[6:], batch.batch["old_log_probs"][6:])
    # row order is preserved
    assert lp.shape == batch.batch["old_log_probs"].shape


def test_selective_warm_start_overwrites_all_rows():
    batch = _make_selective_batch()
    trainer = object.__new__(RayPPOTrainer)
    trainer.tafr_enabled = True
    trainer.tafr_config = TAFRGRPOConfig(enable=True, loss_version="failure_token_adv")
    trainer.tafr_failure_model_ready = False
    RayPPOTrainer._tafr_sync_inactive_failure_log_probs(trainer, batch)
    lp = batch.batch["tafr_failure_log_prob"]
    torch.testing.assert_close(lp, batch.batch["old_log_probs"])


def test_paper_path_has_distinct_anchor_and_failure_slots():
    # The paper path requires both frozen policies under distinct adapter IDs.
    assert "failure" in TAFR_VLLM_ADAPTERS
    assert TAFR_VLLM_ADAPTERS["failure"].int_id == 126
    assert "anchor" in TAFR_VLLM_ADAPTERS
    assert TAFR_VLLM_ADAPTERS["anchor"].int_id != TAFR_VLLM_ADAPTERS["failure"].int_id


# ── TAFR activation regressions ──────────────────────────────────────────────


def _valid_tafr_root_config(*, backend: str, lora_rank: int):
    return OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
            "actor_rollout_ref": {
                "actor": {"use_kl_loss": False},
                "model": {"lora_rank": lora_rank},
            },
            "custom_tafr_grpo": {
                "enable": True,
                "logprob_backend": backend,
                "reuse_rollout_log_probs": backend == "vllm",
            },
        }
    )


def test_backend_validation_full_finetune_hf_succeeds():
    config = validate_tafr_config(_valid_tafr_root_config(backend="hf", lora_rank=0))
    assert config.logprob_backend == "hf"


def test_backend_validation_full_finetune_vllm_fails_early():
    with pytest.raises(ValueError, match="requires actor_rollout_ref.model.lora_rank > 0"):
        validate_tafr_config(_valid_tafr_root_config(backend="vllm", lora_rank=0))


def test_backend_validation_lora_vllm_succeeds():
    config = validate_tafr_config(_valid_tafr_root_config(backend="vllm", lora_rank=8))
    assert config.logprob_backend == "vllm"


def test_backend_validation_rejects_hf_with_rollout_logprob_reuse():
    root = _valid_tafr_root_config(backend="hf", lora_rank=0)
    root.custom_tafr_grpo.reuse_rollout_log_probs = True
    with pytest.raises(ValueError, match="same log-probability backend"):
        validate_tafr_config(root)


def test_backend_validation_rejects_vllm_without_rollout_logprob_reuse():
    root = _valid_tafr_root_config(backend="vllm", lora_rank=8)
    root.custom_tafr_grpo.reuse_rollout_log_probs = False
    with pytest.raises(ValueError, match="same log-probability backend"):
        validate_tafr_config(root)



@pytest.mark.parametrize(
    "worker_output",
    [
        {"tafr_grpo/failure_sft_updates": 1.0},
        [{"tafr_grpo/failure_sft_updates": 1.0}],
    ],
)
def test_rank0_worker_result_accepts_dict_and_list(worker_output):
    result = RayPPOTrainer._tafr_rank0_result(worker_output, operation="test")
    assert result["tafr_grpo/failure_sft_updates"] == 1.0



@pytest.mark.parametrize("worker_output", [None, [], [None]])
def test_rank0_worker_result_rejects_missing_dictionary(worker_output):
    with pytest.raises(RuntimeError, match="returned no rank-0 metrics dictionary"):
        RayPPOTrainer._tafr_rank0_result(worker_output, operation="test")


class _FakeTAFRWorkerGroup:
    def __init__(self, result):
        self.result = result
        self.sft_calls = 0
        self.ema_calls = 0

    def tafr_failure_sft_update(self, records, config):
        self.sft_calls += 1
        assert records
        return self.result

    def tafr_update_failure_ema(self, config):
        self.ema_calls += 1
        return [{"tafr_failure_ema_updated": True}]


def _make_scheduled_sft_trainer(worker_result):
    trainer = object.__new__(RayPPOTrainer)
    trainer.tafr_enabled = True
    trainer.tafr_config = TAFRGRPOConfig(
        enable=True,
        logprob_backend="hf",
        sft_update_interval_grpo_steps=5,
        failure_min_updates_before_use=1,
    )
    trainer.global_steps = 5
    trainer.tafr_failure_collector = FailureDataCollector()
    trainer.tafr_failure_collector.add(prompt="p", response="wrong", reward=0)
    trainer.actor_rollout_wg = _FakeTAFRWorkerGroup(worker_result)
    trainer.tafr_failure_model_update_count = 0
    trainer.tafr_failure_model_ready = False
    trainer.tafr_failure_model_updated_since_save = False
    trainer._tafr_config_dict = lambda: {}
    return trainer


def test_scheduled_failure_sft_activates_and_emits_metrics():
    trainer = _make_scheduled_sft_trainer(
        [{"tafr_grpo/failure_sft_updates": 2.0, "tafr_grpo/failure_sft_loss": 1.25}]
    )

    metrics = RayPPOTrainer._tafr_run_failure_sft_if_due(trainer)

    assert trainer.actor_rollout_wg.sft_calls == 1
    assert trainer.actor_rollout_wg.ema_calls == 1
    assert len(trainer.tafr_failure_collector) == 0
    assert trainer.tafr_failure_model_update_count == 2
    assert trainer.tafr_failure_model_ready
    assert trainer.tafr_failure_model_updated_since_save
    assert metrics["tafr_grpo/failure_sft_updates"] == 2.0
    assert metrics["tafr_grpo/failure_sft_loss"] == 1.25
    assert metrics["tafr_grpo/did_sft_update_this_step"] == 1.0
    assert metrics["tafr_adv/failure_model_ready"] == 1.0


def test_failed_scheduled_failure_sft_keeps_buffer():
    trainer = _make_scheduled_sft_trainer([])

    with pytest.raises(RuntimeError, match="returned no rank-0 metrics dictionary"):
        RayPPOTrainer._tafr_run_failure_sft_if_due(trainer)

    assert len(trainer.tafr_failure_collector) == 1
    assert trainer.tafr_failure_model_update_count == 0
    assert trainer.actor_rollout_wg.ema_calls == 0


def test_tafr_checkpoint_resume_restores_optimizer_readiness_and_count(tmp_path):
    failure_dir = tmp_path / "failure_sft"
    failure_dir.mkdir()
    source_model = torch.nn.Linear(2, 2)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=1e-3)
    source_optimizer.zero_grad()
    source_model(torch.ones(1, 2)).sum().backward()
    source_optimizer.step()
    torch.save(source_model.state_dict(), failure_dir / "pytorch_model.bin")
    torch.save(source_optimizer.state_dict(), failure_dir / "optimizer.pt")
    torch.save(
        {
            "fail_ema_state": {},
            "grpo_ema_state": {},
            "failure_model_ready": True,
            "failure_model_update_count": 7,
        },
        tmp_path / "tafr_state.pt",
    )

    target_model = torch.nn.Linear(2, 2)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=1e-3)
    moved_to = []
    engine = SimpleNamespace(
        _tafr_failure=target_model,
        _tafr_failure_optimizer=target_optimizer,
        _tafr_fail_ema_state={},
        _tafr_grpo_ema_state={},
        _tafr_uses_vllm_logprobs=lambda config: True,
        _is_lora=False,
        _tafr_failure_model_ready=False,
        _tafr_failure_model_update_count=0,
        tafr_init=lambda config: {"tafr_initialized": True},
        _tafr_new_mode=lambda config: True,
        _tafr_move_optimizer_state=lambda optimizer, device: moved_to.append(device),
    )

    worker_result = [FSDPEngine.tafr_load(engine, str(tmp_path), {"loss_version": "failure_token_adv"})]
    result = RayPPOTrainer._tafr_rank0_result(worker_result, operation="checkpoint load")

    assert result["tafr_failure_model_ready"] is True
    assert result["tafr_failure_model_update_count"] == 7
    assert target_optimizer.state
    assert moved_to == [torch.device("cpu")]



def test_actor_worker_exposes_failure_ema_rpc():
    assert callable(ActorRolloutRefWorker.tafr_compute_failure_log_prob)
    calls = []
    worker = SimpleNamespace(
        role="actor",
        actor=SimpleNamespace(
            engine=SimpleNamespace(
                tafr_update_failure_ema=lambda config: calls.append(config) or {"tafr_failure_ema_updated": True}
            )
        ),
    )

    result = ActorRolloutRefWorker.tafr_update_failure_ema(worker, {"ema_gamma": 0.9})

    assert result == {"tafr_failure_ema_updated": True}
    assert calls == [{"ema_gamma": 0.9}]


def test_tafr_clone_preserves_buffer_dtype():
    """Clone weights must land in the destination's dtype, per tensor.

    FSDP keeps buffer_dtype=fp32 while params run bf16. Casting the whole state to
    one dtype gave the anchor clone bf16 RoPE inv_freq against the actor's fp32 --
    a ~0.03 nats/token gap that A_reg read as signal and amplified ~20x.
    """
    from verl.workers.engine.fsdp.transformer_impl import tafr_cast_state_to_dest

    dest = {
        "layer.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
        "rotary.inv_freq": torch.zeros(4, dtype=torch.float32),
        "step": torch.zeros(1, dtype=torch.long),
    }
    src = {
        "layer.weight": torch.ones(2, 2, dtype=torch.float32),
        "rotary.inv_freq": torch.ones(4, dtype=torch.float32),
        "step": torch.ones(1, dtype=torch.long),
    }
    out = tafr_cast_state_to_dest(src, dest, torch.bfloat16, torch.device("cpu"))
    assert out["layer.weight"].dtype == torch.bfloat16
    assert out["rotary.inv_freq"].dtype == torch.float32, "buffers must not be cast to bf16"
    assert out["step"].dtype == torch.long


def test_shuffle_control_preserves_magnitude_and_destroys_alignment():
    """The shuffle arm must differ from the real one ONLY in token alignment."""
    from verl.experimental.tafr_grpo.tafr_loss import shuffle_within_rows

    torch.manual_seed(0)
    values = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0]])
    mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)
    out = shuffle_within_rows(values, mask)

    # masked-out positions untouched; valid entries are a permutation of the originals
    assert out[0, 4] == 5.0
    assert sorted(out[0, :4].tolist()) == sorted(values[0, :4].tolist())
    # magnitude preserved, so RMS normalization sees an identical scale
    torch.testing.assert_close((out * mask).square().sum(), (values * mask).square().sum())


def test_shuffle_control_changes_advantage_but_not_its_rms():
    torch.manual_seed(0)
    kwargs = dict(
        anchor_log_prob=torch.randn(4, 12),
        old_log_prob=torch.randn(4, 12),
        failure_log_prob=torch.randn(4, 12),
        grpo_advantage=torch.ones(4, 12),
        group_reward_mean=torch.tensor([0.0, 0.25, 0.5, 0.75]),
        response_mask=torch.ones(4, 12, dtype=torch.bool),
        advantage_clip=50.0,
        failure_model_ready=True,
    )
    real, real_stats = compute_regularization_token_advantage(**kwargs)
    shuf, shuf_stats = compute_regularization_token_advantage(**kwargs, shuffle_tokens=True)

    assert not torch.allclose(real, shuf), "shuffle must change the per-token assignment"
    # same delivered strength -- the arms differ only in where the signal lands
    torch.testing.assert_close(real_stats["raw_advantage_rms"], shuf_stats["raw_advantage_rms"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="logprobs_from_logits uses a Triton kernel")
def test_tafr_log_probs_batching_matches_naive_implementation():
    """Length-sorted budget batching must be numerically identical to the naive loop.

    This is the exact path A_reg's three log-prob terms share; a re-indexing slip
    here reintroduces a systematic gap that RMS matching amplifies ~20x.
    """
    from verl.utils.torch_functional import logprobs_from_logits

    torch.manual_seed(0)
    vocab, prompt_width, response_len = 23, 5, 7
    width = prompt_width + response_len

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, 16)
            self.out = torch.nn.Linear(16, vocab)

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            return SimpleNamespace(logits=self.out(self.emb(input_ids)))

    dev = torch.device("cuda")
    module = Tiny().eval().to(dev)
    bsz = 9
    input_ids = torch.randint(1, vocab, (bsz, width), device=dev)
    attention_mask = torch.zeros(bsz, width, dtype=torch.long, device=dev)
    response_mask = torch.zeros(bsz, response_len, dtype=torch.long, device=dev)
    for i in range(bsz):
        plen = 2 + i % 3                      # left-padded prompt
        rlen = 1 + (i * 3) % response_len     # right-padded response
        attention_mask[i, prompt_width - plen : prompt_width + rlen] = 1
        response_mask[i, :rlen] = 1
    input_ids = input_ids * attention_mask    # pad ids to 0

    def naive(micro_bsz=4):
        all_lp = []
        with torch.no_grad():
            for s in range(0, bsz, micro_bsz):
                ids = input_ids[s : s + micro_bsz]
                logits = module(input_ids=ids).logits[:, :-1, :].contiguous()
                labels = ids[:, 1:].contiguous()
                lp = logprobs_from_logits(
                    logits=logits.view(-1, logits.size(-1)), labels=labels.view(-1), inplace_backward=False
                ).view(labels.shape)[:, -response_len:]
                all_lp.append(lp)
        return torch.cat(all_lp, dim=0)

    expected = naive()
    for budget in (16, 40, 10_000):  # many small batches through to one big batch
        got = FSDPEngine._tafr_model_log_probs(
            None, module, input_ids, attention_mask, response_mask, max_token_len=budget
        )
        m = response_mask.bool()
        torch.testing.assert_close(got[m], expected[m].float(), rtol=1e-5, atol=1e-6)


def _mix_kwargs(**over):
    torch.manual_seed(3)
    base = dict(
        anchor_log_prob=torch.randn(6, 20) * 0.05,
        old_log_prob=torch.zeros(6, 20),
        failure_log_prob=torch.randn(6, 20),  # deliberately much larger than the anchor term
        grpo_advantage=torch.ones(6, 20),
        group_reward_mean=torch.zeros(6),
        response_mask=torch.ones(6, 20, dtype=torch.bool),
        advantage_clip=1e6,
        failure_model_ready=True,
    )
    base.update(over)
    return base


def test_separate_normalization_makes_mix_match_the_weights():
    """Per-term normalization must decouple the mix from raw magnitudes."""
    _, unnorm = compute_regularization_token_advantage(**_mix_kwargs())
    raw_ratio = float(unnorm["failure_contribution_rms"] / unnorm["anchor_contribution_rms"])
    assert raw_ratio > 5, "fixture should start lopsided toward the failure term"

    _, norm = compute_regularization_token_advantage(
        **_mix_kwargs(), normalize_terms_separately=True, anchor_weight=1.0, failure_weight=1.0
    )
    got = float(norm["failure_contribution_rms"] / norm["anchor_contribution_rms"])
    assert abs(got - 1.0) < 1e-4, f"equal weights must give a 1:1 mix, got {got}"

    _, tilted = compute_regularization_token_advantage(
        **_mix_kwargs(), normalize_terms_separately=True, anchor_weight=0.25, failure_weight=1.0
    )
    got = float(tilted["failure_contribution_rms"] / tilted["anchor_contribution_rms"])
    assert abs(got - 4.0) < 1e-3, f"weights 0.25:1 must give a 4:1 mix, got {got}"


def test_weights_default_to_previous_behaviour():
    """Defaults must reproduce the plain sum, so this is opt-in."""
    a, _ = compute_regularization_token_advantage(**_mix_kwargs())
    b, _ = compute_regularization_token_advantage(**_mix_kwargs(), anchor_weight=1.0, failure_weight=1.0)
    torch.testing.assert_close(a, b)


def test_zero_signal_still_yields_zero_advantage_under_separate_normalization():
    """The step-1 control must survive: identical models in, exactly zero out."""
    kw = _mix_kwargs(anchor_log_prob=torch.zeros(6, 20), failure_log_prob=torch.zeros(6, 20))
    adv, _ = compute_regularization_token_advantage(**kw, normalize_terms_separately=True)
    torch.testing.assert_close(adv, torch.zeros_like(adv))


def test_tail_matching_keeps_areg_subordinate_to_grpo_per_token():
    """Normalizing by the p99 must put A_reg's tail at GRPO's scale, not 4x it."""
    torch.manual_seed(11)
    # Spiky A_reg (a few large tokens) against a flat GRPO advantage, as measured.
    failure = torch.zeros(4, 40)
    failure[:, ::13] = 8.0
    kw = dict(
        anchor_log_prob=torch.zeros(4, 40),
        old_log_prob=torch.zeros(4, 40),
        failure_log_prob=-failure,
        grpo_advantage=torch.full((4, 40), 0.7),
        group_reward_mean=torch.zeros(4),
        response_mask=torch.ones(4, 40, dtype=torch.bool),
        failure_model_ready=True,
    )
    grpo_per_token = 0.7

    adv, stats = compute_regularization_token_advantage(**kw, advantage_clip=5.0)
    ref = float(stats["reference_advantage_rms"])
    p99 = float(torch.quantile(adv.abs().flatten().float(), 0.99))
    assert abs(p99 - ref) < 0.35 * ref, f"tail should land near GRPO's scale, got p99={p99} vs {ref}"
    assert adv.abs().max() < 2.5 * grpo_per_token, "no token may dominate the outcome signal"
