import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.sdc.config import SDCConfig, should_checkpoint, should_run_sft, validate_sdc_config
from verl.experimental.sdc.data_collector import (
    FailureDataCollector,
    OutcomeDataCollector,
    SuccessDataCollector,
    binary_outcome_scores,
)
from verl.experimental.sdc.sdc_loss import compute_sdc_loss, contrast_quality_metrics
from verl.experimental.sdc.sft_trainer import OutcomeSFTTrainer, outcome_sft_loss
from verl.experimental.sdc.vllm_scoring import SDC_VLLM_ADAPTERS, extract_response_logprobs_from_prompt_logprobs


def test_sdc_loss_matches_direct_algebra_and_detaches_references():
    current = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    old = torch.tensor([[-1.5, -2.5]])
    success = torch.tensor([[-3.0, -4.0]], requires_grad=True)
    failure = torch.tensor([[-2.0, -1.0]], requires_grad=True)
    response_mask = torch.tensor([[1, 1]], dtype=torch.bool)
    loss, metrics = compute_sdc_loss(
        log_prob=current,
        old_log_prob=old,
        success_log_prob=success,
        failure_log_prob=failure,
        response_mask=response_mask,
        failure_mask=torch.tensor([True]),
        beta=0.5,
        loss_agg_mode="token-mean",
    )
    expected = 0.5 * (((current - old).exp()) * (failure.detach() - success.detach())).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert current.grad is not None
    assert success.grad is None and failure.grad is None
    assert metrics["sdc/active_token_fraction"] == 1.0


def test_held_out_contrast_quality_uses_separate_success_and_failure_scores():
    metrics = contrast_quality_metrics(
        success_log_prob_under_success=torch.tensor([[-1.0, -1.0], [-2.0, -2.0]]),
        success_log_prob_under_failure=torch.tensor([[-2.0, -2.0], [-3.0, -3.0]]),
        failure_log_prob_under_success=torch.tensor([[-3.0, -3.0], [-4.0, -4.0]]),
        failure_log_prob_under_failure=torch.tensor([[-2.0, -2.0], [-2.0, -2.0]]),
        success_response_mask=torch.ones(2, 2, dtype=torch.bool),
        failure_response_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    assert metrics["sdc/quality_C_S"] == 1.0
    assert metrics["sdc/quality_C_F"] == -1.5
    assert metrics["sdc/quality_auc"] == 1.0


def test_sdc_gradient_matches_importance_weighted_algebra():
    current = torch.tensor([[0.2, -0.1]], requires_grad=True)
    old = torch.tensor([[0.0, 0.1]])
    success = torch.tensor([[-0.4, -0.5]])
    failure = torch.tensor([[0.6, 0.7]])
    beta = 0.25
    loss, _ = compute_sdc_loss(
        log_prob=current,
        old_log_prob=old,
        success_log_prob=success,
        failure_log_prob=failure,
        response_mask=torch.ones_like(current, dtype=torch.bool),
        failure_mask=torch.tensor([True]),
        beta=beta,
        loss_agg_mode="token-mean",
        importance_weight_clip=0.0,
    )
    loss.backward()
    expected = beta * (current.detach() - old).exp() * (failure - success) / 2.0
    torch.testing.assert_close(current.grad, expected)


def test_sdc_reference_swap_negates_loss_and_gradient():
    def run(success, failure):
        current = torch.tensor([[0.2, -0.1]], requires_grad=True)
        loss, _ = compute_sdc_loss(
            log_prob=current,
            old_log_prob=torch.zeros_like(current),
            success_log_prob=success,
            failure_log_prob=failure,
            response_mask=torch.ones_like(current, dtype=torch.bool),
            failure_mask=torch.tensor([True]),
            beta=0.5,
            loss_agg_mode="token-mean",
            importance_weight_clip=0.0,
        )
        loss.backward()
        return loss.detach(), current.grad

    success = torch.tensor([[-0.4, -0.5]])
    failure = torch.tensor([[0.6, 0.7]])
    loss_a, grad_a = run(success, failure)
    loss_b, grad_b = run(failure, success)
    torch.testing.assert_close(loss_b, -loss_a)
    torch.testing.assert_close(grad_b, -grad_a)


def test_sdc_equal_references_have_zero_loss_and_gradient():
    current = torch.tensor([[0.2, -0.1]], requires_grad=True)
    loss, _ = compute_sdc_loss(
        log_prob=current,
        old_log_prob=torch.zeros_like(current),
        success_log_prob=torch.tensor([[-0.4, -0.5]]),
        failure_log_prob=torch.tensor([[-0.4, -0.5]]),
        response_mask=torch.ones_like(current, dtype=torch.bool),
        failure_mask=torch.tensor([True]),
        beta=1.0,
        loss_agg_mode="token-mean",
        importance_weight_clip=0.0,
    )
    loss.backward()
    assert loss.item() == 0.0
    torch.testing.assert_close(current.grad, torch.zeros_like(current))


def test_sdc_rejects_disabled_importance_weight():
    with pytest.raises(ValueError, match="importance weighting"):
        compute_sdc_loss(
            log_prob=torch.zeros(1, 1, requires_grad=True),
            old_log_prob=torch.zeros(1, 1),
            success_log_prob=torch.zeros(1, 1),
            failure_log_prob=torch.ones(1, 1),
            response_mask=torch.ones(1, 1, dtype=torch.bool),
            failure_mask=torch.tensor([True]),
            beta=1.0,
            loss_agg_mode="token-mean",
            use_importance_weight=False,
        )


def test_sdc_uses_active_failed_token_count_for_normalization():
    loss, _ = compute_sdc_loss(
        log_prob=torch.zeros(2, 2, requires_grad=True),
        old_log_prob=torch.zeros(2, 2),
        success_log_prob=torch.zeros(2, 2),
        failure_log_prob=torch.ones(2, 2),
        response_mask=torch.ones(2, 2, dtype=torch.bool),
        failure_mask=torch.tensor([True, False]),
        beta=1.0,
        loss_agg_mode="token-mean",
        global_batch_info={"batch_num_tokens": 100, "dp_size": 1},
        importance_weight_clip=0.0,
    )
    assert loss.item() == 1.0


def test_sdc_row_mask_and_reference_rows_survive_reordering():
    current = torch.tensor([[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]], requires_grad=True)
    old = torch.zeros_like(current)
    success = torch.tensor([[-1.0, -1.0], [-2.0, -2.0], [-3.0, -3.0]])
    failure = torch.tensor([[-1.5, -1.5], [-1.0, -1.0], [-4.0, -4.0]])
    kwargs = dict(
        old_log_prob=old,
        success_log_prob=success,
        failure_log_prob=failure,
        response_mask=torch.ones_like(current, dtype=torch.bool),
        failure_mask=torch.tensor([False, True, False]),
        beta=1.0,
        loss_agg_mode="token-mean",
        importance_weight_clip=0.0,
    )
    loss, _ = compute_sdc_loss(log_prob=current, **kwargs)
    reordered = torch.tensor([2, 0, 1])
    reordered_loss, _ = compute_sdc_loss(
        log_prob=current.detach()[reordered],
        old_log_prob=old[reordered],
        success_log_prob=success[reordered],
        failure_log_prob=failure[reordered],
        response_mask=kwargs["response_mask"][reordered],
        failure_mask=kwargs["failure_mask"][reordered],
        beta=1.0,
        loss_agg_mode="token-mean",
        importance_weight_clip=0.0,
    )
    torch.testing.assert_close(loss, reordered_loss)
    loss.backward()
    assert torch.all(current.grad[[0, 2]] == 0)


def test_all_zero_group_has_zero_grpo_but_nonzero_sdc_gradient():
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    rewards = torch.zeros(2, 2)
    response_mask = torch.ones_like(rewards, dtype=torch.bool)
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.asarray(["group", "group"]),
        norm_adv_by_std_in_grpo=True,
    )
    assert torch.all(advantages == 0)
    grpo_log_prob = torch.zeros_like(rewards, requires_grad=True)
    (-grpo_log_prob.exp() * advantages).mean().backward()
    assert torch.all(grpo_log_prob.grad == 0)

    sdc_log_prob = torch.zeros_like(rewards, requires_grad=True)
    sdc_loss, _ = compute_sdc_loss(
        log_prob=sdc_log_prob,
        old_log_prob=torch.zeros_like(rewards),
        success_log_prob=torch.zeros_like(rewards),
        failure_log_prob=torch.ones_like(rewards),
        response_mask=response_mask,
        failure_mask=torch.tensor([True, True]),
        beta=1.0,
        loss_agg_mode="token-mean",
        importance_weight_clip=0.0,
    )
    sdc_loss.backward()
    assert torch.any(sdc_log_prob.grad != 0)


def test_sdc_uses_only_reward_zero_rows_and_zero_fills_inactive_rows():
    values = torch.tensor([[-1.0, -2.0], [-1.0, -2.0]], requires_grad=True)
    refs = torch.zeros_like(values)
    loss, metrics = compute_sdc_loss(
        log_prob=values,
        old_log_prob=refs,
        success_log_prob=torch.full_like(values, -2.0),
        failure_log_prob=torch.full_like(values, -1.0),
        response_mask=torch.ones_like(values, dtype=torch.bool),
        failure_mask=torch.tensor([True, False]),
        beta=1.0,
        loss_agg_mode="token-mean",
    )
    assert metrics["sdc/active_token_fraction"] == 0.5
    assert loss.item() != 0.0
    loss.backward()
    assert torch.all(values.grad[1] == 0)
    no_failures, _ = compute_sdc_loss(
        log_prob=values,
        old_log_prob=refs,
        success_log_prob=refs,
        failure_log_prob=refs,
        response_mask=torch.ones_like(values, dtype=torch.bool),
        failure_mask=torch.tensor([False, False]),
        beta=1.0,
        loss_agg_mode="token-mean",
    )
    assert no_failures.item() == 0.0


def test_sdc_importance_weight_is_safe_and_clipped():
    loss, metrics = compute_sdc_loss(
        log_prob=torch.tensor([[10.0]], requires_grad=True),
        old_log_prob=torch.tensor([[-10.0]]),
        success_log_prob=torch.tensor([[-2.0]]),
        failure_log_prob=torch.tensor([[-1.0]]),
        response_mask=torch.ones(1, 1, dtype=torch.bool),
        failure_mask=torch.tensor([True]),
        beta=1.0,
        loss_agg_mode="token-mean",
        importance_weight_clip=2.0,
    )
    assert torch.isfinite(loss)
    assert metrics["sdc/importance_weight_mean"] == 2.0
    assert metrics["sdc/importance_weight_clip_fraction"] == 1.0


def test_sdc_clipped_tokens_have_zero_log_ratio_gradient():
    current = torch.tensor([[30.0]], requires_grad=True)
    loss, metrics = compute_sdc_loss(
        log_prob=current,
        old_log_prob=torch.zeros(1, 1),
        success_log_prob=torch.zeros(1, 1),
        failure_log_prob=torch.ones(1, 1),
        response_mask=torch.ones(1, 1, dtype=torch.bool),
        failure_mask=torch.tensor([True]),
        beta=1.0,
        loss_agg_mode="token-mean",
        importance_weight_clip=2.0,
    )
    loss.backward()
    assert metrics["sdc/log_ratio_clipped_fraction"] == 1.0
    assert metrics["sdc/importance_weight_clip_fraction"] == 1.0
    assert current.grad.item() == 0.0


def test_sdc_is_exactly_zero_before_both_models_are_ready():
    current = torch.tensor([[-1.0]], requires_grad=True)
    loss, _ = compute_sdc_loss(
        log_prob=current,
        old_log_prob=torch.tensor([[-2.0]]),
        success_log_prob=torch.tensor([[-3.0]]),
        failure_log_prob=torch.tensor([[-1.0]]),
        response_mask=torch.ones(1, 1, dtype=torch.bool),
        failure_mask=torch.tensor([True]),
        beta=1.0,
        loss_agg_mode="token-mean",
        models_ready=False,
    )
    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros_like(current))


def test_collectors_accept_only_their_binary_outcome():
    success = SuccessDataCollector(max_size=2)
    failure = FailureDataCollector(max_size=2)
    assert success.add(prompt="p", response="ok", reward=1)
    assert not success.add(prompt="p", response="wrong", reward=0)
    assert failure.add(prompt="p", response="wrong", reward=0)
    assert not failure.add(prompt="p", response="fractional", reward=0.5)
    assert success.to_records()[0]["reward"] == 1
    assert failure.to_records()[0]["reward"] == 0
    with pytest.raises(ValueError):
        OutcomeDataCollector(target_reward=2)


def test_binary_reward_validation_rejects_fractional_rows():
    torch.testing.assert_close(
        binary_outcome_scores(torch.tensor([0.0, 1.0])),
        torch.tensor([False, True]),
    )
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        binary_outcome_scores(torch.tensor([[0.5, 0.0]]))


def test_collectors_round_trip_and_consume_oldest_examples():
    collector = FailureDataCollector(max_size=3)
    collector.add(prompt="p0", response="r0", reward=0, metadata={"global_grpo_step": 2})
    collector.add(prompt="p1", response="r1", reward=0, metadata={"global_grpo_step": 4})
    state = collector.state_dict()
    restored = FailureDataCollector(max_size=3)
    restored.load_state_dict(state)
    assert restored.to_records() == collector.to_records()
    assert restored.buffer_age(7) == 5.0
    restored.consume(1)
    assert [row["prompt"] for row in restored.to_records()] == ["p1"]


def test_response_only_sft_loss_masks_prompt_tokens():
    log_probs = torch.tensor([[-100.0, -2.0, -4.0]])
    response_mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)
    assert outcome_sft_loss(log_probs, response_mask).item() == 3.0


class _FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert tokenize
        return list(range(10 if len(messages) == 1 else 15))


def test_sft_prompt_and_response_limits_are_independent():
    from verl.experimental.sdc_grpo.data_collector import OutcomeExample
    from verl.experimental.sdc_grpo.sft_trainer import encode_outcome_example

    ids, response_mask = encode_outcome_example(
        OutcomeExample(prompt="long prompt", response="long response", reward=0),
        _FakeTokenizer(),
        max_prompt_length=4,
        max_response_length=3,
    )
    assert ids.tolist() == [6, 7, 8, 9, 10, 11, 12]
    assert response_mask.tolist() == [0, 0, 0, 0, 1, 1, 1]


def test_lora_sft_optimizer_keeps_base_frozen_and_adapter_trainable():
    from verl.experimental.sdc_grpo.sft_trainer import configure_sft_trainable_parameters

    module = torch.nn.Module()
    module.base = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    module.adapter = torch.nn.Parameter(torch.ones(2), requires_grad=True)
    trainable = configure_sft_trainable_parameters(module, lora_enabled=True)
    assert len(trainable) == 1 and trainable[0] is module.adapter
    assert not module.base.requires_grad and module.adapter.requires_grad
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    assert {id(p) for p in optimizer.param_groups[0]["params"]} == {id(module.adapter)}


def test_pairing_leaves_unmatched_buffer_rows_available():
    from verl.experimental.sdc_grpo.sft_trainer import pair_outcome_records

    success = [{"prompt": "s0"}, {"prompt": "s1"}]
    failure = [{"prompt": "f0"}]
    selected_success, selected_failure = pair_outcome_records(success, failure, max_examples=8)
    assert selected_success == success[:1]
    assert selected_failure == failure
    assert success[1:] == [{"prompt": "s1"}]
    assert pair_outcome_records(success, [], max_examples=8) == ([], [])


def test_sft_optimizer_is_separate_from_actor_optimizer():
    actor = torch.nn.Linear(2, 2)
    success = torch.nn.Linear(2, 2)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    success_optimizer = torch.optim.AdamW(success.parameters(), lr=1e-3)
    trainer = OutcomeSFTTrainer(success, success_optimizer)
    trainer.assert_separate_from(actor_optimizer)


def test_readiness_schedule_and_config_validation():
    cfg = SDCConfig(enable=True, sft_update_interval_policy_steps=5, checkpoint_interval_policy_steps=10)
    assert not should_run_sft(1, cfg)
    assert should_run_sft(5, cfg)
    assert not should_checkpoint(5, cfg)
    assert should_checkpoint(10, cfg)
    root = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
            "actor_rollout_ref": {"actor": {"use_kl_loss": False}, "model": {"lora_rank": 0}},
            "custom_sdc": {"enable": True, "scoring_backend": "hf", "reuse_rollout_log_probs": False},
        }
    )
    assert validate_sdc_config(root).enable


def test_vllm_success_failure_adapters_are_unique():
    assert set(SDC_VLLM_ADAPTERS) == {"success", "failure"}
    assert len({spec.int_id for spec in SDC_VLLM_ADAPTERS.values()}) == 2
    assert {spec.name for spec in SDC_VLLM_ADAPTERS.values()} == {"sdc_success", "sdc_failure"}


def test_hf_and_vllm_response_slice_use_the_same_token_span():
    prompt_logprobs = [[-0.1], [-0.2], [-1.0], [-1.5], [-2.0]]
    assert extract_response_logprobs_from_prompt_logprobs(
        prompt_logprobs=prompt_logprobs, prompt_len=3, response_len=2
    ) == [-1.0, -1.5]
