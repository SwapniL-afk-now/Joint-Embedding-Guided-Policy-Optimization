import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.sdc_grpo.config import SDCGRPOConfig, should_checkpoint, should_run_sft, validate_sdc_config
from verl.experimental.sdc_grpo.data_collector import FailureDataCollector, OutcomeDataCollector, SuccessDataCollector
from verl.experimental.sdc_grpo.sdc_loss import compute_sdc_loss
from verl.experimental.sdc_grpo.sft_trainer import OutcomeSFTTrainer, outcome_sft_loss
from verl.experimental.sdc_grpo.vllm_scoring import SDC_VLLM_ADAPTERS, extract_response_logprobs_from_prompt_logprobs


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


def test_response_only_sft_loss_masks_prompt_tokens():
    log_probs = torch.tensor([[-100.0, -2.0, -4.0]])
    response_mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)
    assert outcome_sft_loss(log_probs, response_mask).item() == 3.0


def test_sft_optimizer_is_separate_from_actor_optimizer():
    actor = torch.nn.Linear(2, 2)
    success = torch.nn.Linear(2, 2)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    success_optimizer = torch.optim.AdamW(success.parameters(), lr=1e-3)
    trainer = OutcomeSFTTrainer(success, success_optimizer)
    trainer.assert_separate_from(actor_optimizer)


def test_readiness_schedule_and_config_validation():
    cfg = SDCGRPOConfig(enable=True, sft_update_interval_grpo_steps=5, checkpoint_interval_grpo_steps=10)
    assert not should_run_sft(1, cfg)
    assert should_run_sft(5, cfg)
    assert not should_checkpoint(5, cfg)
    assert should_checkpoint(10, cfg)
    root = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
            "actor_rollout_ref": {"actor": {"use_kl_loss": False}, "model": {"lora_rank": 0}},
            "custom_sdc_grpo": {"enable": True, "logprob_backend": "hf", "reuse_rollout_log_probs": False},
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
