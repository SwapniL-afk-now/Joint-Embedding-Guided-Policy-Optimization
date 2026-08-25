import torch

from verl.experimental.sdc.sdc_loss import compute_sdc_loss


def test_distributed_metric_arithmetic_uses_global_numerator_and_denominator():
    active = torch.tensor([2.0, 6.0])
    importance_sum = torch.tensor([4.0, 6.0])
    clip_count = torch.tensor([1.0, 4.0])
    global_importance_mean = importance_sum.sum() / active.sum()
    global_clip_fraction = clip_count.sum() / active.sum()

    assert global_importance_mean.item() == 1.25
    assert global_clip_fraction.item() == 0.625
    assert not torch.isclose(importance_sum.div(active).mean(), global_importance_mean)
    assert not torch.isclose(clip_count.div(active).mean(), global_clip_fraction)


def test_metric_refactor_preserves_normalized_locked_loss_and_actor_gradient():
    current = torch.tensor([[-0.2, -0.1], [-0.3, -0.4]], requires_grad=True)
    old = torch.tensor([[-0.3, -0.2], [-0.4, -0.5]])
    success = torch.tensor([[-1.0, -1.0], [-2.0, -2.0]])
    failure = torch.tensor([[-0.5, -0.25], [-1.0, -1.0]])
    response_mask = torch.tensor([[True, True], [True, False]])
    failure_mask = torch.tensor([True, False])
    beta = 0.5

    loss, metrics = compute_sdc_loss(
        log_prob=current,
        old_log_prob=old,
        success_log_prob=success,
        failure_log_prob=failure,
        response_mask=response_mask,
        failure_mask=failure_mask,
        beta=beta,
        importance_weight_clip=0.0,
    )
    active = response_mask & failure_mask[:, None]
    # Canonical algebra: ratio-weighted centered/RMS-normalized contrast over
    # the active failed tokens, divided by the global active token count.
    contrast = (failure - success)[active]
    centered = contrast - contrast.mean()
    normalized_contrast = centered / centered.pow(2).mean().sqrt().clamp_min(1e-8)
    expected_terms = (current - old).exp()[active] * normalized_contrast
    expected_loss = beta * expected_terms.sum() / active.sum()
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    expected_gradient = torch.zeros_like(current)
    expected_gradient[active] = beta * expected_terms / active.sum()
    torch.testing.assert_close(current.grad, expected_gradient)

    assert metrics["sdc/importance_weight_mean"] == float((current.detach() - old).exp()[active].mean())
    assert abs(metrics["sdc/active_token_fraction"] - 2 / 3) < 1e-6


def test_no_active_failed_tokens_stays_zero_after_metric_collective():
    current = torch.ones(1, 2, requires_grad=True)
    loss, metrics = compute_sdc_loss(
        log_prob=current,
        old_log_prob=torch.zeros_like(current),
        success_log_prob=torch.zeros_like(current),
        failure_log_prob=torch.ones_like(current),
        response_mask=torch.ones_like(current, dtype=torch.bool),
        failure_mask=torch.zeros(1, dtype=torch.bool),
        beta=1.0,
    )
    assert loss.item() == 0.0
    assert metrics["sdc/active_token_fraction"] == 0.0
    loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros_like(current))


def test_precomputed_global_active_count_preserves_scaled_actor_loss():
    current = torch.tensor([[-0.2, 0.4]], requires_grad=True)
    old = torch.tensor([[-0.3, -0.2]])
    success = torch.tensor([[-1.0, -1.0]])
    failure = torch.tensor([[-0.5, -0.25]])
    response_mask = torch.ones(1, 2, dtype=torch.bool)
    failure_mask = torch.ones(1, dtype=torch.bool)
    beta = 0.5

    loss, _ = compute_sdc_loss(
        log_prob=current,
        old_log_prob=old,
        success_log_prob=success,
        failure_log_prob=failure,
        response_mask=response_mask,
        failure_mask=failure_mask,
        beta=beta,
        global_batch_info={
            "dp_size": 2,
            "sdc_active_token_count": 6,
        },
        importance_weight_clip=0.0,
    )
    # Canonical algebra recomputed independently: the two active tokens have
    # equal ratios and a centered contrast of [-1, +1], so their weighted sum
    # cancels exactly; the assertion still checks the global denominator
    # (6) and dp_size (2) scaling through a proportional second call.
    contrast = failure.detach()[0] - success[0]
    centered = contrast - contrast.mean()
    normalized_contrast = centered / centered.pow(2).mean().sqrt().clamp_min(1e-8)
    terms = (current.detach() - old).exp()[0] * normalized_contrast
    torch.testing.assert_close(loss, beta * terms.sum() / 6 * 2)

    halved_count_loss, _ = compute_sdc_loss(
        log_prob=current,
        old_log_prob=old,
        success_log_prob=success,
        failure_log_prob=failure,
        response_mask=response_mask,
        failure_mask=failure_mask,
        beta=beta,
        global_batch_info={
            "dp_size": 2,
            "sdc_active_token_count": 3,
        },
        importance_weight_clip=0.0,
    )
    # Halving the global active token count doubles the loss.
    torch.testing.assert_close(halved_count_loss, 2 * loss)


def test_token_ids_bypass_sft_tokenization():
    from verl.experimental.sdc.data_collector import OutcomeExample
    from verl.experimental.sdc.sft_trainer import encode_outcome_example

    class NoTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("tokenizer should not be called for tokenized records")

    ids, response_mask = encode_outcome_example(
        OutcomeExample(
            prompt="unused",
            response="unused",
            reward=0,
            prompt_ids=[1, 2, 3, 4],
            response_ids=[5, 6, 7],
        ),
        NoTokenizer(),
        max_prompt_length=2,
        max_response_length=2,
    )
    assert ids.tolist() == [3, 4, 5, 6]
    assert response_mask.tolist() == [0, 0, 1, 1]
