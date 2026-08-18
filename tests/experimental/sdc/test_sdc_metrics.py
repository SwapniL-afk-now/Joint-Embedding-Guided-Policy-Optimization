import torch

from verl.experimental.sdc.sdc_loss import compute_sdc_loss


def test_distributed_metric_arithmetic_uses_global_numerator_and_denominator():
    active = torch.tensor([2.0, 6.0])
    importance_sum = torch.tensor([4.0, 6.0])
    clip_count = torch.tensor([1.0, 3.0])
    global_importance_mean = importance_sum.sum() / active.sum()
    global_clip_fraction = clip_count.sum() / active.sum()

    assert global_importance_mean.item() == 1.25
    assert global_clip_fraction.item() == 0.5
    assert not torch.isclose(importance_sum.div(active).mean(), global_importance_mean)
    assert not torch.isclose(clip_count.div(active).mean(), global_clip_fraction)


def test_metric_refactor_does_not_change_locked_loss_or_actor_gradient():
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
    expected_terms = (current - old).exp() * (failure - success)
    expected_loss = beta * expected_terms[active].sum() / active.sum()
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    expected_gradient = torch.zeros_like(current)
    expected_gradient[active] = beta * expected_terms[active] / active.sum()
    torch.testing.assert_close(current.grad, expected_gradient)

    assert metrics["sdc/importance_weight_mean"] == float((current.detach() - old).exp()[active].mean())
    assert metrics["sdc/active_token_fraction"] == 2 / 3


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
