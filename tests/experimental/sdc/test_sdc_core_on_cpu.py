import torch

from verl.experimental.sdc.data_collector import FailureDataCollector, SuccessDataCollector, validate_sdc_outcome
from verl.experimental.sdc.sdc_loss import compute_sdc_loss
from verl.experimental.sdc.sft_trainer import outcome_sft_loss


def test_locked_sdc_algebra_masks_failed_rows_and_detaches_references():
    current = torch.tensor([[-0.2, -0.1], [-0.3, -0.4]], requires_grad=True)
    old = torch.zeros_like(current)
    success = torch.tensor([[-1.0, -1.0], [-2.0, -2.0]], requires_grad=True)
    failure = torch.tensor([[-0.5, -0.25], [-1.0, -1.0]], requires_grad=True)
    loss, _ = compute_sdc_loss(
        log_prob=current,
        old_log_prob=old,
        success_log_prob=success,
        failure_log_prob=failure,
        response_mask=torch.ones_like(current, dtype=torch.bool),
        failure_mask=torch.tensor([True, False]),
        beta=0.5,
        importance_weight_clip=0.0,
    )
    expected = 0.5 * ((current[0] - old[0]).exp() * (failure.detach()[0] - success.detach()[0])).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert success.grad is None and failure.grad is None
    assert torch.all(current.grad[1] == 0)


def test_sdc_is_exactly_zero_before_both_references_are_ready():
    current = torch.ones(1, 2, requires_grad=True)
    loss, _ = compute_sdc_loss(
        log_prob=current,
        old_log_prob=torch.zeros_like(current),
        success_log_prob=torch.zeros_like(current),
        failure_log_prob=torch.ones_like(current),
        response_mask=torch.ones_like(current, dtype=torch.bool),
        failure_mask=torch.ones(1, dtype=torch.bool),
        beta=1.0,
        models_ready=False,
    )
    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros_like(current))


def test_outcome_validation_rejects_shaped_or_nonbinary_values():
    torch.testing.assert_close(validate_sdc_outcome(torch.tensor([[0.0], [1.0]])), torch.tensor([False, True]))
    try:
        validate_sdc_outcome(torch.tensor([[0.0, 1.0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("token-level tensors must not be interpreted as semantic outcomes")


def test_success_and_failure_collectors_are_symmetric_and_bounded():
    success, failure = SuccessDataCollector(max_size=1), FailureDataCollector(max_size=1)
    assert success.add(prompt="p", response="ok", reward=1)
    assert not success.add(prompt="p", response="bad", reward=0)
    assert failure.add(prompt="p", response="bad", reward=0)
    assert not failure.add(prompt="p", response="fraction", reward=0.5)
    assert len(success) == len(failure) == 1


def test_response_only_sft_loss_ignores_prompt_tokens():
    log_prob = torch.tensor([[-100.0, -2.0, -4.0]])
    mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)
    assert outcome_sft_loss(log_prob, mask).item() == 3.0
