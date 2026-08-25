import pytest
import torch

from verl.experimental.sdc.data_collector import FailureDataCollector, SuccessDataCollector, validate_sdc_outcome
from verl.experimental.sdc.sdc_loss import compute_sdc_loss
from verl.experimental.sdc.sft_trainer import outcome_sft_loss


def _canonical_normalized_contrast(failure: torch.Tensor, success: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Recompute the canonical centered/RMS-normalized contrast independently."""

    contrast = (failure.detach() - success.detach())[active]
    if contrast.numel() > 1:
        contrast = contrast - contrast.mean()
    return contrast / contrast.pow(2).mean().sqrt().clamp_min(1e-8)


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
    active = torch.tensor([[True, True], [False, False]])
    normalized = _canonical_normalized_contrast(failure, success, active)
    expected = 0.5 * ((current[0] - old[0]).exp() * normalized).sum() / 2
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


# --- Canonical normalization invariants (T1-T4) ---


def _invariant_fixture(scale: float = 1.0, failure_offset: float = 0.0):
    current = torch.tensor([[0.2, -0.1, 0.3]], requires_grad=True)
    old = torch.tensor([[0.05, -0.02, 0.11]])
    success = torch.tensor([[-0.4, -0.5, -0.6]])
    failure = success + scale * (torch.tensor([[0.6, 0.2, 0.9]]) - success) + failure_offset
    return current, old, success, failure


def _run_invariant(current, old, success, failure, *, gate=1.0):
    loss, _ = compute_sdc_loss(
        log_prob=current,
        old_log_prob=old,
        success_log_prob=success,
        failure_log_prob=failure,
        response_mask=torch.ones_like(current, dtype=torch.bool),
        failure_mask=torch.tensor([True]),
        beta=0.5,
        importance_weight_clip=0.0,
        reliability_gate=gate,
    )
    return loss


@pytest.mark.parametrize("k", [-1.5, 0.0, 2.25])
def test_sdc_centering_makes_loss_invariant_to_failure_reference_offset(k):
    base = _run_invariant(*_invariant_fixture())
    shifted = _run_invariant(*_invariant_fixture(failure_offset=k))
    # Centering removes the common-mode offset of log(pi_F/pi_S).
    torch.testing.assert_close(shifted, base, rtol=0.0, atol=1e-5)


def test_sdc_rms_normalization_makes_loss_invariant_to_contrast_scale():
    base = _run_invariant(*_invariant_fixture(scale=1.0))
    scaled = _run_invariant(*_invariant_fixture(scale=4.0))
    # Detached RMS division rescales both numerator and denominator.
    torch.testing.assert_close(scaled, base, rtol=0.0, atol=1e-5)


@pytest.mark.parametrize("gate", [0.0, 0.25, 1.0])
def test_sdc_reliability_gate_scales_loss_and_gradient_linearly(gate):
    def run(gate_value, backward=False):
        current, old, success, failure = _invariant_fixture()
        loss = _run_invariant(current, old, success, failure, gate=gate_value)
        if backward:
            loss.backward()
        return loss.detach(), current.grad

    base_loss, base_grad = run(1.0, backward=True)
    gated_loss, gated_grad = run(gate, backward=True)
    torch.testing.assert_close(gated_loss, gate * base_loss)
    if gate == 0.0:
        assert gated_loss.item() == 0.0
        torch.testing.assert_close(gated_grad, torch.zeros_like(gated_grad))
    else:
        torch.testing.assert_close(gated_grad, gate * base_grad)


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_single_active_token_skips_centering_and_keeps_sign_direction(direction):
    current = torch.tensor([[0.3]], requires_grad=True)
    loss, metrics = compute_sdc_loss(
        log_prob=current,
        old_log_prob=torch.zeros(1, 1),
        success_log_prob=torch.zeros(1, 1),
        failure_log_prob=torch.full((1, 1), direction),
        response_mask=torch.ones(1, 1, dtype=torch.bool),
        failure_mask=torch.tensor([True]),
        beta=0.5,
        importance_weight_clip=0.0,
    )
    # Centering is skipped for a single active token; RMS normalizes the
    # contrast to its sign, so |loss| == beta*|ratio|/N with sign preserved.
    expected = 0.5 * float(current.detach().exp()) * direction
    torch.testing.assert_close(loss, torch.tensor(expected))
    assert metrics["sdc/contrast_center"] == 0.0
    loss.backward()
    torch.testing.assert_close(current.grad, torch.full_like(current, expected))
