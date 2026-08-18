import torch

from verl.experimental.sdc.data_collector import extract_sdc_outcome
from verl.experimental.sdc.sdc_loss import compute_sdc_loss


class _Batch:
    def __init__(self, outcome):
        self.batch = {"sdc_outcome": outcome}


def test_sdc_reads_verifier_outcome_not_shaped_reward():
    batch = _Batch(torch.tensor([1, 0]))
    outcome = extract_sdc_outcome(batch, {"acc": [1, 0]})
    torch.testing.assert_close(outcome, torch.tensor([True, False]))


def test_shaped_reward_cannot_replace_missing_semantic_outcome():
    class ShapedBatch:
        batch = {"token_level_rewards": torch.tensor([[0.5], [-0.2]])}

    try:
        extract_sdc_outcome(ShapedBatch(), {})
    except ValueError as exc:
        assert "explicit verifier correctness" in str(exc)
    else:
        raise AssertionError("SDC must not infer correctness from shaped rewards")


def test_additive_sdc_loss_is_zero_when_not_ready_even_with_shaped_base_data():
    current = torch.tensor([[0.2]], requires_grad=True)
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
