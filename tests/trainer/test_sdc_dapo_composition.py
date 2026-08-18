import torch

from verl.experimental.sdc.sdc_loss import compute_sdc_loss


def test_dapo_final_batch_uses_only_retained_failed_rows_for_sdc():
    current = torch.zeros(2, 2, requires_grad=True)
    loss, metrics = compute_sdc_loss(
        log_prob=current,
        old_log_prob=torch.zeros_like(current),
        success_log_prob=torch.zeros_like(current),
        failure_log_prob=torch.ones_like(current),
        response_mask=torch.ones_like(current, dtype=torch.bool),
        failure_mask=torch.tensor([True, False]),
        beta=1.0,
        importance_weight_clip=0.0,
    )
    assert metrics["sdc/active_token_fraction"] == 0.5
    loss.backward()
    assert torch.all(current.grad[1] == 0)
