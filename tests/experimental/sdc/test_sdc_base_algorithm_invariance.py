import torch

from verl.experimental.sdc.sdc_loss import compute_sdc_loss


def test_sdc_is_invariant_to_base_algorithm_metadata():
    kwargs = dict(
        log_prob=torch.tensor([[-0.2, -0.1]], requires_grad=True),
        old_log_prob=torch.tensor([[-0.3, -0.2]]),
        success_log_prob=torch.tensor([[-1.0, -1.0]]),
        failure_log_prob=torch.tensor([[-0.5, -0.25]]),
        response_mask=torch.ones(1, 2, dtype=torch.bool),
        failure_mask=torch.ones(1, dtype=torch.bool),
        beta=0.5,
        importance_weight_clip=0.0,
    )
    losses, grads = [], []
    for _name in ("grpo", "drgrpo", "dapo", "ngrpo", "avspo"):
        current = kwargs["log_prob"].detach().clone().requires_grad_(True)
        loss, _ = compute_sdc_loss(**{**kwargs, "log_prob": current})
        loss.backward()
        losses.append(loss.detach())
        grads.append(current.grad.detach())
    for value in losses[1:]:
        torch.testing.assert_close(value, losses[0])
    for value in grads[1:]:
        torch.testing.assert_close(value, grads[0])


def test_beta_zero_has_no_sdc_gradient():
    current = torch.tensor([[-0.2, -0.1]], requires_grad=True)
    loss, _ = compute_sdc_loss(
        log_prob=current,
        old_log_prob=torch.zeros_like(current),
        success_log_prob=torch.zeros_like(current),
        failure_log_prob=torch.ones_like(current),
        response_mask=torch.ones_like(current, dtype=torch.bool),
        failure_mask=torch.ones(1, dtype=torch.bool),
        beta=0.0,
    )
    loss.backward()
    assert loss.item() == 0.0
    torch.testing.assert_close(current.grad, torch.zeros_like(current))
