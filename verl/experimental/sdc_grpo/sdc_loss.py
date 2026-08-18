from __future__ import annotations

from typing import Optional

import torch

from verl.trainer.ppo.core_algos import agg_loss


def compute_sdc_loss(
    *,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    success_log_prob: torch.Tensor,
    failure_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    failure_mask: torch.Tensor,
    beta: float,
    loss_agg_mode: str,
    global_batch_info: Optional[dict] = None,
    use_importance_weight: bool = True,
    importance_weight_clip: float = 10.0,
    models_ready: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the fixed success/failure contrastive actor term.

    References are detached because the success and failure models are trained by
    their separate SFT optimizers. Only reward-zero response tokens contribute.
    """

    zero = log_prob.sum() * 0.0
    if not models_ready or beta == 0.0 or failure_mask.numel() == 0 or not bool(failure_mask.any()):
        return zero, {
            "sdc/loss": 0.0,
            "sdc/active_token_fraction": 0.0,
            "sdc/importance_weight_mean": 1.0,
            "sdc/importance_weight_clip_fraction": 0.0,
        }

    old = old_log_prob.detach().to(device=log_prob.device, dtype=log_prob.dtype)
    success = success_log_prob.detach().to(device=log_prob.device, dtype=log_prob.dtype)
    failure = failure_log_prob.detach().to(device=log_prob.device, dtype=log_prob.dtype)
    active = failure_mask.to(device=log_prob.device, dtype=torch.bool).unsqueeze(-1)
    mask = response_mask.to(device=log_prob.device, dtype=torch.bool) & active

    if use_importance_weight:
        log_ratio = (log_prob - old).clamp(min=-20.0, max=20.0)
        importance_weight = log_ratio.exp()
        unclipped = importance_weight
        if importance_weight_clip > 0:
            importance_weight = importance_weight.clamp(max=float(importance_weight_clip))
    else:
        importance_weight = torch.ones_like(log_prob)
        unclipped = importance_weight

    token_loss = importance_weight * (failure - success)
    loss = float(beta) * agg_loss(
        loss_mat=token_loss,
        loss_mask=mask,
        loss_agg_mode=loss_agg_mode,
        **(global_batch_info or {}),
    )
    valid = mask.sum().clamp_min(1).to(log_prob.dtype)
    active_tokens = response_mask.to(log_prob.dtype).sum().clamp_min(1)
    return loss, {
        "sdc/loss": float(loss.detach().cpu()),
        "sdc/active_token_fraction": float(mask.to(log_prob.dtype).sum().detach().cpu() / active_tokens.detach().cpu()),
        "sdc/importance_weight_mean": float((importance_weight.detach() * mask).sum().cpu() / valid.cpu()),
        "sdc/importance_weight_clip_fraction": float(((unclipped > importance_weight).to(log_prob.dtype) * mask).sum().cpu() / valid.cpu()),
    }
