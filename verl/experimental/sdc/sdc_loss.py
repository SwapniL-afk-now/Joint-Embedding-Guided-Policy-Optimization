from __future__ import annotations

from typing import Optional

import torch


def _zero_metrics() -> dict[str, float]:
    return {
        "sdc/loss": 0.0,
        "sdc/active_token_fraction": 0.0,
        "sdc/importance_weight_mean": 1.0,
        "sdc/log_ratio_clipped_fraction": 0.0,
        "sdc/importance_weight_clip_fraction": 0.0,
        "sdc/contrast_p10": 0.0,
        "sdc/contrast_p50": 0.0,
        "sdc/contrast_p90": 0.0,
        "sdc/contrast_max": 0.0,
        "sdc/contrast_sequence_mean": 0.0,
    }


def compute_sdc_loss(
    *,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    success_log_prob: torch.Tensor,
    failure_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    failure_mask: torch.Tensor,
    beta: float,
    loss_agg_mode: str = "token-mean",
    global_batch_info: Optional[dict] = None,
    use_importance_weight: bool = True,
    importance_weight_clip: float = 10.0,
    models_ready: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the locked SDC auxiliary on failed response tokens only.

    The references are detached and the denominator is always the number of
    active failed response tokens.  ``loss_agg_mode`` is retained as an API
    argument for callers during migration but only ``token-mean`` is valid.
    """

    if loss_agg_mode != "token-mean":
        raise ValueError("SDC uses fixed loss aggregation 'token-mean'.")
    zero = log_prob.sum() * 0.0
    if not models_ready or beta == 0.0 or failure_mask.numel() == 0 or not bool(failure_mask.any()):
        return zero, _zero_metrics()
    if not use_importance_weight:
        raise ValueError("SDC requires importance weighting; disabling it removes the actor gradient.")

    old = old_log_prob.detach().to(device=log_prob.device, dtype=log_prob.dtype)
    success = success_log_prob.detach().to(device=log_prob.device, dtype=log_prob.dtype)
    failure = failure_log_prob.detach().to(device=log_prob.device, dtype=log_prob.dtype)
    active = failure_mask.to(device=log_prob.device, dtype=torch.bool).reshape(-1, 1)
    mask = response_mask.to(device=log_prob.device, dtype=torch.bool) & active
    raw_log_ratio = log_prob - old
    safe_log_ratio = raw_log_ratio.clamp(min=-20.0, max=20.0)
    ratio = safe_log_ratio.exp()
    ratio_clipped = raw_log_ratio.ne(safe_log_ratio)
    weight_clipped = torch.zeros_like(ratio, dtype=torch.bool)
    if importance_weight_clip > 0:
        clipped_ratio = ratio.clamp(max=float(importance_weight_clip))
        weight_clipped = ratio.ne(clipped_ratio)
        ratio = clipped_ratio

    # Locked algebra: + beta E[pi_theta/pi_old * log(pi_F/pi_S)].
    token_loss = ratio * (failure - success)
    local_count = mask.to(log_prob.dtype).sum()
    count = local_count.clone()
    info = global_batch_info or {}
    dp_size = int(info.get("dp_size", 1) or 1)
    if dp_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(count)
    count = count.clamp_min(1.0)
    loss = float(beta) * (token_loss * mask).sum() / count * dp_size
    active_tokens = response_mask.to(log_prob.dtype).sum().clamp_min(1.0)
    active_contrast = (success - failure)[mask]
    metrics = {
        "sdc/loss": float(loss.detach().cpu()),
        "sdc/active_token_fraction": float(
            mask.to(log_prob.dtype).sum().detach().cpu() / active_tokens.detach().cpu()
        ),
        "sdc/importance_weight_mean": float((ratio.detach() * mask).sum().cpu() / count.cpu()),
        "sdc/log_ratio_clipped_fraction": float(
            (ratio_clipped.to(log_prob.dtype) * mask).sum().cpu() / count.cpu()
        ),
        "sdc/importance_weight_clip_fraction": float(
            (weight_clipped.to(log_prob.dtype) * mask).sum().cpu() / count.cpu()
        ),
    }
    if active_contrast.numel():
        q = torch.quantile(
            active_contrast.float(),
            torch.tensor([0.10, 0.50, 0.90], device=active_contrast.device),
        )
        sequence_mean = ((success - failure) * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        sequence_mean = sequence_mean[mask.any(dim=-1)].mean()
        metrics.update(
            {
                "sdc/contrast_p10": float(q[0].detach().cpu()),
                "sdc/contrast_p50": float(q[1].detach().cpu()),
                "sdc/contrast_p90": float(q[2].detach().cpu()),
                "sdc/contrast_max": float(active_contrast.max().detach().cpu()),
                "sdc/contrast_sequence_mean": float(sequence_mean.detach().cpu()),
            }
        )
    else:
        metrics.update({key: 0.0 for key in _zero_metrics() if key.startswith("sdc/contrast_")})
    return loss, metrics


def contrast_quality_metrics(
    success_log_prob_under_success: torch.Tensor,
    success_log_prob_under_failure: torch.Tensor,
    failure_log_prob_under_success: torch.Tensor,
    failure_log_prob_under_failure: torch.Tensor,
    *,
    success_response_mask: Optional[torch.Tensor] = None,
    failure_response_mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    if success_log_prob_under_success.shape != success_log_prob_under_failure.shape:
        raise ValueError("Held-out success score tensors must have the same shape.")
    if failure_log_prob_under_success.shape != failure_log_prob_under_failure.shape:
        raise ValueError("Held-out failure score tensors must have the same shape.")
    success_mask = (
        torch.ones_like(success_log_prob_under_success, dtype=torch.bool)
        if success_response_mask is None
        else success_response_mask.bool()
    )
    failure_mask = (
        torch.ones_like(failure_log_prob_under_failure, dtype=torch.bool)
        if failure_response_mask is None
        else failure_response_mask.bool()
    )
    success_contrast = (success_log_prob_under_success - success_log_prob_under_failure).detach()
    failure_contrast = (failure_log_prob_under_success - failure_log_prob_under_failure).detach()
    if success_contrast.ndim > 1:
        success_rows = (success_contrast * success_mask).sum(-1) / success_mask.sum(-1).clamp_min(1)
        failure_rows = (failure_contrast * failure_mask).sum(-1) / failure_mask.sum(-1).clamp_min(1)
        success_rows = success_rows[success_mask.any(dim=-1)]
        failure_rows = failure_rows[failure_mask.any(dim=-1)]
    else:
        success_rows, failure_rows = success_contrast[success_mask], failure_contrast[failure_mask]
    if not success_rows.numel() or not failure_rows.numel():
        return {
            "sdc/quality_C_S": 0.0,
            "sdc/quality_C_F": 0.0,
            "sdc/quality_auc": 0.5,
        }
    pairwise = success_rows[:, None] - failure_rows[None, :]
    auc = (pairwise.gt(0).float() + 0.5 * pairwise.eq(0).float()).mean()
    return {
        "sdc/quality_C_S": float(success_rows.mean().cpu()),
        "sdc/quality_C_F": float(failure_rows.mean().cpu()),
        "sdc/quality_auc": float(auc.cpu()),
    }
