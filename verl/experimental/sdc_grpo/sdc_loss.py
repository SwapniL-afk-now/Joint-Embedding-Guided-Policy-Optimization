from __future__ import annotations

from typing import Optional

import torch

from verl.trainer.ppo.core_algos import agg_loss


def _global_count(count: torch.Tensor, dp_size: int) -> torch.Tensor:
    """Return an active-token count shared by the participating data-parallel ranks."""

    if dp_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        count = count.clone()
        torch.distributed.all_reduce(count)
    return count


def _sdc_aggregate_loss(
    token_loss: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_agg_mode: str,
    global_batch_info: Optional[dict],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate SDC over active failed-response tokens, not all response tokens."""

    info = global_batch_info or {}
    dp_size = int(info.get("dp_size", 1) or 1)
    if loss_agg_mode == "token-mean":
        local_count = mask.to(token_loss.dtype).sum()
        count = _global_count(local_count, dp_size).clamp_min(1.0)
        return (token_loss * mask).sum() / count * dp_size, count
    if loss_agg_mode == "seq-mean-token-mean":
        token_count = mask.to(token_loss.dtype).sum(dim=-1)
        seq_mask = token_count > 0
        seq_loss = (token_loss * mask).sum(dim=-1) / token_count.clamp_min(1.0)
        local_count = seq_mask.to(token_loss.dtype).sum()
        count = _global_count(local_count, dp_size).clamp_min(1.0)
        return (seq_loss * seq_mask).sum() / count * dp_size, count
    # Preserve supported actor aggregation modes, but do not pass GRPO's total
    # response-token denominator into the default SDC token expectation.
    return agg_loss(loss_mat=token_loss, loss_mask=mask, loss_agg_mode=loss_agg_mode, **info), mask.sum().clamp_min(1)


def contrast_quality_metrics(
    success_log_prob_under_success: torch.Tensor,
    success_log_prob_under_failure: torch.Tensor,
    failure_log_prob_under_success: torch.Tensor,
    failure_log_prob_under_failure: torch.Tensor,
    *,
    success_response_mask: Optional[torch.Tensor] = None,
    failure_response_mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Measure held-out success/failure separation.

    The first two tensors are the success and failure model log-probabilities
    on held-out successful responses. The last two are the same scores on
    held-out failed responses. ``C_S`` and ``C_F`` are response-token means of
    ``log(pi_S / pi_F)`` on the two outcome sets; AUC measures how often a
    held-out success receives the larger contrast score.
    """

    if success_log_prob_under_success.shape != success_log_prob_under_failure.shape:
        raise ValueError("Held-out success score tensors must have the same shape.")
    if failure_log_prob_under_success.shape != failure_log_prob_under_failure.shape:
        raise ValueError("Held-out failure score tensors must have the same shape.")
    success_mask = (
        torch.ones_like(success_log_prob_under_success, dtype=torch.bool)
        if success_response_mask is None
        else success_response_mask.to(device=success_log_prob_under_success.device, dtype=torch.bool)
    )
    failure_mask = (
        torch.ones_like(failure_log_prob_under_success, dtype=torch.bool)
        if failure_response_mask is None
        else failure_response_mask.to(device=failure_log_prob_under_success.device, dtype=torch.bool)
    )
    if success_mask.shape != success_log_prob_under_success.shape:
        raise ValueError("Held-out success response mask has the wrong shape.")
    if failure_mask.shape != failure_log_prob_under_success.shape:
        raise ValueError("Held-out failure response mask has the wrong shape.")

    success_contrast = success_log_prob_under_success.detach() - success_log_prob_under_failure.detach()
    failure_contrast = failure_log_prob_under_success.detach() - failure_log_prob_under_failure.detach()
    if success_contrast.ndim > 1:
        success_row_scores = (success_contrast * success_mask).sum(-1) / success_mask.sum(-1).clamp_min(1)
        failure_row_scores = (failure_contrast * failure_mask).sum(-1) / failure_mask.sum(-1).clamp_min(1)
        success_row_scores = success_row_scores[success_mask.any(dim=-1)]
        failure_row_scores = failure_row_scores[failure_mask.any(dim=-1)]
    else:
        success_row_scores = success_contrast[success_mask]
        failure_row_scores = failure_contrast[failure_mask]
    if not success_row_scores.numel() or not failure_row_scores.numel():
        return {"sdc/quality_C_S": 0.0, "sdc/quality_C_F": 0.0, "sdc/quality_auc": 0.5}
    pairwise = success_row_scores[:, None] - failure_row_scores[None, :]
    auc = (pairwise.gt(0).float() + 0.5 * pairwise.eq(0).float()).mean()
    return {
        "sdc/quality_C_S": float(success_row_scores.mean().cpu()),
        "sdc/quality_C_F": float(failure_row_scores.mean().cpu()),
        "sdc/quality_auc": float(auc.cpu()),
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
            "sdc/correct_response_token_nonzero_fraction": 0.0,
            "sdc/importance_weight_mean": 1.0,
            "sdc/log_ratio_clipped_fraction": 0.0,
            "sdc/importance_weight_clip_fraction": 0.0,
            "sdc/contrast_p10": 0.0,
            "sdc/contrast_p50": 0.0,
            "sdc/contrast_p90": 0.0,
            "sdc/contrast_p99": 0.0,
            "sdc/contrast_max": 0.0,
            "sdc/contrast_sequence_mean": 0.0,
        }

    old = old_log_prob.detach().to(device=log_prob.device, dtype=log_prob.dtype)
    success = success_log_prob.detach().to(device=log_prob.device, dtype=log_prob.dtype)
    failure = failure_log_prob.detach().to(device=log_prob.device, dtype=log_prob.dtype)
    active = failure_mask.to(device=log_prob.device, dtype=torch.bool).unsqueeze(-1)
    mask = response_mask.to(device=log_prob.device, dtype=torch.bool) & active

    if not use_importance_weight:
        raise ValueError("SDC requires importance weighting; disabling it removes the actor gradient.")

    raw_log_ratio = log_prob - old
    # These are optimization safeguards, not an exact importance-ratio
    # estimator. Outside either clamp's active interval, PyTorch's derivative
    # is zero, so saturated tokens receive no SDC actor gradient.
    log_ratio = raw_log_ratio.clamp(min=-20.0, max=20.0)
    log_ratio_clipped = raw_log_ratio.ne(log_ratio)
    raw_importance_weight = log_ratio.exp()
    importance_weight = raw_importance_weight
    weight_clipped = torch.zeros_like(importance_weight, dtype=torch.bool)
    if importance_weight_clip > 0:
        importance_weight = importance_weight.clamp(max=float(importance_weight_clip))
        weight_clipped = raw_importance_weight.ne(importance_weight)

    token_loss = importance_weight * (failure - success)
    aggregated_loss, valid = _sdc_aggregate_loss(
        token_loss,
        mask,
        loss_agg_mode=loss_agg_mode,
        global_batch_info=global_batch_info,
    )
    loss = float(beta) * aggregated_loss
    metric_scale = float((global_batch_info or {}).get("dp_size", 1) or 1)
    active_tokens = response_mask.to(log_prob.dtype).sum().clamp_min(1)
    contrast = success - failure  # log(pi_S / pi_F), detached references
    active_contrast = contrast[mask]
    if active_contrast.numel():
        quantiles = torch.quantile(
            active_contrast.float(),
            torch.tensor([0.10, 0.50, 0.90, 0.99], device=active_contrast.device),
        )
        contrast_sequence_mean = (contrast * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        contrast_sequence_mean = contrast_sequence_mean[mask.any(dim=-1)].mean()
        contrast_metrics = {
            "sdc/contrast_p10": float(quantiles[0].detach().cpu()),
            "sdc/contrast_p50": float(quantiles[1].detach().cpu()),
            "sdc/contrast_p90": float(quantiles[2].detach().cpu()),
            "sdc/contrast_p99": float(quantiles[3].detach().cpu()),
            "sdc/contrast_max": float(active_contrast.max().detach().cpu()),
            "sdc/contrast_sequence_mean": float(contrast_sequence_mean.detach().cpu()),
        }
    else:
        contrast_metrics = {
            key: 0.0
            for key in (
                "sdc/contrast_p10",
                "sdc/contrast_p50",
                "sdc/contrast_p90",
                "sdc/contrast_p99",
                "sdc/contrast_max",
                "sdc/contrast_sequence_mean",
            )
        }
    return loss, {
        "sdc/loss": float(loss.detach().cpu()),
        "sdc/active_token_fraction": float(mask.to(log_prob.dtype).sum().detach().cpu() / active_tokens.detach().cpu()),
        # This diagnostic is intentionally always zero: the SDC term never
        # substitutes a token-level correctness/negative-CE objective.
        "sdc/correct_response_token_nonzero_fraction": 0.0,
        "sdc/importance_weight_mean": float((importance_weight.detach() * mask).sum().cpu() * metric_scale / valid.cpu()),
        "sdc/log_ratio_clipped_fraction": float((log_ratio_clipped.to(log_prob.dtype) * mask).sum().cpu() * metric_scale / valid.cpu()),
        "sdc/importance_weight_clip_fraction": float((weight_clipped.to(log_prob.dtype) * mask).sum().cpu() * metric_scale / valid.cpu()),
        **contrast_metrics,
    }
