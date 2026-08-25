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


def _all_reduce_detached(value: torch.Tensor) -> torch.Tensor:
    """Return a globally summed diagnostic tensor without autograd edges."""

    value = value.detach().clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    return value


def _all_reduce_max_detached(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
    return value


def _all_gather_detached_values(value: torch.Tensor) -> torch.Tensor:
    """Gather variable-length diagnostic values for global quantiles."""

    value = value.detach()
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return value
    gathered: list[torch.Tensor | None] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, value.cpu())
    values = [item for item in gathered if item is not None and item.numel()]
    if not values:
        return value.new_empty(0)
    return torch.cat(values).to(device=value.device)


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
    if not models_ready or beta == 0.0:
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
    # The engine computes this once per forward/backward batch, before it
    # creates micro-batches. Reusing it here removes one all-reduce per
    # micro-batch while preserving the exact global token denominator.
    precomputed_count = info.get("sdc_active_token_count")
    if precomputed_count is not None:
        count = torch.as_tensor(precomputed_count, device=log_prob.device, dtype=log_prob.dtype)
    elif dp_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(count)
    count = count.clamp_min(1.0)
    loss = float(beta) * (token_loss * mask).sum() / count * dp_size

    # Metrics are local diagnostics by default. The engine aggregates the
    # returned values once per training batch, so doing the historical
    # all-reduce/all-gather sequence here would add several collectives to
    # every micro-batch. Exact global reduction can still be requested for
    # callers that need it outside the normal actor path.
    active_contrast = (success - failure)[mask]
    valid_sequences = mask.any(dim=-1)
    sequence_values = ((success - failure) * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
    local_metric_sums = torch.stack(
        [
            (token_loss.detach() * mask).sum(),
            mask.to(log_prob.dtype).sum(),
            response_mask.to(log_prob.dtype).sum(),
            (ratio.detach() * mask).sum(),
            (ratio_clipped.to(log_prob.dtype) * mask).sum(),
            (weight_clipped.to(log_prob.dtype) * mask).sum(),
            sequence_values[valid_sequences].sum(),
            valid_sequences.to(log_prob.dtype).sum(),
        ]
    )
    if bool(info.get("reduce_diagnostics", False)):
        (
            global_token_loss_sum,
            global_active_count,
            global_response_count,
            global_importance_sum,
            global_log_ratio_clipped_count,
            global_importance_clip_count,
            global_sequence_sum,
            global_sequence_count,
        ) = _all_reduce_detached(local_metric_sums)
    else:
        (
            global_token_loss_sum,
            global_active_count,
            global_response_count,
            global_importance_sum,
            global_log_ratio_clipped_count,
            global_importance_clip_count,
            global_sequence_sum,
            global_sequence_count,
        ) = local_metric_sums
    global_has_active = bool(global_active_count.item() > 0)
    if not global_has_active:
        return zero, _zero_metrics()
    global_active_count = global_active_count.clamp_min(1.0)
    global_response_count = global_response_count.clamp_min(1.0)
    global_sequence_count = global_sequence_count.clamp_min(1.0)
    local_contrast_max = (
        active_contrast.max()
        if active_contrast.numel()
        else torch.tensor(float("-inf"), device=log_prob.device)
    )
    global_contrast_max = (
        _all_reduce_max_detached(local_contrast_max)
        if bool(info.get("reduce_diagnostics", False))
        else local_contrast_max.detach()
    )
    global_active_contrast = (
        _all_gather_detached_values(active_contrast)
        if bool(info.get("reduce_diagnostics", False))
        else active_contrast.detach()
    )
    metrics = {
        "sdc/loss": float((float(beta) * global_token_loss_sum / global_active_count).cpu()),
        "sdc/active_token_fraction": float((global_active_count / global_response_count).cpu()),
        "sdc/importance_weight_mean": float((global_importance_sum / global_active_count).cpu()),
        "sdc/log_ratio_clipped_fraction": float(
            (global_log_ratio_clipped_count / global_active_count).cpu()
        ),
        "sdc/importance_weight_clip_fraction": float(
            (global_importance_clip_count / global_active_count).cpu()
        ),
    }
    if global_has_active:
        q = torch.quantile(
            global_active_contrast.float(),
            torch.tensor([0.10, 0.50, 0.90], device=global_active_contrast.device),
        )
        sequence_mean = global_sequence_sum / global_sequence_count
        metrics.update(
            {
                "sdc/contrast_p10": float(q[0].detach().cpu()),
                "sdc/contrast_p50": float(q[1].detach().cpu()),
                "sdc/contrast_p90": float(q[2].detach().cpu()),
                "sdc/contrast_max": float(global_contrast_max.cpu()),
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
