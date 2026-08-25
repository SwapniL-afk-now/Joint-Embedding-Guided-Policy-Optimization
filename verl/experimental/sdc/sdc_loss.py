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
        "sdc/contrast_center": 0.0,
        "sdc/contrast_rms": 0.0,
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
    use_centering: bool = False,
    use_scale_normalize: bool = False,
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

    info = global_batch_info or {}
    dp_size = int(info.get("dp_size", 1) or 1)
    reduce_diagnostics = bool(info.get("reduce_diagnostics", False))
    # The engine computes this once per forward/backward batch, before it
    # creates micro-batches. Reusing it here removes one all-reduce per
    # micro-batch while preserving the exact global token denominator.
    precomputed_count = info.get("sdc_active_token_count")
    if precomputed_count is not None and float(precomputed_count) == 0.0:
        # The global active token count is known host-side to be zero: no rank
        # can have active tokens, so the historical diagnostics path would also
        # return the zero metrics. Exit before any tensor work.
        return zero, _zero_metrics()

    old = old_log_prob.detach().to(device=log_prob.device, dtype=log_prob.dtype)
    success = success_log_prob.detach().to(device=log_prob.device, dtype=log_prob.dtype)
    failure = failure_log_prob.detach().to(device=log_prob.device, dtype=log_prob.dtype)
    response_mask_bool = response_mask.to(device=log_prob.device, dtype=torch.bool)
    # failure_mask is per-sequence [bsz]; every tensor below is restricted to
    # the failed rows so the hot path never materializes full-batch temporaries.
    rows = failure_mask.to(device=log_prob.device, dtype=torch.bool).reshape(-1).nonzero(as_tuple=True)[0]
    resp = response_mask_bool[rows]
    log_prob_a = log_prob[rows]
    old_a = old[rows]
    success_a = success[rows]
    failure_a = failure[rows]
    raw_log_ratio_a = log_prob_a - old_a
    safe_log_ratio_a = raw_log_ratio_a.clamp(min=-20.0, max=20.0)
    ratio_a = safe_log_ratio_a.exp()
    ratio_clipped_a = raw_log_ratio_a.ne(safe_log_ratio_a)
    weight_clipped_a = torch.zeros_like(ratio_a, dtype=torch.bool)
    if importance_weight_clip > 0:
        clipped_ratio_a = ratio_a.clamp(max=float(importance_weight_clip))
        weight_clipped_a = ratio_a.ne(clipped_ratio_a)
        ratio_a = clipped_ratio_a

    # Locked algebra: + beta E[pi_theta/pi_old * log(pi_F/pi_S)].
    contrast = failure_a - success_a
    contrast_center_t = torch.zeros((), device=log_prob.device, dtype=torch.float32)
    contrast_rms_t = torch.zeros((), device=log_prob.device, dtype=torch.float32)
    if use_centering or use_scale_normalize:
        # Opt-in normalization over the SAME active mask used by the loss sum.
        # Statistics are detached: they rescale the gradient, never feed it.
        norm_mask = resp.to(log_prob.dtype)
        norm_count = norm_mask.sum().clamp_min(1.0)
        if use_centering:
            # Strip the common-mode offset of log(pi_F/pi_S); sign structure
            # is preserved and the uniform failed-token-mass gradient that
            # overlaps the KL-to-old term is removed.
            contrast_center_t = (contrast.detach() * norm_mask).sum() / norm_count
            contrast = contrast - contrast_center_t
        if use_scale_normalize:
            # Applied after centering so beta stays scale-stable as the
            # teacher gap sharpens.
            contrast_rms_t = ((contrast.detach() ** 2) * norm_mask).sum().div(norm_count).sqrt().clamp_min(1e-8)
            contrast = contrast / contrast_rms_t
    token_loss_a = ratio_a * contrast
    local_count = resp.to(log_prob.dtype).sum()
    if precomputed_count is not None:
        count = torch.as_tensor(precomputed_count, device=log_prob.device, dtype=log_prob.dtype)
    else:
        count = local_count.detach().clone()
        if dp_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(count)
    loss = float(beta) * (token_loss_a * resp).sum() / count.clamp_min(1.0) * dp_size

    # Metrics are local diagnostics by default. The engine aggregates the
    # returned values once per training batch, so doing the historical
    # all-reduce/all-gather sequence here would add several collectives to
    # every micro-batch. Exact global reduction can still be requested for
    # callers that need it outside the normal actor path.
    #
    # All scalar diagnostics are stacked into one detached fp32 tensor and
    # synced with a single .cpu() instead of ~10 blocking host transfers.
    valid_rows = resp.any(dim=-1)
    contrast_full = success_a - failure_a
    sequence_values = (contrast_full * resp).sum(dim=-1) / resp.sum(dim=-1).clamp_min(1)
    active_contrast = contrast_full[resp]
    local_metric_sums = torch.stack(
        [
            (token_loss_a.detach() * resp).to(torch.float32).sum(),
            local_count.to(torch.float32),
            response_mask_bool.to(torch.float32).sum(),
            (ratio_a.detach() * resp).to(torch.float32).sum(),
            ratio_clipped_a.to(torch.float32).sum(),
            weight_clipped_a.to(torch.float32).sum(),
            sequence_values[valid_rows].to(torch.float32).sum(),
            valid_rows.to(torch.float32).sum(),
        ]
    ).detach()
    metric_sums = _all_reduce_detached(local_metric_sums) if reduce_diagnostics else local_metric_sums
    (
        global_token_loss_sum,
        global_active_count,
        global_response_count,
        global_importance_sum,
        global_log_ratio_clipped_count,
        global_importance_clip_count,
        global_sequence_sum,
        global_sequence_count,
    ) = metric_sums.cpu().tolist()
    global_has_active = global_active_count > 0
    if not global_has_active:
        return zero, _zero_metrics()
    global_active_count = max(global_active_count, 1.0)
    global_response_count = max(global_response_count, 1.0)
    global_sequence_count = max(global_sequence_count, 1.0)
    local_contrast_max = (
        active_contrast.max()
        if active_contrast.numel()
        else torch.tensor(float("-inf"), device=log_prob.device, dtype=torch.float32)
    )
    global_contrast_max = (
        _all_reduce_max_detached(local_contrast_max) if reduce_diagnostics else local_contrast_max.detach()
    ).to(torch.float32)
    global_active_contrast = (
        _all_gather_detached_values(active_contrast) if reduce_diagnostics else active_contrast.detach()
    )
    q = torch.quantile(
        global_active_contrast.float(),
        torch.tensor([0.10, 0.50, 0.90], device=global_active_contrast.device),
    )
    sequence_mean_t = metric_sums[6] / metric_sums[7].clamp_min(1.0)
    # One further host sync batches the quantiles, the max, and the sequence mean.
    tail = torch.stack(
        [
            q[0],
            q[1],
            q[2],
            global_contrast_max.reshape(()),
            sequence_mean_t.to(torch.float32).reshape(()),
            contrast_center_t.to(torch.float32).reshape(()),
            contrast_rms_t.to(torch.float32).reshape(()),
        ]
    ).detach()
    (
        contrast_p10,
        contrast_p50,
        contrast_p90,
        contrast_max,
        contrast_sequence_mean,
        contrast_center,
        contrast_rms,
    ) = tail.cpu().tolist()
    metrics = {
        "sdc/loss": float(beta) * global_token_loss_sum / global_active_count,
        "sdc/active_token_fraction": global_active_count / global_response_count,
        "sdc/importance_weight_mean": global_importance_sum / global_active_count,
        "sdc/log_ratio_clipped_fraction": global_log_ratio_clipped_count / global_active_count,
        "sdc/importance_weight_clip_fraction": global_importance_clip_count / global_active_count,
        "sdc/contrast_p10": contrast_p10,
        "sdc/contrast_p50": contrast_p50,
        "sdc/contrast_p90": contrast_p90,
        "sdc/contrast_max": contrast_max,
        "sdc/contrast_sequence_mean": contrast_sequence_mean,
        "sdc/contrast_center": contrast_center,
        "sdc/contrast_rms": contrast_rms,
    }
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
