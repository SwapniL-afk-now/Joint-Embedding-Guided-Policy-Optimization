# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from tensordict import TensorDict

from verl.experimental.sharpening_grpo.loss import compute_sharpening_grpo_loss
from verl.experimental.tafr_grpo.tafr_loss import (
    compute_regularization_policy_loss,
    compute_tafr_grpo_auxiliary_loss,
    replay_gate,
    response_length_normalized_mean,
)
from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding
from verl.workers.utils.wasserstein_guidance import _as_list, compute_wasserstein_guidance_loss


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def _tafr_regularization_metrics(
    *,
    loss: torch.Tensor,
    regularization_loss: torch.Tensor,
    regularization_advantage: torch.Tensor,
    stability_signal: torch.Tensor,
    failure_signal: torch.Tensor,
    failure_component: torch.Tensor,
    raw_advantage: torch.Tensor,
    scaled_advantage: torch.Tensor,
    grpo_advantage: torch.Tensor,
    normalization_scale: torch.Tensor,
    fail_stats: dict[str, torch.Tensor],
    group_reward_mean: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float,
    failure_model_ready: bool,
    update_count: int,
    advantage_clip: float,
) -> dict[str, float]:
    """Metrics for the paper anchor-plus-failure regularization advantage."""
    mask = response_mask.to(dtype=regularization_advantage.dtype)
    rmean = group_reward_mean.detach().float().clamp(0.0, 1.0).reshape(-1)
    failure_rows = (rmean < 1.0).float() * float(failure_model_ready)
    failure_mask = (failure_rows.unsqueeze(-1) * mask).bool()
    token_mask = mask.bool()
    valid_tokens = mask.sum().clamp_min(1.0)

    def _masked_stats(values: torch.Tensor, selected: torch.Tensor) -> tuple[float, float, float]:
        count = selected.sum()
        if count.item() == 0:
            return 0.0, 0.0, 0.0
        selected_f = selected.to(values.dtype)
        mean = (values * selected_f).sum() / count
        var = ((values - mean).square() * selected_f).sum() / count
        positive = ((values > 0).to(values.dtype) * selected_f).sum() / count
        return float(mean), float(var.clamp_min(0.0).sqrt()), float(positive)

    def _masked_rms(values: torch.Tensor, selected: torch.Tensor) -> float:
        count = selected.sum()
        if count.item() == 0:
            return 0.0
        return float(((values.square() * selected.to(values.dtype)).sum() / count).sqrt())

    stability_mean, stability_std, _ = _masked_stats(stability_signal, token_mask)
    failure_mean, failure_std, failure_pos = _masked_stats(failure_signal, failure_mask)
    adv_mean, adv_std, adv_pos = _masked_stats(regularization_advantage, token_mask)
    raw_rms = _masked_rms(raw_advantage, token_mask)
    reg_rms = _masked_rms(regularization_advantage, token_mask)
    grpo_rms = _masked_rms(grpo_advantage, token_mask)
    weighted_rms = abs(beta) * reg_rms
    rms_ratio = weighted_rms / grpo_rms if grpo_rms > 1.0e-8 else 0.0
    scale = float(normalization_scale.reshape(-1)[0]) if normalization_scale.numel() else 0.0
    ratio = fail_stats["ppo_ratio"]
    ratio_mean = float((ratio * mask).sum() / valid_tokens)
    clip_fraction = float(((scaled_advantage.abs() > advantage_clip).to(mask.dtype) * mask).sum() / valid_tokens)

    def _within_between_std(values: torch.Tensor) -> tuple[float, float]:
        """Split token-level variation into within-sequence and between-sequence parts.

        A_reg is per-token by construction, but if it were near-constant along each
        rollout it would be per-token in shape and per-sequence in effect -- i.e. no
        better than GRPO's broadcast advantage. within >> between means it really is
        doing token-level credit assignment.
        """
        counts = mask.sum(dim=-1)
        rows = counts > 0
        if not bool(rows.any()):
            return 0.0, 0.0
        counts = counts[rows].clamp_min(1.0)
        vals = values[rows]
        row_mask = mask[rows]
        row_mean = (vals * row_mask).sum(dim=-1) / counts
        within_var = ((vals - row_mean.unsqueeze(-1)).square() * row_mask).sum(dim=-1) / counts
        within = float(within_var.clamp_min(0.0).sqrt().mean())
        between = float(row_mean.std(unbiased=False)) if row_mean.numel() > 1 else 0.0
        return within, between

    adv_within_std, adv_between_std = _within_between_std(regularization_advantage)
    failure_within_std, failure_between_std = _within_between_std(failure_signal)

    def _quantiles(values: torch.Tensor) -> dict[str, float]:
        """Distribution of the per-token advantage actually delivered.

        A std alone hides whether the signal is a few saturated spikes or a broad
        spread -- which is the difference between shaping most tokens and shouting
        at a handful.
        """
        sel = values[token_mask]
        if sel.numel() == 0:
            return dict.fromkeys(("p50", "p90", "p99", "max"), 0.0)
        a = sel.abs().float()
        q = torch.quantile(a, torch.tensor([0.5, 0.9, 0.99], device=a.device)) if a.numel() > 1 else a.repeat(3)
        return {"p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]), "max": float(a.max())}

    adv_q = _quantiles(regularization_advantage)

    def _regime_mean(regime: torch.Tensor) -> float:
        selected = regime.to(mask.dtype).unsqueeze(-1).bool() & token_mask
        return _masked_stats(regularization_advantage, selected)[0]

    all_correct = (rmean == 1.0).to(mask.dtype).unsqueeze(-1).bool() & token_mask
    all_correct_reg_max = float((regularization_advantage.abs() * all_correct).max()) if all_correct.any() else 0.0
    all_correct_failure_max = float((failure_component.abs() * all_correct).max()) if all_correct.any() else 0.0

    values = {
        "tafr_adv/advantage_abs_p50": adv_q["p50"],
        "tafr_adv/advantage_abs_p90": adv_q["p90"],
        "tafr_adv/advantage_abs_p99": adv_q["p99"],
        "tafr_adv/advantage_abs_max_all": adv_q["max"],
        "tafr_adv/advantage_within_seq_std": adv_within_std,
        "tafr_adv/advantage_between_seq_std": adv_between_std,
        "tafr_adv/failure_signal_within_seq_std": failure_within_std,
        "tafr_adv/failure_signal_between_seq_std": failure_between_std,
        "tafr_adv/regularization_loss": float(regularization_loss.detach().cpu()),
        "tafr_adv/weighted_regularization_loss": float((beta * regularization_loss).detach().cpu()),
        # Compatibility aliases for existing dashboards.
        "tafr_adv/failure_loss": float(regularization_loss.detach().cpu()),
        "tafr_adv/weighted_failure_loss": float((beta * regularization_loss).detach().cpu()),
        "tafr_adv/total_policy_loss": float(loss.detach().cpu()),
        "tafr_adv/group_difficulty_mean": float((1.0 - rmean).mean().cpu()),
        "tafr_adv/active_group_fraction": float(failure_rows.mean().cpu()),
        "tafr_adv/scored_token_fraction": float(failure_mask.to(mask.dtype).sum() / valid_tokens),
        "tafr_adv/stability_signal_mean": stability_mean,
        "tafr_adv/stability_signal_std": stability_std,
        "tafr_adv/failure_signal_mean": failure_mean,
        "tafr_adv/failure_signal_std": failure_std,
        "tafr_adv/failure_signal_positive_fraction": failure_pos,
        "tafr_adv/advantage_mean": adv_mean,
        "tafr_adv/advantage_std": adv_std,
        "tafr_adv/advantage_positive_fraction": adv_pos,
        "tafr_adv/raw_regularization_advantage_rms": raw_rms,
        "tafr_adv/regularization_advantage_rms": reg_rms,
        "tafr_adv/grpo_advantage_rms": grpo_rms,
        "tafr_adv/weighted_regularization_advantage_rms": weighted_rms,
        "tafr_adv/regularization_to_grpo_rms_ratio": rms_ratio,
        "tafr_adv/normalization_scale": scale,
        "tafr_adv/advantage_clip_fraction": clip_fraction,
        "tafr_adv/ppo_ratio_mean": ratio_mean,
        "tafr_adv/ppo_ratio_clip_fraction": float(fail_stats["ppo_ratio_clip_fraction"].detach().cpu()),
        "tafr_adv/failure_model_ready": 1.0 if failure_model_ready else 0.0,
        "tafr_adv/failure_model_update_count": float(update_count),
        "tafr_adv/all_wrong/advantage_mean": _regime_mean(rmean == 0.0),
        "tafr_adv/mixed/advantage_mean": _regime_mean((rmean > 0.0) & (rmean < 1.0)),
        "tafr_adv/all_correct/advantage_abs_max": all_correct_reg_max,
        "tafr_adv/all_correct/regularization_advantage_abs_max": all_correct_reg_max,
        "tafr_adv/all_correct/failure_component_abs_max": all_correct_failure_max,
    }
    return values


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None, exploration=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    ema_log_prob = None
    if "ema_log_probs" in model_output:
        ema_log_prob = no_padding_2_padding(model_output["ema_log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    tafr_config = tu.get_non_tensor_data(data=data, key="custom_tafr_grpo", default=None)
    tafr_group_ids = _as_list(data.get("uid", None)) if tafr_config and tafr_config.get("enable", False) else []

    gpi_config = tu.get_non_tensor_data(data=data, key="custom_gpi_ce", default=None)
    gpi_enabled = bool(gpi_config and gpi_config.get("enable", False))

    sharpening_config = tu.get_non_tensor_data(data=data, key="custom_sharpening_grpo", default=None)
    sharpening_enabled = bool(sharpening_config and sharpening_config.get("enable", False))

    abstract_config = tu.get_non_tensor_data(data=data, key="custom_abstract_rl", default=None)
    abstract_enabled = bool(abstract_config and abstract_config.get("enable", False))

    wasserstein_guidance = getattr(config, "wasserstein_guidance", {})
    wg_enabled = bool(wasserstein_guidance.get("enable", False))
    wg_group_ids = _as_list(data.get("uid", None)) if wg_enabled else []

    # select fields and convert to padded tensor
    # old_log_probs is optional: when the trainer fuses the old-log-prob forward
    # into the update (ppo_epochs==1 and a single optimizer step per batch), it is
    # absent and we default it to log_prob.detach() below -> ratio == 1 exactly.
    has_old_log_probs = "old_log_probs" in data
    fields = ["response_mask", "advantages"]
    if has_old_log_probs:
        fields.append("old_log_probs")
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    if wg_enabled:
        for field in ("rm_scores", "token_level_rewards"):
            if field in data:
                fields.append(field)
    tafr_enabled = bool(tafr_config and tafr_config.get("enable", False))
    if tafr_enabled:
        for field in (
            "tafr_group_reward_mean",
            "tafr_failure_log_prob",
            "tafr_anchor_log_probs",
            "tafr_replay_log_probs",
            "tafr_regularization_advantage",
            "tafr_stability_signal",
            "tafr_failure_signal",
            "tafr_failure_component",
            "tafr_raw_regularization_advantage",
            "tafr_scaled_regularization_advantage",
            "tafr_regularization_scale",
        ):
            if field in data:
                fields.append(field)
    if sharpening_enabled:
        # The Sequence Sharpening loss needs ref_log_prob on top of the
        # standard fields. Make sure it is included in the padded payload.
        if "ref_log_prob" in data and "ref_log_prob" not in fields:
            fields.append("ref_log_prob")
    if gpi_enabled:
        for field in ("gpi_q_target", "gpi_group_id", "gpi_scalar_reward", "gpi_is_offline"):
            if field in data:
                fields.append(field)
    if abstract_enabled:
        for field in (
            "abstract_mask",
            "abstract_reasoning_mask",
            "abstract_prior_log_probs",
            "abstract_kl_weight",
        ):
            if field in data:
                fields.append(field)
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    policy_response_mask = response_mask
    # compute policy loss
    # When old_log_probs was not materialized (fused path), old == new so the PPO
    # ratio is exactly 1.0 -- valid only for a single optimizer step (ppo_epochs==1
    # and ppo_mini_batch_size >= train_batch_size), which the trainer enforces.
    old_log_prob = data["old_log_probs"] if has_old_log_probs else log_prob.detach()
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    if gpi_enabled:
        # GPI-CE is a finite-group cross entropy, not a policy-gradient surrogate:
        # it needs the candidate group and q*, neither of which the registered
        # policy-loss interface carries, and it uses no ratio, clip or advantage.
        from verl.experimental.gpi_ce import compute_gpi_ce_loss

        pg_loss, pg_metrics = compute_gpi_ce_loss(
            log_prob=log_prob,
            response_mask=policy_response_mask,
            group_size=int(gpi_config["group_size"]),
            groups_per_rank=int(gpi_config["groups_per_rank"]),
            # Absent on the fused path -> q* is built here from the detached scores,
            # which saves the whole separate old-policy forward.
            q_target=data.get("gpi_q_target", None),
            group_ids=data.get("gpi_group_id", None),
            rewards=data.get("gpi_scalar_reward", None),
            temperature=float(gpi_config["temperature"]),
            is_offline=data.get("gpi_is_offline", None),
        )
    else:
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=policy_response_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
        )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    ppo_loss_coef = float(getattr(config, "ppo_loss_coef", 1.0))
    policy_loss = ppo_loss_coef * pg_loss
    metrics["actor/ppo_loss_coef"] = ppo_loss_coef

    if sharpening_enabled:
        if "ref_log_prob" not in data:
            raise ValueError(
                "Sequence Sharpening requires 'ref_log_prob' in the actor batch. "
                "Ensure a reference policy is configured (use_reference_policy=true)."
            )
        sharpening_use_grpo_reward = bool(sharpening_config.get("use_grpo_reward", True))
        sharpening_group_size = int(sharpening_config.get("group_size", 8))
        sharpening_clip_ratio = float(sharpening_config.get("clip_ratio", config.clip_ratio))
        sharpening_gamma = float(sharpening_config.get("gamma", 0.1))
        sharpening_alpha = float(sharpening_config.get("alpha", 1.0))
        sharpening_beta = float(sharpening_config.get("beta", 0.1))

        sharpen_output = compute_sharpening_grpo_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=policy_response_mask,
            ref_log_prob=data["ref_log_prob"],
            group_size=sharpening_group_size,
            clip_ratio=sharpening_clip_ratio,
            use_grpo_reward=sharpening_use_grpo_reward,
        )

        # Replace the GRPO contribution with the toggleable sharpening term.
        # When use_grpo_reward=False, the GRPO contribution is exactly 0 but the
        # term is still computed and reported in metrics for diagnostics.
        grpo_scale = ppo_loss_coef if sharpening_use_grpo_reward else 0.0
        policy_loss = (
            grpo_scale * sharpen_output.grpo_term
            + sharpening_gamma * sharpening_alpha * sharpen_output.seq_term
            + sharpening_beta * sharpen_output.kl_term
        )

        sharpen_metrics = Metric.from_dict(sharpen_output.metrics, aggregation=AggregationType.MEAN)
        metrics.update(sharpen_metrics)
        metrics["sharpen/grpo_term"] = Metric(value=sharpen_output.grpo_term, aggregation=metric_aggregation)
        metrics["sharpen/seq_term"] = Metric(value=sharpen_output.seq_term, aggregation=metric_aggregation)
        metrics["sharpen/kl_term"] = Metric(value=sharpen_output.kl_term, aggregation=metric_aggregation)
        metrics["sharpen/gamma"] = Metric(value=sharpening_gamma, aggregation=AggregationType.MAX)
        metrics["sharpen/alpha"] = Metric(value=sharpening_alpha, aggregation=AggregationType.MAX)
        metrics["sharpen/beta"] = Metric(value=sharpening_beta, aggregation=AggregationType.MAX)

    if tafr_enabled:
        if "tafr_group_reward_mean" not in data:
            raise ValueError("TAFR-GRPO batch is missing required field: tafr_group_reward_mean")
        loss_version = str(tafr_config.get("loss_version", "failure_token_adv"))
        if loss_version == "failure_token_adv":
            required = (
                "tafr_regularization_advantage",
                "tafr_stability_signal",
                "tafr_failure_signal",
                "tafr_failure_component",
                "tafr_raw_regularization_advantage",
                "tafr_scaled_regularization_advantage",
                "tafr_regularization_scale",
            )
            missing = [field for field in required if field not in data]
            if missing:
                raise ValueError(f"TAFR paper advantage is missing required fields: {missing}")
            failure_model_ready = bool(tafr_config.get("tafr_failure_model_ready", False))
            update_count = int(tafr_config.get("tafr_failure_model_update_count", 0))
            advantage_clip = float(tafr_config.get("failure_advantage_clip", 5.0))
            clip_ratio_raw = tafr_config.get("failure_clip_ratio", None)
            regularization_clip_ratio = float(clip_ratio_raw) if clip_ratio_raw is not None else config.clip_ratio
            beta = float(tafr_config.get("beta", 0.5))
            regularization_advantage = data["tafr_regularization_advantage"]
            regularization_loss, reg_stats = compute_regularization_policy_loss(
                current_log_prob=log_prob,
                old_log_prob=old_log_prob,
                regularization_advantage=regularization_advantage,
                response_mask=response_mask,
                clip_ratio=regularization_clip_ratio,
                loss_agg_mode=loss_agg_mode,
                global_batch_info=config.global_batch_info,
            )
            combined_advantages = advantages + beta * regularization_advantage
            combined_pg_loss, _ = policy_loss_fn(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=combined_advantages,
                response_mask=policy_response_mask,
                loss_agg_mode=loss_agg_mode,
                config=config,
                rollout_is_weights=rollout_is_weights,
            )
            policy_loss = ppo_loss_coef * combined_pg_loss
            tafr_metrics = _tafr_regularization_metrics(
                loss=policy_loss,
                regularization_loss=regularization_loss,
                regularization_advantage=regularization_advantage,
                stability_signal=data["tafr_stability_signal"],
                failure_signal=data["tafr_failure_signal"],
                failure_component=data["tafr_failure_component"],
                raw_advantage=data["tafr_raw_regularization_advantage"],
                scaled_advantage=data["tafr_scaled_regularization_advantage"],
                grpo_advantage=advantages,
                normalization_scale=data["tafr_regularization_scale"],
                fail_stats=reg_stats,
                group_reward_mean=data["tafr_group_reward_mean"],
                response_mask=response_mask,
                beta=beta,
                failure_model_ready=failure_model_ready,
                update_count=update_count,
                advantage_clip=advantage_clip,
            )
            tafr_metrics = Metric.from_dict(tafr_metrics, aggregation=AggregationType.MEAN)
            metrics.update(tafr_metrics)
            metrics["tafr_adv/loss_grpo"] = Metric(value=pg_loss, aggregation=metric_aggregation)
            metrics["tafr_adv/combined_policy_loss"] = Metric(value=combined_pg_loss, aggregation=metric_aggregation)
            metrics["tafr_adv/loss_delta_vs_grpo"] = Metric(
                value=combined_pg_loss - pg_loss, aggregation=metric_aggregation
            )
        else:
            tafr_output = compute_tafr_grpo_auxiliary_loss(
                log_prob=log_prob,
                response_mask=response_mask,
                group_reward_mean=data["tafr_group_reward_mean"],
                beta=float(tafr_config.get("beta", 0.0)),
                beta_anchor=tafr_config.get("beta_anchor", None),
                beta_replay=tafr_config.get("beta_replay", None),
                variant=str(tafr_config.get("variant", "full")),
                anchor_log_prob=data.get("tafr_anchor_log_probs", None),
                replay_log_prob=data.get("tafr_replay_log_probs", None),
                group_ids=tafr_group_ids,
                gate_mode=str(tafr_config.get("gate_mode", "difficulty")),
                anchor_kl_direction=str(tafr_config.get("anchor_kl_direction", "forward")),
                old_log_prob=old_log_prob,
                use_importance_weight=bool(tafr_config.get("use_importance_weight", True)),
                importance_weight_clip=float(tafr_config.get("importance_weight_clip", 10.0)),
            )
            if bool(tafr_config.get("diagnostic_only", False)):
                # Baseline rows: the anchor and failure models were built and their
                # KLs measured above, but nothing may reach the actor. Multiplying by
                # zero (rather than skipping) keeps the term in the autograd graph, so
                # any accidental gradient path shows up as a non-zero grad rather than
                # silently vanishing.
                policy_loss = policy_loss + 0.0 * tafr_output.loss
                metrics["tafr_grpo/diagnostic_only"] = Metric(
                    value=torch.ones_like(pg_loss), aggregation=metric_aggregation
                )
            elif bool(tafr_config.get("negative_ce", False)):
                # Table 7 control: same failed replay buffer, but the learned failure
                # density is replaced by uniform negative cross-entropy on the failed
                # rollouts. Anchor term is untouched, so the only difference from full
                # FEPO is *what* provides the repulsion direction.
                gate = replay_gate(
                    data["tafr_group_reward_mean"].to(log_prob.dtype).clamp(0.0, 1.0),
                    str(tafr_config.get("gate_mode", "difficulty")),
                )
                seq_nll = response_length_normalized_mean(log_prob, response_mask)
                neg_ce = (gate * seq_nll).mean()
                coef = float(tafr_config.get("negative_ce_coef", 1.0))
                b_r = tafr_config.get("beta_replay", None)
                b_r = float(tafr_config.get("beta", 0.0)) if b_r is None else float(b_r)
                # tafr_output.loss carries b_a*anchor - b_r*replay; drop the replay half
                # by re-adding it, then subtract the negative-CE term in its place.
                # replay_loss is None only for variant=anchor_only, where there is no
                # replay half to remove.
                replay_half = tafr_output.replay_loss
                if replay_half is None:
                    replay_half = torch.zeros_like(tafr_output.loss)
                policy_loss += tafr_output.loss + b_r * replay_half + coef * b_r * neg_ce
                metrics["tafr_grpo/negative_ce"] = Metric(value=neg_ce, aggregation=metric_aggregation)
            else:
                policy_loss += tafr_output.loss
            tafr_metrics = Metric.from_dict(tafr_output.metrics, aggregation=AggregationType.MEAN)
            metrics.update(tafr_metrics)
            metrics["tafr_grpo/loss_grpo"] = Metric(value=pg_loss, aggregation=metric_aggregation)
            # Share of the update attributable to TAFR rather than the policy gradient.
            # Log the two magnitudes separately and divide post-hoc: mean(a/b) != mean(a)/mean(b),
            # and a single microbatch with pg_loss ~ 0 sent the averaged ratio to 1463 on the
            # beta=1.0 arm while grad_norm stayed at a normal 0.18. Plot
            # tafr_advantage_abs / loss_grpo_abs in the UI instead.
            metrics["tafr_grpo/loss_grpo_abs"] = Metric(value=pg_loss.detach().abs(), aggregation=metric_aggregation)

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if wg_enabled:
        rewards = None
        if "rm_scores" in data:
            rewards = data["rm_scores"].sum(dim=-1)
        elif "token_level_rewards" in data:
            rewards = data["token_level_rewards"].sum(dim=-1)
        wg_stats = compute_wasserstein_guidance_loss(
            log_prob=log_prob,
            response_mask=response_mask,
            rewards=rewards,
            group_ids=wg_group_ids,
            alpha_transport=float(wasserstein_guidance.get("alpha_transport", 0.2)),
        )
        lambda_wg = float(wasserstein_guidance.get("lambda_wg", 0.01))
        policy_loss += lambda_wg * wg_stats.loss
        metrics["wasserstein_guidance/loss"] = Metric(value=wg_stats.loss, aggregation=metric_aggregation)
        metrics["wasserstein_guidance/lambda_wg"] = lambda_wg
        metrics["wasserstein_guidance/active_groups"] = wg_stats.active_groups
        metrics["wasserstein_guidance/mixed_group_fraction"] = wg_stats.mixed_group_fraction
        metrics["wasserstein_guidance/mean_wrong_mass"] = wg_stats.mean_wrong_mass

    # add the abstraction compression KL (correct + valid abstraction tokens only)
    if abstract_enabled and "abstract_mask" in data:
        from verl.experimental.abstract_rl.loss import (
            compute_abstract_sft_loss,
            compute_compression_loss,
            compute_span_diagnostics,
        )

        abstract_mask = data["abstract_mask"].to(bool) & response_mask
        reasoning_mask = data["abstract_reasoning_mask"].to(bool) & response_mask
        sft_warmup = bool(abstract_config.get("sft_warmup", False))
        if sft_warmup:
            # Format warm-up: imitate the grafted abstractions outright. The policy
            # loss is dropped (its ratio is 1 by construction on grafted rows) and
            # the compression KL is off until the format exists.
            sft_loss, sft_metrics = compute_abstract_sft_loss(
                log_prob=log_prob, abstract_mask=abstract_mask, row_weight=data["abstract_kl_weight"]
            )
            policy_loss = sft_loss
            metrics.update(
                Metric.from_dict(
                    {f"abstract_rl/{k}": v for k, v in sft_metrics.items()}, aggregation=AggregationType.MEAN
                )
            )
            metrics["abstract_rl/sft_warmup"] = 1.0
        compression_loss, compression_metrics = compute_compression_loss(
            log_prob=log_prob,
            prior_log_prob=data["abstract_prior_log_probs"],
            abstract_mask=abstract_mask,
            row_weight=data["abstract_kl_weight"],
            kl_type=str(abstract_config.get("kl_type", "low_var_kl")),
            kl_clip=float(abstract_config.get("kl_clip", 0.0)),
        )
        kl_coef = 0.0 if sft_warmup else float(abstract_config.get("kl_coef", 0.0))
        policy_loss = policy_loss + kl_coef * compression_loss
        compression_metrics.update(
            compute_span_diagnostics(
                log_prob=log_prob,
                old_log_prob=old_log_prob,
                entropy=entropy,
                reasoning_mask=reasoning_mask,
                abstract_mask=abstract_mask,
                response_mask=response_mask,
                clip_low=config.clip_ratio_low,
                clip_high=config.clip_ratio_high,
            )
        )
        metrics.update(
            Metric.from_dict(
                {f"abstract_rl/{k}": v for k, v in compression_metrics.items()}, aggregation=AggregationType.MEAN
            )
        )
        metrics["abstract_rl/kl_coef"] = kl_coef

    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    if exploration is not None:
        explore_loss, explore_metrics = exploration.compute_loss(
            current_log_probs=log_prob,
            ema_log_probs=ema_log_prob,
            entropy=entropy,
            response_mask=response_mask,
        )
        policy_loss += explore_loss
        explore_metrics = Metric.from_dict(explore_metrics, aggregation=AggregationType.MEAN)
        explore_metrics["explore/status"] = Metric(
            value=explore_metrics["explore/status"].values[0], aggregation=AggregationType.MAX
        )
        metrics.update(explore_metrics)

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
