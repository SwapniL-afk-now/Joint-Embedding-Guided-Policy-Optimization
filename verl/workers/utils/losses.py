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
from verl.experimental.sdc.sdc_loss import compute_sdc_loss
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

    sdc_config = tu.get_non_tensor_data(data=data, key="custom_sdc", default=None)

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
    sdc_enabled = bool(sdc_config and sdc_config.get("enable", False))
    if sdc_enabled:
        for field in ("sdc_failure_mask", "sdc_success_log_probs", "sdc_failure_log_probs"):
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

    if sdc_enabled:
        required = ("sdc_failure_mask", "sdc_success_log_probs", "sdc_failure_log_probs")
        missing = [field for field in required if field not in data]
        if missing:
            raise ValueError(f"SDC batch is missing required fields: {missing}")
        sdc_loss, sdc_stats = compute_sdc_loss(
            log_prob=log_prob,
            old_log_prob=old_log_prob,
            success_log_prob=data["sdc_success_log_probs"],
            failure_log_prob=data["sdc_failure_log_probs"],
            response_mask=response_mask,
            failure_mask=data["sdc_failure_mask"],
            beta=float(sdc_config.get("beta", 0.0)),
            loss_agg_mode="token-mean",
            global_batch_info=config.global_batch_info,
            use_importance_weight=bool(sdc_config.get("use_importance_weight", True)),
            importance_weight_clip=float(sdc_config.get("importance_weight_clip", 10.0)),
            models_ready=bool(sdc_config.get("sdc_models_ready", False)),
        )
        policy_loss = policy_loss + sdc_loss
        metrics.update(Metric.from_dict(sdc_stats, aggregation=AggregationType.MEAN))
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
