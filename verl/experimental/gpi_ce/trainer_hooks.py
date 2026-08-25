# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Driver-side glue between RayPPOTrainer.fit() and GPI-CE.

Three call sites in the stock synchronous loop, which already implements the
required lifecycle (generate -> sleep_replicas -> old_log_prob -> update_actor ->
update_weights, i.e. one parameter sync per rollout batch):

    1. ``build_mixed_batch``   right after the generated rollouts are unioned into
       the batch, BEFORE reward scoring and DP balancing, so the offline
       candidates are verified by the same reward path and travel with their group.
    2. ``compute_targets``     in place of ``compute_advantage``.
    3. ``stamp_actor_batch``   just before ``update_actor``, to pin the group size
       through micro-batching and hand the loss its normalizer.
"""

from __future__ import annotations

import numpy as np
import torch

from verl.protocol import DataProto

from .mixed_candidate_builder import MixedCandidateBuilder
from .target import compute_gpi_targets, target_metrics


def make_builder(trainer) -> MixedCandidateBuilder:
    pad_token_id = trainer.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = trainer.tokenizer.eos_token_id
    return MixedCandidateBuilder(
        pad_token_id=pad_token_id,
        allow_short_offline=trainer.gpi_ce_config.allow_short_offline,
    )


def score_offline_rows(trainer, batch: DataProto) -> dict[str, float]:
    """Score the spliced-in offline rows with the training verifier.

    Rewards are produced by the agent loop DURING generation, so ``rm_scores`` already
    exists by the time the offline candidates are appended -- and because each offline
    row is cloned from its group's first online row, it would otherwise silently
    inherit that row's reward. (``use_rm`` is false for rule-based math, so the trainer
    never recomputes.) Left unfixed, verified-correct R1 solutions get whatever the
    model's own first rollout happened to score, and q* tilts on noise.

    Scores are written with the same convention as the agent loop: the scalar lands on
    the last valid response token, so ``rm_scores.sum(-1)`` is the sequence reward.
    """
    if "rm_scores" not in batch.batch.keys():
        raise ValueError("GPI-CE expected rm_scores from the agent loop before scoring offline rows.")

    from verl.utils.reward_score import default_compute_score

    is_offline = batch.batch["gpi_is_offline"].bool()
    rm_scores = batch.batch["rm_scores"]
    response_mask = batch.batch["response_mask"]
    responses = batch.batch["responses"]
    data_sources = batch.non_tensor_batch["data_source"]
    reward_models = batch.non_tensor_batch["reward_model"]

    scored, correct = 0, 0
    for i in torch.nonzero(is_offline, as_tuple=False).flatten().tolist():
        valid = int(response_mask[i].sum())
        if valid == 0:
            rm_scores[i] = 0.0
            continue
        text = trainer.tokenizer.decode(responses[i][:valid], skip_special_tokens=True)
        ground_truth = reward_models[i]["ground_truth"]
        result = default_compute_score(str(data_sources[i]), text, ground_truth)
        score = float(result["score"]) if isinstance(result, dict) else float(result)
        rm_scores[i] = 0.0
        rm_scores[i, valid - 1] = score
        scored += 1
        correct += int(score > 0)

    return {
        "gpi/offline_rows_scored": float(scored),
        # Expect this near 1.0: these are dataset-verified solutions. A low value means
        # the splice is corrupting them (truncation, wrong tokens, bad mask).
        "gpi/offline_verifier_pass_rate": float(correct / scored) if scored else 0.0,
    }


def build_mixed_batch(trainer, batch: DataProto) -> DataProto:
    """Append the offline candidates, turning B x K_on rows into B x K."""
    cfg = trainer.gpi_ce_config
    mixed = trainer.gpi_ce_builder.build(
        online_batch=batch,
        online_per_prompt=cfg.online_per_prompt,
        offline_per_prompt=cfg.offline_per_prompt,
    )
    mixed.meta_info = dict(batch.meta_info)
    return mixed


def compute_targets(trainer, batch: DataProto, reward_tensor: torch.Tensor) -> dict[str, float]:
    """Build q* from the pre-update actor's scores and the verifier rewards.

    Replaces ``compute_advantage``. Writes ``gpi_q_target``, ``gpi_old_group_prob``
    and ``gpi_old_seq_score`` into the batch, plus a zero ``advantages`` tensor so
    the unchanged actor plumbing (which always selects that field) stays happy.
    """
    cfg = trainer.gpi_ce_config
    group_size = cfg.group_size
    scalar_reward = reward_tensor.sum(dim=-1).float()
    batch.batch["gpi_scalar_reward"] = scalar_reward
    # GPI-CE has neither advantages nor returns, but the shared actor path selects
    # "advantages" unconditionally and compute_data_metrics reads "returns"; both are
    # normally produced by compute_advantage, which this path replaces. Zeros keep the
    # untouched plumbing working -- the GPI-CE loss ignores them entirely.
    zeros = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
    batch.batch["advantages"] = zeros
    batch.batch["returns"] = zeros.clone()

    if cfg.fuse_old_log_prob:
        # No old-policy forward ran, so there is no s_old here: the loss builds q*
        # from its own detached scores. Only reward-side diagnostics are available
        # on the driver; the q*-dependent ones are emitted by the loss instead.
        r = scalar_reward.view(-1, group_size)
        spread = r.max(dim=-1).values - r.min(dim=-1).values
        metrics = {
            "gpi/reward_mean": float(scalar_reward.mean()),
            "gpi/degenerate_group_fraction": float((spread <= 0).float().mean()),
            "gpi/group_size": float(group_size),
            "system/total_actor_tokens": float(batch.batch["response_mask"].sum()),
        }
        is_off = batch.batch.get("gpi_is_offline", None)
        if is_off is not None:
            off = is_off.bool()
            metrics["gpi/reward_offline_mean"] = float(scalar_reward[off].mean()) if bool(off.any()) else 0.0
            metrics["gpi/reward_online_mean"] = float(scalar_reward[~off].mean()) if bool((~off).any()) else 0.0
            metrics["system/offline_tokens_trained"] = float(batch.batch["response_mask"][off].sum())
        return metrics

    targets = compute_gpi_targets(
        old_log_probs=batch.batch["old_log_probs"],
        response_mask=batch.batch["response_mask"],
        rewards=scalar_reward,
        group_size=group_size,
        temperature=cfg.temperature,
    )
    for key, value in targets.items():
        batch.batch[key] = value

    is_offline = batch.batch.get("gpi_is_offline", None)
    metrics = target_metrics(
        q_target=targets["gpi_q_target"],
        old_group_prob=targets["gpi_old_group_prob"],
        rewards=scalar_reward,
        group_size=group_size,
        is_offline=is_offline,
    )

    seq_score = targets["gpi_old_seq_score"]
    if is_offline is not None:
        off = is_offline.bool()
        if bool(off.any()):
            metrics["gpi/seq_score_offline_mean"] = float(seq_score[off].mean())
        if bool((~off).any()):
            metrics["gpi/seq_score_online_mean"] = float(seq_score[~off].mean())
        metrics["system/offline_tokens_trained"] = float(batch.batch["response_mask"][off].sum())
    metrics["system/total_actor_tokens"] = float(batch.batch["response_mask"].sum())
    metrics["gpi/group_size"] = float(group_size)
    return metrics


def stamp_actor_batch(trainer, batch_td) -> None:
    """Pin group atomicity through micro-batching and pass the loss its normalizer."""
    import verl.utils.tensordict_utils as tu

    cfg = trainer.gpi_ce_config
    group_size = cfg.group_size
    num_rows = batch_td.batch_size[0]
    num_groups = num_rows // group_size
    dp_size = trainer._get_dp_size(trainer.actor_rollout_wg, "actor")
    if num_groups % dp_size != 0:
        raise ValueError(
            f"GPI-CE needs the group count ({num_groups}) divisible by dp_size ({dp_size}) so every rank "
            "holds the same number of groups; adjust data.train_batch_size."
        )
    tu.assign_non_tensor(
        batch_td,
        force_group_size=group_size,
        custom_gpi_ce={**cfg.to_dict(), "groups_per_rank": num_groups // dp_size},
    )


def check_batch_integrity(batch: DataProto, group_size: int) -> None:
    """Fail loudly if DP balancing or filtering broke group contiguity."""
    uids = np.asarray(batch.non_tensor_batch["uid"], dtype=object)
    num_rows = len(uids)
    if num_rows % group_size != 0:
        raise ValueError(f"GPI-CE batch has {num_rows} rows, not a multiple of group_size {group_size}")
    grouped = uids.reshape(-1, group_size)
    bad = [i for i, row in enumerate(grouped) if len(set(row)) != 1]
    if bad:
        raise ValueError(
            f"GPI-CE candidate groups were split (rows {bad[:5]}). The group softmax is only defined over a "
            "complete group -- check balance_batch group preservation and DAPO filtering."
        )
    cand = batch.non_tensor_batch.get("gpi_candidate_id", None)
    if cand is not None:
        expected = np.tile(np.arange(group_size), len(grouped))
        if not np.array_equal(np.asarray(cand), expected):
            raise ValueError("GPI-CE candidate ids are not 0..K-1 within every group after batching.")
