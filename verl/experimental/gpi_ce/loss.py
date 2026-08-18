# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Group Policy-Improvement Cross Entropy loss.

    L(theta) = -(1/B) sum_b sum_i q*_b,i log p^theta_b,i

with p^theta_b = softmax_i(s^theta_b,i) over the K candidates of prompt b and
s^theta the response-token mean log-probability under the CURRENT actor.

The score-level gradient is exactly

    dL / ds^theta_b,i = (1/B) (p^theta_b,i - q*_b,i)

so a candidate whose target mass exceeds its current mass is pushed up and one
whose target mass is below is pushed down. There is no PPO ratio, no clipping,
no advantage and no critic.

Normalization: each micro-batch returns the SUM over its groups divided by
``groups_per_rank`` (the number of groups this DP rank holds for the whole
mini-batch). Summing across micro-batches therefore yields the rank mean, and
FSDP's gradient all-reduce (mean over the DP group) yields the global mean.
"""

from __future__ import annotations

import torch

from .target import masked_mean_sequence_score

_EPS = 1e-12


def compute_gpi_ce_loss(
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    group_size: int,
    groups_per_rank: int,
    q_target: torch.Tensor | None = None,
    group_ids: torch.Tensor | None = None,
    rewards: torch.Tensor | None = None,
    temperature: float | None = None,
    is_offline: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the GPI-CE policy loss for one micro-batch.

    Two ways to supply the target:

    * ``q_target`` given -- built on the driver from a separate old-policy forward.
    * ``q_target=None`` -- FUSED path: q* is built here from ``seq_score.detach()``
      and ``rewards``, removing that forward entirely. Exact rather than approximate,
      because with ppo_epochs=1 and a single optimizer step theta_old == theta while
      this runs, and force_group_size=K guarantees the group is complete right here.

    Args:
        log_prob: [N, T] token log-probs under the current actor (grad-connected).
        response_mask: [N, T] response-token mask.
        group_size: K.
        groups_per_rank: groups this DP rank holds for the whole mini-batch; the
            denominator that makes micro-batch sums add up to the rank mean.
        q_target: [N] detached target mass, or None to fuse.
        group_ids: optional [N] integer group id, used only to assert group integrity.
        rewards: [N] scalar verifier reward. Required when fusing.
        temperature: tau. Required when fusing.
        is_offline: optional [N] bool, enables the online/offline mass diagnostics.

    Returns:
        (loss, metrics)
    """
    num_rows = log_prob.shape[0]
    if num_rows % group_size != 0:
        raise ValueError(
            f"GPI-CE micro-batch has {num_rows} rows, not a multiple of group_size {group_size}: "
            "a candidate group was split across a micro-batch."
        )

    seq_score = masked_mean_sequence_score(log_prob.float(), response_mask).view(-1, group_size)

    if q_target is None:
        if rewards is None or temperature is None:
            raise ValueError("GPI-CE fused path needs both `rewards` and `temperature`.")
        if temperature <= 0.0:
            raise ValueError(f"gpi_ce temperature must be positive, got {temperature}")
        # theta_old == theta here, so the detached current score IS the old score.
        old_seq_score = seq_score.detach()
        reward_grouped = rewards.float().view(-1, group_size).to(seq_score.device)
        # Centering the reward within the group is a no-op for softmax (shift
        # invariance) but makes the tied-reward case land on q* == p_old EXACTLY
        # rather than within ~1e-9, so degenerate groups contribute a true zero.
        reward_grouped = reward_grouped - reward_grouped.mean(dim=-1, keepdim=True)
        q = torch.log_softmax(old_seq_score + reward_grouped / temperature, dim=-1).exp()
        p_old = torch.log_softmax(old_seq_score, dim=-1).exp()
    else:
        q = q_target.float().view(-1, group_size).to(seq_score.device).detach()
        p_old = None

    if group_ids is not None:
        gid = group_ids.view(-1, group_size)
        if not bool(torch.all(gid == gid[:, :1])):
            raise ValueError("A GPI-CE candidate group was split across a micro-batch (group ids differ within a row).")

    q_sums = q.sum(dim=-1)
    if not bool(torch.allclose(q_sums, torch.ones_like(q_sums), atol=1e-4)):
        raise ValueError(f"GPI-CE q_target rows must sum to 1; got min={float(q_sums.min())} max={float(q_sums.max())}")

    log_group_prob = torch.log_softmax(seq_score, dim=-1)
    group_loss = -(q * log_group_prob).sum(dim=-1)  # [num_groups_in_micro_batch]
    loss = group_loss.sum() / max(groups_per_rank, 1)

    with torch.no_grad():
        group_prob = log_group_prob.exp()
        metrics = {
            "gpi/loss": float(group_loss.detach().mean()),
            "gpi/policy_entropy": float(-(group_prob * log_group_prob).sum(-1).mean()),
            # KL(q* || p_theta): how far the actor still is from the improved target.
            "gpi/projection_kl": float((q * (q.clamp_min(_EPS).log() - log_group_prob)).sum(-1).mean()),
            "gpi/policy_mass_argmax_target": float(
                group_prob.gather(1, q.argmax(dim=-1, keepdim=True)).squeeze(-1).mean()
            ),
            "gpi/seq_score_mean": float(seq_score.detach().mean()),
            "gpi/target_entropy": float(-(q * q.clamp_min(_EPS).log()).sum(-1).mean()),
        }
        if p_old is not None:
            # The driver cannot compute these on the fused path -- it never sees p_old.
            r = rewards.float().view(-1, group_size).to(q.device)
            metrics["gpi/target_reward"] = float((q * r).sum(-1).mean())
            metrics["gpi/old_policy_reward"] = float((p_old * r).sum(-1).mean())
            metrics["gpi/target_reward_improvement"] = float(((q - p_old) * r).sum(-1).mean())
            metrics["gpi/old_group_entropy"] = float(-(p_old * p_old.clamp_min(_EPS).log()).sum(-1).mean())
            metrics["gpi/reward_mean"] = float(r.mean())
            # Zero-gradient groups: q* == p_old exactly when rewards tie inside a group.
            metrics["gpi/degenerate_group_fraction"] = float(
                (r.max(dim=-1).values - r.min(dim=-1).values <= 0).float().mean()
            )
            if is_offline is not None:
                off = is_offline.to(q.device).bool().view(-1, group_size)
                metrics["gpi/target_mass_offline"] = float((q * off).sum(-1).mean())
                metrics["gpi/old_mass_offline"] = float((p_old * off).sum(-1).mean())
                metrics["gpi/target_mass_online"] = float((q * ~off).sum(-1).mean())
                metrics["gpi/old_mass_online"] = float((p_old * ~off).sum(-1).mean())
                if bool(off.any()):
                    metrics["gpi/seq_score_offline_mean"] = float(seq_score.detach()[off].mean())
                if bool((~off).any()):
                    metrics["gpi/seq_score_online_mean"] = float(seq_score.detach()[~off].mean())
    return loss, metrics
