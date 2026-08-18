# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Reward-improved finite-group target distribution q* for GPI-CE.

For each prompt x_b with a candidate group G_b = {y_b,1 ... y_b,K}, the old-policy
sequence score is the response-token MEAN log-probability

    s_b,i = sum_t m_b,i,t log pi_old(y_b,i,t | x_b, y_b,i,<t) / sum_t m_b,i,t

(mean, not sum: a sum lets short responses win purely by accumulating fewer
negative terms). The old finite-group distribution is p_b = softmax_i(s_b,i).

q* is the solution of the KL-regularized reward maximization over the simplex

    q*_b = argmax_q  sum_i q_i r_b,i - tau * KL(q || p_b)

whose closed form is

    q*_b,i = p_b,i exp(r_b,i / tau) / sum_j p_b,j exp(r_b,j / tau)
           = softmax_i(s_b,i + r_b,i / tau).

Constant rewards within a group give q* == p_old exactly, hence zero gradient at
the pre-update policy -- the intended finite-group policy-improvement behaviour.
"""

from __future__ import annotations

import torch


def masked_mean_sequence_score(log_probs: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Response-token mean log-probability per sequence.

    Args:
        log_probs: [N, T] per-token log pi(y_t | ...).
        response_mask: [N, T] 1 for real response tokens, 0 for padding.

    Returns:
        [N] float32 sequence scores. Rows with no valid token score 0.
    """
    mask = response_mask.to(log_probs.dtype)
    token_count = mask.sum(dim=-1)
    total = (log_probs * mask).sum(dim=-1)
    return total / token_count.clamp_min(1.0)


def compute_gpi_targets(
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int,
    temperature: float,
) -> dict[str, torch.Tensor]:
    """Build q*, p_old and the old sequence scores for every candidate group.

    Rows must already be laid out group-contiguously: candidates 0..K-1 of prompt 0,
    then candidates 0..K-1 of prompt 1, and so on.

    Args:
        old_log_probs: [N, T] token log-probs under the PRE-update actor.
        response_mask: [N, T] response-token mask.
        rewards: [N] scalar verifier reward per candidate.
        group_size: K.
        temperature: tau > 0.

    Returns:
        dict with flat [N] tensors ``q_target``, ``old_group_prob``, ``old_seq_score``.
    """
    if temperature <= 0.0:
        raise ValueError(f"gpi_ce temperature must be positive, got {temperature}")
    num_rows = old_log_probs.shape[0]
    if num_rows % group_size != 0:
        raise ValueError(f"batch rows ({num_rows}) must be divisible by group_size ({group_size})")

    # float32 throughout: the softmax is over K terms of a mean log-prob that can be
    # a few hundred in magnitude, and bf16 would quantize the reward offset away.
    seq_score = masked_mean_sequence_score(old_log_probs.float(), response_mask).view(-1, group_size)
    reward = rewards.float().view(-1, group_size).to(seq_score.device)

    # Shift-invariant for softmax, but keeps q* numerically identical to p_old when
    # rewards tie inside a group (the exactly-zero-gradient case). Must match the
    # fused path in loss.py or the two would disagree at the 1e-9 level.
    reward = reward - reward.mean(dim=-1, keepdim=True)

    log_p_old = torch.log_softmax(seq_score, dim=-1)
    log_q = torch.log_softmax(seq_score + reward / temperature, dim=-1)

    return {
        "gpi_q_target": log_q.exp().reshape(-1).detach(),
        "gpi_old_group_prob": log_p_old.exp().reshape(-1).detach(),
        "gpi_old_seq_score": seq_score.reshape(-1).detach(),
    }


def target_metrics(
    q_target: torch.Tensor,
    old_group_prob: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int,
    is_offline: torch.Tensor | None = None,
) -> dict[str, float]:
    """Driver-side diagnostics for the constructed target.

    ``target_reward_improvement`` is the guaranteed finite-group improvement
    E_q*[r] - E_p_old[r], which is non-negative by construction of q*.
    """
    q = q_target.float().view(-1, group_size)
    p = old_group_prob.float().view(-1, group_size)
    r = rewards.float().view(-1, group_size).to(q.device)

    q_reward = (q * r).sum(-1)
    p_reward = (p * r).sum(-1)
    eps = 1e-12

    metrics = {
        "gpi/target_reward": float(q_reward.mean()),
        "gpi/old_policy_reward": float(p_reward.mean()),
        "gpi/target_reward_improvement": float((q_reward - p_reward).mean()),
        "gpi/target_entropy": float(-(q * q.clamp_min(eps).log()).sum(-1).mean()),
        "gpi/old_group_entropy": float(-(p * p.clamp_min(eps).log()).sum(-1).mean()),
        "gpi/target_old_kl": float((q * (q.clamp_min(eps).log() - p.clamp_min(eps).log())).sum(-1).mean()),
        "gpi/reward_mean": float(r.mean()),
    }

    if is_offline is not None:
        off = is_offline.to(q.device).bool().view(-1, group_size)
        on = ~off
        metrics.update(
            {
                # The key diagnostic: if target_mass_offline stays ~0 the actor prior
                # suppresses offline traces before reward weighting can lift them.
                "gpi/target_mass_offline": float((q * off).sum(-1).mean()),
                "gpi/target_mass_online": float((q * on).sum(-1).mean()),
                "gpi/old_mass_offline": float((p * off).sum(-1).mean()),
                "gpi/old_mass_online": float((p * on).sum(-1).mean()),
                "gpi/reward_offline_mean": float(r[off].mean()) if bool(off.any()) else 0.0,
                "gpi/reward_online_mean": float(r[on].mean()) if bool(on.any()) else 0.0,
            }
        )
    return metrics
