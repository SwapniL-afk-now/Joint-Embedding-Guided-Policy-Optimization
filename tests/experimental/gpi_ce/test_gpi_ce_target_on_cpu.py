# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Target-distribution properties of q* (plan section 12)."""

import pytest
import torch

from verl.experimental.gpi_ce import compute_gpi_targets, masked_mean_sequence_score, target_metrics

K = 4


def _mask(lengths, total):
    mask = torch.zeros(len(lengths), total)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return mask


def test_q_is_a_distribution():
    torch.manual_seed(0)
    log_probs = torch.randn(3 * K, 7) - 1.0
    mask = _mask([3, 5, 7, 4] * 3, 7)
    rewards = torch.randint(0, 2, (3 * K,)).float()

    out = compute_gpi_targets(log_probs, mask, rewards, group_size=K, temperature=0.5)
    q = out["gpi_q_target"].view(-1, K)

    assert torch.all(q >= 0)
    torch.testing.assert_close(q.sum(-1), torch.ones(3))


def test_constant_reward_reproduces_old_policy():
    """r constant within a group => q* == p_old => zero gradient at theta_old."""
    torch.manual_seed(1)
    log_probs = torch.randn(2 * K, 6) - 1.0
    mask = _mask([6] * 2 * K, 6)
    rewards = torch.full((2 * K,), 1.0)

    out = compute_gpi_targets(log_probs, mask, rewards, group_size=K, temperature=0.5)
    torch.testing.assert_close(out["gpi_q_target"], out["gpi_old_group_prob"])

    metrics = target_metrics(out["gpi_q_target"], out["gpi_old_group_prob"], rewards, K)
    assert metrics["gpi/target_reward_improvement"] == pytest.approx(0.0, abs=1e-6)


def test_reward_ratio_is_exponential_in_delta_over_tau():
    """Equal old probability + reward gap dr => q1/q2 == exp(dr/tau)."""
    tau = 0.5
    log_probs = torch.zeros(K, 4)  # identical scores => identical p_old
    mask = _mask([4] * K, 4)
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])

    out = compute_gpi_targets(log_probs, mask, rewards, group_size=K, temperature=tau)
    q = out["gpi_q_target"]
    torch.testing.assert_close(out["gpi_old_group_prob"], torch.full((K,), 1.0 / K))
    assert float(q[0] / q[1]) == pytest.approx(torch.exp(torch.tensor(1.0 / tau)).item(), rel=1e-5)


def test_improvement_is_non_negative():
    """E_q*[r] - E_p_old[r] >= 0 is guaranteed by construction."""
    torch.manual_seed(2)
    for _ in range(20):
        log_probs = torch.randn(5 * K, 8) * 2 - 1.0
        mask = _mask(torch.randint(1, 9, (5 * K,)).tolist(), 8)
        rewards = torch.randint(0, 2, (5 * K,)).float()
        out = compute_gpi_targets(log_probs, mask, rewards, group_size=K, temperature=0.5)
        m = target_metrics(out["gpi_q_target"], out["gpi_old_group_prob"], rewards, K)
        assert m["gpi/target_reward_improvement"] >= -1e-6


def test_score_is_length_normalized():
    """A long and a short response with the same per-token log-prob score equally.

    A sequence SUM would rank the short one strictly higher for free.
    """
    log_probs = torch.full((2, 10), -0.5)
    mask = _mask([2, 10], 10)
    scores = masked_mean_sequence_score(log_probs, mask)
    torch.testing.assert_close(scores, torch.full((2,), -0.5))


def test_rejects_bad_shapes_and_temperature():
    log_probs = torch.zeros(6, 3)
    mask = torch.ones(6, 3)
    rewards = torch.zeros(6)
    with pytest.raises(ValueError, match="divisible"):
        compute_gpi_targets(log_probs, mask, rewards, group_size=4, temperature=0.5)
    with pytest.raises(ValueError, match="positive"):
        compute_gpi_targets(log_probs, mask, rewards, group_size=3, temperature=0.0)
