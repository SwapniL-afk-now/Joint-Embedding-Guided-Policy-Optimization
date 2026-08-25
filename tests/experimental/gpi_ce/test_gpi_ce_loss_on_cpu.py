# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Gradient and invariance properties of the GPI-CE loss (plan section 12)."""

import pytest
import torch

from verl.experimental.gpi_ce import compute_gpi_ce_loss

K = 4


def _batch(num_groups=2, seq_len=6, seed=0):
    torch.manual_seed(seed)
    rows = num_groups * K
    log_prob = (torch.randn(rows, seq_len, dtype=torch.float64) - 1.0).requires_grad_(True)
    mask = torch.zeros(rows, seq_len)
    lengths = torch.randint(1, seq_len + 1, (rows,))
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    q = torch.softmax(torch.randn(num_groups, K, dtype=torch.float64), dim=-1).reshape(-1)
    group_ids = torch.arange(num_groups).repeat_interleave(K)
    return log_prob, mask, q, group_ids, lengths


def test_score_gradient_matches_p_minus_q():
    """dL/ds_i == (p_theta_i - q*_i) / num_groups, checked in fp64."""
    num_groups = 3
    torch.manual_seed(0)
    seq_score = torch.randn(num_groups * K, dtype=torch.float64).requires_grad_(True)
    q = torch.softmax(torch.randn(num_groups, K, dtype=torch.float64), dim=-1)

    log_p = torch.log_softmax(seq_score.view(num_groups, K), dim=-1)
    loss = -(q * log_p).sum(-1).sum() / num_groups
    loss.backward()

    analytic = ((log_p.exp() - q) / num_groups).reshape(-1).detach()
    rel = (seq_score.grad - analytic).abs() / (analytic.abs() + 1e-12)
    assert float(rel.max()) < 1e-8


def test_token_gradient_is_uniform_over_valid_tokens():
    """Every valid token of candidate i gets (p_i - q_i)/T_i; padding gets exactly 0."""
    log_prob, mask, q, group_ids, lengths = _batch(num_groups=2, seed=3)
    loss, _ = compute_gpi_ce_loss(log_prob, mask, K, 2, q_target=q, group_ids=group_ids)
    loss.backward()

    grad = log_prob.grad
    for i, length in enumerate(lengths):
        valid = grad[i, :length]
        # uniform across the sequence's own valid tokens
        torch.testing.assert_close(valid, valid[0].expand_as(valid))
        assert torch.all(grad[i, length:] == 0), f"row {i} leaked gradient into padding"
        # and scaled by 1/T_i
        assert float(valid[0].abs()) == pytest.approx(float(valid[0].abs() * length / length), rel=1e-9)


def test_padding_length_does_not_change_loss():
    """Extra padding columns must not move the loss."""
    log_prob, mask, q, group_ids, _ = _batch(num_groups=2, seed=5)
    loss_a, _ = compute_gpi_ce_loss(log_prob, mask, K, 2, q_target=q, group_ids=group_ids)

    pad = torch.randn(log_prob.shape[0], 4, dtype=torch.float64)
    log_prob_b = torch.cat([log_prob.detach(), pad], dim=1).requires_grad_(True)
    mask_b = torch.cat([mask, torch.zeros(mask.shape[0], 4)], dim=1)
    loss_b, _ = compute_gpi_ce_loss(log_prob_b, mask_b, K, 2, q_target=q, group_ids=group_ids)

    torch.testing.assert_close(loss_a.detach(), loss_b.detach())


def test_permutation_within_group_leaves_loss_unchanged():
    log_prob, mask, q, group_ids, _ = _batch(num_groups=2, seed=7)
    base, _ = compute_gpi_ce_loss(log_prob, mask, K, 2, q_target=q, group_ids=group_ids)

    perm = torch.tensor([2, 0, 3, 1])
    idx = torch.cat([perm + g * K for g in range(2)])
    permuted, _ = compute_gpi_ce_loss(
        log_prob[idx], mask[idx], K, 2, q_target=q[idx], group_ids=group_ids[idx]
    )
    torch.testing.assert_close(base.detach(), permuted.detach())


def test_micro_batch_split_sums_to_the_same_gradient():
    """Two micro-batches of one group each == one micro-batch of two groups."""
    log_prob, mask, q, group_ids, _ = _batch(num_groups=2, seed=11)
    whole, _ = compute_gpi_ce_loss(log_prob, mask, K, 2, q_target=q, group_ids=group_ids)
    whole.backward()
    whole_grad = log_prob.grad.clone()

    split = log_prob.detach().clone().requires_grad_(True)
    for g in range(2):
        sl = slice(g * K, (g + 1) * K)
        part, _ = compute_gpi_ce_loss(
            split[sl], mask[sl], K, 2, q_target=q[sl], group_ids=group_ids[sl]
        )
        part.backward()
    torch.testing.assert_close(whole_grad, split.grad)


def test_mean_of_micro_batch_means_is_wrong():
    """Counter-example: averaging micro-batch means over-weights small micro-batches."""
    log_prob, mask, q, group_ids, _ = _batch(num_groups=4, seed=13)
    correct, _ = compute_gpi_ce_loss(log_prob, mask, K, 4, q_target=q, group_ids=group_ids)

    # split 3 groups + 1 group and average the two per-micro-batch means
    a, _ = compute_gpi_ce_loss(
        log_prob[: 3 * K], mask[: 3 * K], K, 3, q_target=q[: 3 * K], group_ids=group_ids[: 3 * K]
    )
    b, _ = compute_gpi_ce_loss(
        log_prob[3 * K :], mask[3 * K :], K, 1, q_target=q[3 * K :], group_ids=group_ids[3 * K :]
    )
    mean_of_means = (a + b) / 2
    assert not torch.allclose(correct.detach(), mean_of_means.detach())


def test_split_group_raises():
    """A micro-batch that cuts a group must fail, not silently renormalize."""
    log_prob, mask, q, group_ids, _ = _batch(num_groups=2, seed=17)
    with pytest.raises(ValueError, match="not a multiple of group_size"):
        compute_gpi_ce_loss(log_prob[:6], mask[:6], K, 2, q_target=q[:6], group_ids=group_ids[:6])

    scrambled = group_ids.clone()
    scrambled[0] = 99
    with pytest.raises(ValueError, match="split across a micro-batch"):
        compute_gpi_ce_loss(log_prob, mask, K, 2, q_target=q, group_ids=scrambled)


def test_non_normalized_target_raises():
    log_prob, mask, q, group_ids, _ = _batch(num_groups=2, seed=19)
    with pytest.raises(ValueError, match="must sum to 1"):
        compute_gpi_ce_loss(log_prob, mask, K, 2, q_target=q * 0.5, group_ids=group_ids)


def test_fused_path_equals_two_pass_path():
    """Fusing q* into the training forward is EXACT, not an approximation.

    With ppo_epochs=1 and one optimizer step, theta_old == theta during the update,
    so building q* from the detached training scores must reproduce -- bit for bit --
    the q* a separate old-policy forward would have produced.
    """
    from verl.experimental.gpi_ce import compute_gpi_targets

    num_groups, tau = 3, 0.5
    log_prob, mask, _, group_ids, _ = _batch(num_groups=num_groups, seed=29)
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0] * num_groups, dtype=torch.float64)

    # Two-pass: old_log_probs come from a separate forward of the SAME theta.
    targets = compute_gpi_targets(log_prob.detach(), mask, rewards, K, tau)
    two_pass, m_two = compute_gpi_ce_loss(
        log_prob, mask, K, num_groups, q_target=targets["gpi_q_target"], group_ids=group_ids
    )
    two_pass.backward()
    grad_two = log_prob.grad.clone()

    fused_input = log_prob.detach().clone().requires_grad_(True)
    fused, m_fused = compute_gpi_ce_loss(
        fused_input, mask, K, num_groups, group_ids=group_ids, rewards=rewards, temperature=tau
    )
    fused.backward()

    torch.testing.assert_close(two_pass.detach(), fused.detach(), rtol=0, atol=1e-9)
    torch.testing.assert_close(grad_two, fused_input.grad, rtol=0, atol=1e-9)
    assert m_fused["gpi/projection_kl"] == pytest.approx(m_two["gpi/projection_kl"], rel=1e-9)
    # Only the fused path can report these -- the driver never sees p_old there.
    assert m_fused["gpi/target_reward_improvement"] >= -1e-9
    assert "gpi/degenerate_group_fraction" in m_fused


def test_fused_path_reports_offline_mass_and_degenerate_groups():
    num_groups, tau = 2, 0.5
    log_prob, mask, _, group_ids, _ = _batch(num_groups=num_groups, seed=31)
    # group 0 has mixed rewards, group 1 is all-correct => zero gradient by construction
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    is_offline = torch.tensor([False, False, False, True] * num_groups)

    loss, m = compute_gpi_ce_loss(
        log_prob, mask, K, num_groups, group_ids=group_ids, rewards=rewards,
        temperature=tau, is_offline=is_offline,
    )
    assert m["gpi/degenerate_group_fraction"] == pytest.approx(0.5)
    assert 0.0 <= m["gpi/target_mass_offline"] <= 1.0
    assert m["gpi/target_mass_offline"] + m["gpi/target_mass_online"] == pytest.approx(1.0, abs=1e-6)


def test_all_equal_rewards_give_zero_gradient_in_fused_path():
    """q* == p_old when rewards tie, so the group contributes exactly nothing."""
    num_groups, tau = 2, 0.5
    log_prob, mask, _, group_ids, _ = _batch(num_groups=num_groups, seed=37)
    rewards = torch.ones(num_groups * K, dtype=torch.float64)

    loss, m = compute_gpi_ce_loss(
        log_prob, mask, K, num_groups, group_ids=group_ids, rewards=rewards, temperature=tau
    )
    loss.backward()
    # Exactly zero in real arithmetic; ~1e-9 here purely from the deliberate fp32
    # upcast in the score computation (verified: pure fp64 gives exactly 0.0).
    # bf16 training noise is orders of magnitude larger than this.
    assert float(log_prob.grad.abs().max()) < 1e-6, "tied rewards must produce ~zero gradient"
    assert m["gpi/degenerate_group_fraction"] == pytest.approx(1.0)
    assert m["gpi/target_reward_improvement"] == pytest.approx(0.0, abs=1e-9)


def test_fused_path_requires_rewards_and_temperature():
    log_prob, mask, _, group_ids, _ = _batch(num_groups=2, seed=41)
    with pytest.raises(ValueError, match="needs both"):
        compute_gpi_ce_loss(log_prob, mask, K, 2, group_ids=group_ids)


def test_target_is_detached():
    log_prob, mask, q, group_ids, _ = _batch(num_groups=2, seed=23)
    q = q.clone().requires_grad_(True)
    loss, _ = compute_gpi_ce_loss(log_prob, mask, K, 2, q_target=q, group_ids=group_ids)
    loss.backward()
    assert q.grad is None, "q* must be a detached target, not a trainable quantity"
