import math

import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_sdc_tr, compute_policy_loss_vanilla


class _PolicyLoss:
    def __init__(self, sdc_tr_alpha, sdc_tr_tilt_clip=10.0):
        self.sdc_tr_alpha = sdc_tr_alpha
        self.sdc_tr_tilt_clip = sdc_tr_tilt_clip


class Config:
    clip_ratio = 0.2
    clip_ratio_low = 0.2
    clip_ratio_high = 0.2
    global_batch_info = {}

    def __init__(self, sdc_tr_alpha, sdc_tr_tilt_clip=10.0):
        self.policy_loss = _PolicyLoss(sdc_tr_alpha, sdc_tr_tilt_clip)

    @staticmethod
    def get(key, default=None):
        return default


def test_sdc_tr_alpha_one_matches_vanilla_bit_for_bit():
    old = torch.zeros(2, 1)
    current = torch.log(torch.tensor([[1.5], [0.7]]))
    advantages = torch.tensor([[1.0], [-1.0]])
    response_mask = torch.ones(2, 1, dtype=torch.bool)
    failure_mask = torch.tensor([True, True])

    vanilla_loss, _ = compute_policy_loss_vanilla(
        old_log_prob=old, log_prob=current, advantages=advantages, response_mask=response_mask, config=Config(1.0)
    )

    for success, failure in [(-1.0, -0.5), (-5.0, -0.01), (0.0, 0.0)]:
        loss, metrics = compute_policy_loss_sdc_tr(
            old_log_prob=old,
            log_prob=current,
            advantages=advantages,
            response_mask=response_mask,
            config=Config(1.0),
            success_log_prob=torch.full_like(old, success),
            failure_log_prob=torch.full_like(old, failure),
            failure_mask=failure_mask,
            reliability_gate=1.0,
        )
        torch.testing.assert_close(loss, vanilla_loss)
        assert metrics["sdc_tr/lambda_eff"] == 0.0


def test_sdc_tr_zero_reliability_gate_matches_vanilla_regardless_of_alpha():
    old = torch.zeros(2, 1)
    current = torch.log(torch.tensor([[1.5], [0.7]]))
    advantages = torch.tensor([[1.0], [-1.0]])
    response_mask = torch.ones(2, 1, dtype=torch.bool)
    failure_mask = torch.tensor([True, True])

    vanilla_loss, _ = compute_policy_loss_vanilla(
        old_log_prob=old, log_prob=current, advantages=advantages, response_mask=response_mask, config=Config(0.3)
    )
    loss, metrics = compute_policy_loss_sdc_tr(
        old_log_prob=old,
        log_prob=current,
        advantages=advantages,
        response_mask=response_mask,
        config=Config(0.3),
        success_log_prob=torch.tensor([[-1.0], [-4.0]]),
        failure_log_prob=torch.tensor([[-3.0], [-0.2]]),
        failure_mask=failure_mask,
        reliability_gate=0.0,
    )
    torch.testing.assert_close(loss, vanilla_loss)
    assert metrics["sdc_tr/lambda_eff"] == 0.0


def test_sdc_tr_sign_matches_effect_table():
    old = torch.zeros(1, 2)
    current = torch.zeros(1, 2)  # ratio_base == 1 for both tokens
    advantages = torch.tensor([[-1.0, -1.0]])
    response_mask = torch.ones(1, 2, dtype=torch.bool)
    failure_mask = torch.tensor([True])
    # token0: success_log_prob > failure_log_prob -> success-like (contrast < 0)
    # token1: failure_log_prob > success_log_prob -> failure-like (contrast > 0)
    success_log_prob = torch.tensor([[-1.0, -3.0]])
    failure_log_prob = torch.tensor([[-3.0, -1.0]])

    loss, metrics = compute_policy_loss_sdc_tr(
        old_log_prob=old,
        log_prob=current,
        advantages=advantages,
        response_mask=response_mask,
        config=Config(0.5),
        success_log_prob=success_log_prob,
        failure_log_prob=failure_log_prob,
        failure_mask=failure_mask,
        reliability_gate=1.0,
    )

    lambda_eff = 0.5
    r0 = math.exp(-lambda_eff)  # success-like token: tilted ratio pulled below 1
    r1 = math.exp(lambda_eff)  # failure-like token: tilted ratio pushed above 1
    assert r0 < 1.0 < r1

    def dual_clip_loss(r):
        pg1 = -(-1.0) * r
        pg2 = -(-1.0) * min(max(r, 0.8), 1.2)
        clipped = max(pg1, pg2)
        return min(clipped, 3.0)  # advantages < 0 -> dual-clip lower bound applies

    expected0 = dual_clip_loss(r0)  # success-like + A<0 -> clipped earlier (floor 0.8 binds)
    expected1 = dual_clip_loss(r1)  # failure-like + A<0 -> more room to decrease (unclipped)
    assert math.isclose(expected0, 0.8, rel_tol=1e-6)
    assert expected1 > expected0
    expected = (expected0 + expected1) / 2
    torch.testing.assert_close(loss, torch.tensor(expected))
    assert metrics["sdc_tr/active_failed_token_fraction"] == 1.0
