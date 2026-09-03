# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Zero-advantage ("all-wrong group") repair for the SDC-TR policy loss.

Dr.GRPO/GRPO advantages are group-centered, so a group in which every sampled
response is wrong has ``A == 0`` on every token and the tilted-ratio loss
``-A * r`` produces no gradient at all. ``policy_loss.sdc_tr_degenerate_coef``
opts into injecting a detached pseudo-advantage there:

    A_eff = A - mu_eff * c,   mu_eff = sdc_tr_degenerate_coef * reliability_gate

These tests pin (a) that the default 0.0 is bit-for-bit inert, and (b) that when
enabled the gradient w.r.t. ``log_prob`` has the sign the derivation demands.
CPU only.
"""

import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_sdc_tr, compute_policy_loss_vanilla


class _PolicyLoss:
    def __init__(self, sdc_tr_alpha, sdc_tr_tilt_clip=10.0, sdc_tr_degenerate_coef=None):
        self.sdc_tr_alpha = sdc_tr_alpha
        self.sdc_tr_tilt_clip = sdc_tr_tilt_clip
        if sdc_tr_degenerate_coef is not None:
            # Left unset on purpose when None: exercises the getattr(..., 0.0)
            # fallback for configs predating the knob.
            self.sdc_tr_degenerate_coef = sdc_tr_degenerate_coef


class Config:
    clip_ratio = 0.2
    clip_ratio_low = 0.2
    clip_ratio_high = 0.2
    global_batch_info = {}

    def __init__(self, sdc_tr_alpha, sdc_tr_tilt_clip=10.0, sdc_tr_degenerate_coef=None):
        self.policy_loss = _PolicyLoss(sdc_tr_alpha, sdc_tr_tilt_clip, sdc_tr_degenerate_coef)

    @staticmethod
    def get(key, default=None):
        return default


def _all_wrong_batch(requires_grad=False):
    """One failed response, two tokens, group-degenerate advantages (A == 0).

    Token 0: success teacher wins (normalized contrast c < 0).
    Token 1: failure teacher wins (normalized contrast c > 0).
    ``log_prob == old_log_prob`` so the base ratio is exactly 1 and no PPO clip
    boundary is active; the observed gradient is therefore purely the repair.
    """
    old = torch.zeros(1, 2)
    log_prob = torch.zeros(1, 2, requires_grad=requires_grad)
    advantages = torch.zeros(1, 2)
    response_mask = torch.ones(1, 2, dtype=torch.bool)
    failure_mask = torch.tensor([True])
    success_log_prob = torch.tensor([[-1.0, -3.0]])
    failure_log_prob = torch.tensor([[-3.0, -1.0]])
    return old, log_prob, advantages, response_mask, failure_mask, success_log_prob, failure_log_prob


def test_default_degenerate_coef_is_bit_for_bit_vanilla():
    """coef unset / 0.0 must not perturb the loss at all, teachers present or not."""
    old = torch.zeros(2, 2)
    log_prob = torch.log(torch.tensor([[1.5, 0.9], [0.7, 1.1]]))
    advantages = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    response_mask = torch.ones(2, 2, dtype=torch.bool)
    failure_mask = torch.tensor([True, True])
    success_log_prob = torch.tensor([[-1.0, -3.0], [-0.5, -2.0]])
    failure_log_prob = torch.tensor([[-3.0, -1.0], [-2.5, -0.2]])

    vanilla_loss, _ = compute_policy_loss_vanilla(
        old_log_prob=old,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=Config(1.0),
    )

    for cfg in (Config(1.0), Config(1.0, sdc_tr_degenerate_coef=0.0)):
        loss, metrics = compute_policy_loss_sdc_tr(
            old_log_prob=old,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=cfg,
            success_log_prob=success_log_prob,
            failure_log_prob=failure_log_prob,
            failure_mask=failure_mask,
            reliability_gate=1.0,
        )
        assert loss.item() == vanilla_loss.item()  # bit-for-bit, not just close
        assert metrics["sdc_tr/degenerate_coef"] == 0.0
        assert metrics["sdc_tr/mu_eff"] == 0.0
        assert metrics["sdc_tr/degenerate_token_fraction"] == 0.0
        assert metrics["sdc_tr/pseudo_advantage_abs_mean"] == 0.0

    # ... and with no teacher tensors at all (repair cannot fire without them).
    loss_no_teacher, metrics_no_teacher = compute_policy_loss_sdc_tr(
        old_log_prob=old,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=Config(1.0, sdc_tr_degenerate_coef=1.0),
    )
    assert loss_no_teacher.item() == vanilla_loss.item()
    assert metrics_no_teacher["sdc_tr/degenerate_token_fraction"] == 0.0


def test_all_wrong_group_gets_nonzero_loss_and_correct_gradient_sign():
    old, log_prob, adv, mask, fmask, s_lp, f_lp = _all_wrong_batch(requires_grad=True)

    baseline, _ = compute_policy_loss_sdc_tr(
        old_log_prob=old,
        log_prob=log_prob,
        advantages=adv,
        response_mask=mask,
        config=Config(1.0),
        success_log_prob=s_lp,
        failure_log_prob=f_lp,
        failure_mask=fmask,
        reliability_gate=1.0,
    )
    assert baseline.item() == 0.0  # no repair -> all-wrong group is a dead gradient
    assert torch.autograd.grad(baseline, log_prob, retain_graph=False)[0].abs().max().item() == 0.0

    loss, metrics = compute_policy_loss_sdc_tr(
        old_log_prob=old,
        log_prob=log_prob,
        advantages=adv,
        response_mask=mask,
        config=Config(1.0, sdc_tr_degenerate_coef=0.5),
        success_log_prob=s_lp,
        failure_log_prob=f_lp,
        failure_mask=fmask,
        reliability_gate=1.0,
    )
    # The loss VALUE can still cancel to 0 here (symmetric contrast at ratio 1),
    # but the gradient -- the thing that was dead before -- must not.
    (grad,) = torch.autograd.grad(loss, log_prob)
    assert grad.abs().max().item() > 0.0
    # Descent step is -grad. Token 1 (failure teacher wins, c > 0) must have its
    # log-prob pushed DOWN, i.e. positive gradient; token 0 (c < 0) pushed UP.
    assert grad[0, 1] > 0.0, "failure-preferred token must be suppressed"
    assert grad[0, 0] < 0.0, "success-preferred token must be reinforced"
    # Symmetric contrast (c = +-1 after centering + RMS) -> symmetric magnitudes.
    torch.testing.assert_close(grad[0, 0].abs(), grad[0, 1].abs())

    assert metrics["sdc_tr/degenerate_token_fraction"] == 1.0
    assert metrics["sdc_tr/pseudo_advantage_abs_mean"] > 0.0


def test_degenerate_repair_scales_with_coef_and_gate():
    """mu_eff = coef * gate, so doubling either doubles the pseudo-advantage."""
    old, log_prob, adv, mask, fmask, s_lp, f_lp = _all_wrong_batch()

    def run(coef, gate):
        return compute_policy_loss_sdc_tr(
            old_log_prob=old,
            log_prob=log_prob,
            advantages=adv,
            response_mask=mask,
            config=Config(1.0, sdc_tr_degenerate_coef=coef),
            success_log_prob=s_lp,
            failure_log_prob=f_lp,
            failure_mask=fmask,
            reliability_gate=gate,
        )[1]

    base = run(0.25, 1.0)
    assert base["sdc_tr/mu_eff"] == 0.25
    torch.testing.assert_close(
        torch.tensor(run(0.5, 1.0)["sdc_tr/pseudo_advantage_abs_mean"]),
        torch.tensor(2 * base["sdc_tr/pseudo_advantage_abs_mean"]),
    )
    torch.testing.assert_close(
        torch.tensor(run(1.0, 0.25)["sdc_tr/pseudo_advantage_abs_mean"]),
        torch.tensor(base["sdc_tr/pseudo_advantage_abs_mean"]),
    )


def test_zero_reliability_gate_is_vanilla_regardless_of_coef():
    old, log_prob, adv, mask, fmask, s_lp, f_lp = _all_wrong_batch()

    vanilla_loss, _ = compute_policy_loss_vanilla(
        old_log_prob=old,
        log_prob=log_prob,
        advantages=adv,
        response_mask=mask,
        config=Config(0.3),
    )
    for coef in (0.0, 0.5, 10.0):
        loss, metrics = compute_policy_loss_sdc_tr(
            old_log_prob=old,
            log_prob=log_prob,
            advantages=adv,
            response_mask=mask,
            config=Config(0.3, sdc_tr_degenerate_coef=coef),
            success_log_prob=s_lp,
            failure_log_prob=f_lp,
            failure_mask=fmask,
            reliability_gate=0.0,
        )
        assert loss.item() == vanilla_loss.item()
        assert metrics["sdc_tr/mu_eff"] == 0.0
        assert metrics["sdc_tr/lambda_eff"] == 0.0
        assert metrics["sdc_tr/degenerate_token_fraction"] == 0.0


def test_nonzero_advantages_are_untouched_by_the_repair():
    old = torch.zeros(2, 2)
    log_prob = torch.log(torch.tensor([[1.05, 0.95], [0.9, 1.1]]))
    advantages = torch.tensor([[1.0, -0.5], [0.25, -2.0]])  # nothing is exactly 0
    response_mask = torch.ones(2, 2, dtype=torch.bool)
    failure_mask = torch.tensor([True, True])
    success_log_prob = torch.tensor([[-1.0, -3.0], [-0.5, -2.0]])
    failure_log_prob = torch.tensor([[-3.0, -1.0], [-2.5, -0.2]])

    def run(coef, alpha=1.0):
        return compute_policy_loss_sdc_tr(
            old_log_prob=old,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=Config(alpha, sdc_tr_degenerate_coef=coef),
            success_log_prob=success_log_prob,
            failure_log_prob=failure_log_prob,
            failure_mask=failure_mask,
            reliability_gate=1.0,
        )

    off_loss, off_metrics = run(0.0)
    on_loss, on_metrics = run(2.0)
    assert on_loss.item() == off_loss.item()
    assert on_metrics["sdc_tr/degenerate_token_fraction"] == 0.0
    assert on_metrics["sdc_tr/pseudo_advantage_abs_mean"] == 0.0
    assert off_metrics["sdc_tr/degenerate_token_fraction"] == 0.0

    # The repair also must not disturb the tilt path when advantages are live.
    tilt_off, _ = run(0.0, alpha=0.5)
    tilt_on, _ = run(2.0, alpha=0.5)
    assert tilt_on.item() == tilt_off.item()


def test_repair_applies_only_to_failed_rows_and_response_tokens():
    old = torch.zeros(2, 3)
    log_prob = torch.zeros(2, 3, requires_grad=True)
    advantages = torch.zeros(2, 3)
    # Row 0 is a failed response; row 1 succeeded. Token 2 of each row is padding.
    response_mask = torch.tensor([[True, True, False], [True, True, False]])
    failure_mask = torch.tensor([True, False])
    success_log_prob = torch.tensor([[-1.0, -3.0, -1.0], [-1.0, -3.0, -1.0]])
    failure_log_prob = torch.tensor([[-3.0, -1.0, -9.0], [-3.0, -1.0, -9.0]])

    loss, metrics = compute_policy_loss_sdc_tr(
        old_log_prob=old,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=Config(1.0, sdc_tr_degenerate_coef=0.5),
        success_log_prob=success_log_prob,
        failure_log_prob=failure_log_prob,
        failure_mask=failure_mask,
        reliability_gate=1.0,
    )

    (grad,) = torch.autograd.grad(loss, log_prob)
    assert grad[0, 0] < 0.0 and grad[0, 1] > 0.0  # failed row: repaired
    assert torch.all(grad[1] == 0.0), "successful row must stay untouched"
    assert grad[0, 2].item() == 0.0, "padding token must stay untouched"

    # 2 repaired tokens out of 4 unmasked response tokens.
    assert metrics["sdc_tr/degenerate_token_fraction"] == 0.5
    assert metrics["sdc_tr/active_failed_token_fraction"] == 0.5


def test_degenerate_metrics_keys_and_fraction_are_reported():
    old = torch.zeros(1, 4)
    log_prob = torch.zeros(1, 4)
    # Half the tokens are group-degenerate (A == 0), half carry reward signal.
    advantages = torch.tensor([[0.0, 0.0, 1.0, -1.0]])
    response_mask = torch.ones(1, 4, dtype=torch.bool)
    failure_mask = torch.tensor([True])
    success_log_prob = torch.tensor([[-1.0, -3.0, -1.0, -3.0]])
    failure_log_prob = torch.tensor([[-3.0, -1.0, -3.0, -1.0]])

    loss, metrics = compute_policy_loss_sdc_tr(
        old_log_prob=old,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=Config(1.0, sdc_tr_degenerate_coef=0.75),
        success_log_prob=success_log_prob,
        failure_log_prob=failure_log_prob,
        failure_mask=failure_mask,
        reliability_gate=0.8,
    )

    for key in (
        "sdc_tr/degenerate_coef",
        "sdc_tr/mu_eff",
        "sdc_tr/degenerate_token_fraction",
        "sdc_tr/pseudo_advantage_abs_mean",
    ):
        assert key in metrics, key
        assert isinstance(metrics[key], float)

    assert metrics["sdc_tr/degenerate_coef"] == 0.75
    torch.testing.assert_close(metrics["sdc_tr/mu_eff"], 0.75 * 0.8)
    assert metrics["sdc_tr/degenerate_token_fraction"] == 0.5
    assert torch.isfinite(loss)


def test_config_default_degenerate_coef_is_zero():
    """The knob must be opt-in: the dataclass default keeps the loss unchanged."""
    from verl.workers.config.actor import PolicyLossConfig

    assert PolicyLossConfig().sdc_tr_degenerate_coef == 0.0


def test_all_wrong_group_loss_value_is_nonzero_off_ratio_one():
    """Away from ratio == 1 the repaired all-wrong group also has a nonzero loss."""
    old = torch.zeros(1, 2)
    log_prob = torch.log(torch.tensor([[1.05, 0.9]]))
    advantages = torch.zeros(1, 2)
    response_mask = torch.ones(1, 2, dtype=torch.bool)
    failure_mask = torch.tensor([True])
    success_log_prob = torch.tensor([[-1.0, -3.0]])
    failure_log_prob = torch.tensor([[-3.0, -1.0]])

    kwargs = dict(
        old_log_prob=old,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        success_log_prob=success_log_prob,
        failure_log_prob=failure_log_prob,
        failure_mask=failure_mask,
        reliability_gate=1.0,
    )
    off, _ = compute_policy_loss_sdc_tr(config=Config(1.0, sdc_tr_degenerate_coef=0.0), **kwargs)
    on, _ = compute_policy_loss_sdc_tr(config=Config(1.0, sdc_tr_degenerate_coef=0.5), **kwargs)
    assert off.item() == 0.0
    assert on.item() != 0.0
