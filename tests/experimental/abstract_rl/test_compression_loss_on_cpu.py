# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Compression KL: masking, weighting, and gradient isolation."""

import pytest
import torch

from verl.experimental.abstract_rl.loss import compute_compression_loss, compute_span_diagnostics
from verl.trainer.ppo.core_algos import kl_penalty

B, T = 3, 8


def _fixture(requires_grad=True):
    torch.manual_seed(0)
    log_prob = torch.randn(B, T, requires_grad=requires_grad)
    prior = torch.randn(B, T)
    abstract_mask = torch.zeros(B, T, dtype=torch.bool)
    abstract_mask[:, 5:7] = True  # A = positions 5,6 in every row
    reasoning_mask = torch.zeros(B, T, dtype=torch.bool)
    reasoning_mask[:, :4] = True
    response_mask = torch.ones(B, T, dtype=torch.bool)
    return log_prob, prior, abstract_mask, reasoning_mask, response_mask


def test_gradient_reaches_only_abstraction_tokens():
    log_prob, prior, abstract_mask, _, _ = _fixture()
    row_weight = torch.ones(B)
    loss, _ = compute_compression_loss(log_prob, prior, abstract_mask, row_weight)
    loss.backward()

    grad = log_prob.grad
    assert (grad[abstract_mask] != 0).any(), "abstraction tokens must receive compression gradient"
    assert torch.equal(grad[~abstract_mask], torch.zeros_like(grad[~abstract_mask])), (
        "no compression gradient outside A"
    )


def test_no_gradient_reaches_the_prior():
    log_prob, prior, abstract_mask, _, _ = _fixture()
    prior.requires_grad_(True)
    loss, _ = compute_compression_loss(log_prob, prior, abstract_mask, torch.ones(B))
    loss.backward()
    assert prior.grad is None, "the question-only prior must be stop-gradiented"


def test_incorrect_or_invalid_rows_contribute_nothing():
    log_prob, prior, abstract_mask, _, _ = _fixture()
    # only row 1 is correct AND has a valid abstraction
    row_weight = torch.tensor([0.0, 1.0, 0.0])
    loss, metrics = compute_compression_loss(log_prob, prior, abstract_mask, row_weight)
    loss.backward()

    grad = log_prob.grad
    assert (grad[1, 5:7] != 0).any()
    assert torch.equal(grad[0], torch.zeros(T)) and torch.equal(grad[2], torch.zeros(T))
    assert float(metrics["abstract_kl_tokens"]) == 2.0


def test_matches_the_masked_k3_estimator():
    log_prob, prior, abstract_mask, _, _ = _fixture(requires_grad=False)
    row_weight = torch.tensor([1.0, 0.0, 1.0])
    loss, _ = compute_compression_loss(log_prob, prior, abstract_mask, row_weight)

    kld = kl_penalty(logprob=log_prob, ref_logprob=prior, kl_penalty="low_var_kl")
    weight = abstract_mask.float() * row_weight.unsqueeze(-1)
    expected = (kld * weight).sum() / weight.sum()
    assert loss == pytest.approx(float(expected), abs=1e-6)


def test_kl_clip_caps_outlier_tokens_and_lowers_the_gradient():
    """A token whose k3 sits well inside the shared +-10 clamp (so this is testing
    kl_clip itself, not core_algos' own clamp) should have its loss AND gradient
    contribution suppressed once kl_clip cuts below that value.
    """
    torch.manual_seed(0)
    log_prob = torch.randn(1, T, requires_grad=True)
    prior = torch.randn(1, T)
    with torch.no_grad():
        prior[0, 5] = log_prob[0, 5].detach() + 2.0  # k3 = exp(2)-2-1 ~= 4.4, unsaturated
    abstract_mask = torch.zeros(1, T, dtype=torch.bool)
    abstract_mask[:, 5:7] = True
    row_weight = torch.ones(1)

    unclipped, _ = compute_compression_loss(log_prob, prior, abstract_mask, row_weight, kl_clip=0.0)
    unclipped.backward()
    grad_unclipped = log_prob.grad[0, 5].item()

    log_prob.grad = None
    clipped, _ = compute_compression_loss(log_prob, prior, abstract_mask, row_weight, kl_clip=2.0)
    clipped.backward()
    grad_clipped = log_prob.grad[0, 5].item()

    assert clipped < unclipped, "capping the outlier token must lower the batch loss"
    assert grad_unclipped != 0, "sanity: the unclipped token has a real gradient below the shared clamp"
    assert grad_clipped == 0, "clamped past kl_clip, the gradient at that token must vanish"


def test_empty_batch_is_zero_and_finite():
    log_prob, prior, abstract_mask, _, _ = _fixture()
    for mask, weight in ((torch.zeros_like(abstract_mask), torch.ones(B)), (abstract_mask, torch.zeros(B))):
        loss, metrics = compute_compression_loss(log_prob, prior, mask, weight)
        assert float(loss) == 0.0 and torch.isfinite(loss)
        assert float(metrics["compression_loss"]) == 0.0
        loss.backward()  # must stay differentiable so every rank contributes
        assert torch.isfinite(log_prob.grad).all()
        log_prob.grad = None


def test_nan_free_with_degenerate_inputs():
    log_prob, prior, abstract_mask, _, _ = _fixture(requires_grad=False)
    prior = torch.full_like(prior, -40.0)  # far-apart distributions
    loss, _ = compute_compression_loss(log_prob, prior, abstract_mask, torch.ones(B))
    assert torch.isfinite(loss)


def test_grpo_gradient_reaches_both_spans():
    """The abstraction stays inside the GRPO loss via the unchanged response_mask."""
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    from verl.workers.config.actor import ActorConfig

    log_prob, _, abstract_mask, reasoning_mask, response_mask = _fixture()
    config = ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1)
    pg_loss, _ = compute_policy_loss_vanilla(
        old_log_prob=log_prob.detach(),
        log_prob=log_prob,
        advantages=torch.ones(B, T),
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=config,
    )
    pg_loss.backward()
    assert (log_prob.grad[reasoning_mask] != 0).any(), "GRPO must update reasoning tokens"
    assert (log_prob.grad[abstract_mask] != 0).any(), "GRPO must update abstraction tokens"


def test_span_diagnostics_report_participation():
    log_prob, _, abstract_mask, reasoning_mask, response_mask = _fixture(requires_grad=False)
    old_log_prob = log_prob + 0.1
    out = compute_span_diagnostics(
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        entropy=torch.ones(B, T),
        reasoning_mask=reasoning_mask,
        abstract_mask=abstract_mask,
        response_mask=response_mask,
        clip_low=0.2,
        clip_high=0.2,
    )
    assert out["abstract_gradient_token_fraction"] == pytest.approx(2 / T)
    assert out["reasoning_gradient_token_fraction"] == pytest.approx(4 / T)
    assert out["abstract_policy_ratio_mean"] == pytest.approx(float(torch.exp(torch.tensor(-0.1))), abs=1e-5)
    assert out["abstract_entropy"] == pytest.approx(1.0)
    assert all(torch.isfinite(v) for v in out.values())


def test_span_diagnostics_survive_empty_masks():
    log_prob, _, _, _, response_mask = _fixture(requires_grad=False)
    empty = torch.zeros(B, T, dtype=torch.bool)
    out = compute_span_diagnostics(
        log_prob=log_prob,
        old_log_prob=log_prob,
        entropy=None,
        reasoning_mask=empty,
        abstract_mask=empty,
        response_mask=response_mask,
        clip_low=0.2,
        clip_high=0.2,
    )
    assert all(torch.isfinite(v) for v in out.values())
    assert out["abstract_gradient_token_fraction"] == 0.0


# --- format warm-up SFT -------------------------------------------------------


def test_sft_loss_gradient_reaches_only_the_abstraction():
    from verl.experimental.abstract_rl.loss import compute_abstract_sft_loss

    log_prob, _, abstract_mask, _, _ = _fixture()
    loss, metrics = compute_abstract_sft_loss(log_prob, abstract_mask, torch.ones(B))
    loss.backward()
    grad = log_prob.grad
    assert (grad[abstract_mask] != 0).any(), "the warm-up must train the abstraction span"
    assert torch.equal(grad[~abstract_mask], torch.zeros_like(grad[~abstract_mask])), (
        "the warm-up must not fine-tune the reasoning trace"
    )
    assert float(metrics["sft_tokens"]) == float(abstract_mask.sum())


def test_sft_loss_is_negative_log_likelihood():
    from verl.experimental.abstract_rl.loss import compute_abstract_sft_loss

    log_prob, _, abstract_mask, _, _ = _fixture(requires_grad=False)
    row_weight = torch.tensor([1.0, 0.0, 1.0])
    loss, _ = compute_abstract_sft_loss(log_prob, abstract_mask, row_weight)
    weight = abstract_mask.float() * row_weight.unsqueeze(-1)
    expected = -(log_prob * weight).sum() / weight.sum()
    assert loss == pytest.approx(float(expected), abs=1e-6)


def test_sft_loss_skips_incorrect_and_invalid_rows():
    from verl.experimental.abstract_rl.loss import compute_abstract_sft_loss

    log_prob, _, abstract_mask, _, _ = _fixture()
    compute_abstract_sft_loss(log_prob, abstract_mask, torch.tensor([0.0, 1.0, 0.0]))[0].backward()
    assert (log_prob.grad[1] != 0).any()
    assert torch.equal(log_prob.grad[0], torch.zeros(T)) and torch.equal(log_prob.grad[2], torch.zeros(T))


def test_sft_loss_is_zero_and_safe_with_nothing_to_imitate():
    from verl.experimental.abstract_rl.loss import compute_abstract_sft_loss

    log_prob, _, abstract_mask, _, _ = _fixture()
    loss, metrics = compute_abstract_sft_loss(log_prob, abstract_mask, torch.zeros(B))
    assert float(loss) == 0.0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(log_prob.grad).all()


def test_warmup_optimizer_swaps_lr_and_clip_then_restores():
    from verl.experimental.abstract_rl.trainer_hooks import sft_warmup_optimizer

    class _Cfg:
        clip_grad = 1.0

    class _Engine:
        optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1e-6)
        optimizer_config = _Cfg()

    engine = _Engine()
    warm = {"sft_warmup": True, "sft_lr": 1e-5, "sft_clip_grad": 20.0}
    with sft_warmup_optimizer(engine, warm):
        assert engine.optimizer.param_groups[0]["lr"] == 1e-5
        assert engine.optimizer_config.clip_grad == 20.0
    assert engine.optimizer.param_groups[0]["lr"] == 1e-6
    assert engine.optimizer_config.clip_grad == 1.0

    # RL steps and the disabled path must not touch the optimizer at all
    for cfg in ({"sft_warmup": False, "sft_lr": 1e-5, "sft_clip_grad": 20.0}, None, {}):
        with sft_warmup_optimizer(engine, cfg):
            assert engine.optimizer.param_groups[0]["lr"] == 1e-6
            assert engine.optimizer_config.clip_grad == 1.0

    # restored even if the update raises
    with pytest.raises(RuntimeError), sft_warmup_optimizer(engine, warm):
        raise RuntimeError("boom")
    assert engine.optimizer.param_groups[0]["lr"] == 1e-6
    assert engine.optimizer_config.clip_grad == 1.0


def test_warmup_aborts_when_the_solver_degrades():
    from verl.experimental.abstract_rl.bootstrap import BootstrapState
    from verl.experimental.abstract_rl.config import AbstractRLConfig
    from verl.experimental.abstract_rl.trainer_hooks import config_dict

    cfg = AbstractRLConfig(
        enable=True, bootstrap_steps=150, bootstrap_sft_steps=30, bootstrap_sft_accuracy_drop=0.08
    )
    st = BootstrapState(cfg)
    for a in (0.56, 0.55, 0.57, 0.56, 0.55):  # baseline window
        st.observe_accuracy(a)
    assert config_dict(cfg, 5, st)["sft_warmup"] is True
    for a in (0.40, 0.38, 0.39, 0.40, 0.38):  # sustained collapse
        st.observe_accuracy(a)
    assert st.sft_disabled and config_dict(cfg, 10, st)["sft_warmup"] is False
    # latched: recovery does not re-enable it
    st.observe_accuracy(0.60)
    assert config_dict(cfg, 11, st)["sft_warmup"] is False

    # A single low outlier is noise, not damage. This is the measured case that
    # killed a real run: 0.570 -> 0.443 -> 0.551 on consecutive steps, with
    # held-out pass@1 showing no degradation at all.
    noisy = BootstrapState(cfg)
    for a in (0.570, 0.570, 0.551, 0.594, 0.443, 0.551):
        noisy.observe_accuracy(a)
    assert not noisy.sft_disabled, "a one-step dip must not abort the warm-up"

    # abstractrl-warm30-20260806 steps 1-10 verbatim, at the shipped default
    # tolerance. It aborted at step 7 and then ran 240 steps with grad_norm 0,
    # yet accuracy over the following 240 steps averaged 0.4995 -- pure noise.
    real = BootstrapState(AbstractRLConfig(enable=True, bootstrap_steps=150, bootstrap_sft_steps=30))
    for a in (0.5605, 0.5664, 0.541, 0.5781, 0.4531, 0.5332, 0.543, 0.5391, 0.5195, 0.5508):
        real.observe_accuracy(a)
    assert not real.sft_disabled, "the measured step 1-10 trace must not abort the warm-up"

    # guard off -> never aborts
    off = AbstractRLConfig(enable=True, bootstrap_steps=150, bootstrap_sft_steps=30, bootstrap_sft_accuracy_drop=0.0)
    st2 = BootstrapState(off)
    for a in (0.56, 0.56, 0.56, 0.01, 0.01, 0.01):
        st2.observe_accuracy(a)
    assert config_dict(off, 6, st2)["sft_warmup"] is True


def test_warmup_extends_until_the_policy_emits_natively():
    from verl.experimental.abstract_rl.bootstrap import BootstrapState
    from verl.experimental.abstract_rl.config import AbstractRLConfig
    from verl.experimental.abstract_rl.trainer_hooks import config_dict

    cfg = AbstractRLConfig(
        enable=True, bootstrap_steps=150, bootstrap_sft_steps=30, bootstrap_native_threshold=0.1, bootstrap_patience=2
    )
    st = BootstrapState(cfg)
    # past bootstrap_sft_steps but still emitting nothing -> keep warming up
    assert config_dict(cfg, 40, st)["sft_warmup"] is True
    for _ in range(cfg.bootstrap_patience):
        st.observe(0.3)
    assert st.disabled and config_dict(cfg, 41, st)["sft_warmup"] is False
    # bootstrap_steps is still the hard ceiling: no grafts, so no SFT targets
    assert config_dict(cfg, 151, BootstrapState(cfg))["sft_warmup"] is False


def test_warmup_aborts_when_abstracts_collapse_before_accuracy_moves():
    from verl.experimental.abstract_rl.bootstrap import BootstrapState
    from verl.experimental.abstract_rl.config import AbstractRLConfig
    from verl.experimental.abstract_rl.trainer_hooks import config_dict

    cfg = AbstractRLConfig(
        enable=True, bootstrap_steps=150, bootstrap_sft_steps=30, bootstrap_sft_min_length_fraction=0.6
    )
    # Length-only, so this asserts the length guard on its own rather than racing
    # the accuracy guard. Numbers are the measured lr=3e-6 collapse.
    st = BootstrapState(cfg)
    st.observe_abstract_length(80.4)
    st.observe_abstract_length(60.3)
    assert not st.sft_disabled, "60/80 is above the floor -- must not fire yet"
    st.observe_abstract_length(41.8)  # 0.52 of baseline
    assert st.sft_disabled and st.sft_abort_reason == "length"
    assert config_dict(cfg, 3, st)["sft_warmup"] is False

    off = AbstractRLConfig(enable=True, bootstrap_steps=150, bootstrap_sft_steps=30)
    off.bootstrap_sft_min_length_fraction = 0.0
    st2 = BootstrapState(off)
    st2.observe_abstract_length(79.3)
    st2.observe_abstract_length(1.0)
    assert not st2.sft_disabled


def test_short_abstracts_are_rejected_as_degenerate():
    from verl.experimental.abstract_rl.bootstrap import is_degenerate

    assert is_degenerate("Use algebra.")  # the 3.7-token collapse mode
    assert not is_degenerate(
        "Set up an equation relating the unknown to the given totals, then solve for the unknown and verify it."
    )


def test_warmup_flag_follows_the_step():
    from verl.experimental.abstract_rl.config import AbstractRLConfig
    from verl.experimental.abstract_rl.trainer_hooks import config_dict

    cfg = AbstractRLConfig(enable=True, bootstrap_steps=150, bootstrap_sft_steps=30)
    assert config_dict(cfg, 1)["sft_warmup"] is True
    assert config_dict(cfg, 30)["sft_warmup"] is True
    assert config_dict(cfg, 31)["sft_warmup"] is False
    assert config_dict(AbstractRLConfig(enable=True), 1)["sft_warmup"] is False
