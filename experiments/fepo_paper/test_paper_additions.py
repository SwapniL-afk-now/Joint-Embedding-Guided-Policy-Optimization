#!/usr/bin/env python3
"""Self-checks for every code addition the FEPO paper program depends on.

No ray, no GPU, no model. Covers the pieces where a silent failure would produce a
plausible-looking but wrong table cell:

  * forward-KL anchor fix   -- the estimator must actually recover KL(theta||anchor)
  * gate_mode               -- Table 3 "no gate" row
  * NGRPO / AVSPO           -- Table 1 baseline rows
  * instruction-following   -- Table 5 reward
  * config validation       -- the guards that stop a misconfigured 13 GPU-h run

    python experiments/fepo_paper/test_paper_additions.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verl.experimental.tafr_grpo.config import TAFRGRPOConfig, validate_tafr_config  # noqa: E402
from verl.experimental.tafr_grpo.tafr_loss import (  # noqa: E402
    compute_tafr_grpo_auxiliary_loss,
    replay_gate,
)
from verl.trainer.ppo.core_algos import (  # noqa: E402
    compute_avspo_outcome_advantage,
    compute_grpo_outcome_advantage,
    compute_ngrpo_outcome_advantage,
)
from verl.utils.reward_score.instruction_following import compute_score, compute_score_mmlu  # noqa: E402


# --------------------------------------------------------------------------- KL
def test_forward_anchor_kl_recovers_true_kl():
    """The whole point of the fix: k3 must estimate KL(pi_theta || pi_anchor).

    The legacy orientation evaluated to chi^2 - KL, which is not a divergence the
    paper's Proposition (well-posedness) applies to.
    """
    torch.manual_seed(0)
    V, N = 64, 200_000
    logp = torch.log_softmax(torch.randn(N, V), -1)
    logq = torch.log_softmax(torch.randn(1, V).expand(N, V), -1)
    a = torch.multinomial(logp.exp(), 1).squeeze(-1)
    lp = logp.gather(1, a[:, None]).squeeze()
    lq = logq.gather(1, a[:, None]).squeeze()
    true = (logp.exp() * (logp - logq)).sum(-1).mean()

    def k3(lr):
        return (torch.exp(lr) - 1 - lr).mean()

    forward, legacy = k3(lq - lp), k3(lp - lq)
    assert abs(forward - true) < 0.02 * true, f"forward {forward:.4f} != true {true:.4f}"
    assert legacy > 2 * true, "legacy orientation should be badly biased, not close"


def _kl_inputs(group_reward_mean):
    torch.manual_seed(0)
    n, t = len(group_reward_mean), 6
    return dict(
        log_prob=torch.randn(n, t, requires_grad=True) * 0.1 - 1.0,
        response_mask=torch.ones(n, t),
        group_reward_mean=torch.tensor(group_reward_mean),
        anchor_log_prob=torch.randn(n, t) * 0.1 - 1.0,
        replay_log_prob=torch.randn(n, t) * 0.1 - 1.0,
    )


def test_anchor_kl_is_non_negative():
    """A KL estimate that goes negative would silently flip the anchor into a push."""
    out = compute_tafr_grpo_auxiliary_loss(**_kl_inputs([0.5] * 8), beta=0.1, variant="anchor_only")
    assert out.metrics["tafr_grpo/kl_anchor"] >= 0.0


# ------------------------------------------------------------------------- gate
def test_kl_is_measured_for_every_variant():
    """Table 3's KL column must be populated on all five rows.

    variant=replay_only drops the anchor from the LOSS, but the ablation table still
    reports KL(actor, anchor) for that row -- so measurement must not be gated on the
    variant. This regressed once already.
    """
    kw = _kl_inputs([0.5] * 8)
    for v in ("full", "anchor_only", "replay_only"):
        m = compute_tafr_grpo_auxiliary_loss(**kw, beta=0.1, variant=v).metrics
        assert m["tafr_grpo/kl_anchor"] > 0.0, f"{v}: kl_anchor missing"
        assert m["tafr_grpo/kl_replay"] > 0.0, f"{v}: kl_replay missing"


def test_variant_still_controls_the_loss():
    """Measuring both KLs must not accidentally put both into the loss."""
    kw = _kl_inputs([0.5] * 8)
    full = compute_tafr_grpo_auxiliary_loss(**kw, beta=0.1, variant="full")
    a_only = compute_tafr_grpo_auxiliary_loss(**kw, beta=0.1, variant="anchor_only")
    r_only = compute_tafr_grpo_auxiliary_loss(**kw, beta=0.1, variant="replay_only")
    assert not torch.allclose(full.loss, a_only.loss)
    assert not torch.allclose(full.loss, r_only.loss)
    assert torch.allclose(a_only.loss + r_only.loss, full.loss, atol=1e-6)


def test_importance_weight_is_applied():
    """Eq. (iwk3): without w the estimator is biased once the actor leaves pi_old."""
    kw = _kl_inputs([0.5] * 8)
    old = kw["log_prob"].detach() - 0.5      # actor has moved well off pi_old
    with_iw = compute_tafr_grpo_auxiliary_loss(**kw, beta=0.1, old_log_prob=old)
    without = compute_tafr_grpo_auxiliary_loss(**kw, beta=0.1, old_log_prob=old,
                                               use_importance_weight=False)
    assert not torch.allclose(with_iw.loss, without.loss), "importance weight had no effect"
    assert abs(with_iw.metrics["tafr_grpo/iw_mean"] - 1.0) > 0.1, "w should not be ~1 here"
    assert without.metrics["tafr_grpo/iw_mean"] == 1.0


def test_estimator_diagnostics_are_reported():
    """The paper's diagnostics subsection needs all four of these."""
    kw = _kl_inputs([0.5] * 8)
    m = compute_tafr_grpo_auxiliary_loss(**kw, beta=0.1, old_log_prob=kw["log_prob"].detach()).metrics
    for k in ("iwk3_variance", "iw_clip_fraction", "logratio_clamp_fraction", "iw_mean"):
        assert f"tafr_grpo/{k}" in m, f"missing tafr_grpo/{k}"


def test_importance_weight_clipping_activates_and_is_counted():
    kw = _kl_inputs([0.5] * 8)
    old = kw["log_prob"].detach() - 5.0     # w = e^5 ~ 148, far above the clip
    m = compute_tafr_grpo_auxiliary_loss(**kw, beta=0.1, old_log_prob=old,
                                         importance_weight_clip=2.0).metrics
    assert m["tafr_grpo/iw_clip_fraction"] > 0.9, "clip should fire on nearly every token"
    assert m["tafr_grpo/iw_mean"] <= 2.0 + 1e-6, "clip must actually bound w"


def test_gate_modes():
    r = torch.tensor([0.0, 0.25, 0.5, 1.0])
    assert torch.allclose(replay_gate(r, "difficulty"), 1.0 - r)
    assert torch.allclose(replay_gate(r, "constant"), torch.ones_like(r)), "no-gate row must be g=1 everywhere"


def test_gate_mode_changes_the_loss():
    """Table 3's no-gate row must actually differ from full FEPO."""
    kw = _kl_inputs([0.0, 0.5, 0.75, 1.0])
    a = compute_tafr_grpo_auxiliary_loss(**kw, beta=0.5, gate_mode="difficulty")
    b = compute_tafr_grpo_auxiliary_loss(**kw, beta=0.5, gate_mode="constant")
    assert not torch.allclose(a.loss, b.loss), "gate_mode had no effect -- the ablation would be a duplicate run"


# -------------------------------------------------------------- NGRPO / AVSPO
def _adv_inputs():
    # 3 groups of 4: mixed, all-wrong, all-correct
    scores = [1.0, 0.0, 1.0, 0.0] + [0.0] * 4 + [1.0] * 4
    rewards = torch.zeros(12, 5)
    rewards[:, 0] = torch.tensor(scores)
    return rewards, torch.ones(12, 5), np.array(["g0"] * 4 + ["g1"] * 4 + ["g2"] * 4)


def test_grpo_gives_zero_on_homogeneous_groups():
    """The baseline failure mode the other two estimators exist to repair."""
    adv, _ = compute_grpo_outcome_advantage(*_adv_inputs())
    assert torch.allclose(adv[4:8], torch.zeros(4, 5)), "GRPO all-wrong group should be exactly zero"
    assert torch.allclose(adv[8:12], torch.zeros(4, 5)), "GRPO all-correct group should be exactly zero"


def test_ngrpo_repairs_all_wrong_groups():
    adv, _ = compute_ngrpo_outcome_advantage(*_adv_inputs())
    allwrong = adv[4:8, 0]
    assert not torch.allclose(allwrong, torch.zeros(4)), "NGRPO must give all-wrong groups a gradient"
    assert (allwrong < 0).all(), "all-wrong advantage must be negative (push the group down)"
    assert torch.allclose(adv[:4], compute_grpo_outcome_advantage(*_adv_inputs())[0][:4]), (
        "NGRPO must leave mixed groups identical to GRPO, or it is not a controlled comparison"
    )


def test_avspo_repairs_both_homogeneous_directions():
    adv, _ = compute_avspo_outcome_advantage(*_adv_inputs())
    assert (adv[4:8, 0] < 0).all(), "all-wrong should score below the batch mean"
    assert (adv[8:12, 0] > 0).all(), "all-correct should score above the batch mean"
    assert torch.allclose(adv[:4], compute_grpo_outcome_advantage(*_adv_inputs())[0][:4]), (
        "AVSPO must leave mixed groups identical to GRPO"
    )


def test_avspo_std_floor_bounds_the_advantage():
    """A fully homogeneous batch has zero reward std; without the floor this explodes."""
    rewards = torch.zeros(8, 3)
    adv, _ = compute_avspo_outcome_advantage(rewards, torch.ones(8, 3), np.array(["g0"] * 4 + ["g1"] * 4))
    assert torch.isfinite(adv).all() and adv.abs().max() < 100


# ------------------------------------------------------- instruction following
def test_hard_success_rate_is_all_or_nothing():
    specs = [{"type": "word_count", "count": 3, "relation": "at least"}, {"type": "no_comma"}]
    assert compute_score("alpha beta gamma delta", json.dumps(specs))["score"] == 1.0
    assert compute_score("alpha, beta gamma delta", json.dumps(specs))["score"] == 0.0, "one violation must zero it"
    assert compute_score("alpha", json.dumps(specs))["score"] == 0.0


def test_unsupported_constraint_fails_closed():
    """Never award credit for a constraint the verifier cannot check."""
    assert compute_score("anything", json.dumps([{"type": "detect_language"}]))["score"] == 0.0


def test_checkers_are_total_on_hostile_input():
    specs = json.dumps([{"type": "bullet_count", "count": 2}, {"type": "json_format"}])
    for bad in ["", "\x00\x00", "```json{{{", "*" * 5000]:
        assert compute_score(bad, specs)["score"] in (0.0, 1.0)


def test_reasoning_block_is_stripped_before_checking():
    specs = json.dumps([{"type": "no_comma"}])
    assert compute_score("<think>a, b, c</think>clean answer", specs)["score"] == 1.0, (
        "constraints must apply to the answer, not the reasoning trace"
    )


def test_mmlu_takes_the_last_letter():
    assert compute_score_mmlu("Let me think... A is wrong. The answer is C", "C")["score"] == 1.0
    assert compute_score_mmlu("The answer is B", "C")["score"] == 0.0


# ------------------------------------------------------------------ validation
def _cfg(**tafr):
    base = {"enable": True, "loss_version": "k3_legacy"}
    base.update(tafr)
    return {
        "custom_tafr_grpo": base,
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "actor_rollout_ref": {"actor": {"use_kl_loss": False}, "model": {}},
    }


def _rejects(cfg, why):
    try:
        validate_tafr_config(cfg)
    except ValueError:
        return
    raise AssertionError(why)


def test_config_defaults_are_the_corrected_ones():
    c = TAFRGRPOConfig()
    assert c.anchor_kl_direction == "forward" and c.gate_mode == "difficulty"
    assert c.diagnostic_only is False and c.negative_ce is False


def test_validation_rejects_bad_values():
    _rejects(_cfg(gate_mode="sometimes"), "bad gate_mode accepted")
    _rejects(_cfg(anchor_kl_direction="sideways"), "bad anchor_kl_direction accepted")
    _rejects(_cfg(negative_ce=True, diagnostic_only=True), "mutually exclusive pair accepted")
    _rejects(_cfg(negative_ce=True, loss_version="failure_token_adv"), "negative_ce on wrong loss_version accepted")


def test_diagnostic_only_permits_non_grpo_estimators():
    cfg = _cfg(diagnostic_only=True)
    cfg["algorithm"]["adv_estimator"] = "ngrpo"
    validate_tafr_config(cfg)  # must not raise: this is how m07/m08 get their KL column


def test_k3_legacy_composes_with_non_grpo_estimators():
    """FEPO[NGRPO] and FEPO[AVSPO] (Table 1 m09/m10): k3_legacy's gate and KL terms
    come from raw per-uid reward and log-probs only (verified: _tafr_group_reward_mean
    never touches the advantage tensor), so composing with a non-GRPO estimator is a
    real composition, not just a diagnostic measurement."""
    for est in ("ngrpo", "avspo"):
        cfg = _cfg(loss_version="k3_legacy")  # diagnostic_only defaults False here
        cfg["algorithm"]["adv_estimator"] = est
        validate_tafr_config(cfg)  # must not raise


def test_failure_token_adv_still_requires_grpo():
    """The RMS-matching path DOES read the advantage tensor directly -- unlike
    k3_legacy, it must stay restricted to adv_estimator=grpo."""
    cfg = _cfg(loss_version="failure_token_adv")
    cfg["algorithm"]["adv_estimator"] = "ngrpo"
    _rejects(cfg, "failure_token_adv accepted a non-grpo estimator")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(tests)} passed")
