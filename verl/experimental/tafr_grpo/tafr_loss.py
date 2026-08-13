# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch


@dataclass(frozen=True)
class TAFRLossOutput:
    loss: torch.Tensor
    metrics: dict[str, float]
    # The two halves, already group-averaged but NOT yet weighted by b_a/b_r. Exposed
    # so a caller can substitute one half (the Table 7 negative-CE control swaps out
    # the replay push) without recomputing the other.
    anchor_loss: Optional[torch.Tensor] = None
    replay_loss: Optional[torch.Tensor] = None


def response_length_normalized_mean(values: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    mask = response_mask.to(dtype=values.dtype)
    lengths = mask.sum(dim=-1).clamp_min(1.0)
    return (values * mask).sum(dim=-1) / lengths


def replay_gate(group_reward_mean: torch.Tensor, gate_mode: str = "difficulty") -> torch.Tensor:
    """g_x, the weight on the replay push.

    "difficulty" is the paper's gate g_x = 1 - p_x, making the push proportional to
    observed failure. "constant" pins g = 1 for every group: the Table 3 ablation
    that tests whether conditioning on empirical failure frequency is necessary at
    all, or whether an always-on push does just as well.
    """
    if gate_mode == "constant":
        return torch.ones_like(group_reward_mean)
    return 1.0 - group_reward_mean


def shuffle_within_rows(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Permute each row's valid entries among its own valid positions.

    Ablation control: preserves every marginal property A_reg is normalized on --
    per-row mean, per-row std, global RMS -- and destroys ONLY the alignment
    between the signal and the token it was computed at. If training with the
    shuffled advantage matches the real one, the per-token structure carries no
    error-localization information and the mechanism is decorative.
    """
    out = values.clone()
    for i in range(values.shape[0]):
        idx = mask[i].nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() > 1:
            out[i, idx] = values[i, idx[torch.randperm(idx.numel(), device=values.device)]]
    return out


def compute_regularization_token_advantage(
    *,
    anchor_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    failure_log_prob: torch.Tensor,
    grpo_advantage: torch.Tensor,
    group_reward_mean: torch.Tensor,
    response_mask: torch.Tensor,
    advantage_clip: float,
    failure_model_ready: bool,
    shuffle_tokens: bool = False,
    anchor_weight: float = 1.0,
    failure_weight: float = 1.0,
    normalize_terms_separately: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Construct the detached paper TAFR regularization advantage.

    A_reg = log(pi_anchor / pi_old)
          + c_x * log(pi_old / pi_failure),  c_x = 1 - group_reward_mean.

    The complete A_reg is RMS-matched to GRPO once on the logical batch. The
    outer beta controls the requested relative strength (beta=0.5 targets 50%).
    """
    mask = response_mask.to(dtype=torch.float32)
    reward_mean = group_reward_mean.float().reshape(-1).clamp(0.0, 1.0)
    difficulty = 1.0 - reward_mean
    if not failure_model_ready:
        difficulty = torch.zeros_like(difficulty)
    difficulty = difficulty.unsqueeze(-1)

    stability_signal = anchor_log_prob.detach().float() - old_log_prob.detach().float()
    failure_signal = old_log_prob.detach().float() - failure_log_prob.detach().float()
    failure_component = difficulty * failure_signal

    valid_tokens = mask.sum().clamp_min(1.0)

    def _rms(values: torch.Tensor) -> torch.Tensor:
        return ((values.square() * mask).sum() / valid_tokens).sqrt()

    anchor_part, failure_part = stability_signal, failure_component
    # Natural magnitudes, before any weighting or normalization. These are what
    # drift on their own (anchor ramps between refreshes, failure grows with SFT),
    # so watch them to choose the weights.
    raw_anchor_rms = _rms(stability_signal * mask)
    raw_failure_rms = _rms(failure_component * mask)
    if normalize_terms_separately:
        # Without this the mix is set by raw magnitudes, which drift for unrelated
        # reasons: the anchor term ramps between refreshes (frozen anchor, moving
        # policy) and the failure term grows as the failure model accumulates SFT
        # updates. Normalizing each to unit RMS first makes the realized mix exactly
        # anchor_weight : failure_weight.
        a_rms, f_rms = _rms(anchor_part), _rms(failure_part)
        anchor_part = torch.where(a_rms > 1.0e-8, anchor_part / a_rms.clamp_min(1.0e-8), torch.zeros_like(anchor_part))
        failure_part = torch.where(
            f_rms > 1.0e-8, failure_part / f_rms.clamp_min(1.0e-8), torch.zeros_like(failure_part)
        )

    anchor_contribution = float(anchor_weight) * anchor_part * mask
    failure_contribution = float(failure_weight) * failure_part * mask
    raw_advantage = anchor_contribution + failure_contribution
    if shuffle_tokens:
        # Shuffle AFTER composing and BEFORE normalizing, so the arm differs from
        # the real one only in token alignment.
        raw_advantage = shuffle_within_rows(raw_advantage, response_mask.bool()) * mask

    valid = mask.sum().clamp_min(1.0)
    # Match the TAIL, not the RMS. A_reg is a heavy-tailed log-prob difference
    # (p50 ~2e-4, p99 ~3.5) while GRPO's advantage is flat, so RMS-matching leaves
    # the top 1% of tokens at ~4.5x GRPO -- overriding the outcome signal on exactly
    # the tokens it fires on. Scaling by |A_reg|'s p99 puts the tail at GRPO's scale
    # by construction, so A_reg shapes within a rollout without ever dominating it.
    sel = raw_advantage[mask.bool()].abs()
    raw_rms = torch.quantile(sel.float(), 0.99) if sel.numel() > 1 else sel.sum()
    grpo = grpo_advantage.detach().float() * mask
    grpo_rms = ((grpo.square() * mask).sum() / valid).sqrt()
    # Homogeneous batches have zero GRPO advantage but are exactly where TAFR
    # must still learn, so use GRPOs unit-normalized reference scale there.
    reference_rms = torch.where(grpo_rms > 1.0e-8, grpo_rms, torch.ones_like(grpo_rms))
    scale = torch.where(raw_rms > 1.0e-8, reference_rms / raw_rms.clamp_min(1.0e-8), torch.zeros_like(raw_rms))
    scaled_advantage = raw_advantage * scale
    # After scaling, A_reg's RMS equals GRPO's by construction -- but A_reg is sparse
    # and spiky where GRPO's is flat, so its tail tokens still overwhelm the outcome
    # signal. An absolute clip of 5.0 is ~7x GRPO's per-token magnitude. Expressing
    # the cap in units of the GRPO advantage keeps A_reg subordinate per token, not
    # merely on average, and keeps that guarantee when GRPO's scale changes.
    clip_limit = advantage_clip  # now a rare safety net, not the alignment mechanism
    advantage = scaled_advantage.clamp(min=-clip_limit, max=clip_limit) * mask
    advantage = advantage.detach()

    stats = {
        "difficulty": difficulty.detach(),
        "stability_signal": stability_signal.detach(),
        "failure_signal": failure_signal.detach(),
        "failure_component": failure_component.detach(),
        "raw_advantage": raw_advantage.detach(),
        "scaled_advantage": scaled_advantage.detach(),
        "advantage": advantage,
        # Realized mix, so the anchor:failure balance is observable rather than inferred.
        "raw_anchor_rms": raw_anchor_rms.detach(),
        "raw_failure_rms": raw_failure_rms.detach(),
        "anchor_contribution_rms": _rms(anchor_contribution).detach(),
        "failure_contribution_rms": _rms(failure_contribution).detach(),
        "raw_advantage_rms": raw_rms.detach(),
        "grpo_advantage_rms": grpo_rms.detach(),
        "reference_advantage_rms": reference_rms.detach(),
        "normalization_scale": scale.detach(),
        "advantage_clip_limit": torch.as_tensor(clip_limit).detach(),
    }
    return advantage, stats


def masked_loss_aggregate(
    token_loss: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    global_batch_info: Optional[dict] = None,
) -> torch.Tensor:
    """Aggregate a [batch, response_length] token loss with the verl GRPO modes.

    Keep the auxiliary term on exactly the same normalization convention as
    GRPO, including DP/global-batch scaling.
    """
    from verl.trainer.ppo.core_algos import agg_loss

    return agg_loss(
        loss_mat=token_loss,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **(global_batch_info or {}),
    )


def compute_regularization_policy_loss(
    *,
    current_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    regularization_advantage: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio: float,
    loss_agg_mode: str,
    global_batch_info: Optional[dict] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute a PPO-clipped token policy loss from detached A_reg.

    The clipping decision is fully independent of the GRPO clipping: only the
    This helper is diagnostic; training fuses beta-scaled A_reg into A_GRPO before one PPO clip.
    """
    old_log_prob_d = old_log_prob.detach()
    log_ratio = current_log_prob - old_log_prob_d
    ratio = torch.exp(log_ratio)

    clipped_ratio = ratio.clamp(min=1.0 - clip_ratio, max=1.0 + clip_ratio)

    unclipped_objective = ratio * regularization_advantage
    clipped_objective = clipped_ratio * regularization_advantage

    token_loss = -torch.minimum(unclipped_objective, clipped_objective)

    regularization_loss = masked_loss_aggregate(
        token_loss,
        response_mask,
        loss_agg_mode,
        global_batch_info=global_batch_info,
    )

    mask = response_mask.to(dtype=regularization_advantage.dtype)
    valid = mask.sum().clamp_min(1.0)
    stats = {
        "regularization_loss": regularization_loss.detach(),
        "token_loss": token_loss.detach(),
        "ppo_ratio": ratio.detach(),
        "ppo_ratio_clip_fraction": (
            ((ratio < (1.0 - clip_ratio)) | (ratio > (1.0 + clip_ratio))).to(regularization_advantage.dtype) * mask
        ).sum() / valid,
    }
    return regularization_loss, stats


def _group_mean(values: torch.Tensor, group_ids: Optional[Iterable[object]]) -> torch.Tensor:
    if values.numel() == 0:
        return values.sum()
    if group_ids is None:
        return values.mean()
    ids = list(group_ids)
    if len(ids) != values.shape[0]:
        return values.mean()
    grouped = []
    for uid in dict.fromkeys(ids):
        idx = [i for i, item in enumerate(ids) if item == uid]
        if idx:
            grouped.append(values[torch.as_tensor(idx, device=values.device, dtype=torch.long)].mean())
    if not grouped:
        return values.mean()
    return torch.stack(grouped).mean()


def compute_tafr_grpo_auxiliary_loss(
    *,
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    group_reward_mean: torch.Tensor,
    beta: float,
    beta_anchor: Optional[float] = None,
    beta_replay: Optional[float] = None,
    variant: str = "full",
    anchor_log_prob: Optional[torch.Tensor] = None,
    replay_log_prob: Optional[torch.Tensor] = None,
    group_ids: Optional[Iterable[object]] = None,
    gate_mode: str = "difficulty",
    anchor_kl_direction: str = "forward",
    old_log_prob: Optional[torch.Tensor] = None,
    use_importance_weight: bool = True,
    importance_weight_clip: float = 10.0,
    sequence_level: bool = False,
) -> TAFRLossOutput:
    """Compute TAFR-GRPO KL terms on the pi_old rollout samples.

    All rows are actor (pi_old) rollout samples — no separate replay rows.
    Both anchor and replay log-probs are evaluated on the same y_i ~ pi_old.

    Both terms are FORWARD KLs from the actor to a fixed reference, matching
    Eq. (state_kls) of the paper:

        Anchor KL:   D_KL(pi_theta || pi_anchor)
        Replay KL:   D_KL(pi_theta || pi_failure)

    The k3 estimator (rho - 1 - log rho) recovers D_KL(p || q) from samples of p
    only when rho = q/p, i.e. the log-ratio is (reference - actor). The replay term
    always used that form. The anchor term historically used its inverse
    (actor - reference), which is not a KL in either direction -- it evaluates to
    chi^2 - KL, measured at 4.65 against a true KL of 1.03. anchor_kl_direction
    ="legacy" reproduces that older behaviour for A/B comparison against runs made
    before the fix; "forward" is correct and is the default.

    Actor loss adds:
        + b_a * anchor_kl   (stay close to stable anchor)
        - b_r * (1 - r_bar_x) * replay_kl   (move away from failure modes)

    b_a/b_r default to `beta` (single-knob behaviour). Pass beta_anchor/beta_replay
    to weight the two opposing pushes independently -- with one knob, raising the
    replay push always raises the anchor pull that cancels it.

    Only log_prob carries actor gradients; anchor_log_prob and replay_log_prob
    must be detached (frozen models).
    """

    if variant not in {"full", "anchor_only", "replay_only"}:
        raise ValueError("variant must be 'full', 'anchor_only', or 'replay_only'.")

    group_reward_mean = group_reward_mean.to(device=log_prob.device, dtype=log_prob.dtype).clamp(0.0, 1.0)
    gate = replay_gate(group_reward_mean, gate_mode)

    zero = log_prob.sum() * 0.0
    anchor_loss = zero
    replay_loss = zero

    anchor_kl_metric = 0.0
    replay_kl_metric = 0.0
    actor_lp_metric = 0.0
    anchor_lp_metric = 0.0
    replay_lp_metric = 0.0
    actor_lp_on_replay_metric = 0.0
    kl_var_metric = 0.0
    iw_clip_fraction = 0.0
    clamp_fraction = 0.0
    iw_mean_metric = 1.0

    # Importance weight w = pi_theta(a)/pi_old(a) for Eq. (iwk3) of the paper. The
    # actions were drawn from pi_old, so without w the k3 estimator is biased for
    # D_KL(pi_theta || r) as soon as the actor moves off pi_old -- which it does within
    # every PPO epoch. Gradients are deliberately retained through w (Cor. iwk3_gradient).
    if use_importance_weight and old_log_prob is not None:
        old_lp = old_log_prob.to(device=log_prob.device, dtype=log_prob.dtype).detach()
        iw_log_ratio = log_prob - old_lp
        iw = torch.exp(iw_log_ratio.clamp(min=-20, max=20))
        if importance_weight_clip and importance_weight_clip > 0:
            clipped = iw.clamp(max=float(importance_weight_clip))
            iw_clip_fraction = float(
                (((iw > float(importance_weight_clip)).to(response_mask.dtype) * response_mask).sum()
                 / response_mask.sum().clamp_min(1.0)).detach().cpu()
            )
            iw = clipped
        iw_mean_metric = float(
            ((iw.detach() * response_mask).sum() / response_mask.sum().clamp_min(1.0)).cpu()
        )
    else:
        iw_log_ratio = torch.zeros_like(log_prob)
        iw = torch.ones_like(log_prob)

    def _k3(log_ratio: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Return a per-response k3 estimate.

        GSPO uses a geometric (sequence-level) importance ratio.  Its FEPO companion
        must not add an independently clipped token-level policy objective, so this
        path applies k3 to the length-normalized sequence log-ratio as well.  This is
        deliberately the GSPO-style geometric ratio, not exp(sum(token log-ratios)),
        which is unusably high variance for long completions.
        """
        if sequence_level:
            seq_log_ratio = response_length_normalized_mean(log_ratio, response_mask)
            seq_iw = response_length_normalized_mean(iw_log_ratio, response_mask).exp()
            clamped = seq_log_ratio.clamp(min=-20, max=20)
            frac = float((seq_log_ratio != clamped).float().mean().detach().cpu())
            return (seq_iw * ((torch.exp(clamped) - 1) - clamped)).clamp(min=-10, max=10), frac

        clamped = log_ratio.clamp(min=-20, max=20)
        frac = float(
            (((log_ratio != clamped).to(response_mask.dtype) * response_mask).sum()
             / response_mask.sum().clamp_min(1.0)).detach().cpu()
        )
        per_token = (iw * ((torch.exp(clamped) - 1) - clamped)).clamp(min=-10, max=10)
        return response_length_normalized_mean(per_token, response_mask), frac

    # Both KLs are MEASURED whenever their reference log-probs exist; `variant` only
    # decides which enters the loss. Otherwise the Table 3 "failure repulsion only"
    # row (variant=replay_only) would report KL(actor, anchor) = 0 and leave that
    # column of the ablation empty.
    if anchor_log_prob is not None:
        frozen_anchor = anchor_log_prob.to(device=log_prob.device, dtype=log_prob.dtype).detach()
        # rho = pi_anchor/pi_theta gives D_KL(pi_theta || pi_anchor) -- the same
        # orientation the replay term below uses. See the docstring for why the
        # inverted "legacy" form is not a KL.
        if anchor_kl_direction == "legacy":
            log_ratio_anchor = log_prob - frozen_anchor
        else:
            log_ratio_anchor = frozen_anchor - log_prob
        anchor_seq_kl, anchor_clamp = _k3(log_ratio_anchor)
        clamp_fraction = max(clamp_fraction, anchor_clamp)
        anchor_kl_metric = float(anchor_seq_kl.detach().mean().cpu())
        kl_var_metric = float(anchor_seq_kl.detach().var(unbiased=False).cpu()) if anchor_seq_kl.numel() > 1 else 0.0
        actor_lp_metric = float(response_length_normalized_mean(log_prob.detach(), response_mask).mean().cpu())
        anchor_lp_metric = float(response_length_normalized_mean(frozen_anchor, response_mask).mean().cpu())
        if variant in {"full", "anchor_only"}:
            # REVERTED: tried gating this symmetrically with replay_loss (theory:
            # anchor's full-strength pull was unopposed on all-correct groups). It
            # matched GRPO's mean val over steps 100-190 (0.6567 vs 0.6568) but ran
            # much noisier than either GRPO or the old gated baseline -- val range
            # 0.076 over steps 250-260 vs ~0.02-0.026 for those two -- and the
            # instability wasn't damping between checkpoints. Reverting to
            # ungated: only replay_loss (the failure/repulsion term) is gated by
            # difficulty; anchor_loss stays full-strength always, as in every
            # prior stable run.
            anchor_loss = _group_mean(anchor_seq_kl, group_ids)

    if replay_log_prob is not None:
        frozen_replay = replay_log_prob.to(device=log_prob.device, dtype=log_prob.dtype).detach()
        replay_seq_kl, replay_clamp = _k3(frozen_replay - log_prob)
        clamp_fraction = max(clamp_fraction, replay_clamp)
        replay_kl_metric = float(replay_seq_kl.detach().mean().cpu())
        replay_lp_metric = float(response_length_normalized_mean(frozen_replay, response_mask).mean().cpu())
        actor_lp_on_replay_metric = float(response_length_normalized_mean(log_prob.detach(), response_mask).mean().cpu())
        if variant in {"full", "replay_only"}:
            # REVERTED (measured regression): ungating this made replay_loss fire at
            # full strength on all-correct groups too, on the theory that it would
            # self-cancel against anchor_loss per-token wherever pi_anchor(t) ~=
            # pi_failure(t). That assumption isn't enforced anywhere -- the failure
            # clone can diverge from the anchor model broadly, not just at error
            # tokens -- so ungating added an uncorrelated, ungated gradient source
            # even on the model's best groups. Measured effect on GPU1 (42r70off):
            # val/math500/pass_at_1 dropped to 0.498 at step 160 and never recovered
            # to the 0.63-0.68 band both the GRPO baseline and the gated runs held
            # steady in over the same span. gate = 1 - r_bar_x concentrates the
            # replay/repulsion signal on groups that are actually failing, which is
            # where repulsion-from-failure-model should matter; that concentration,
            # not an oversight, is what the group-level gate was doing.
            gated_replay_seq_kl = gate * replay_seq_kl
            replay_loss = _group_mean(gated_replay_seq_kl, group_ids)

    b_a = float(beta if beta_anchor is None else beta_anchor)
    b_r = float(beta if beta_replay is None else beta_replay)
    loss = b_a * anchor_loss - b_r * replay_loss
    group_reward_mean_d = group_reward_mean.detach()

    metrics = {
        "tafr_grpo/loss_total": float(loss.detach().cpu()),
        "tafr_grpo/kl_anchor": anchor_kl_metric,
        "tafr_grpo/kl_replay": replay_kl_metric,
        "tafr_grpo/replay_gate_mean": float(gate.detach().mean().cpu()),
        "tafr_grpo/mean_group_reward": float(group_reward_mean_d.mean().cpu()),
        "tafr_grpo/fraction_all_wrong_groups": float((group_reward_mean_d == 0).float().mean().cpu()),
        "tafr_grpo/fraction_all_correct_groups": float((group_reward_mean_d == 1).float().mean().cpu()),
        "tafr_grpo/fraction_mixed_groups": float(
            ((group_reward_mean_d > 0) & (group_reward_mean_d < 1)).float().mean().cpu()
        ),
        "tafr_grpo/beta": float(beta),
        "tafr_grpo/beta_anchor": b_a,
        "tafr_grpo/beta_replay": b_r,
        # The TAFR term itself, decomposed. loss = anchor_term - replay_term, so
        # tafr_advantage is the signed quantity TAFR adds on top of the GRPO objective:
        # positive means the net push is toward the failure/replay model, negative means
        # the anchor pull dominates. These are the numbers to plot against beta in an
        # ablation -- kl_anchor/kl_replay above are beta-independent, so they cannot show
        # how much of the update TAFR is actually responsible for.
        "tafr_grpo/anchor_term": b_a * anchor_kl_metric,
        "tafr_grpo/replay_term": b_r * replay_kl_metric,
        "tafr_grpo/tafr_advantage": -float(loss.detach().cpu()),
        "tafr_grpo/tafr_advantage_abs": abs(float(loss.detach().cpu())),
        # Estimator diagnostics. The paper reports these rather than claiming the
        # numerical safeguards are free: both clipping paths bias the estimator
        # relative to Prop. (iwk3), so their activation rates must be visible.
        "tafr_grpo/iwk3_variance": kl_var_metric,
        "tafr_grpo/iw_clip_fraction": iw_clip_fraction,
        "tafr_grpo/logratio_clamp_fraction": clamp_fraction,
        "tafr_grpo/iw_mean": iw_mean_metric,
    }
    if anchor_log_prob is not None:
        metrics["tafr_grpo/actor_logprob"] = actor_lp_metric
        metrics["tafr_grpo/anchor_logprob"] = anchor_lp_metric
    if replay_log_prob is not None:
        metrics["tafr_grpo/replay_logprob"] = replay_lp_metric
        metrics["tafr_grpo/actor_logprob_on_replay_samples"] = actor_lp_on_replay_metric
    return TAFRLossOutput(loss=loss, metrics=metrics, anchor_loss=anchor_loss, replay_loss=replay_loss)
