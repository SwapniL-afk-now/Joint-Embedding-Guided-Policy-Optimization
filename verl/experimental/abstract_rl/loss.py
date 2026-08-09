# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Abstraction compression KL and the reasoning/abstraction GRPO diagnostics."""

from __future__ import annotations

import torch

from verl.trainer.ppo.core_algos import kl_penalty

EPS = 1e-8


def compute_compression_loss(
    log_prob: torch.Tensor,
    prior_log_prob: torch.Tensor,
    abstract_mask: torch.Tensor,
    row_weight: torch.Tensor,
    kl_type: str = "low_var_kl",
    kl_clip: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compression KL over abstraction tokens of correct, valid rollouts.

        L = mean_i [ r_i v_i * mean_{t in A_i} KL(pi_theta(.|X,Y,A_<t) || pi_bar(.|X,A_<t)) ]
            over rows with r_i v_i |A_i| > 0

    Per-ROW mean-then-mean-of-rows, not one global token-weighted mean (plan
    section 15): a row's KL cost must not be diluteable by other rows in the same
    micro-batch having short or empty abstracts, or the marginal cost of a longer
    `A_i` on THIS row goes to ~0 as the batch grows -- exactly what let
    `abstractrl-fillgap-strong-20260807-v3` triple its abstract length for free.

    ``prior_log_prob`` comes from the frozen scoring pass and carries no grad, so
    the gradient flows only through the current-policy side, and only on
    abstraction positions. Returns ``(loss, metrics)``; the loss is exactly ``0``
    (and still connected to no graph) when nothing qualifies.

    Args:
        log_prob: (B, T) current-policy log-probs of the generated response tokens.
        prior_log_prob: (B, T) question-only prior log-probs, aligned to A positions.
        abstract_mask: (B, T) bool, abstraction tokens only (tags excluded).
        row_weight: (B,) r_i * v_i.
        kl_type: verl KL estimator name; ``low_var_kl`` is k3.
        kl_clip: per-token cap on top of k3's own +-10 clamp (core_algos.py). 10 nats
            is a huge ceiling on a fresh graft -- a handful of tokens the policy has
            never produced in this context sit near it and dominate the batch mean
            and its gradient (measured grad_norm climbing 2.0 -> 2.6 over the first 4
            steps of abstractrl-twopass-math15b-20260806 vs plain GRPO's ~0.2, with
            abstract_gradient_token_fraction only ~0.14 -- a few outliers, not the
            span). 0 disables (falls back to the shared clamp only).
    """
    mask = abstract_mask.float()
    row_tokens = mask.sum(dim=-1)
    row_qualifies = (row_weight.to(log_prob.dtype) > 0) & (row_tokens > 0)
    n_rows = row_qualifies.float().sum()
    if n_rows <= 0:
        # Keep the graph connected so every rank contributes a (zero) gradient.
        zero = (log_prob * 0.0).sum()
        return zero, {
            "compression_loss": torch.zeros((), device=log_prob.device),
            "abstract_kl_tokens": torch.zeros((), device=log_prob.device),
        }

    kld = kl_penalty(logprob=log_prob, ref_logprob=prior_log_prob.detach(), kl_penalty=kl_type)
    if kl_clip > 0:
        kld = kld.clamp(max=kl_clip)
    per_row_kl = (kld * mask).sum(dim=-1) / row_tokens.clamp(min=1.0)  # mean over each row's own A_i
    loss = (per_row_kl * row_qualifies.float()).sum() / (n_rows + EPS)
    qualifying_tokens = (row_tokens * row_qualifies.float()).sum()
    return loss, {
        "compression_loss": loss.detach(),
        "abstract_kl_tokens": qualifying_tokens.detach(),
        "abstract_kl_rows": n_rows.detach(),
    }


def compute_abstract_sft_loss(
    log_prob: torch.Tensor,
    abstract_mask: torch.Tensor,
    row_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Plain NLL on the abstraction span, for the format warm-up.

        L = - sum_i w_i sum_{t in A_i} log pi_theta(a_t | X, Y_i, A_<t)
            / (sum_i w_i |A_i| + eps)

    Replaces the policy loss for the first ``bootstrap_sft_steps`` steps. The
    grafted sequences were not sampled from ``pi_theta(.|X_train)``, so a policy
    gradient over them is a fiction -- the ratio is 1 by construction. Maximising
    their likelihood directly is the same intent, stated honestly, and much
    stronger: no mean-centering, no clipping, every qualifying row contributes.

    Masked to the abstraction so the warm-up teaches the transition into
    ``<abstract>`` without fine-tuning on the reasoning trace and disturbing a
    solver that already works. ``row_weight`` is ``r_i * v_i``, so only correct
    rollouts with a well-formed abstraction are imitated.
    """
    weight = (abstract_mask.float() * row_weight.to(log_prob.dtype).unsqueeze(-1)).detach()
    denom = weight.sum()
    if denom <= 0:
        zero = (log_prob * 0.0).sum()
        return zero, {"sft_loss": torch.zeros((), device=log_prob.device), "sft_tokens": denom.detach()}
    loss = -(log_prob * weight).sum() / (denom + EPS)
    return loss, {"sft_loss": loss.detach(), "sft_tokens": denom.detach()}


def compute_span_diagnostics(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    entropy: torch.Tensor | None,
    reasoning_mask: torch.Tensor,
    abstract_mask: torch.Tensor,
    response_mask: torch.Tensor,
    clip_low: float,
    clip_high: float,
) -> dict[str, torch.Tensor]:
    """Per-span policy ratio / clip fraction / entropy / token share.

    Verifies that abstraction tokens really do participate in the GRPO update:
    ``abstract_gradient_token_fraction`` is the share of loss-carrying tokens that
    are abstraction tokens, and it must be > 0 whenever abstractions are emitted.

    The ``_mean`` keys are per-micro-batch masked means, and verl averages them
    across micro-batches -- so a micro-batch with no abstraction token contributes
    a 0 and pulls ``abstract_*_mean`` down. The ``_sum`` / ``_tokens`` keys are
    SUM-aggregated, so their ratio is the exact batch-level value.
    """
    ratio = torch.exp((log_prob - old_log_prob).clamp(-20, 20))
    clipped = ((ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)).float()
    total = response_mask.float().sum().clamp(min=1.0)

    out: dict[str, torch.Tensor] = {}
    for name, mask in (("reasoning", reasoning_mask), ("abstract", abstract_mask)):
        m = mask.float()
        n = m.sum()
        safe = n.clamp(min=1.0)
        out[f"{name}_policy_ratio_mean"] = (ratio * m).sum() / safe
        out[f"{name}_clip_fraction"] = (clipped * m).sum() / safe
        out[f"{name}_gradient_token_fraction"] = n / total
        out[f"{name}_policy_ratio_sum"] = (ratio * m).sum()
        out[f"{name}_span_tokens"] = n
        if entropy is not None:
            out[f"{name}_entropy"] = (entropy * m).sum() / safe
    return out
