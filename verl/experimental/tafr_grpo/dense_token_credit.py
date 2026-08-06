# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Dense Token Credit: per-token credit reallocation for GRPO.

The GRPO advantage is one scalar per response. This module turns anchor/failure
log-probability disagreement into a per-token *residual* that redistributes that
scalar across the response's tokens without changing what the response is worth.

Two properties make it safe to add on top of a sequence-level advantage:

1. **Zero mean within each sequence.** The residual subtracts its own masked
   sequence mean, so ``mean_t(A_dense[i, :]) == 0`` for every row. The
   sequence-level GRPO preference is therefore untouched -- this only decides
   which tokens inside a response absorb the credit.

2. **Per-rollout relative scaling.** The residual is scaled by
   ``max(|A_grpo[i]|, eps_floor)``, i.e. by that rollout's *own* advantage, not
   by a batch-wide statistic. A token perturbation is then the same *fraction*
   of every rollout's advantage regardless of how easy or hard its group was.
   A batch-wide scale cannot do this: in a group of 8 with one correct rollout
   the advantages are +1.75 (correct) and -0.25 (the other seven), so a single
   global perturbation is a small ripple on one and a sign flip on the rest.
   ``eps_floor`` keeps the scale from vanishing on homogeneous groups (where
   ``A_grpo == 0``) or exploding as a ratio near zero.

Token residual RMS is tracked as an EMA across batches so that ``lambda_a`` and
``lambda_f`` mean "fraction of the rollout's advantage reallocated", stably,
rather than tracking whatever the current batch's spread happens to be.
"""

from __future__ import annotations

import torch
from torch import nn


class DenseTokenCredit(nn.Module):
    """Per-token, zero-mean-per-sequence credit reallocation for GRPO advantages.

    Returns the dense *delta* only. The caller forms the full advantage as::

        A_total = A_grpo.unsqueeze(-1) + dense_credit(A_grpo, ...)

    Args:
        lambda_a: Relative weight on the anchor (stability) residual. Only the ratio
            ``lambda_a : lambda_f`` matters -- the mix is renormalized by ``dose``.
        lambda_f: Relative weight on the failure (repulsion) residual.
        dose: The guarantee. ``rms_t(A_dense[i, :]) == dose * |A_grpo[i]|``, so
            ``dose=0.5`` means the dense term is 50% of that rollout's own advantage.
            Holds from step 1, at any signal scale, for any mix.
        tau: Symmetric clip applied to the raw signed signals before any mean,
            residual or RMS is computed. Bounds the influence of a single
            pathological log-prob.
        eps: Denominator guard when dividing by the EMA RMS.
        ema_decay: Decay for the token-residual RMS EMA. Higher is slower.
        eps_floor: Floor on ``|A_grpo[i]|`` used as the per-rollout scale. Groups
            whose rollouts all agree have ``A_grpo == 0``; this keeps their dense
            term small but finite instead of zero-scaled or divergent.
    """

    def __init__(
        self,
        lambda_a: float,
        lambda_f: float,
        dose: float = 0.5,
        pbrs: bool = False,
        tau: float = 5.0,
        eps: float = 1e-8,
        ema_decay: float = 0.99,
        eps_floor: float = 0.01,
    ) -> None:
        super().__init__()
        self.lambda_a = float(lambda_a)
        self.lambda_f = float(lambda_f)
        self.dose = float(dose)
        self.pbrs = bool(pbrs)
        self.tau = float(tau)
        self.eps = float(eps)
        self.ema_decay = float(ema_decay)
        self.eps_floor = float(eps_floor)
        # Zero-init + Adam-style bias correction (see _update_sigma). A 1.0 init instead
        # of this is a trap: real residual RMS is ~0.02, so at decay 0.99 sigma stays
        # pinned near 1.0 for hundreds of steps and the dose runs ~40x under lambda,
        # ramping the whole time. Bias correction makes step 1 already unbiased.
        self.register_buffer("sigma_a", torch.zeros(()), persistent=False)
        self.register_buffer("sigma_f", torch.zeros(()), persistent=False)
        self.register_buffer("sigma_c", torch.zeros(()), persistent=False)
        self.register_buffer("ema_steps", torch.zeros((), dtype=torch.long), persistent=False)

    def _residual(self, signal: torch.Tensor, mask: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Clip, then (unless PBRS mode) subtract the masked per-sequence mean."""
        signal = signal.clamp(min=-self.tau, max=self.tau) * mask
        if self.pbrs:
            # Strict potential-based shaping: the per-token terms must telescope to
            # Phi(end) - Phi(start), so nothing trajectory-dependent may be subtracted.
            return signal
        seq_mean = signal.sum(dim=-1, keepdim=True) / counts
        return (signal - seq_mean) * mask

    def _update_sigma(self, buffer: torch.Tensor, residual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Bias-corrected EMA of the masked mean square of the residual; returns its sqrt."""
        valid = mask.sum().clamp_min(1.0)
        mean_sq = (residual.square() * mask).sum() / valid
        buffer.mul_(self.ema_decay).add_((1.0 - self.ema_decay) * mean_sq.detach().to(buffer.dtype))
        correction = 1.0 - self.ema_decay ** int(self.ema_steps)
        return (buffer / correction).sqrt().to(residual.dtype)

    @torch.no_grad()
    def forward(
        self,
        A_grpo: torch.Tensor,
        logp_old: torch.Tensor,
        logp_anchor: torch.Tensor,
        logp_failure: torch.Tensor,
        c_x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the dense per-token advantage delta.

        Args:
            A_grpo: ``[B]`` sequence-level GRPO advantages (already group-normalized).
            logp_old: ``[B, T]`` log-probs of the sampled tokens under the sampling policy.
            logp_anchor: ``[B, T]`` same tokens under the frozen anchor.
            logp_failure: ``[B, T]`` same tokens under the failure model.
            c_x: ``[B]`` per-prompt scaling for the failure term (group difficulty).
            mask: ``[B, T]`` boolean, True on valid response tokens.

        Returns:
            ``[B, T]`` detached dense delta. Exactly zero where ``mask`` is False, and
            zero-mean over the valid tokens of every row.
        """
        mask = mask.to(dtype=logp_old.dtype)
        # clamp_min(1.0) so a fully-masked row divides by 1 instead of 0; its signal is
        # already zeroed by the mask, so the residual comes out exactly 0.
        counts = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)

        s_a = logp_anchor - logp_old
        s_f = c_x.reshape(-1, 1) * (logp_old - logp_failure)

        r_a = self._residual(s_a, mask, counts)
        r_f = self._residual(s_f, mask, counts)

        self.ema_steps += 1

        sigma_a = self._update_sigma(self.sigma_a, r_a, mask)
        sigma_f = self._update_sigma(self.sigma_f, r_f, mask)

        # Per-term normalization first, so lambda_a/lambda_f weigh two signals that live
        # on different natural scales. Only their *ratio* matters -- the combined term is
        # renormalized below, so doubling both changes nothing.
        combined = self.lambda_a * r_a / (sigma_a + self.eps) + self.lambda_f * r_f / (sigma_f + self.eps)
        sigma_c = self._update_sigma(self.sigma_c, combined, mask)

        if self.pbrs:
            # A per-rollout scale would make the potential depend on the group's reward
            # composition, which is not a fixed function of state. One global scale only.
            scale = torch.ones((), dtype=combined.dtype, device=combined.device)
        else:
            scale = A_grpo.reshape(-1, 1).abs().clamp_min(self.eps_floor)
        dense = scale * self.dose * combined / (sigma_c + self.eps)

        # Bias-corrected values actually used, for logging; the raw buffers are not these.
        self.last_sigma_a = float(sigma_a)
        self.last_sigma_f = float(sigma_f)
        self.last_sigma_c = float(sigma_c)
        return (dense * mask).detach()


def _selftest() -> None:
    torch.manual_seed(0)
    B, T = 6, 40
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 25:] = False  # ragged row
    mask[4] = False  # T_i = 0 row

    module = DenseTokenCredit(lambda_a=0.5, lambda_f=0.5)
    kwargs = dict(
        A_grpo=torch.tensor([1.75, -0.25, 0.0, -1.75, 0.5, 0.25]),
        logp_old=torch.randn(B, T, requires_grad=True),
        logp_anchor=torch.randn(B, T),
        logp_failure=torch.randn(B, T),
        c_x=torch.rand(B),
        mask=mask,
    )
    dense = module(**kwargs)

    assert dense.shape == (B, T), f"expected [{B}, {T}], got {tuple(dense.shape)}"
    assert not dense.requires_grad, "dense advantage must be detached"

    counts = mask.sum(dim=-1).clamp_min(1)
    row_mean = (dense * mask).sum(dim=-1) / counts
    assert row_mean.abs().max() < 1e-5, f"per-sequence mean must vanish, got {row_mean.abs().max():.3e}"

    assert (dense[~mask] == 0).all(), "dense terms must be exactly zero on masked-out tokens"
    assert (dense[4] == 0).all(), "a row with no valid tokens must return zeros"

    # The tau clip must bite before the mean, so a single wild log-prob cannot
    # drag a whole sequence's residual with it.
    wild = dict(kwargs)
    wild["logp_anchor"] = kwargs["logp_anchor"].clone()
    wild["logp_anchor"][0, 0] = 1e6
    bounded = DenseTokenCredit(lambda_a=0.5, lambda_f=0.0, tau=5.0)
    out = bounded(**wild)
    assert torch.isfinite(out).all(), "tau clip must keep the output finite"

    # Per-rollout scaling: rows with a bigger |A_grpo| get a proportionally bigger
    # perturbation from the same underlying signal.
    same = dict(kwargs)
    same["A_grpo"] = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    same["logp_anchor"] = kwargs["logp_anchor"][0].repeat(B, 1)
    same["logp_old"] = kwargs["logp_old"][0].detach().repeat(B, 1)
    same["logp_failure"] = kwargs["logp_failure"][0].repeat(B, 1)
    same["c_x"] = torch.ones(B)
    same["mask"] = torch.ones(B, T, dtype=torch.bool)
    out = DenseTokenCredit(lambda_a=0.5, lambda_f=0.5)(**same)
    ratio = out[1, :5] / out[0, :5]
    assert torch.allclose(ratio, torch.full_like(ratio, 2.0), atol=1e-4), f"scale must track |A_grpo|, got {ratio}"

    # The dose guarantee: rms(A_dense) == dose * |A_grpo|, from step 1, at any signal
    # scale and any mix. Run d0itf2ok violated this by ~40x (sigma pinned at its init),
    # which is the whole reason `dose` exists as a knob instead of a calibrated lambda.
    for signal_rms in (0.02, 1.0):
        for la, lf in ((0.5, 0.5), (1.0, 0.0), (3.0, 3.0)):
            for dose in (0.1, 0.5):
                m = DenseTokenCredit(lambda_a=la, lambda_f=lf, dose=dose)
                for _ in range(3):
                    out = m(
                        A_grpo=torch.full((B,), 1.75),
                        logp_old=torch.zeros(B, T),
                        logp_anchor=torch.randn(B, T) * signal_rms,
                        logp_failure=torch.randn(B, T) * signal_rms,
                        c_x=torch.ones(B),
                        mask=torch.ones(B, T, dtype=torch.bool),
                    )
                got = out.square().mean().sqrt().item() / 1.75
                assert abs(got - dose) < 0.15 * dose + 0.02, (
                    f"dose broken: signal_rms={signal_rms} mix=({la},{lf}) want {dose}, got {got:.4f}"
                )

    print("dense_token_credit selftest OK")


if __name__ == "__main__":
    _selftest()
