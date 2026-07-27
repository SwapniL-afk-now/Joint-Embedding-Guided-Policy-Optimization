# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""JEPA-GRPO algorithm primitives.

All pure differentiable math lives here:
  - Dr.GRPO policy loss (group-centered, no std normalization)
  - Epps-Pulley CF test statistic (vectorized)
  - SIGReg (Sketched Isotropic Gaussian Regularization, LeJEPA)
  - LeJEPA loss (squared-Euclidean alignment + SIGReg)
  - LLM-JEPA loss (cosine-distance prediction alignment + SIGReg)

References:
  - Balestriero et al. 2025 "LeJEPA" arXiv:2511.08544
  - Huang, LeCun, Balestriero 2025 "LLM-JEPA" arXiv:2509.14252
  - Equation doc: JEPA-GRPO-Equation.md
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

# Re-export for trainer convenience
from verl.experimental.fepo.core_algos import compute_group_advantages  # noqa: F401


# ---------------------------------------------------------------------------
# Dr.GRPO policy loss
# ---------------------------------------------------------------------------

def dr_grpo_loss(
    current_logps: torch.Tensor,    # (T_total,) flat token log-probs, current policy
    old_logps: torch.Tensor,        # (T_total,) flat, detached, from rollout time
    advantages: torch.Tensor,       # (T_total,) flat, per-token broadcast of group advantage
    loss_mask: torch.Tensor,        # (T_total,) bool — True = valid completion token
    clip_eps: float = 0.2,
    token_log_ratio_clip: float = 8.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dr.GRPO clipped policy loss.

    Group-centered advantages, NO std normalization (Dr.GRPO variant).
    Advantages are pre-computed by the caller via compute_group_advantages.
    """
    if current_logps.numel() == 0 or loss_mask.sum() == 0:
        zero = current_logps.sum() * 0.0
        return zero, {
            "grpo_pg_loss": 0.0,
            "grpo_clip_fraction": 0.0,
            "grpo_token_ratio_mean": 1.0,
            "grpo_log_ratio_abs_mean": 0.0,
            "grpo_loss_token_count": 0.0,
        }

    old_logps = old_logps.to(device=current_logps.device, dtype=current_logps.dtype).detach()
    advantages = advantages.to(device=current_logps.device, dtype=current_logps.dtype).detach()
    mask = loss_mask.to(device=current_logps.device).bool().detach()

    cur = current_logps[mask]
    old = old_logps[mask]
    adv = advantages[mask]

    log_ratio = (cur - old).clamp(-token_log_ratio_clip, token_log_ratio_clip)
    ratio = log_ratio.exp()

    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
    pg_loss = -torch.min(unclipped, clipped).mean()

    clip_frac = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float().mean()

    return pg_loss, {
        "grpo_pg_loss": float(pg_loss.detach().cpu()),
        "grpo_clip_fraction": float(clip_frac.detach().cpu()),
        "grpo_token_ratio_mean": float(ratio.detach().mean().cpu()),
        "grpo_log_ratio_abs_mean": float(log_ratio.detach().abs().mean().cpu()),
        "grpo_loss_token_count": float(mask.sum().cpu()),
    }


# ---------------------------------------------------------------------------
# Epps-Pulley CF test statistic (vectorized, single direction)
# ---------------------------------------------------------------------------

def epps_pulley_statistic(
    z: torch.Tensor,    # (N,) 1-D projected sample
    t: torch.Tensor,    # (K,) frequency grid
    s: float = 1.0,     # Gaussian-tapered kernel bandwidth (LeJEPA default)
) -> torch.Tensor:
    """Epps-Pulley test statistic comparing empirical CF of z to CF of N(0,1).

    Uses a Gaussian-tapered kernel w(t) = exp(-t²/s²) as in the LeJEPA paper
    (NOT the classical 1/t² weighting).
    """
    angles = z.unsqueeze(-1) * t          # (N, K)
    emp_cf_real = torch.cos(angles).mean(dim=0)   # (K,)
    emp_cf_imag = torch.sin(angles).mean(dim=0)   # (K,)

    gaussian_cf = torch.exp(-0.5 * t ** 2)        # (K,)  CF of N(0,1) is real-valued
    weights = torch.exp(-t ** 2 / s ** 2)          # (K,)

    diff_real = emp_cf_real - gaussian_cf
    return (weights * (diff_real ** 2 + emp_cf_imag ** 2)).sum()


# ---------------------------------------------------------------------------
# SIGReg — Sketched Isotropic Gaussian Regularization (LeJEPA)
# ---------------------------------------------------------------------------

def sigreg_loss(
    embeddings: torch.Tensor,   # (N, d) L2-normalized embeddings on unit sphere
    M: int = 1024,              # Random projection directions (LeJEPA default)
    n_freq: int = 17,           # Epps-Pulley quadrature points (LeJEPA default)
    t_min: float = -5.0,        # Frequency range min (LeJEPA default)
    t_max: float = 5.0,         # Frequency range max (LeJEPA default)
    s: float = 1.0,             # Gaussian kernel bandwidth (LeJEPA default)
) -> torch.Tensor:
    """SIGReg: fully vectorized over all M directions simultaneously.

    Memory: (N, M, K) tensor. With N=512, M=1024, K=17 in fp32 ≈ 35 MB.

    LeJEPA defaults (arXiv 2511.08544 §6.1):
        M=1024, n_freq=17, t∈[−5,+5], s=1.0
    """
    N, d = embeddings.shape
    device = embeddings.device
    dtype = embeddings.dtype

    # SIGReg (Epps-Pulley) tests the embedding distribution against N(0, I_d) via
    # random 1-D projections, and that test only has dynamic range when the
    # projected coordinates are O(1)-variance. Callers pass L2-normalized
    # (unit-sphere) embeddings — required for the cosine alignment term — but the
    # projection of a unit-norm vector onto a random unit direction has variance
    # ~1/d (std ~0.026 at d=1536). The empirical characteristic function then sits
    # pinned at ~1 across t∈[t_min,t_max] for *every* arrangement on the sphere, so
    # the statistic degenerates into a near-constant scale mismatch that is blind to
    # the actual anisotropy/collapse it is supposed to penalize.
    #
    # Fix: rescale by the global radius only — divide by rms_norm/sqrt(d), where
    # rms_norm = sqrt(mean_i ||x_i||²). For isotropic unit-sphere data this maps the
    # mean projected variance to 1 (E_v[(x·v)²] = ||x||²/d), so projections look like
    # N(0,1) and the loss is low; a collapsed cone keeps its near-constant projection
    # along most directions (variance ≪ 1, nonzero mean), so the CF stays pinned at 1
    # and mismatches the Gaussian — high loss, real gradient. Crucially we do NOT
    # center and do NOT per-dim standardize: both would whiten away the rank/anisotropy
    # signal (subtracting the mean of a tight cone leaves only its ~isotropic jitter,
    # which then looks exactly like a well-spread set). Scaling is differentiable.
    rms_norm = embeddings.pow(2).sum(dim=-1).mean().sqrt()
    embeddings = embeddings * (d ** 0.5) / (rms_norm + 1e-6)

    # M random unit projection directions on S^{d-1}
    v = F.normalize(torch.randn(M, d, device=device, dtype=dtype), dim=-1)  # (M, d)

    # All 1-D projections: z[n, m] = dot(embeddings[n], v[m])
    z = embeddings @ v.T   # (N, M)

    # Frequency grid
    t = torch.linspace(t_min, t_max, n_freq, device=device, dtype=dtype)  # (K,)

    # Batch CF across all M directions: angles[n, m, k] = z[n, m] * t[k]
    angles = z.unsqueeze(-1) * t   # (N, M, K)

    emp_cf_real = torch.cos(angles).mean(dim=0)   # (M, K)
    emp_cf_imag = torch.sin(angles).mean(dim=0)   # (M, K)

    gaussian_cf = torch.exp(-0.5 * t ** 2)        # (K,)
    weights = torch.exp(-t ** 2 / s ** 2)          # (K,)

    diff_real = emp_cf_real - gaussian_cf          # (M, K)
    # Epps-Pulley statistic per direction (M,), then average
    stats = (weights * (diff_real ** 2 + emp_cf_imag ** 2)).sum(dim=-1)   # (M,)
    return stats.mean()


# ---------------------------------------------------------------------------
# LeJEPA loss: squared-Euclidean alignment + SIGReg
# ---------------------------------------------------------------------------

def lejepa_loss(
    enc_q_cot: torch.Tensor,        # (B_joint, d) L2-normalized CoT question embeddings
    enc_a_code: torch.Tensor,       # (B_joint, d) L2-normalized correct code response embeddings
    all_embeddings: torch.Tensor,   # (N_pool, d)  L2-normalized pool for SIGReg
    lambda_: float = 0.05,          # SIGReg vs align mixing (LeJEPA default)
    M: int = 1024,
    n_freq: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
    s: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """LeJEPA loss = (1-λ)·L_align + λ·L_SIGReg.

    L_align: squared-Euclidean distance on the unit sphere between paired views.
    L_SIGReg: Epps-Pulley test that joint embedding distribution matches N(0, I_d).

    Per LeJEPA Theorem 1: the isotropic Gaussian is the unique minimizer of the
    integrated square bias — no other regularizer (InfoNCE, VICReg) has this guarantee.
    """
    # L_align: 0.5 * ||enc_q_cot - enc_a_code||² averaged over pairs
    diff = enc_q_cot - enc_a_code       # (B_joint, d)
    align = 0.5 * (diff ** 2).sum(dim=-1).mean()

    # L_SIGReg: on the full pool (both views, all correct)
    sig = sigreg_loss(all_embeddings, M=M, n_freq=n_freq, t_min=t_min, t_max=t_max, s=s)

    loss = (1.0 - lambda_) * align + lambda_ * sig

    return loss, {
        "jepa/align_loss": float(align.detach().cpu()),
        "jepa/sigreg_loss": float(sig.detach().cpu()),
        "jepa/lejepa_loss": float(loss.detach().cpu()),
        "jepa/lambda": float(lambda_),
        "jepa/n_pairs": int(enc_q_cot.shape[0]),
        "jepa/pool_size": int(all_embeddings.shape[0]),
    }


# ---------------------------------------------------------------------------
# LLM-JEPA (paper-exact) loss — Huang, LeCun, Balestriero 2025, arXiv:2509.14252
# ---------------------------------------------------------------------------

def llm_jepa_paper_loss(
    pred: torch.Tensor,          # (N, d) Pred(Enc([x, y_S, [PRED]×k])) — student CoT reads, L2-normalized
    target_cot: torch.Tensor,    # (N, d) Enc([x, y_T_cot]) — teacher CoT view, L2-normalized, NOT detached
    target_code: torch.Tensor,   # (N, d) Enc([x, y_T_code]) — teacher Code view, L2-normalized, NOT detached
    has_code: torch.Tensor,      # (N,) bool — rows with a code target (False -> code arm skips that row)
    align_cot_on: bool = True,
    align_code_on: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Exact replication of the official LLM-JEPA fine-tuning loss (finetune.py):

        L_JEPA = 1 - mean_i cos(Pred(Enc(Text_i)), Enc(Code_i))

    Paper/reference-implementation properties preserved verbatim:
      - metric d(.,.) = cosine distance via F.cosine_similarity (here: dot product
        of pre-L2-normalized embeddings — mathematically identical, including the
        gradient, since normalization is part of the cosine derivative);
      - a FLAT mean over pairs (no reward stratification, no prompt averaging);
      - NO stop-gradient on either branch: the reference concatenates all views
        into one differentiable forward, so Enc(target) receives gradient exactly
        like Pred(Enc(text)) does. Callers must therefore pass NON-detached
        target embeddings from a grad-enabled forward;
      - no SIGReg / InfoNCE / variance term (those are ablations, not the loss).

    Deviation (per project design, not the paper): the paper's single (Text, Code)
    view pair becomes TWO pairs sharing one prediction — the student's generated
    CoT predicts BOTH the teacher's CoT response and the teacher's Code response:

        L = d(p, Enc([x, y^T_cot])) + d(p, Enc([x, y^T_code]))

    Each arm is the unmodified paper loss; they are summed with equal weight.
    """
    device, dtype = pred.device, pred.dtype
    N = pred.shape[0]
    if N == 0:
        zero = torch.zeros((), device=device, dtype=dtype)
        return zero, {
            "jepa/llm_jepa_loss": 0.0, "jepa/align_cot": 0.0, "jepa/align_code": 0.0,
            "jepa/cos_cot": 0.0, "jepa/cos_code": 0.0,
            "jepa/n_anchors_cot": 0, "jepa/n_anchors_code": 0, "jepa/pool_size": 0,
        }

    # Embeddings arrive unit-norm; dot == F.cosine_similarity of the raw states.
    cos_cot = (pred * target_cot).sum(dim=-1)                  # (N,)
    align_cot = (1.0 - cos_cot).mean()

    hc = has_code.to(device=device, dtype=torch.bool)
    n_code = int(hc.sum().item())
    if n_code > 0:
        cos_code = (pred[hc] * target_code[hc]).sum(dim=-1)    # (n_code,)
        align_code = (1.0 - cos_code).mean()
    else:
        cos_code = pred.new_zeros((0,))
        align_code = torch.zeros((), device=device, dtype=dtype)

    c_cot = 1.0 if align_cot_on else 0.0
    c_code = 1.0 if align_code_on else 0.0
    loss = c_cot * align_cot + c_code * align_code

    metrics = {
        "jepa/llm_jepa_loss": float(loss.detach().cpu()),
        "jepa/tcr_loss": float(loss.detach().cpu()),  # alias: keeps wandb key parity with prior runs
        "jepa/align_cot": float(align_cot.detach().cpu()),
        "jepa/align_code": float(align_code.detach().cpu()),
        "jepa/cos_cot": float(cos_cot.detach().mean().cpu()),
        "jepa/cos_code": float(cos_code.detach().mean().cpu()) if n_code > 0 else 0.0,
        "jepa/n_anchors_cot": int(N),
        "jepa/n_anchors_code": int(n_code),
        "jepa/pool_size": int(N),
        "jepa/arm_cot_on": float(align_cot_on),
        "jepa/arm_code_on": float(align_code_on),
        # Constant-zero keys kept ONLY for wandb panel parity with prior
        # jepa-tcr-dual runs (e.g. grpo-qwen-3b/jxorb8w4): the paper loss has no
        # SIGReg, self-consistency, or lambda mixing.
        "jepa/self_consist_loss": 0.0,
        "jepa/tcr_sigreg_loss": 0.0,
        "jepa/cos_self": 0.0,
        "jepa/n_self_pairs": 0,
        "jepa/tcr_lambda": 0.0,
        "jepa/arm_self_on": 0.0,
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# LLM-JEPA loss: cosine-distance prediction alignment + SIGReg
# ---------------------------------------------------------------------------

def _prompt_stratified_align(
    ell: torch.Tensor,
    group_id: torch.Tensor | None,
    is_correct: torch.Tensor | None,
) -> torch.Tensor:
    """Reward-stratified, PROMPT-averaged mean of per-anchor losses ``ell``.

    Mirrors the aggregation inside ``llm_jepa_tcr_loss``: per prompt,
    ``½·mean(correct ℓ) + ½·mean(wrong ℓ)`` when both classes are present, else the
    single available conditional mean; then mean over PROMPTS (not raw anchors) so
    anchor counts never implicitly weight the loss. Falls back to a flat mean when
    ``group_id``/``is_correct`` are None. Pure function of its tensors (unit-testable).
    """
    A = ell.shape[0]
    device, dtype = ell.device, ell.dtype
    if A == 0:
        return torch.zeros((), device=device, dtype=dtype)
    if group_id is None or is_correct is None:
        return ell.mean()
    gid = group_id.to(device=device, dtype=torch.long)
    corr = is_correct.to(device=device, dtype=torch.bool)
    G = int(gid.max().item()) + 1
    ones = torch.ones(A, device=device, dtype=dtype)

    def _group_mean(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = mask.to(dtype)
        ssum = torch.zeros(G, device=device, dtype=dtype).index_add_(0, gid, ell * m)
        cnt = torch.zeros(G, device=device, dtype=dtype).index_add_(0, gid, ones * m)
        present = cnt > 0
        return ssum / cnt.clamp(min=1.0), present

    cmean, cpresent = _group_mean(corr)
    wmean, wpresent = _group_mean(~corr)
    both = cpresent & wpresent
    per_prompt = torch.where(both, 0.5 * cmean + 0.5 * wmean,
                             torch.where(cpresent, cmean, wmean))
    any_anchor = cpresent | wpresent
    n_prompts = any_anchor.sum().clamp(min=1)
    return (per_prompt * any_anchor.to(dtype)).sum() / n_prompts


def llm_jepa_tcr_dual_loss(
    pred_cot: torch.Tensor,             # (A_c, d) CoT student [PRED] reads, L2-normalized
    teacher_target_cot: torch.Tensor,  # (A_c, d) CoT z^+, recomputed online by current policy (sg)
    pred_code: torch.Tensor,           # (A_k, d) Code student [PRED] reads, L2-normalized
    teacher_target_code: torch.Tensor, # (A_k, d) Code z^+, recomputed online by current policy (sg)
    self_target: torch.Tensor,         # (A_c, d) Code_S BOUNDARY reads paired to CoT rows (raw, will be detached)
    self_mask: torch.Tensor,           # (A_c,) bool: True where a CoT row has a Code partner
    cot_group_id: torch.Tensor | None = None,
    cot_is_correct: torch.Tensor | None = None,
    code_group_id: torch.Tensor | None = None,
    code_is_correct: torch.Tensor | None = None,
    self_consist_w: float = 1.0,
    align_cot_on: bool = True,
    align_code_on: bool = True,
    self_on: bool = True,
    lambda_: float = 0.5,
    M: int = 1024,
    n_freq: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
    s: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dual-target, self-consistent TCR loss (loss_type='jepa-tcr-dual').

    L = (1-λ)·( align_cot + align_code + w·L_self ) + λ·SIGReg([pred_cot, pred_code])

      align_cot  = stratified mean of 1 - <pred_cot_i,  sg(z_cot_i)>      (CoT teacher pull)
      align_code = stratified mean of 1 - <pred_code_j, sg(z_code_j)>     (Code teacher pull)
      L_self     = mean over partnered CoT rows of 1 - <pred_cot_i, sg(b_i)>

    where b_i is the *boundary read* of the paired Code_S block (the raw last-real-token
    embedding BEFORE the [PRED] token) — a free second read of a block already in the
    forward. self_target is stop-gradiented here (only pred_cot moves toward it), the
    direction specified for the self-consistency arm; this differs from the symmetric
    crossview_partner term in llm_jepa_tcr_loss (which moves both [PRED] reads). L_self
    self-gates to 0 when no CoT row has a code partner (self_mask all-False).

    Both teacher targets are z^+ = normalize(f_theta([x, y^+])) recomputed online by the
    current policy (no EMA) and stop-gradiented here (sg), so they track the policy's
    representation space instead of being stale cached frozen-ref latents. SIGReg over the student
    preds of BOTH views supplies anti-collapse, preserving the load-bearing global-radius
    rescale on unit-sphere inputs (see sigreg_loss / project_jepa_sigreg_collapse).
    """
    A_c = pred_cot.shape[0]
    A_k = pred_code.shape[0]
    device = pred_cot.device
    dtype = pred_cot.dtype

    pool = torch.cat([pred_cot, pred_code], dim=0)
    if pool.shape[0] == 0:
        zero = torch.zeros((), device=device, dtype=dtype)
        return zero, {
            "jepa/tcr_loss": 0.0, "jepa/llm_jepa_loss": 0.0, "jepa/tcr_lambda": float(lambda_),
            "jepa/align_cot": 0.0, "jepa/align_code": 0.0, "jepa/self_consist_loss": 0.0,
            "jepa/tcr_sigreg_loss": 0.0, "jepa/n_anchors_cot": 0, "jepa/n_anchors_code": 0,
            "jepa/n_self_pairs": 0, "jepa/cos_cot": 0.0, "jepa/cos_code": 0.0, "jepa/cos_self": 0.0,
            "jepa/pool_size": 0,
        }

    # Teacher pulls (each stratified + prompt-averaged independently).
    cot_pos = (pred_cot * teacher_target_cot.detach()).sum(dim=-1) if A_c > 0 else pred_cot.new_zeros((0,))
    code_pos = (pred_code * teacher_target_code.detach()).sum(dim=-1) if A_k > 0 else pred_code.new_zeros((0,))
    align_cot = _prompt_stratified_align(1.0 - cot_pos, cot_group_id, cot_is_correct)
    align_code = _prompt_stratified_align(1.0 - code_pos, code_group_id, code_is_correct)

    # Self-consistency: pred_CoT pulled toward the stop-grad Code_S boundary read.
    self_loss = torch.zeros((), device=device, dtype=dtype)
    n_self = 0
    self_pos_mean = 0.0
    if A_c > 0 and self_mask is not None:
        m = self_mask.to(device=device, dtype=torch.bool)
        if m.any():
            self_pos = (pred_cot[m] * self_target[m].detach()).sum(dim=-1)
            self_loss = (1.0 - self_pos).mean()
            n_self = int(m.sum().item())
            self_pos_mean = float(self_pos.detach().mean().cpu())

    sig = sigreg_loss(pool, M=M, n_freq=n_freq, t_min=t_min, t_max=t_max, s=s)
    # Per-arm plateau latches: a disabled arm contributes 0 gradient (coefficient 0),
    # but its preds STILL feed the SIGReg pool above (anti-collapse pressure preserved).
    c_cot = 1.0 if align_cot_on else 0.0
    c_code = 1.0 if align_code_on else 0.0
    c_self = 1.0 if self_on else 0.0
    loss = (1.0 - lambda_) * (
        c_cot * align_cot + c_code * align_code + c_self * self_consist_w * self_loss
    ) + lambda_ * sig

    metrics = {
        "jepa/align_cot": float(align_cot.detach().cpu()),
        "jepa/align_code": float(align_code.detach().cpu()),
        "jepa/self_consist_loss": float(self_loss.detach().cpu()),
        "jepa/tcr_sigreg_loss": float(sig.detach().cpu()),
        "jepa/tcr_loss": float(loss.detach().cpu()),
        "jepa/llm_jepa_loss": float(loss.detach().cpu()),  # alias read back by worker
        "jepa/tcr_lambda": float(lambda_),
        "jepa/n_anchors_cot": int(A_c),
        "jepa/n_anchors_code": int(A_k),
        "jepa/n_self_pairs": int(n_self),
        "jepa/cos_cot": float(cot_pos.detach().mean().cpu()) if A_c > 0 else 0.0,
        "jepa/cos_code": float(code_pos.detach().mean().cpu()) if A_k > 0 else 0.0,
        "jepa/cos_self": self_pos_mean,
        "jepa/pool_size": int(pool.shape[0]),
        "jepa/arm_cot_on": float(align_cot_on),
        "jepa/arm_code_on": float(align_code_on),
        "jepa/arm_self_on": float(self_on),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# LLM-JEPA contrastive loss (loss_type='llm-jepa-contrastive')
# ---------------------------------------------------------------------------

def _prompt_mean(vals: torch.Tensor, group_id: torch.Tensor) -> torch.Tensor:
    """Collapse a per-distance vector to ONE scalar per group (prompt).

    Returns a (n_groups_present,) vector of per-group means — the per-prompt unit
    used by the contrastive loss's pooled average. Empty input -> empty vector.
    Differentiable (index_add_ + divide).
    """
    if vals.numel() == 0:
        return vals.new_zeros((0,))
    gid = group_id.to(device=vals.device, dtype=torch.long)
    G = int(gid.max().item()) + 1
    ssum = torch.zeros(G, device=vals.device, dtype=vals.dtype).index_add_(0, gid, vals)
    cnt = torch.zeros(G, device=vals.device, dtype=vals.dtype).index_add_(
        0, gid, torch.ones_like(vals))
    present = cnt > 0
    return ssum[present] / cnt[present]


def llm_jepa_contrastive_loss(
    good_pred: torch.Tensor,        # (Ng, d) correct-anchor <good_pred> reads, L2-normalized
    target_cot: torch.Tensor,       # (Ng, d) teacher CoT view per good anchor, L2-norm, NOT detached
    target_code: torch.Tensor,      # (Ng, d) teacher Code view per good anchor, L2-norm, NOT detached
    has_code: torch.Tensor,         # (Ng,) bool — good anchors with a code target
    good_group_id: torch.Tensor,    # (Ng,) long — prompt id per good anchor
    bad_pred: torch.Tensor,         # (Nb, d) wrong-anchor <bad_pred> reads, L2-normalized
    bad_target: torch.Tensor,       # (Nb, d) wrong target per bad anchor (no stop-grad), L2-norm
    bad_group_id: torch.Tensor,     # (Nb,) long — prompt id per bad anchor
    contrast_good: torch.Tensor,    # (Nc, d) good_pred read per contrast pair (gathered)
    contrast_bad: torch.Tensor,     # (Nc, d) bad_pred read per contrast pair (gathered)
    contrast_group_id: torch.Tensor,  # (Nc,) long — prompt id per contrast pair
    sig_pool: "torch.Tensor | None" = None,  # (Npool, d) L2-normalized reads for anti-collapse
    sigreg_lambda: float = 0.0,     # SIGReg vs arms convex mix; 0 => SIGReg off (back-compat)
    sig_M: int = 1024, sig_n_freq: int = 17,
    sig_t_min: float = -5.0, sig_t_max: float = 5.0, sig_s: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Three-arm LLM-JEPA loss that also uses the student's WRONG rollouts, plus a
    SIGReg anti-collapse term.

    Arms (each an unmodified cosine metric, all in [0,1]):
      A  align_cot / align_code : correct anchor <good_pred> pulled toward the teacher
                                  CoT / Code views (paper loss, no stop-grad).
      B  bad_align              : wrong anchor <bad_pred> pulled toward ONE wrong response
                                  chosen as target (no stop-grad); (1 - cos), CoT-only.
      C  contrastive            : (1 + cos(good_pred, bad_pred)) / 2 — pushes the good/bad
                                  reads apart on prompts that have both.

    Averaging (imbalance fix): each arm is first collapsed to ONE scalar PER PROMPT
    (mean over that prompt's anchors/pairs), then EVERY per-prompt unit from all arms is
    pooled into a single mean:  arms = mean(concat(a_cot, a_code, b, c)).  So an arm covering
    more prompts weighs proportionally more, no arm dominates by raw rollout count, and the
    arms term stays in [0,1]. Arms with no data this step contribute no units.

    Anti-collapse (SIGReg): the official LLM-JEPA is SUPERVISED — its LM loss grounds the
    reads so they cannot collapse. Under RL we have no such anchor, so the cosine-alignment
    arms drive every read into one anisotropic cone (all cosines -> ~0.9, contrastive fails).
    LeJEPA's SIGReg is the designed fix for this "no supervised anchor" regime: it tests the
    pooled reads against an isotropic Gaussian and its gradient spreads them out. Combined
    convex:  L = (1 - lambda)*arms + lambda*SIGReg(sig_pool).  Pass sig_pool = all unique
    reads (good_pred + bad_pred + teacher + wrong). sigreg_lambda=0 disables it.
    """
    dev, dtype = good_pred.device, good_pred.dtype
    hc = has_code.to(device=dev, dtype=torch.bool)

    # Per-distance cosine terms (embeddings arrive unit-norm -> dot == cosine).
    cos_cot = (good_pred * target_cot).sum(dim=-1) if good_pred.numel() else good_pred.new_zeros((0,))
    a_cot_d = 1.0 - cos_cot
    cos_code = (good_pred[hc] * target_code[hc]).sum(dim=-1)
    a_code_d = 1.0 - cos_code
    cos_bad = (bad_pred * bad_target).sum(dim=-1) if bad_pred.numel() else bad_pred.new_zeros((0,))
    b_d = 1.0 - cos_bad
    cos_ctr = (contrast_good * contrast_bad).sum(dim=-1) if contrast_good.numel() else good_pred.new_zeros((0,))
    c_d = (1.0 + cos_ctr) / 2.0

    # Collapse each arm to one unit per prompt, then pool all units into one mean.
    a_cot = _prompt_mean(a_cot_d, good_group_id)
    a_code = _prompt_mean(a_code_d, good_group_id[hc])
    b = _prompt_mean(b_d, bad_group_id)
    c = _prompt_mean(c_d, contrast_group_id)

    parts = [v for v in (a_cot, a_code, b, c) if v.numel() > 0]
    arms = torch.cat(parts).mean() if parts else good_pred.sum() * 0.0

    # SIGReg anti-collapse on the pooled reads (convex mix with the arms term).
    lam = float(sigreg_lambda)
    if sig_pool is not None and sig_pool.numel() and lam > 0.0:
        sig = sigreg_loss(sig_pool, M=sig_M, n_freq=sig_n_freq,
                          t_min=sig_t_min, t_max=sig_t_max, s=sig_s)
        loss = (1.0 - lam) * arms + lam * sig
    else:
        sig = arms.new_zeros(())
        loss = arms

    def _m(t):
        return float(t.detach().mean().cpu()) if t.numel() else 0.0

    metrics = {
        "jepa/llm_jepa_loss": float(loss.detach().cpu()),
        "jepa/tcr_loss": float(loss.detach().cpu()),  # alias: wandb key parity
        "jepa/arms_loss": float(arms.detach().cpu()),
        "jepa/sigreg_loss": float(sig.detach().cpu()),
        "jepa/sigreg_lambda": lam,
        "jepa/sig_pool_size": int(sig_pool.shape[0]) if sig_pool is not None else 0,
        "jepa/align_cot": _m(a_cot),
        "jepa/align_code": _m(a_code),
        "jepa/bad_align": _m(b),
        "jepa/contrastive": _m(c),
        "jepa/cos_cot": _m(cos_cot),
        "jepa/cos_code": _m(cos_code),
        "jepa/cos_bad": _m(cos_bad),
        "jepa/cos_contrast": _m(cos_ctr),
        "jepa/n_anchors_cot": int(good_pred.shape[0]),
        "jepa/n_anchors_code": int(hc.sum().item()),
        "jepa/n_bad_anchors": int(bad_pred.shape[0]),
        "jepa/n_contrast_pairs": int(contrast_good.shape[0]),
        "jepa/n_prompts_cot": int(a_cot.numel()),
        "jepa/n_prompts_code": int(a_code.numel()),
        "jepa/n_prompts_bad": int(b.numel()),
        "jepa/n_prompts_contrast": int(c.numel()),
        "jepa/pool_size": int(sum(v.numel() for v in parts)),
        # Constant-zero keys kept for wandb panel parity with prior jepa runs.
        "jepa/self_consist_loss": 0.0,
        "jepa/tcr_sigreg_loss": 0.0,
        "jepa/cos_self": 0.0,
        "jepa/n_self_pairs": 0,
        "jepa/tcr_lambda": 0.0,
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# LLM-JEPA InfoNCE loss (loss_type='llm-jepa-infoNCE')
# ---------------------------------------------------------------------------

def llm_jepa_infonce_loss(
    pred_pos: torch.Tensor,   # (M, d) p+ = f(x, y+, [PRED]) — correct-rollout reads, L2-normalized
    pred_neg: torch.Tensor,   # (M, d) p- = f(x, y-, [PRED]) — wrong-rollout reads, L2-normalized
    teacher: torch.Tensor,    # (M, d) z+_cot — teacher CoT reads, L2-norm. May or may not carry
                              # grad depending on the caller (worker.py's llm-jepa-infoNCE branch
                              # now encodes this in the same differentiable joint forward as
                              # pred_pos/pred_neg — no stop-grad, per user request); this function
                              # itself never calls .detach() on `teacher`/`teacher_code`.
    teacher_code: torch.Tensor | None = None,  # (M, d) z+_code, same caveat as `teacher`
    has_code: torch.Tensor | None = None,      # (M,) bool — rows with a code teacher view
    tau: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Teacher-anchored discriminative InfoNCE over mixed-reward prompts.

    Per pair and per available teacher view v ∈ {cot, code}:

        L_iv = -log( exp(cos(p+,z_v)/τ) / (exp(cos(p+,z_v)/τ) + exp(cos(p-,z_v)/τ)) )

    i.e. "which read-out matches the teacher?" — BOTH the attraction (p+ → z_v)
    and the repulsion (p- ← z_v) are measured against the same per-step teacher
    read. The final loss is the mean over all (pair, view) terms present; rows
    without a code view contribute only cot.

    All embeddings are re-centered by the (detached) shared batch mean and
    re-normalized before the cosines: raw same-model reads share a large
    anisotropic component (cos≈0.96 between arbitrary reads before any
    training), and centering removes it so the logits measure content, not the
    common style direction — the model can no longer satisfy the softmax with a
    tiny relative margin riding on a saturated base similarity.

    Gradient reaches p+ and p- always. Whether it also reaches `teacher`/
    `teacher_code` depends entirely on the caller: pass detached tensors for
    the original stop-grad/EMA-teacher contract, or tensors from the same
    differentiable graph as pred_pos/pred_neg for a "no stop-grad" variant
    (worker.py's llm-jepa-infoNCE branch now does the latter) — this function
    imposes no constraint either way.
    """
    device, dtype = pred_pos.device, pred_pos.dtype
    M = pred_pos.shape[0]
    empty_metrics = {
        "jepa/llm_jepa_loss": 0.0, "jepa/tcr_loss": 0.0, "jepa/infonce_loss": 0.0,
        "jepa/cos_cot": 0.0, "jepa/cos_code": 0.0, "jepa/cos_neg": 0.0,
        "jepa/margin": 0.0, "jepa/infonce_acc": 0.0, "jepa/infonce_tau": float(tau),
        "jepa/n_anchors_cot": 0, "jepa/n_anchors_code": 0, "jepa/pool_size": 0,
    }
    if M == 0:
        return torch.zeros((), device=device, dtype=dtype), empty_metrics

    if has_code is None or teacher_code is None:
        has_code = torch.zeros(M, dtype=torch.bool, device=device)
    else:
        has_code = has_code.to(device=device, dtype=torch.bool)

    # Center by the shared batch mean (detached: centering must not become a
    # gradient path into the teacher), then re-normalize to the unit sphere.
    views = [pred_pos, pred_neg, teacher] + ([teacher_code[has_code]] if has_code.any() else [])
    mu = torch.cat(views, dim=0).mean(dim=0, keepdim=True).detach()
    pred_pos = F.normalize(pred_pos - mu, dim=-1)
    pred_neg = F.normalize(pred_neg - mu, dim=-1)
    teacher = F.normalize(teacher - mu, dim=-1)

    sim_pos = (pred_pos * teacher).sum(dim=-1)    # (M,) cos(p+, z_cot)
    sim_neg = (pred_neg * teacher).sum(dim=-1)    # (M,) cos(p-, z_cot)
    sp_terms, sn_terms = [sim_pos], [sim_neg]
    cos_code = 0.0
    n_code = int(has_code.sum().item())
    if n_code > 0:
        z_code = F.normalize(teacher_code[has_code] - mu, dim=-1)
        sp_c = (pred_pos[has_code] * z_code).sum(dim=-1)
        sn_c = (pred_neg[has_code] * z_code).sum(dim=-1)
        sp_terms.append(sp_c)
        sn_terms.append(sn_c)
        cos_code = float(sp_c.detach().mean().cpu())

    sp = torch.cat(sp_terms)   # (M + n_code,)
    sn = torch.cat(sn_terms)
    logits = torch.stack([sp, sn], dim=-1) / tau
    loss = F.cross_entropy(logits, torch.zeros(sp.shape[0], dtype=torch.long, device=device))

    metrics = {
        "jepa/llm_jepa_loss": float(loss.detach().cpu()),
        "jepa/tcr_loss": float(loss.detach().cpu()),  # alias: wandb key parity
        "jepa/infonce_loss": float(loss.detach().cpu()),
        "jepa/cos_cot": float(sim_pos.detach().mean().cpu()),
        "jepa/cos_code": cos_code,
        "jepa/cos_neg": float(sn.detach().mean().cpu()),
        "jepa/margin": float((sp - sn).detach().mean().cpu()),
        "jepa/infonce_acc": float((sp > sn).float().mean().cpu()),
        "jepa/infonce_tau": float(tau),
        "jepa/n_anchors_cot": int(M),
        "jepa/n_anchors_code": n_code,
        "jepa/pool_size": int(2 * M),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Explicit correct/teacher/wrong geometry (loss_type='llm-jepa-geometry')
# ---------------------------------------------------------------------------

def representation_effective_rank(embeddings: torch.Tensor) -> tuple[float, float]:
    """Return raw and normalized effective rank for a representation pool."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return 0.0, 0.0
    with torch.no_grad():
        # Do not center: centering a collapsed cone removes its common direction
        # and leaves only isotropic jitter, falsely reporting healthy rank.
        x = F.normalize(embeddings.detach().float(), dim=-1)
        eig = torch.linalg.eigvalsh(x @ x.T).clamp_min_(0)
        total = eig.sum()
        max_rank = min(int(embeddings.shape[0]), int(embeddings.shape[1]))
        if max_rank <= 0 or float(total) <= 1e-12:
            return 0.0, 0.0
        p = eig / total
        p = p[p > 1e-12]
        erank = torch.exp(-(p * p.log()).sum())
        raw = float(erank.cpu())
        return raw, min(1.0, raw / max_rank)


def llm_jepa_geometry_loss(
    correct: torch.Tensor,
    wrong: torch.Tensor,
    teacher: torch.Tensor,
    group_id: torch.Tensor,
    teacher_code: torch.Tensor | None = None,
    has_code: torch.Tensor | None = None,
    tau: float = 0.1,
    margin: float = 0.1,
    align_weight: float = 1.0,
    wrong_teacher_weight: float = 1.0,
    wrong_correct_weight: float = 1.0,
    sig_pool: torch.Tensor | None = None,
    sigreg_lambda: float = 0.0,
    sig_M: int = 1024,
    sig_n_freq: int = 17,
    sig_t_min: float = -5.0,
    sig_t_max: float = 5.0,
    sig_s: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Attract correct student reads to fixed CoT and Code teacher anchors.

    Both teacher views are stop-gradient references. Attraction therefore moves
    only the correct student, while both repulsion terms move only the wrong
    student. Pairs are averaged within prompt before prompts are averaged.
    """
    device, dtype = correct.device, correct.dtype
    count = int(correct.shape[0])
    empty = {
        "jepa/llm_jepa_loss": 0.0, "jepa/tcr_loss": 0.0,
        "jepa/geometry_loss": 0.0, "jepa/geometry_align_loss": 0.0,
        "jepa/geometry_wrong_teacher_loss": 0.0,
        "jepa/geometry_wrong_correct_loss": 0.0,
        "jepa/cos_correct_teacher": 0.0, "jepa/cos_wrong_teacher": 0.0,
        "jepa/cos_wrong_correct": 0.0, "jepa/margin_teacher": 0.0,
        "jepa/margin_correct": 0.0, "jepa/violation_teacher": 0.0,
        "jepa/violation_correct": 0.0, "jepa/sigreg_loss": 0.0,
        "jepa/effective_rank": 0.0, "jepa/effective_rank_norm": 0.0,
        "jepa/n_anchors_cot": 0, "jepa/n_anchors_code": 0, "jepa/pool_size": 0,
    }
    if count == 0:
        return torch.zeros((), device=device, dtype=dtype), empty
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    correct = F.normalize(correct, dim=-1)
    wrong = F.normalize(wrong, dim=-1)
    teacher = F.normalize(teacher.detach(), dim=-1)
    has_code = (torch.zeros(count, dtype=torch.bool, device=device) if has_code is None
                else has_code.to(device=device, dtype=torch.bool))

    cos_ct_cot = (correct * teacher).sum(dim=-1)
    cos_wt_cot = (wrong * teacher.detach()).sum(dim=-1)
    align_sum = 1.0 - cos_ct_cot
    repel_teacher_sum = tau * F.softplus(
        (cos_wt_cot - cos_ct_cot.detach() + margin) / tau
    )
    reference_sum = cos_ct_cot
    view_count = torch.ones(count, device=device, dtype=dtype)
    cos_ct_views = [cos_ct_cot]
    cos_wt_views = [cos_wt_cot]

    n_code = int(has_code.sum().item()) if teacher_code is not None else 0
    if n_code > 0:
        code = F.normalize(teacher_code[has_code].detach(), dim=-1)
        cos_ct_code = (correct[has_code] * code).sum(dim=-1)
        cos_wt_code = (wrong[has_code] * code.detach()).sum(dim=-1)
        align_sum = align_sum.clone()
        repel_teacher_sum = repel_teacher_sum.clone()
        reference_sum = reference_sum.clone()
        align_sum[has_code] += 1.0 - cos_ct_code
        repel_teacher_sum[has_code] += tau * F.softplus(
            (cos_wt_code - cos_ct_code.detach() + margin) / tau
        )
        reference_sum[has_code] += cos_ct_code
        view_count[has_code] += 1.0
        cos_ct_views.append(cos_ct_code)
        cos_wt_views.append(cos_wt_code)

    align_pair = align_sum / view_count
    repel_teacher_pair = repel_teacher_sum / view_count
    reference_pair = (reference_sum / view_count).detach()
    cos_wc = (wrong * correct.detach()).sum(dim=-1)
    repel_correct_pair = tau * F.softplus(
        (cos_wc - reference_pair + margin) / tau
    )

    group_id = group_id.to(device=device, dtype=torch.long)
    align = _prompt_mean(align_pair, group_id).mean()
    repel_teacher = _prompt_mean(repel_teacher_pair, group_id).mean()
    repel_correct = _prompt_mean(repel_correct_pair, group_id).mean()
    geometry = (
        float(align_weight) * align
        + float(wrong_teacher_weight) * repel_teacher
        + float(wrong_correct_weight) * repel_correct
    )

    lam = float(sigreg_lambda)
    if sig_pool is not None and sig_pool.numel() and lam > 0.0:
        sig = sigreg_loss(sig_pool, M=sig_M, n_freq=sig_n_freq,
                          t_min=sig_t_min, t_max=sig_t_max, s=sig_s)
        loss = (1.0 - lam) * geometry + lam * sig
    else:
        sig = geometry.new_zeros(())
        loss = geometry

    rank_pool = sig_pool if sig_pool is not None and sig_pool.numel() else torch.cat(
        [correct, wrong], dim=0
    )
    effective_rank, effective_rank_norm = representation_effective_rank(rank_pool)
    cos_ct = torch.cat(cos_ct_views)
    cos_wt = torch.cat(cos_wt_views)
    teacher_margin = cos_ct - cos_wt
    correct_margin = reference_pair - cos_wc
    metrics = {
        "jepa/llm_jepa_loss": float(loss.detach().cpu()),
        "jepa/tcr_loss": float(loss.detach().cpu()),
        "jepa/geometry_loss": float(geometry.detach().cpu()),
        "jepa/geometry_align_loss": float(align.detach().cpu()),
        "jepa/geometry_wrong_teacher_loss": float(repel_teacher.detach().cpu()),
        "jepa/geometry_wrong_correct_loss": float(repel_correct.detach().cpu()),
        "jepa/cos_correct_teacher": float(cos_ct.detach().mean().cpu()),
        "jepa/cos_wrong_teacher": float(cos_wt.detach().mean().cpu()),
        "jepa/cos_wrong_correct": float(cos_wc.detach().mean().cpu()),
        "jepa/margin_teacher": float(teacher_margin.detach().mean().cpu()),
        "jepa/margin_correct": float(correct_margin.detach().mean().cpu()),
        "jepa/violation_teacher": float((teacher_margin.detach() < margin).float().mean().cpu()),
        "jepa/violation_correct": float((correct_margin.detach() < margin).float().mean().cpu()),
        "jepa/sigreg_loss": float(sig.detach().cpu()),
        "jepa/effective_rank": effective_rank,
        "jepa/effective_rank_norm": effective_rank_norm,
        "jepa/cos_cot": float(cos_ct_cot.detach().mean().cpu()),
        "jepa/cos_code": float(cos_ct_views[1].detach().mean().cpu()) if n_code > 0 else 0.0,
        "jepa/cos_neg": float(cos_wt.detach().mean().cpu()),
        "jepa/margin": float(teacher_margin.detach().mean().cpu()),
        "jepa/n_anchors_cot": count,
        "jepa/n_anchors_code": n_code,
        "jepa/pool_size": int(rank_pool.shape[0]),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Teacher-anchored triplet (loss_type='llm-jepa-triplet')
# ---------------------------------------------------------------------------

def llm_jepa_triplet_loss(
    correct: torch.Tensor,       # (M, d) p+ correct-rollout reads, L2-normalized
    wrong: torch.Tensor,         # (M, d) p- wrong-rollout reads, L2-normalized
    teacher: torch.Tensor,       # (M, d) z+_cot teacher CoT reads, L2-norm, NOT detached
    group_id: torch.Tensor,      # (M,) long — prompt id per pair
    teacher_code: torch.Tensor | None = None,  # (M, d) z+_code, NOT detached
    has_code: torch.Tensor | None = None,      # (M,) bool — rows with a code teacher view
    tau: float = 0.1,
    margin: float = 0.1,
    align_weight: float = 1.0,
    repel_weight: float = 1.0,
    include_cot: bool = True,   # False = Code-only (paper-faithful cross-view target;
                                # CoT-vs-CoT is same-modality and near-saturated, see
                                # core_algos module docstring / llm_jepa_paper_loss)
    detach_teacher: bool = False,  # True = a_i is sg(f_theta(y^T)); see below
    sig_pool: torch.Tensor | None = None,
    sigreg_lambda: float = 0.0,
    sig_M: int = 1024, sig_n_freq: int = 17,
    sig_t_min: float = -5.0, sig_t_max: float = 5.0, sig_s: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Minimal discriminative extension of the paper LLM-JEPA cosine loss.

    Keeps the paper's ALIGN term (pull the correct student read onto the teacher
    view(s)) and adds ONE hinge that pushes the wrong read farther from the teacher
    than the correct read:

        L = align_weight · mean_v (1 − cos(p+, z_v))
          + repel_weight · mean_v τ·softplus( (cos(p-, z_v) − cos(p+, z_v) + m)/τ )

    over teacher views v ∈ {cot (if include_cot), code (if present)}, averaged
    within prompt then across prompts (``_prompt_mean``). Rows with NO active view
    (include_cot=False and no code teacher) contribute nothing (excluded from the
    prompt-mean, not zero-filled). Unlike ``llm_jepa_geometry_loss`` there is no
    wrong↔correct repel arm. The optional SIGReg convex mix (student pool only) is off
    by default.

    ``detach_teacher`` — THE anchor-dragging fix, and the reason this loss failed before.
    The teacher view is ``z_v = f_theta(y^T)``: the SAME policy encoding the cached teacher
    tokens, so with detach_teacher=False (paper fidelity, as in the reference finetune.py)
    it stays in the differentiable graph. The cheapest way to descend is then to move the
    ANCHOR toward the student cloud, which raises cos(p+,z) and cos(p-,z) by the same
    amount and leaves the margin at zero. Measured, run 0ntqn3zc (170 steps, 3B):
    cos_correct 0.8525 -> 0.9393 and cos_wrong 0.8550 -> 0.9445 rose in lockstep,
    jepa/margin stayed at ~0, and jepa/violation was pinned at 1.00 for every step.
    detach_teacher=True makes a_i = sg(f_theta(y^T)) a fixed point, so raising s+ while
    lowering s- requires actually separating the two student clouds. It also removes the
    need for a SIGReg guard: uniform contraction toward a frozen anchor cannot satisfy a
    RELATIVE objective, so there is no degenerate optimum left to guard against.
    """
    device, dtype = correct.device, correct.dtype
    count = int(correct.shape[0])
    empty = {
        "jepa/llm_jepa_loss": 0.0, "jepa/tcr_loss": 0.0,
        "jepa/triplet_loss": 0.0, "jepa/align_cot": 0.0, "jepa/align_code": 0.0,
        "jepa/repel_teacher": 0.0, "jepa/cos_cot": 0.0, "jepa/cos_code": 0.0,
        "jepa/cos_neg": 0.0, "jepa/margin": 0.0, "jepa/violation": 0.0,
        "jepa/sigreg_loss": 0.0, "jepa/effective_rank": 0.0,
        "jepa/effective_rank_norm": 0.0, "jepa/n_anchors_cot": 0,
        "jepa/n_anchors_code": 0, "jepa/pool_size": 0,
    }
    if count == 0:
        return torch.zeros((), device=device, dtype=dtype), empty
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    correct = F.normalize(correct, dim=-1)
    wrong = F.normalize(wrong, dim=-1)
    teacher = F.normalize(teacher, dim=-1)
    if detach_teacher:
        teacher = teacher.detach()          # a_i = sg(f_theta(y^T))
    has_code = (torch.zeros(count, dtype=torch.bool, device=device) if has_code is None
                else has_code.to(device=device, dtype=torch.bool))

    align_sum = torch.zeros(count, device=device, dtype=dtype)
    repel_sum = torch.zeros(count, device=device, dtype=dtype)
    view_count = torch.zeros(count, device=device, dtype=dtype)
    cos_ct_views: list[torch.Tensor] = []
    cos_wt_views: list[torch.Tensor] = []
    cos_ct_cot = cos_wt_cot = None

    if include_cot:
        cos_ct_cot = (correct * teacher).sum(dim=-1)
        cos_wt_cot = (wrong * teacher).sum(dim=-1)
        align_sum = align_sum + (1.0 - cos_ct_cot)
        # Hinge reference is the LIVE correct cosine (no .detach): the wrong read is
        # pushed below cos(p+, z) − margin, and gradient is free to move p+, p- and z.
        repel_sum = repel_sum + tau * F.softplus((cos_wt_cot - cos_ct_cot + margin) / tau)
        view_count = view_count + 1.0
        cos_ct_views.append(cos_ct_cot)
        cos_wt_views.append(cos_wt_cot)

    n_code = int(has_code.sum().item()) if teacher_code is not None else 0
    cos_ct_code = cos_wt_code = None
    if n_code > 0:
        code = F.normalize(teacher_code[has_code], dim=-1)
        if detach_teacher:
            code = code.detach()
        cos_ct_code = (correct[has_code] * code).sum(dim=-1)
        cos_wt_code = (wrong[has_code] * code).sum(dim=-1)
        align_sum = align_sum.clone()
        repel_sum = repel_sum.clone()
        view_count = view_count.clone()
        align_sum[has_code] += 1.0 - cos_ct_code
        repel_sum[has_code] += tau * F.softplus((cos_wt_code - cos_ct_code + margin) / tau)
        view_count[has_code] += 1.0
        cos_ct_views.append(cos_ct_code)
        cos_wt_views.append(cos_wt_code)

    # Rows with no active view (possible when include_cot=False and that row has no
    # code teacher) are excluded from the prompt-mean rather than zero-filled, so
    # they neither help nor hurt the loss. If NO row has any active view this step
    # (e.g. include_cot=False and no code teachers sampled), fall back to the same
    # zero-loss empty-batch path as count == 0.
    valid = view_count > 0
    if not bool(valid.any()):
        return torch.zeros((), device=device, dtype=dtype), empty
    align_pair = align_sum[valid] / view_count[valid]
    repel_pair = repel_sum[valid] / view_count[valid]
    group_id = group_id.to(device=device, dtype=torch.long)[valid]
    align = _prompt_mean(align_pair, group_id).mean()
    repel = _prompt_mean(repel_pair, group_id).mean()
    triplet = float(align_weight) * align + float(repel_weight) * repel

    lam = float(sigreg_lambda)
    if sig_pool is not None and sig_pool.numel() and lam > 0.0:
        sig = sigreg_loss(sig_pool, M=sig_M, n_freq=sig_n_freq,
                          t_min=sig_t_min, t_max=sig_t_max, s=sig_s)
        loss = (1.0 - lam) * triplet + lam * sig
    else:
        sig = triplet.new_zeros(())
        loss = triplet

    rank_pool = sig_pool if sig_pool is not None and sig_pool.numel() else torch.cat(
        [correct, wrong], dim=0
    )
    effective_rank, effective_rank_norm = representation_effective_rank(rank_pool)
    cos_ct = torch.cat(cos_ct_views) if cos_ct_views else correct.new_zeros((0,))
    cos_wt = torch.cat(cos_wt_views) if cos_wt_views else correct.new_zeros((0,))
    teacher_margin = cos_ct - cos_wt
    metrics = {
        "jepa/llm_jepa_loss": float(loss.detach().cpu()),
        "jepa/tcr_loss": float(loss.detach().cpu()),
        "jepa/triplet_loss": float(triplet.detach().cpu()),
        "jepa/align_cot": float((1.0 - cos_ct_cot).detach().mean().cpu()) if include_cot else 0.0,
        "jepa/align_code": float((1.0 - cos_ct_code).detach().mean().cpu()) if n_code > 0 else 0.0,
        "jepa/repel_teacher": float(repel.detach().cpu()),
        "jepa/cos_cot": float(cos_ct_cot.detach().mean().cpu()) if include_cot else 0.0,
        "jepa/cos_code": float(cos_ct_code.detach().mean().cpu()) if n_code > 0 else 0.0,
        "jepa/cos_neg": float(cos_wt.detach().mean().cpu()) if cos_wt.numel() else 0.0,
        "jepa/margin": float(teacher_margin.detach().mean().cpu()) if teacher_margin.numel() else 0.0,
        "jepa/violation": float((teacher_margin.detach() < margin).float().mean().cpu()) if teacher_margin.numel() else 0.0,
        "jepa/sigreg_loss": float(sig.detach().cpu()),
        "jepa/effective_rank": effective_rank,
        "jepa/effective_rank_norm": effective_rank_norm,
        "jepa/n_anchors_cot": count,
        "jepa/n_anchors_code": n_code,
        "jepa/pool_size": int(rank_pool.shape[0]),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Hidden-State GRPO (loss_type='h-grpo')
# ---------------------------------------------------------------------------

def hgrpo_loss(
    student: torch.Tensor,      # (M, d) ALL rollout reads of the contributing prompts
    label: torch.Tensor,        # (M,) float verifier correctness c_ij in {0, 1}
    teacher: torch.Tensor,      # (P, d) one anchor per prompt — DETACHED here
    group_id: torch.Tensor,     # (M,) long, compact prompt id in 0..P-1
    eps_a: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """H-GRPO: GRPO's group-relative surrogate with the token policy replaced by a
    group-normalised distribution over hidden-state compatibility.

        a_i  = sg(normalize(f_theta(x_i, y_i^T)))        frozen correct-solution anchor
        s_ij = a_i . z_ij                                compatibility of rollout j
        q_i  = softmax_j(s_ij)                           the "representation policy"
        A_ij = (c_ij - cbar_i) / (sigma_i + eps)         group-relative advantage, cbar/sigma
                                                         q_old-weighted; sum_j q_old A = 0
        L    = -mean_i sum_j A_ij q_ij(theta)

    WHY A SOFTMAX AND NOT A MARGIN. q is exactly invariant to s_j -> s_j + gamma, so
    raising every cosine by the same amount is not a descent direction — it is not a
    direction in the objective at all. That common mode is exactly how the sibling
    ``llm_jepa_triplet_loss`` failed (run 0ntqn3zc, 170 steps at 3B): its absolute
    ``1 - cos(p+, a)`` align term was satisfied by dragging everything toward the anchor,
    cos_correct 0.8525->0.9393 and cos_wrong 0.8550->0.9445 in lockstep, margin ~0 and
    violation pinned at 1.00 throughout. There is no absolute attraction term here.

    THE MISSING CLIP IS NOT MISSING. The full objective is the clipped ratio surrogate
    ``-mean_i sum_j q_old_ij min[r A, clip(r, 1+-eps) A]`` with ``r = q_theta/q_old``.
    Here q_old is the DETACHED VALUE OF THIS SAME FORWARD (``q.detach()``), not a stale
    policy's -- there is one JEPA forward per rollout batch and the surrogate is evaluated
    at the very point where it is differentiated. So r == 1 identically, the clip is
    inactive by construction, and substituting collapses the surrogate algebraically:

        -sum_j q_old (q/q_old) A = -sum_j A q
        dL/ds_il = -q_il (A_il - sum_j q_j A_j) = -q_il^old A_il

    the last step because sum_j q_old A = 0 by construction of A. Note also that A comes
    from the verifier LABEL, which is theta-independent: the PPO ratio corrects for stale
    policy ESTIMATES, and there is no estimate here to correct.

    This identity holds however the optimizer is scheduled, but the TRAINING DYNAMICS do
    not: jepa.fuse_optimizer_step=True is required (enforced in config_ray) so that
    L_GRPO + alpha*L_H produces ONE step over the summed gradient, as in LLM-JEPA. Two
    sequential Adam steps on the same optimizer are not equivalent -- Adam rescales each
    gradient by its own second moment, so alpha would stop controlling the relative
    contribution and the shared m/v buffers would be updated alternately by two
    differently scaled gradients.

    THE VIRTUAL TEACHER MEMBER. Column 0 of every group is the anchor itself: s_i0 = 1
    (constant, no gradient) with c_i0 = 1. It costs nothing (no extra forward — the anchor
    is already encoded) and it is what makes ALL-WRONG groups trainable: with students-only
    labels all equal, sigma = 0 and every A vanishes, whereas the teacher member gives the
    wrong rollouts a negative advantage and pushes their scores down. All-correct groups
    still yield A == 0 and are dropped by the batch builder rather than wasting a forward.

    No temperature: q = exp(s)/sum exp(s) as specified, so the only coefficient this
    objective adds to L_GRPO is the jepa.alpha that scales it.
    """
    device, dtype = student.device, student.dtype
    M = int(student.shape[0])
    empty = {
        "jepa/llm_jepa_loss": 0.0, "jepa/tcr_loss": 0.0, "jepa/hgrpo_loss": 0.0,
        "jepa/cos_cot": 0.0, "jepa/cos_neg": 0.0, "jepa/margin": 0.0,
        "jepa/violation": 0.0, "jepa/hgrpo_q_correct": 0.0,
        "jepa/hgrpo_q_teacher": 0.0, "jepa/n_allwrong_groups": 0,
        "jepa/effective_rank": 0.0, "jepa/effective_rank_norm": 0.0,
        "jepa/n_anchors_cot": 0, "jepa/n_anchors_code": 0, "jepa/pool_size": 0,
    }
    if M == 0:
        return torch.zeros((), device=device, dtype=dtype), empty

    gid = group_id.to(device=device, dtype=torch.long)
    P = int(teacher.shape[0])
    c = label.to(device=device, dtype=dtype)

    student = F.normalize(student, dim=-1)
    a = F.normalize(teacher, dim=-1).detach()      # a_i = sg(f_theta(y^T)) — mandatory
    s = (student * a[gid]).sum(dim=-1)             # (M,)

    # Scatter the ragged groups into a padded (P, Gmax) matrix. Column 0 is the virtual
    # teacher member (s=1, c=1); students occupy columns 1.. in arrival order. Absent
    # slots carry -inf so softmax assigns them exactly zero mass and they drop out of
    # every q-weighted statistic with no manual renormalisation.
    counts = torch.bincount(gid, minlength=P)
    offsets = counts.cumsum(0) - counts               # first slot of each group
    rank_in_group = torch.empty(M, dtype=torch.long, device=device)
    order = torch.argsort(gid, stable=True)
    rank_in_group[order] = torch.arange(M, device=device) - offsets[gid[order]]
    col = rank_in_group + 1                           # column 0 is the teacher
    Gmax = int(counts.max().item()) + 1

    s_mat = torch.full((P, Gmax), float("-inf"), device=device, dtype=dtype)
    s_mat[:, 0] = 1.0                                 # a_i . a_i = 1, a constant
    s_mat = s_mat.index_put((gid, col), s)            # out-of-place: differentiable
    c_mat = torch.zeros((P, Gmax), device=device, dtype=dtype)
    c_mat[:, 0] = 1.0
    c_mat = c_mat.index_put((gid, col), c)
    slot = torch.zeros((P, Gmax), dtype=torch.bool, device=device).index_put(
        (gid, col), torch.ones(M, dtype=torch.bool, device=device))

    q = torch.softmax(s_mat, dim=1)                   # (P, Gmax), differentiable via s
    q_old = q.detach()
    cbar = (q_old * c_mat).sum(dim=1, keepdim=True)
    sigma = (q_old * (c_mat - cbar).pow(2)).sum(dim=1, keepdim=True).clamp_min(0).sqrt()
    adv = ((c_mat - cbar) / (sigma + eps_a)).detach()

    loss = -(adv * q).sum(dim=1).mean()

    # ---- metrics (no gradient, no host sync per prompt) ----
    with torch.no_grad():
        sd, cd = s.detach(), c.detach()
        pos, neg = cd > 0.5, cd <= 0.5
        pos_m = slot & (c_mat > 0.5)
        neg_m = slot & (c_mat <= 0.5)
        n_pos, n_neg = pos_m.sum(1), neg_m.sum(1)
        mixed = (n_pos > 0) & (n_neg > 0)
        if bool(mixed.any()):
            s_pad = s_mat.detach().masked_fill(~slot, 0.0)
            # Delta_i of the spec: per-prompt mean(s | correct) - mean(s | wrong),
            # averaged over the prompts that actually have both.
            mu_p = (s_pad * pos_m).sum(1) / n_pos.clamp_min(1)
            mu_n = (s_pad * neg_m).sum(1) / n_neg.clamp_min(1)
            margin = float((mu_p - mu_n)[mixed].mean().cpu())
            # violation = fraction of within-prompt (correct, wrong) pairs the anchor
            # ranks the WRONG way; 1 - violation is nearest-anchor retrieval accuracy.
            bad = ((s_pad[:, :, None] <= s_pad[:, None, :])
                   & pos_m[:, :, None] & neg_m[:, None, :]).sum()
            margin_pairs = int((n_pos * n_neg).sum())
            violation = float(bad) / max(margin_pairs, 1)
        else:
            margin, violation = 0.0, 0.0
        erank, erank_norm = representation_effective_rank(student)
        q_correct = float((q_old * pos_m).sum(dim=1).mean().cpu())
        n_allwrong = int((n_pos == 0).sum())

    metrics = {
        "jepa/llm_jepa_loss": float(loss.detach().cpu()),
        "jepa/tcr_loss": float(loss.detach().cpu()),
        "jepa/hgrpo_loss": float(loss.detach().cpu()),
        "jepa/cos_cot": float(sd[pos].mean().cpu()) if bool(pos.any()) else 0.0,
        "jepa/cos_neg": float(sd[neg].mean().cpu()) if bool(neg.any()) else 0.0,
        "jepa/margin": margin,
        "jepa/violation": violation,
        "jepa/hgrpo_q_correct": q_correct,
        "jepa/hgrpo_q_teacher": float(q_old[:, 0].mean().cpu()),
        "jepa/n_allwrong_groups": n_allwrong,
        "jepa/effective_rank": erank,
        "jepa/effective_rank_norm": erank_norm,
        "jepa/n_anchors_cot": M,
        "jepa/n_anchors_code": 0,
        "jepa/pool_size": M,
    }
    return loss, metrics


if __name__ == "__main__":
    # Self-check for llm_jepa_contrastive_loss: gating, [0,1] bounds, per-prompt
    # collapse (anchor count must not change a prompt's unit weight), and arm-A
    # equivalence with the paper loss on a good-only, one-anchor-per-prompt batch.
    torch.manual_seed(0)
    d = 16

    def _unit(n):
        return F.normalize(torch.randn(n, d), dim=-1)

    empty = torch.zeros(0, d)
    empty_g = torch.zeros(0, dtype=torch.long)

    # (1) good-only, one anchor per prompt == paper loss (align_cot + align_code averaged
    #     as two pooled units per prompt). Compare per-arm means to the paper fn.
    gp, tc, tk = _unit(4), _unit(4), _unit(4)
    hcode = torch.ones(4, dtype=torch.bool)
    gg = torch.arange(4)
    loss, m = llm_jepa_contrastive_loss(
        gp, tc, tk, hcode, gg, empty, empty, empty_g, empty, empty, empty_g)
    paper_loss, pm = llm_jepa_paper_loss(gp, tc, tk, hcode)
    # paper = align_cot + align_code (sum); contrastive pools the two as separate units
    # -> mean of the two, i.e. paper/2. Check the per-arm means match the paper's arms.
    assert abs(m["jepa/align_cot"] - pm["jepa/align_cot"]) < 1e-5, (m["jepa/align_cot"], pm["jepa/align_cot"])
    assert abs(m["jepa/align_code"] - pm["jepa/align_code"]) < 1e-5
    assert abs(float(loss) - (pm["jepa/align_cot"] + pm["jepa/align_code"]) / 2.0) < 1e-5
    assert 0.0 <= float(loss) <= 2.0  # cosine distance 1-cos in [0,2]; contrast (1+cos)/2 in [0,1]
    assert m["jepa/n_bad_anchors"] == 0 and m["jepa/n_contrast_pairs"] == 0

    # (2) per-prompt collapse: duplicating one prompt's good anchors must NOT change loss.
    gp2 = torch.cat([gp, gp[:1]], 0); tc2 = torch.cat([tc, tc[:1]], 0)
    tk2 = torch.cat([tk, tk[:1]], 0); hc2 = torch.cat([hcode, hcode[:1]], 0)
    gg2 = torch.cat([gg, gg[:1]], 0)  # extra anchor shares prompt 0
    loss2, _ = llm_jepa_contrastive_loss(
        gp2, tc2, tk2, hc2, gg2, empty, empty, empty_g, empty, empty, empty_g)
    assert abs(float(loss) - float(loss2)) < 1e-5, (float(loss), float(loss2))

    # (3) all-wrong: only bad arm fires, still in [0,1], good/contrast absent.
    bp, bt, bgid = _unit(3), _unit(3), torch.tensor([0, 0, 1])
    lb, mb = llm_jepa_contrastive_loss(
        empty, empty, empty, torch.zeros(0, dtype=torch.bool), empty_g,
        bp, bt, bgid, empty, empty, empty_g)
    assert 0.0 <= float(lb) <= 2.0
    assert mb["jepa/n_prompts_bad"] == 2 and mb["jepa/n_prompts_cot"] == 0

    # (4) contrastive: identical good/bad reads -> cos=1 -> c=1; opposite -> c=0.
    g = _unit(2)
    lc_same, mc = llm_jepa_contrastive_loss(
        g, _unit(2), _unit(2), torch.ones(2, dtype=torch.bool), torch.tensor([0, 1]),
        empty, empty, empty_g, g, g, torch.tensor([0, 1]))
    assert abs(mc["jepa/contrastive"] - 1.0) < 1e-5
    lc_opp, mc2 = llm_jepa_contrastive_loss(
        g, _unit(2), _unit(2), torch.ones(2, dtype=torch.bool), torch.tensor([0, 1]),
        empty, empty, empty_g, g, -g, torch.tensor([0, 1]))
    assert abs(mc2["jepa/contrastive"] - 0.0) < 1e-5

    # (5) gradient flows to both branches (no stop-grad).
    gp.requires_grad_(True); tc.requires_grad_(True)
    lg, _ = llm_jepa_contrastive_loss(gp, tc, tk, hcode, gg, empty, empty, empty_g, empty, empty, empty_g)
    lg.backward()
    assert gp.grad is not None and tc.grad is not None and tc.grad.abs().sum() > 0

    # (6) SIGReg anti-collapse: a COLLAPSED (coned) pool must score HIGHER sigreg than a
    #     SPREAD (isotropic) pool, the convex mix must land between arms and sig, and the
    #     sig gradient must reach the pool reads (this is what breaks the cone in training).
    torch.manual_seed(1)
    dd = 64
    spread = F.normalize(torch.randn(96, dd), dim=-1)                     # isotropic
    cone = F.normalize(F.normalize(torch.randn(dd), dim=0) * 6 + torch.randn(96, dd) * 0.3, dim=-1)  # tight cone
    gp6, tc6, tk6 = F.normalize(torch.randn(4, dd), dim=-1), F.normalize(torch.randn(4, dd), dim=-1), F.normalize(torch.randn(4, dd), dim=-1)
    hc6, gg6 = torch.ones(4, dtype=torch.bool), torch.arange(4)
    e6, eg6 = torch.zeros(0, dd), torch.zeros(0, dtype=torch.long)
    _, ms = llm_jepa_contrastive_loss(gp6, tc6, tk6, hc6, gg6, e6, e6, eg6, e6, e6, eg6,
                                      sig_pool=spread, sigreg_lambda=0.1)
    _, mc6 = llm_jepa_contrastive_loss(gp6, tc6, tk6, hc6, gg6, e6, e6, eg6, e6, e6, eg6,
                                       sig_pool=cone, sigreg_lambda=0.1)
    assert mc6["jepa/sigreg_loss"] > ms["jepa/sigreg_loss"], (mc6["jepa/sigreg_loss"], ms["jepa/sigreg_loss"])
    # convex mix: loss between arms and sig when lambda in (0,1)
    assert min(mc6["jepa/arms_loss"], mc6["jepa/sigreg_loss"]) - 1e-6 <= mc6["jepa/llm_jepa_loss"] <= max(mc6["jepa/arms_loss"], mc6["jepa/sigreg_loss"]) + 1e-6
    # sig gradient reaches the pool
    pool_g = F.normalize(torch.randn(96, dd), dim=-1).requires_grad_(True)
    lsig, _ = llm_jepa_contrastive_loss(gp6, tc6, tk6, hc6, gg6, e6, e6, eg6, e6, e6, eg6,
                                        sig_pool=pool_g, sigreg_lambda=1.0)
    lsig.backward()
    assert pool_g.grad is not None and pool_g.grad.abs().sum() > 0

    print("llm_jepa_contrastive_loss self-check passed")

    # Self-check for llm_jepa_infonce_loss (teacher-anchored + centered): p+ near
    # z / p- far -> low loss, swapped -> high loss; p- ON the teacher -> higher
    # loss than p- random; the code view adds terms; grad reaches p+ and p- but
    # NEVER the teacher; empty input gates to 0.
    z = _unit(4)
    pp = z.clone()                                     # p+ == z+ (cos=1 survives shared centering)
    pn = F.normalize(torch.randn(4, d), dim=-1)
    l_good, mi = llm_jepa_infonce_loss(pp, pn, z, tau=0.1)
    l_bad, _ = llm_jepa_infonce_loss(pn, pp, z, tau=0.1)  # p+ random, p- == z+
    assert float(l_good) < float(l_bad)
    assert abs(mi["jepa/cos_cot"] - 1.0) < 1e-5 and mi["jepa/infonce_acc"] == 1.0
    # monotone in the negative: same p+/z+, p- sitting on the teacher -> higher loss
    l_close, _ = llm_jepa_infonce_loss(pp, z.clone(), z, tau=0.1)   # p- == z+
    assert float(l_close) > float(l_good)
    # code view: extra (pair, view) terms flow into the same softmax
    zc = _unit(4)
    hc = torch.tensor([True, True, False, False])
    l_dual, md = llm_jepa_infonce_loss(pp, pn, z, teacher_code=zc, has_code=hc, tau=0.1)
    assert md["jepa/n_anchors_code"] == 2 and md["jepa/n_anchors_cot"] == 4
    pp_g = pp.clone().requires_grad_(True)
    pn_g = pn.clone().requires_grad_(True)
    z_g = z.clone().requires_grad_(True)
    li, _ = llm_jepa_infonce_loss(pp_g, pn_g, z_g.detach(), tau=0.1)
    li.backward()
    assert pp_g.grad is not None and pp_g.grad.abs().sum() > 0
    assert pn_g.grad is not None and pn_g.grad.abs().sum() > 0
    assert z_g.grad is None
    l0, m0 = llm_jepa_infonce_loss(empty, empty, empty)
    assert float(l0) == 0.0 and m0["jepa/n_anchors_cot"] == 0
    print("llm_jepa_infonce_loss self-check passed")
