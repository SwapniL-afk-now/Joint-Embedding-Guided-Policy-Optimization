# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from omegaconf import DictConfig, OmegaConf


TAFR_VARIANTS = {"full", "anchor_only", "replay_only"}

TAFR_LOSS_VERSIONS = {"k3_legacy", "failure_token_adv"}

TAFR_ADV_NORMS = {"p99", "dense_token_credit", "pbrs_pure"}

TAFR_GATE_MODES = {"difficulty", "constant"}

TAFR_ANCHOR_KL_DIRECTIONS = {"forward", "legacy"}


@dataclass
class TAFRGRPOConfig:
    """Typed config for TAFR-GRPO.

    Two loss versions:

    k3_legacy (kept for ablation and existing checkpoints):

        L_t(theta) = L_GRPO(theta)
            + beta D_KL(pi_theta || pi_anchor^t)
            - beta(1 - r_bar_x) D_KL(pi_replay^t || pi_theta)

        where pi_anchor and pi_replay are frozen lagged EMA/reference mixtures.

    failure_token_adv (paper token-advantage path):

        L_t(theta) = PPO(A_GRPO + beta A_reg), where

            A^reg_{i,t} = log pi_anchor(y_{i,t} | s_{i,t})
                          - log pi_old(y_{i,t} | s_{i,t})
                          + (1 - r_bar_x) [log pi_old(y_{i,t} | s_{i,t})
                                           - log pi_failure(y_{i,t} | s_{i,t})].

        The complete detached A_reg is RMS-matched to the GRPO advantage; beta
        sets its relative RMS target (0.5 means approximately 50%).
    """

    enable: bool = False
    loss_version: str = "failure_token_adv"
    beta: float = 0.5
    # k3_legacy only: split `beta` into its two opposing halves. One knob cannot
    # strengthen the replay push without also strengthening the anchor pull that
    # suppresses it -- measured at beta 0.5/1.0, failure_sft_loss stays at ~0.29
    # (clone never fits) vs 0.073 at beta 0.1, i.e. replay goes inert at high beta.
    # None means "use beta", so existing arms are unchanged.
    beta_anchor: float | None = None
    beta_replay: float | None = None
    # Paper regularization-advantage construction.
    failure_advantage_clip: float = 5.0
    # None falls back to the GRPO clip_ratio.
    failure_clip_ratio: float | None = None
    # Reuse the exact chosen-token log-probabilities captured during vLLM
    # rollout instead of running a separate old-policy forward.
    reuse_rollout_log_probs: bool = False
    # When enabled, recompute the HF old log-probs AND use the rollout ones,
    # logging parity metrics until they are validated.
    verify_rollout_log_probs: bool = False
    # Ablation control: permute A_reg within each sequence's valid tokens, keeping
    # its magnitude and per-row statistics but destroying token alignment. If this
    # arm matches the real one, the per-token structure carries no information.
    shuffle_advantage_tokens: bool = False
    # Relative strength of A_reg's two halves. Summed BEFORE the joint RMS match,
    # so without these the mix is dictated by raw magnitudes that drift on their own
    # (anchor ramps between refreshes; failure grows with SFT updates).
    anchor_weight: float = 1.0
    failure_weight: float = 1.0
    # Normalize each half to unit RMS before weighting, making the realized mix
    # exactly anchor_weight : failure_weight regardless of that drift.
    normalize_terms_separately: bool = False
    # How A_reg is normalized before it is added to the GRPO advantage.
    #   p99                -- legacy: per-part unit RMS, then a p99-quantile match against
    #                         the batch RMS of the GRPO advantage. One global scalar, and
    #                         the result is NOT zero-mean within a sequence, so it shifts
    #                         the sequence-level GRPO preference.
    #   dense_token_credit -- residuals are zero-mean per sequence (sequence-level
    #                         preference untouched) and scaled by each rollout's own
    #                         |A_GRPO|, so the perturbation is the same fraction of every
    #                         rollout's advantage. anchor_weight/failure_weight act as
    #                         lambda_a/lambda_f; failure_advantage_clip is unused.
    adv_norm: str = "p99"
    # dense_token_credit knobs (ignored when adv_norm == "p99").
    dtc_dose: float = 0.5   # rms(A_dense) = dtc_dose * |A_grpo| per rollout
    dtc_tau: float = 5.0
    dtc_eps: float = 1e-8
    # Floor on the per-rollout |A_GRPO| scale. Homogeneous groups have A_GRPO == 0;
    # this keeps their dense term small but finite. Set to ~RMS(A_GRPO) (0.75 as
    # measured) to make those groups carry a full-strength dense signal instead.
    dtc_eps_floor: float = 0.01
    dtc_ema_decay: float = 0.99
    # Diagnostics: write decoded tokens next to their A_reg components here.
    # null disables. Dumps 2 wrong + 2 correct rows per step (~200 KB/step).
    token_advantage_dump_dir: Optional[str] = None
    token_advantage_dump_every: int = 1
    # Only run the failure-model prefill on groups with group_reward_mean < 1.
    score_only_active_groups: bool = True
    # Number of successful failure-SFT updates before the failure advantage
    # is activated (warm start). Persisted in tafr_state.pt.
    failure_min_updates_before_use: int = 1
    ema_gamma: float = 0.99
    mix_eta: float = 1.0
    sft_update_interval_grpo_steps: int = 5
    checkpoint_interval_grpo_steps: int = 10
    disable_builtin_kl: bool = True
    failure_sft_lr: float = 1e-6
    failure_sft_batch_size: int = 8
    # Token-budgeted micro-batching for failure-SFT. When > 0, records are packed
    # greedily so each micro-batch's padded token count (n_rows * max_len) stays
    # under this budget, raising GPU utilization vs the fixed failure_sft_batch_size
    # (which is used as a fallback / row cap when this is 0).
    failure_sft_max_token_len_per_gpu: int = 0
    failure_sft_max_updates_per_interval: int = 1
    # Throttle heavy TAFR disk writes independently of the EMA refresh cadence.
    # 0 = write on every refresh (legacy). When > 0, the EMA refresh + vLLM adapter
    # export still run every checkpoint_interval, but the failure model / optimizer /
    # tafr_state.pt are only written when global_step % this == 0. Old dirs are still
    # rotated away, so the latest saved state replaces the previous one.
    save_to_disk_interval_grpo_steps: int = 0
    # Skip persisting the failure-SFT Adam optimizer state (the largest TAFR file).
    # Set False to halve TAFR checkpoint size at the cost of losing failure-SFT
    # optimizer momentum across a resume.
    save_failure_sft_optimizer: bool = True
    failure_data_max_size: Optional[int] = None
    failure_data_sampling: str = "recent"
    anchor_checkpoint_dir: Optional[str] = None
    replay_checkpoint_dir: Optional[str] = None
    variant: str = "full"
    logprob_backend: str = "hf"
    vllm_score_micro_batch_size: int = 16
    # --- paper-program additions -------------------------------------------------
    # Build the anchor and failure models and log their KLs, but contribute exactly
    # zero to the actor loss. This is what lets a plain GRPO/GSPO/DAPO baseline row
    # report KL(actor, anchor) without any regularizer acting on it -- otherwise each
    # baseline needs a second instrumented run purely to fill that column.
    diagnostic_only: bool = False
    # How the replay push is weighted by observed group failure.
    #   difficulty -- g = 1 - r_bar_x, the paper's gate.
    #   constant   -- g = 1 for every group. The Table 3 "no gate" ablation: tests
    #                 whether conditioning on empirical failure frequency is needed.
    gate_mode: str = "difficulty"
    # Table 7 negative control. Replaces the learned failure density with uniform
    # negative cross-entropy on the SAME failed replay buffer, so the comparison
    # isolates "learned failure distribution" from "push down failed tokens".
    negative_ce: bool = False
    negative_ce_coef: float = 1.0
    # k3_legacy anchor direction. The k3 estimator (rho - 1 - log rho) recovers
    # KL(p||q) from samples of p only when rho = q/p. The legacy anchor term used
    # rho = pi_theta/pi_anchor, which is not a KL in either direction (it evaluates
    # to chi^2 - KL; measured 4.65 against a true KL of 1.03). "forward" uses
    # rho = pi_anchor/pi_theta and estimates KL(pi_theta || pi_anchor), which is what
    # Eq. (state_kls) of the paper specifies and what the replay term already does.
    # "legacy" reproduces the pre-fix behaviour for A/B against earlier runs.
    anchor_kl_direction: str = "forward"
    # Eq. (iwk3): the actions were sampled from pi_old, so recovering
    # D_KL(pi_theta || r) needs the importance weight w = pi_theta/pi_old. Without it
    # the estimator is biased the moment the actor moves off pi_old inside a PPO
    # epoch. Set False only to reproduce pre-fix runs.
    use_importance_weight: bool = True
    # Safeguard on w. Reported as tafr_grpo/iw_clip_fraction because clipping biases
    # the estimator; 0 disables.
    importance_weight_clip: float = 10.0

    @classmethod
    def from_config(cls, config: DictConfig | dict | None) -> "TAFRGRPOConfig":
        if not config:
            return cls()
        merged = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.create(config))
        return OmegaConf.to_object(merged)


def tafr_enabled(config: DictConfig | dict) -> bool:
    custom = config.get("custom_tafr_grpo", {}) if config is not None else {}
    return bool(custom and custom.get("enable", False))


def should_run_failure_sft(global_grpo_step: int, config: TAFRGRPOConfig) -> bool:
    return bool(config.enable and global_grpo_step % config.sft_update_interval_grpo_steps == 0)


def should_checkpoint_and_refresh(global_grpo_step: int, config: TAFRGRPOConfig) -> bool:
    return bool(config.enable and global_grpo_step % config.checkpoint_interval_grpo_steps == 0)


def validate_tafr_config(config: DictConfig | dict) -> TAFRGRPOConfig:
    """Validate root verl config and return the typed TAFR config."""

    custom = TAFRGRPOConfig.from_config(config.get("custom_tafr_grpo", {}) if config is not None else {})
    if not custom.enable:
        return custom

    algorithm = config.get("algorithm", {})
    actor_rollout_ref = config.get("actor_rollout_ref", {})
    actor = actor_rollout_ref.get("actor", {})
    model = actor_rollout_ref.get("model", {})

    raw_adv_estimator = algorithm.get("adv_estimator", "")
    adv_estimator = str(getattr(raw_adv_estimator, "value", raw_adv_estimator)).lower()
    if adv_estimator != "grpo" and not custom.diagnostic_only:
        if custom.loss_version == "failure_token_adv":
            # This path RMS-matches A_reg directly against the batch's GRPO advantage
            # tensor (compute_regularization_token_advantage), so it needs that tensor
            # to actually be a GRPO-style advantage.
            raise ValueError(
                "custom_tafr_grpo.loss_version='failure_token_adv' requires "
                "algorithm.adv_estimator='grpo' (it RMS-matches against the GRPO "
                "advantage tensor directly). Use loss_version='k3_legacy' to compose "
                "with a different adv_estimator, or diagnostic_only=true for a "
                "measurement-only baseline row."
            )
        # k3_legacy is otherwise fine composed with any estimator: its gate and KL
        # terms come from _tafr_group_reward_mean (raw per-uid reward, independent of
        # which estimator produced the policy loss's advantages) and from log-probs
        # only -- nothing here reads the advantage tensor. So FEPO[NGRPO]/FEPO[AVSPO]
        # is a valid composition, not just a diagnostic no-op.

    if custom.disable_builtin_kl:
        if bool(algorithm.get("use_kl_in_reward", False)):
            raise ValueError("TAFR-GRPO requires algorithm.use_kl_in_reward=false when disable_builtin_kl=true.")
        if bool(actor.get("use_kl_loss", False)):
            raise ValueError("TAFR-GRPO requires actor_rollout_ref.actor.use_kl_loss=false.")

    if custom.beta < 0:
        raise ValueError("custom_tafr_grpo.beta must be non-negative.")
    for _name in ("beta_anchor", "beta_replay"):
        _v = getattr(custom, _name, None)
        if _v is not None and float(_v) < 0:
            raise ValueError(f"custom_tafr_grpo.{_name} must be non-negative.")
        if _v is not None and custom.loss_version != "k3_legacy":
            raise ValueError(f"custom_tafr_grpo.{_name} only applies to loss_version=k3_legacy.")
    if custom.loss_version not in TAFR_LOSS_VERSIONS:
        raise ValueError(f"custom_tafr_grpo.loss_version must be one of {sorted(TAFR_LOSS_VERSIONS)}.")
    if custom.adv_norm not in TAFR_ADV_NORMS:
        raise ValueError(f"custom_tafr_grpo.adv_norm must be one of {sorted(TAFR_ADV_NORMS)}.")
    if custom.adv_norm != "p99" and custom.loss_version != "failure_token_adv":
        raise ValueError("custom_tafr_grpo.adv_norm only applies to loss_version='failure_token_adv'.")
    if custom.dtc_eps_floor <= 0:
        raise ValueError("custom_tafr_grpo.dtc_eps_floor must be positive.")
    if not 0.0 <= custom.dtc_ema_decay < 1.0:
        raise ValueError("custom_tafr_grpo.dtc_ema_decay must be in [0, 1).")
    if custom.dtc_dose < 0:
        raise ValueError("custom_tafr_grpo.dtc_dose must be non-negative.")
    if custom.dtc_tau <= 0:
        raise ValueError("custom_tafr_grpo.dtc_tau must be positive.")
    if custom.failure_advantage_clip <= 0:
        raise ValueError("custom_tafr_grpo.failure_advantage_clip must be positive.")
    if custom.failure_clip_ratio is not None and custom.failure_clip_ratio <= 0:
        raise ValueError("custom_tafr_grpo.failure_clip_ratio must be positive or null.")
    if custom.failure_min_updates_before_use < 1:
        raise ValueError("custom_tafr_grpo.failure_min_updates_before_use must be >= 1.")
    if not 0 <= custom.ema_gamma <= 1:
        raise ValueError("custom_tafr_grpo.ema_gamma must be in [0, 1].")
    if not 0 <= custom.mix_eta <= 1:
        raise ValueError("custom_tafr_grpo.mix_eta must be in [0, 1].")
    if custom.sft_update_interval_grpo_steps <= 0:
        raise ValueError("custom_tafr_grpo.sft_update_interval_grpo_steps must be positive.")
    if custom.checkpoint_interval_grpo_steps <= 0:
        raise ValueError("custom_tafr_grpo.checkpoint_interval_grpo_steps must be positive.")
    if custom.failure_sft_batch_size <= 0:
        raise ValueError("custom_tafr_grpo.failure_sft_batch_size must be positive.")
    if custom.failure_sft_max_token_len_per_gpu < 0:
        raise ValueError("custom_tafr_grpo.failure_sft_max_token_len_per_gpu must be >= 0 (0 disables token batching).")
    if custom.save_to_disk_interval_grpo_steps < 0:
        raise ValueError("custom_tafr_grpo.save_to_disk_interval_grpo_steps must be >= 0 (0 saves every refresh).")
    if custom.failure_sft_max_updates_per_interval <= 0:
        raise ValueError("custom_tafr_grpo.failure_sft_max_updates_per_interval must be positive (use 9999 for unlimited).")
    if custom.failure_data_max_size is not None and custom.failure_data_max_size <= 0:
        raise ValueError("custom_tafr_grpo.failure_data_max_size must be positive or null.")
    if custom.failure_data_sampling not in {"recent", "uniform"}:
        raise ValueError("custom_tafr_grpo.failure_data_sampling must be 'recent' or 'uniform'.")
    if custom.variant not in TAFR_VARIANTS:
        raise ValueError(f"custom_tafr_grpo.variant must be one of {sorted(TAFR_VARIANTS)}.")
    if custom.logprob_backend not in {"hf", "vllm"}:
        raise ValueError("custom_tafr_grpo.logprob_backend must be 'hf' or 'vllm'.")
    lora_rank = int(model.get("lora_rank", 0) or 0)
    if custom.logprob_backend == "vllm" and lora_rank <= 0:
        raise ValueError(
            "custom_tafr_grpo.logprob_backend='vllm' requires actor_rollout_ref.model.lora_rank > 0; "
            "use logprob_backend='hf' for full fine-tuning."
        )
    if custom.logprob_backend == "hf" and custom.reuse_rollout_log_probs:
        raise ValueError(
            "HF TAFR scoring requires reuse_rollout_log_probs=false so pi_old, pi_anchor, and pi_failure "
            "use the same log-probability backend."
        )
    if custom.logprob_backend == "vllm" and not custom.reuse_rollout_log_probs:
        raise ValueError(
            "vLLM TAFR scoring requires reuse_rollout_log_probs=true so pi_old, pi_anchor, and pi_failure "
            "use the same log-probability backend."
        )
    if custom.vllm_score_micro_batch_size <= 0:
        raise ValueError("custom_tafr_grpo.vllm_score_micro_batch_size must be positive.")
    if custom.gate_mode not in TAFR_GATE_MODES:
        raise ValueError(f"custom_tafr_grpo.gate_mode must be one of {sorted(TAFR_GATE_MODES)}.")
    if custom.anchor_kl_direction not in TAFR_ANCHOR_KL_DIRECTIONS:
        raise ValueError(
            f"custom_tafr_grpo.anchor_kl_direction must be one of {sorted(TAFR_ANCHOR_KL_DIRECTIONS)}."
        )
    if custom.negative_ce and custom.loss_version != "k3_legacy":
        raise ValueError("custom_tafr_grpo.negative_ce requires loss_version='k3_legacy'.")
    if custom.negative_ce and custom.diagnostic_only:
        raise ValueError("custom_tafr_grpo.negative_ce and diagnostic_only are mutually exclusive.")
    if custom.negative_ce_coef < 0:
        raise ValueError("custom_tafr_grpo.negative_ce_coef must be non-negative.")
    if custom.importance_weight_clip < 0:
        raise ValueError("custom_tafr_grpo.importance_weight_clip must be non-negative (0 disables).")

    return custom
