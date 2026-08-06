# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Typed config for GPI-CE and its validation against the root verl config."""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf


@dataclass
class GPICEConfig:
    """Group Policy-Improvement Cross Entropy.

    ``online_per_prompt`` is what vLLM generates (``actor_rollout_ref.rollout.n``);
    ``offline_per_prompt`` is appended from the dataset after the rollout replicas
    sleep. The candidate group the loss sees has ``group_size`` = the sum.
    """

    enable: bool = False
    online_per_prompt: int = 4
    offline_per_prompt: int = 4
    # tau in q* = softmax(s_old + r/tau). 0.5 for binary rewards.
    temperature: float = 0.5
    # Only "token_mean" is implemented: a sequence sum would let short responses
    # win the group softmax by accumulating fewer negative log-prob terms.
    score_normalization: str = "token_mean"
    # Re-run the training verifier on offline responses instead of trusting the
    # dataset's stored reward. Keeps online and offline reward semantics identical.
    verify_offline: bool = True
    # Fail instead of padding when a prompt has fewer than offline_per_prompt
    # stored responses.
    allow_short_offline: bool = False
    # Skip the separate old-policy forward and build q* inside the training forward.
    # Exact (not an approximation) because ppo_epochs=1 + one optimizer step means
    # theta_old == theta during the update, and force_group_size=K puts every complete
    # group inside one micro-batch, so q* is computable there from detached scores.
    fuse_old_log_prob: bool = True

    @property
    def group_size(self) -> int:
        return int(self.online_per_prompt) + int(self.offline_per_prompt)

    @classmethod
    def from_config(cls, cfg: DictConfig | dict | None) -> "GPICEConfig":  # noqa: UP037
        if cfg is None:
            return cls()
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in dict(cfg).items() if k in known})

    def to_dict(self) -> dict:
        return {
            "enable": bool(self.enable),
            "online_per_prompt": int(self.online_per_prompt),
            "offline_per_prompt": int(self.offline_per_prompt),
            "group_size": int(self.group_size),
            "temperature": float(self.temperature),
            "score_normalization": str(self.score_normalization),
            "verify_offline": bool(self.verify_offline),
            "allow_short_offline": bool(self.allow_short_offline),
            "fuse_old_log_prob": bool(self.fuse_old_log_prob),
        }


def validate_gpi_ce_config(config: DictConfig | dict) -> GPICEConfig:
    """Validate the root verl config and return the typed GPI-CE config."""
    custom = GPICEConfig.from_config(config.get("custom_gpi_ce", {}) if config is not None else {})
    if not custom.enable:
        return custom

    actor_rollout_ref = config.get("actor_rollout_ref", {})
    actor = actor_rollout_ref.get("actor", {})
    rollout = actor_rollout_ref.get("rollout", {})
    algorithm = config.get("algorithm", {})

    if custom.online_per_prompt < 1:
        raise ValueError("custom_gpi_ce.online_per_prompt must be >= 1 (the actor must stay on-policy).")
    if custom.offline_per_prompt < 0:
        raise ValueError("custom_gpi_ce.offline_per_prompt must be >= 0.")
    if custom.group_size < 2:
        raise ValueError("custom_gpi_ce.group_size must be >= 2; a one-candidate softmax has no gradient.")
    if custom.temperature <= 0:
        raise ValueError("custom_gpi_ce.temperature must be positive.")
    if custom.score_normalization != "token_mean":
        raise ValueError("custom_gpi_ce.score_normalization must be 'token_mean'.")

    rollout_n = int(rollout.get("n", 1))
    if rollout_n != custom.online_per_prompt:
        raise ValueError(
            f"actor_rollout_ref.rollout.n ({rollout_n}) must equal custom_gpi_ce.online_per_prompt "
            f"({custom.online_per_prompt}); the offline candidates are appended after generation, not sampled."
        )

    # One optimizer step per mixed batch: the target q* is built from theta_old, so a
    # second step would optimize a stale target.
    if int(actor.get("ppo_epochs", 1)) != 1:
        raise ValueError("custom_gpi_ce requires actor_rollout_ref.actor.ppo_epochs=1.")
    if bool(actor.get("shuffle", False)):
        raise ValueError("custom_gpi_ce requires actor_rollout_ref.actor.shuffle=false to keep groups contiguous.")
    if not bool(actor.get("accumulate_minibatch_grads", False)):
        raise ValueError(
            "custom_gpi_ce requires actor_rollout_ref.actor.accumulate_minibatch_grads=true "
            "so the whole mixed batch produces exactly one optimizer step."
        )
    # use_dynamic_bsz is ALLOWED: rearrange_micro_batches partitions whole groups when
    # force_group_size > 1 (seqlen_balancing.py), so token-budget batching keeps groups
    # atomic. It is also the safer choice here -- a group carrying a 20k-token offline
    # trace gets its own micro-batch instead of being packed by a fixed group count.
    if bool(actor.get("use_dynamic_bsz", False)):
        max_len = int(config.get("data", {}).get("max_prompt_length", 0)) + int(
            config.get("data", {}).get("max_response_length", 0)
        )
        budget = int(actor.get("ppo_max_token_len_per_gpu", 0))
        if budget < max_len:
            raise ValueError(
                f"custom_gpi_ce with use_dynamic_bsz=true needs ppo_max_token_len_per_gpu ({budget}) >= "
                f"max_prompt_length + max_response_length ({max_len}); rearrange_micro_batches asserts "
                "the budget can hold the longest single sequence."
            )

    # The KL trust region is already inside q* (the tau-weighted KL to p_old).
    if bool(actor.get("use_kl_loss", False)):
        raise ValueError("custom_gpi_ce requires actor_rollout_ref.actor.use_kl_loss=false (q* carries the KL).")
    if bool(algorithm.get("use_kl_in_reward", False)):
        raise ValueError("custom_gpi_ce requires algorithm.use_kl_in_reward=false.")

    loss_mode = actor.get("policy_loss", {}).get("loss_mode", "vanilla")
    if loss_mode != "gpi_ce":
        raise ValueError(
            "custom_gpi_ce.enable=true requires actor_rollout_ref.actor.policy_loss.loss_mode='gpi_ce', "
            f"got {loss_mode!r}."
        )
    return custom
