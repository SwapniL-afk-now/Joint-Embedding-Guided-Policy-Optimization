# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from omegaconf import DictConfig, OmegaConf


@dataclass
class SDCGRPOConfig:
    """Configuration for the success/failure contrastive GRPO objective."""

    enable: bool = False
    beta: float = 0.5
    logprob_backend: str = "hf"
    vllm_score_micro_batch_size: int = 16
    reuse_rollout_log_probs: bool = False
    verify_rollout_log_probs: bool = False
    use_importance_weight: bool = True
    importance_weight_clip: float = 10.0
    base_mix_gamma: float = 0.9

    sft_update_interval_grpo_steps: int = 5
    checkpoint_interval_grpo_steps: int = 10
    min_sft_updates_before_use: int = 1
    sft_lr: float = 1e-6
    sft_batch_size: int = 8
    sft_max_token_len_per_gpu: int = 0
    sft_max_updates_per_interval: int = 1

    data_max_size: Optional[int] = None
    data_sampling: str = "recent"
    save_to_disk_interval_grpo_steps: int = 0

    @classmethod
    def from_config(cls, config: DictConfig | dict | None) -> "SDCGRPOConfig":
        if not config:
            return cls()
        merged = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.create(config))
        return OmegaConf.to_object(merged)


def sdc_enabled(config: DictConfig | dict) -> bool:
    custom = config.get("custom_sdc_grpo", {}) if config is not None else {}
    return bool(custom and custom.get("enable", False))


def should_run_sft(global_grpo_step: int, config: SDCGRPOConfig) -> bool:
    return bool(config.enable and global_grpo_step % config.sft_update_interval_grpo_steps == 0)


def should_checkpoint(global_grpo_step: int, config: SDCGRPOConfig) -> bool:
    return bool(config.enable and global_grpo_step % config.checkpoint_interval_grpo_steps == 0)


def validate_sdc_config(config: DictConfig | dict) -> SDCGRPOConfig:
    """Validate root verl configuration and return typed SDC settings."""

    custom = SDCGRPOConfig.from_config(config.get("custom_sdc_grpo", {}) if config is not None else {})
    if not custom.enable:
        return custom

    algorithm = config.get("algorithm", {})
    actor_rollout_ref = config.get("actor_rollout_ref", {})
    actor = actor_rollout_ref.get("actor", {})
    model = actor_rollout_ref.get("model", {})
    raw_estimator = algorithm.get("adv_estimator", "")
    estimator = str(getattr(raw_estimator, "value", raw_estimator)).lower()

    if estimator != "grpo":
        raise ValueError("custom_sdc_grpo requires algorithm.adv_estimator='grpo'.")
    if bool(algorithm.get("use_kl_in_reward", False)):
        raise ValueError("SDC-GRPO requires algorithm.use_kl_in_reward=false.")
    if bool(actor.get("use_kl_loss", False)):
        raise ValueError("SDC-GRPO requires actor_rollout_ref.actor.use_kl_loss=false.")
    if custom.beta < 0:
        raise ValueError("custom_sdc_grpo.beta must be non-negative.")
    if custom.logprob_backend not in {"hf", "vllm"}:
        raise ValueError("custom_sdc_grpo.logprob_backend must be 'hf' or 'vllm'.")
    if custom.logprob_backend == "vllm" and int(model.get("lora_rank", 0) or 0) <= 0:
        raise ValueError("custom_sdc_grpo.logprob_backend='vllm' requires a positive model.lora_rank.")
    if custom.logprob_backend == "hf" and custom.reuse_rollout_log_probs:
        raise ValueError("HF SDC scoring requires reuse_rollout_log_probs=false.")
    if custom.logprob_backend == "vllm" and not custom.reuse_rollout_log_probs:
        raise ValueError("vLLM SDC scoring requires reuse_rollout_log_probs=true.")
    if custom.vllm_score_micro_batch_size <= 0:
        raise ValueError("custom_sdc_grpo.vllm_score_micro_batch_size must be positive.")
    if custom.importance_weight_clip < 0:
        raise ValueError("custom_sdc_grpo.importance_weight_clip must be non-negative.")
    if not custom.use_importance_weight:
        raise ValueError("custom_sdc_grpo.use_importance_weight=false would remove the actor-dependent SDC gradient; keep it true.")
    if not 0.0 < custom.base_mix_gamma <= 1.0:
        raise ValueError("custom_sdc_grpo.base_mix_gamma must be in (0, 1].")
    if custom.min_sft_updates_before_use < 1:
        raise ValueError("custom_sdc_grpo.min_sft_updates_before_use must be >= 1.")
    if custom.sft_update_interval_grpo_steps <= 0 or custom.checkpoint_interval_grpo_steps <= 0:
        raise ValueError("SDC SFT and checkpoint intervals must be positive.")
    if custom.sft_lr <= 0 or custom.sft_batch_size <= 0:
        raise ValueError("SDC SFT learning rate and batch size must be positive.")
    if custom.sft_max_token_len_per_gpu < 0 or custom.sft_max_updates_per_interval <= 0:
        raise ValueError("SDC SFT token budget must be non-negative and update budget positive.")
    if custom.data_max_size is not None and custom.data_max_size <= 0:
        raise ValueError("custom_sdc_grpo.data_max_size must be positive or null.")
    if custom.data_sampling not in {"recent", "uniform"}:
        raise ValueError("custom_sdc_grpo.data_sampling must be 'recent' or 'uniform'.")
    if custom.save_to_disk_interval_grpo_steps < 0:
        raise ValueError("custom_sdc_grpo.save_to_disk_interval_grpo_steps must be non-negative.")
    return custom
