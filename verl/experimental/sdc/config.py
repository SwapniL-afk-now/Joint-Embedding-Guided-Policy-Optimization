from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf


@dataclass
class SDCConfig:
    """Configuration for the additive, base-algorithm-independent SDC term."""

    enable: bool = False
    beta: float = 0.5
    scoring_backend: str = "hf"
    vllm_score_micro_batch_size: int = 16
    reuse_rollout_log_probs: bool = False
    verify_rollout_log_probs: bool = False
    use_importance_weight: bool = True
    importance_weight_clip: float = 10.0
    base_mix_gamma: float = 0.9

    # Algorithm-strength variants. All default to the locked legacy behavior;
    # enabling any of them changes training dynamics (never the actor-loss
    # algebra's identity) and must be validated via A/B before adoption.
    # Train success/failure teachers on prompt-matched subsets so both see the
    # same prompt distribution (removes coverage-asymmetry bias in c_t).
    match_prompts: bool = False
    # Subtract the detached mean contrast over active tokens before weighting:
    # removes the common-mode component that uniformly scales failed-row mass.
    contrast_centering: bool = False
    # Divide the centered contrast by its detached RMS over active tokens so
    # beta keeps a stable effective scale as teachers sharpen.
    contrast_scale_normalize: bool = False

    # S/F reference training. Both references use the same paired response-token
    # budget, optimizer-step count, and learning rate.
    sft_update_interval_policy_steps: int = 5
    checkpoint_interval_policy_steps: int = 10
    min_sft_updates_before_use: int = 1
    sft_lr: float = 1e-6
    sft_batch_size: int = 8
    sft_max_token_len_per_gpu: int = 0
    sft_max_updates_per_interval: int = 1
    fused_adamw: bool = False

    # Sidecar residency policy. 'auto_offload' (default) moves both SDC sidecar
    # clones back to host memory after every scoring/SFT pass; this matches the
    # historical behavior exactly. 'resident' keeps the clones on the actor GPU
    # across scoring/SFT calls until an explicit release point; trainers using
    # resident mode MUST call the actor engine's sdc_release_residency()
    # method before vLLM wake-up / rollout weight sync, otherwise vLLM will
    # not have enough free GPU memory.
    sidecar_residency: str = "auto_offload"
    # Sidecar gradient-checkpointing override. None (default) inherits the
    # actor model's gradient-checkpointing setting, matching historical
    # behavior. Explicit True/False enables/disables activation
    # checkpointing on the two SDC sidecar clones ONLY; the actor module is
    # never touched. Recompute-or-not is numerically identical for scoring.
    sidecar_gradient_checkpointing: Optional[bool] = None

    data_max_size: Optional[int] = None
    data_sampling: str = "recent"
    save_to_disk_interval_policy_steps: int = 0
    loss_agg_mode: str = "token-mean"
    # Keep exact distributed diagnostic reduction opt-in. The actor engine
    # already aggregates returned metrics once per training batch; reducing
    # them inside every micro-batch is unnecessarily expensive.
    distributed_metrics: bool = False

    def __post_init__(self):
        if not 0.0 <= float(self.base_mix_gamma) <= 1.0:
            raise ValueError(
                f"custom_sdc.base_mix_gamma must be in [0, 1], got {self.base_mix_gamma}."
            )
        if self.sidecar_residency not in {"auto_offload", "resident"}:
            raise ValueError(
                "custom_sdc.sidecar_residency must be 'auto_offload' or 'resident', "
                f"got {self.sidecar_residency!r}."
            )
        if self.sidecar_gradient_checkpointing is not None:
            self.sidecar_gradient_checkpointing = bool(self.sidecar_gradient_checkpointing)

    @classmethod
    def from_config(cls, config: DictConfig | dict[str, Any] | None) -> SDCConfig:
        if not config:
            return cls()
        merged = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.create(config))
        return OmegaConf.to_object(merged)


def _root_custom(config: DictConfig | dict | None) -> tuple[dict | DictConfig, bool]:
    if config is None:
        return {}, False
    canonical = config.get("custom_sdc", {})
    legacy = config.get("custom_sdc_grpo", {})
    canonical_enabled = bool(canonical and canonical.get("enable", False))
    legacy_enabled = bool(legacy and legacy.get("enable", False))
    if canonical_enabled and legacy_enabled:
        raise ValueError("Set only one of custom_sdc or the deprecated custom_sdc_grpo alias.")
    if canonical_enabled:
        return canonical, True
    if legacy_enabled:
        return legacy, True
    return canonical or legacy or {}, False


def sdc_enabled(config: DictConfig | dict) -> bool:
    return _root_custom(config)[1]


def _get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    return value.get(key, default)


def should_run_sft(global_policy_step: int, config: SDCConfig) -> bool:
    interval = int(config.sft_update_interval_policy_steps)
    return bool(config.enable and interval > 0 and global_policy_step % interval == 0)


def should_checkpoint(global_policy_step: int, config: SDCConfig) -> bool:
    interval = int(config.checkpoint_interval_policy_steps)
    return bool(config.enable and interval > 0 and global_policy_step % interval == 0)


def validate_sdc_config(config: DictConfig | dict) -> SDCConfig:
    """Validate only SDC requirements; the base estimator is deliberately opaque."""

    custom, enabled = _root_custom(config)
    settings = SDCConfig.from_config(_canonicalize_custom(custom))
    if not enabled:
        return settings

    actor_rollout_ref = config.get("actor_rollout_ref", {})
    actor = actor_rollout_ref.get("actor", {})
    model = actor_rollout_ref.get("model", {})
    if settings.beta < 0:
        raise ValueError("custom_sdc.beta must be non-negative.")
    if settings.scoring_backend not in {"hf", "vllm"}:
        raise ValueError("custom_sdc.scoring_backend must be 'hf' or 'vllm'.")
    if settings.scoring_backend == "vllm" and int(model.get("lora_rank", 0) or 0) <= 0:
        raise ValueError("custom_sdc.scoring_backend='vllm' requires a positive model.lora_rank.")
    if settings.scoring_backend == "hf" and settings.reuse_rollout_log_probs:
        raise ValueError("HF SDC scoring requires reuse_rollout_log_probs=false.")
    if settings.scoring_backend == "vllm" and not settings.reuse_rollout_log_probs:
        raise ValueError("vLLM SDC scoring requires reuse_rollout_log_probs=true.")
    if settings.vllm_score_micro_batch_size <= 0:
        raise ValueError("custom_sdc.vllm_score_micro_batch_size must be positive.")
    if settings.importance_weight_clip < 0:
        raise ValueError("custom_sdc.importance_weight_clip must be non-negative.")
    if not 0.0 <= float(settings.base_mix_gamma) <= 1.0:
        raise ValueError("custom_sdc.base_mix_gamma must be between 0.0 and 1.0.")
    if not settings.use_importance_weight:
        raise ValueError("SDC requires importance weighting to retain the actor-dependent gradient.")
    if settings.loss_agg_mode != "token-mean":
        raise ValueError("SDC uses fixed loss aggregation 'token-mean'.")
    if settings.min_sft_updates_before_use < 1:
        raise ValueError("custom_sdc.min_sft_updates_before_use must be >= 1.")
    if settings.sft_update_interval_policy_steps <= 0 or settings.checkpoint_interval_policy_steps <= 0:
        raise ValueError("SDC SFT and checkpoint intervals must be positive.")
    if settings.sft_lr <= 0 or settings.sft_batch_size <= 0:
        raise ValueError("SDC SFT learning rate and batch size must be positive.")
    if settings.sft_max_token_len_per_gpu < 0 or settings.sft_max_updates_per_interval <= 0:
        raise ValueError("SDC token budget must be non-negative and update budget positive.")
    if settings.data_max_size is None:
        # Advisory only: behavior is unchanged (unbounded buffers remain legal).
        warnings.warn(
            "custom_sdc.data_max_size is null: SDC success/failure outcome "
            "buffers will grow without bound over training. Set a positive "
            "integer cap to bound memory use.",
            stacklevel=2,
        )
    if settings.data_max_size is not None and settings.data_max_size <= 0:
        raise ValueError("custom_sdc.data_max_size must be positive or null.")
    if settings.data_sampling not in {"recent", "uniform"}:
        raise ValueError("custom_sdc.data_sampling must be 'recent' or 'uniform'.")
    if settings.save_to_disk_interval_policy_steps < 0:
        raise ValueError("custom_sdc.save_to_disk_interval_policy_steps must be non-negative.")
    # SDC can coexist with any supported base reward/loss protocol.  These KL
    # checks are intentionally not mathematical SDC requirements.
    _ = actor
    return settings


def canonical_sdc_config_dict(config: DictConfig | dict) -> dict[str, Any]:
    custom, _ = _root_custom(config)
    result = dict(custom)
    # Accept old spellings only at the boundary; all internal state is generic.
    aliases = {
        "logprob_backend": "scoring_backend",
        "sft_update_interval_grpo_steps": "sft_update_interval_policy_steps",
        "checkpoint_interval_grpo_steps": "checkpoint_interval_policy_steps",
        "save_to_disk_interval_grpo_steps": "save_to_disk_interval_policy_steps",
    }
    for old, new in aliases.items():
        if old in result and new not in result:
            result[new] = result[old]
    return result


def _canonicalize_custom(custom: DictConfig | dict) -> dict[str, Any]:
    result = dict(custom)
    aliases = {
        "logprob_backend": "scoring_backend",
        "sft_update_interval_grpo_steps": "sft_update_interval_policy_steps",
        "checkpoint_interval_grpo_steps": "checkpoint_interval_policy_steps",
        "save_to_disk_interval_grpo_steps": "save_to_disk_interval_policy_steps",
    }
    for old, new in aliases.items():
        if old in result and new not in result:
            result[new] = result[old]
        result.pop(old, None)
    return result
