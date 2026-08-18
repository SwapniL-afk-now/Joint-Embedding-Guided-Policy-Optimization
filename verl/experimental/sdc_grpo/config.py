"""Deprecated compatibility boundary for the pre-modular SDC package."""

from __future__ import annotations

from dataclasses import dataclass

from verl.experimental.sdc.config import SDCConfig, canonical_sdc_config_dict
from verl.experimental.sdc.config import validate_sdc_config as _validate


@dataclass(init=False)
class SDCGRPOConfig(SDCConfig):
    """Deprecated alias; new configurations must use :class:`SDCConfig`."""

    def __init__(self, **kwargs):
        aliases = {
            "logprob_backend": "scoring_backend",
            "sft_update_interval_grpo_steps": "sft_update_interval_policy_steps",
            "checkpoint_interval_grpo_steps": "checkpoint_interval_policy_steps",
            "save_to_disk_interval_grpo_steps": "save_to_disk_interval_policy_steps",
        }
        for old, new in aliases.items():
            if old in kwargs and new not in kwargs:
                kwargs[new] = kwargs.pop(old)
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        return cls(**canonical_sdc_config_dict({"custom_sdc": config}))


def should_run_sft(global_grpo_step: int, config: SDCGRPOConfig) -> bool:
    from verl.experimental.sdc.config import should_run_sft as _should
    return _should(global_grpo_step, config)


def should_checkpoint(global_grpo_step: int, config: SDCGRPOConfig) -> bool:
    from verl.experimental.sdc.config import should_checkpoint as _should
    return _should(global_grpo_step, config)


def validate_sdc_config(config):
    return _validate(config)
