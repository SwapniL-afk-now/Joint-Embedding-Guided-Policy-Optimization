# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""The config guards that keep the one-step / atomic-group contract (plan section 14)."""

import pytest
from omegaconf import OmegaConf

from verl.experimental.gpi_ce import validate_gpi_ce_config


def _config(**overrides):
    cfg = OmegaConf.create(
        {
            "custom_gpi_ce": {
                "enable": True,
                "online_per_prompt": 4,
                "offline_per_prompt": 4,
                "temperature": 0.5,
            },
            "algorithm": {"use_kl_in_reward": False},
            "actor_rollout_ref": {
                "rollout": {"n": 4},
                "actor": {
                    "ppo_epochs": 1,
                    "shuffle": False,
                    "accumulate_minibatch_grads": True,
                    "use_dynamic_bsz": False,
                    "use_kl_loss": False,
                    "policy_loss": {"loss_mode": "gpi_ce"},
                },
            },
        }
    )
    for dotted, value in overrides.items():
        OmegaConf.update(cfg, dotted, value, merge=True)
    return cfg


def test_valid_config_passes():
    cfg = validate_gpi_ce_config(_config())
    assert cfg.group_size == 8


def test_disabled_config_skips_every_check():
    cfg = validate_gpi_ce_config(_config(**{"custom_gpi_ce.enable": False, "actor_rollout_ref.actor.ppo_epochs": 4}))
    assert not cfg.enable


@pytest.mark.parametrize(
    "override,message",
    [
        ({"actor_rollout_ref.rollout.n": 8}, "must equal custom_gpi_ce.online_per_prompt"),
        ({"actor_rollout_ref.actor.ppo_epochs": 2}, "ppo_epochs=1"),
        ({"actor_rollout_ref.actor.shuffle": True}, "shuffle=false"),
        ({"actor_rollout_ref.actor.accumulate_minibatch_grads": False}, "accumulate_minibatch_grads=true"),
        ({"actor_rollout_ref.actor.use_kl_loss": True}, "use_kl_loss=false"),
        ({"algorithm.use_kl_in_reward": True}, "use_kl_in_reward=false"),
        ({"actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla"}, "loss_mode='gpi_ce'"),
        ({"custom_gpi_ce.temperature": 0.0}, "temperature must be positive"),
        ({"custom_gpi_ce.online_per_prompt": 0}, "online_per_prompt must be >= 1"),
    ],
)
def test_invalid_configs_are_rejected(override, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        validate_gpi_ce_config(_config(**override))


def test_dynamic_bsz_is_allowed_with_a_big_enough_token_budget():
    """Token-budget batching keeps groups atomic via force_group_size, so it is legal."""
    cfg = _config(
        **{
            "actor_rollout_ref.actor.use_dynamic_bsz": True,
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 32768,
            "data": {"max_prompt_length": 1024, "max_response_length": 31744},
        }
    )
    assert validate_gpi_ce_config(cfg).group_size == 8


def test_dynamic_bsz_budget_below_one_sequence_is_rejected():
    cfg = _config(
        **{
            "actor_rollout_ref.actor.use_dynamic_bsz": True,
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 8192,
            "data": {"max_prompt_length": 1024, "max_response_length": 31744},
        }
    )
    with pytest.raises(ValueError, match="ppo_max_token_len_per_gpu"):
        validate_gpi_ce_config(cfg)


def test_group_size_of_one_is_rejected():
    cfg = _config(**{"custom_gpi_ce.online_per_prompt": 1, "custom_gpi_ce.offline_per_prompt": 0})
    cfg.actor_rollout_ref.rollout.n = 1
    with pytest.raises(ValueError, match="group_size must be >= 2"):
        validate_gpi_ce_config(cfg)


def test_ppo_trainer_yaml_declares_the_block():
    """Hydra needs the key present in the shipped config or overrides are rejected."""
    cfg = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    assert "custom_gpi_ce" in cfg
    assert cfg.custom_gpi_ce.enable is False
    assert cfg.custom_gpi_ce.online_per_prompt == 4
