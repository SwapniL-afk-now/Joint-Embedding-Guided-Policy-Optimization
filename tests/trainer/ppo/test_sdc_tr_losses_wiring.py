import math

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss


def make_actor_config(loss_mode="vanilla", **policy_loss_kwargs):
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(loss_mode=loss_mode, **policy_loss_kwargs),
    )


def make_ppo_data(advantages, **extra_tensors):
    tensors = {
        "prompts": torch.tensor([[1], [2]]),
        "responses": torch.tensor([[3, 4], [5, 6]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "response_mask": torch.ones(2, 2, dtype=torch.bool),
        "old_log_probs": torch.zeros(2, 2),
        "advantages": advantages,
    }
    tensors.update(extra_tensors)
    data = TensorDict(tensors, batch_size=[2])
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=None, global_batch_size=None)
    return data


def test_ppo_loss_sdc_tr_matches_vanilla_before_teachers_ready():
    """loss_mode='sdc_tr' with sdc.enable=true but sdc_models_ready=false must
    match plain vanilla PPO exactly: the wiring forces reliability_gate=0.0
    (hence lambda_eff=0.0) until both SDC teachers are trained."""
    advantages = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    model_output = {"log_probs": torch.zeros(6, requires_grad=True)}

    vanilla_data = make_ppo_data(advantages)
    vanilla_loss, _ = ppo_loss(config=make_actor_config("vanilla"), model_output=model_output, data=vanilla_data)

    sdc_tr_data = make_ppo_data(advantages)
    tu.assign_non_tensor(sdc_tr_data, custom_sdc={"enable": True, "sdc_models_ready": False, "beta": 0.5})
    sdc_tr_loss, sdc_tr_metrics = ppo_loss(
        config=make_actor_config("sdc_tr", sdc_tr_alpha=0.3),
        model_output={"log_probs": torch.zeros(6, requires_grad=True)},
        data=sdc_tr_data,
    )

    torch.testing.assert_close(sdc_tr_loss, vanilla_loss)
    assert sdc_tr_metrics["sdc_tr/lambda_eff"].aggregate() == 0.0
    # The additive compute_sdc_loss() path must be fully skipped for sdc_tr.
    assert "sdc/loss" not in sdc_tr_metrics


def test_ppo_loss_sdc_tr_applies_teacher_contrast_once_ready():
    advantages = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
    data = make_ppo_data(
        advantages,
        sdc_success_log_probs=torch.tensor([[-1.0, -3.0], [-1.0, -3.0]]),
        sdc_failure_log_probs=torch.tensor([[-3.0, -1.0], [-3.0, -1.0]]),
        sdc_failure_mask=torch.tensor([True, True]),
    )
    tu.assign_non_tensor(
        data, custom_sdc={"enable": True, "sdc_models_ready": True, "beta": 0.5, "sdc_reliability_gate": 1.0}
    )

    loss, metrics = ppo_loss(
        config=make_actor_config("sdc_tr", sdc_tr_alpha=0.5),
        model_output={"log_probs": torch.zeros(6, requires_grad=True)},
        data=data,
    )

    lambda_eff = 0.5
    r_success_like = math.exp(-lambda_eff)
    r_failure_like = math.exp(lambda_eff)

    def dual_clip_loss(r):
        pg1, pg2 = -(-1.0) * r, -(-1.0) * min(max(r, 0.8), 1.2)
        return min(max(pg1, pg2), 3.0)

    expected = (dual_clip_loss(r_success_like) + dual_clip_loss(r_failure_like)) / 2
    torch.testing.assert_close(loss, torch.tensor(expected))
    assert metrics["sdc_tr/lambda_eff"].aggregate() == 0.5
    assert metrics["sdc_tr/active_failed_token_fraction"].aggregate() == 1.0
    assert "sdc/loss" not in metrics
