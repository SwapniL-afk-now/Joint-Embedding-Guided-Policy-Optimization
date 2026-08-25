import numpy as np
import torch

from verl.trainer.ppo.core_algos import AVSPOState, compute_avspo_outcome_advantage


def test_avspo_augments_collapsed_groups_with_paper_virtual_rewards():
    rewards = torch.zeros(8, 1)
    state = AVSPOState()
    advantages, _ = compute_avspo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=torch.ones_like(rewards, dtype=torch.bool),
        index=np.asarray(["g"] * 8),
        state=state,
    )
    assert state.last_metrics["avspo/acr"] == 1.0
    assert state.last_metrics["avspo/triggered"] == 1.0
    assert state.last_metrics["avspo/k_virtual"] == 8.0
    assert torch.all(advantages < 0)


def test_avspo_state_update_and_round_trip():
    state = AVSPOState()
    rewards = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    compute_avspo_outcome_advantage(
        rewards,
        torch.ones_like(rewards, dtype=torch.bool),
        np.asarray(["a", "a", "b", "b"]),
        state=state,
    )
    restored = AVSPOState()
    restored.load_state_dict(state.state_dict())
    assert restored.state_dict() == state.state_dict()
