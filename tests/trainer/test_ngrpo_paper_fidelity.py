import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_ngrpo_outcome_advantage, compute_policy_loss_ngrpo


def test_ngrpo_augments_every_group_with_r_max_and_uses_population_std():
    rewards = torch.tensor([[0.0], [0.0], [1.0], [0.0], [1.0], [1.0], [1.0], [1.0]])
    advantages, _ = compute_ngrpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=torch.ones_like(rewards, dtype=torch.bool),
        index=np.asarray(["g"] * 8),
    )
    augmented = torch.cat([rewards[:, 0], torch.ones(1)])
    expected = (rewards[:, 0] - augmented.mean()) / augmented.std(unbiased=False).add(1e-6)
    torch.testing.assert_close(advantages[:, 0], expected)
    assert float(advantages[:, 0].sum()) < 0


def test_ngrpo_clipping_uses_asymmetric_boundaries():
    class Config:
        clip_ratio = 0.2
        clip_ratio_low = 0.16
        clip_ratio_high = 0.24
        global_batch_info = {}

        @staticmethod
        def get(key, default=None):
            return default

    old = torch.zeros(2, 1)
    current = torch.log(torch.tensor([[1.5], [0.7]]))
    advantages = torch.tensor([[1.0], [-1.0]])
    loss, _ = compute_policy_loss_ngrpo(
        old_log_prob=old,
        log_prob=current,
        advantages=advantages,
        response_mask=torch.ones(2, 1, dtype=torch.bool),
        config=Config(),
    )
    expected = torch.tensor([-(1.24), -(-1.0 * 0.84)]).mean()
    torch.testing.assert_close(loss, expected)
