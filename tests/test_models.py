"""tests/test_models.py – Unit tests for RL agent models (PPO, TD3)."""

from __future__ import annotations

import pytest
import torch
import numpy as np

from src.models.ppo import PPOAgent, PPOConfig
from src.models.td3 import TD3Agent, TD3Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OBS_DIM = 8
ACT_DIM = 1
DEVICE = torch.device("cpu")


@pytest.fixture
def ppo_agent() -> PPOAgent:
    cfg = PPOConfig(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=[64, 64],
        lr_actor=3e-4,
        lr_critic=3e-4,
        clip_ratio=0.2,
        gamma=0.99,
        lam=0.95,
        n_epochs=4,
        batch_size=32,
    )
    return PPOAgent(cfg, device=DEVICE)


@pytest.fixture
def td3_agent() -> TD3Agent:
    cfg = TD3Config(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=[64, 64],
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
    )
    return TD3Agent(cfg, device=DEVICE)


# ---------------------------------------------------------------------------
# PPO tests
# ---------------------------------------------------------------------------

class TestPPOAgent:
    def test_act_returns_correct_shape(self, ppo_agent: PPOAgent) -> None:
        obs = torch.randn(1, OBS_DIM)
        action = ppo_agent.act(obs)
        assert action.shape == (1, ACT_DIM) or action.shape == (ACT_DIM,)

    def test_act_batch(self, ppo_agent: PPOAgent) -> None:
        obs = torch.randn(16, OBS_DIM)
        action = ppo_agent.act(obs)
        assert action.shape[0] == 16 or action.shape[-1] == ACT_DIM

    def test_get_value_returns_scalar_per_obs(self, ppo_agent: PPOAgent) -> None:
        obs = torch.randn(4, OBS_DIM)
        values = ppo_agent.get_value(obs)
        assert values.shape == (4, 1) or values.shape == (4,)

    def test_update_does_not_raise(self, ppo_agent: PPOAgent) -> None:
        batch_size = 32
        obs = torch.randn(batch_size, OBS_DIM)
        actions = torch.randn(batch_size, ACT_DIM)
        returns = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        log_probs_old = torch.randn(batch_size)
        # Should not raise
        ppo_agent.update(
            obs=obs,
            actions=actions,
            returns=returns,
            advantages=advantages,
            log_probs_old=log_probs_old,
        )

    def test_eval_mode_no_grad(self, ppo_agent: PPOAgent) -> None:
        ppo_agent.eval()
        obs = torch.randn(1, OBS_DIM)
        with torch.no_grad():
            action = ppo_agent.act(obs)
        assert action is not None


# ---------------------------------------------------------------------------
# TD3 tests
# ---------------------------------------------------------------------------

class TestTD3Agent:
    def test_act_returns_correct_shape(self, td3_agent: TD3Agent) -> None:
        obs = torch.randn(1, OBS_DIM)
        action = td3_agent.act(obs)
        assert action.shape == (1, ACT_DIM) or action.shape == (ACT_DIM,)

    def test_act_batch(self, td3_agent: TD3Agent) -> None:
        obs = torch.randn(16, OBS_DIM)
        action = td3_agent.act(obs)
        assert action.shape[0] == 16 or action.shape[-1] == ACT_DIM

    def test_update_does_not_raise(self, td3_agent: TD3Agent) -> None:
        batch_size = 32
        obs = torch.randn(batch_size, OBS_DIM)
        actions = torch.randn(batch_size, ACT_DIM)
        rewards = torch.randn(batch_size)
        next_obs = torch.randn(batch_size, OBS_DIM)
        dones = torch.zeros(batch_size)
        # Should not raise
        td3_agent.update(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            step=1,
        )

    def test_target_networks_initialized(self, td3_agent: TD3Agent) -> None:
        """Target networks should start with same weights as live networks."""
        actor_params = list(td3_agent.actor.parameters())
        target_params = list(td3_agent.actor_target.parameters())
        for p, tp in zip(actor_params, target_params):
            torch.testing.assert_close(p.data, tp.data)

    def test_eval_mode_no_grad(self, td3_agent: TD3Agent) -> None:
        td3_agent.eval()
        obs = torch.randn(1, OBS_DIM)
        with torch.no_grad():
            action = td3_agent.act(obs)
        assert action is not None

    def test_save_load_state_dict(self, td3_agent: TD3Agent, tmp_path) -> None:
        path = tmp_path / "td3_checkpoint.pt"
        torch.save(td3_agent.state_dict(), str(path))
        state = torch.load(str(path), map_location=DEVICE)
        td3_agent.load_state_dict(state)
        # Re-run act after reload
        obs = torch.randn(1, OBS_DIM)
        action = td3_agent.act(obs)
        assert action is not None
