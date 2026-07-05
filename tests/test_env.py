"""tests/test_env.py – Unit tests for the HedgingEnv Gymnasium environment."""

from __future__ import annotations

import numpy as np
import pytest

from src.envs.hedging_env import HedgingEnv
from src.pricing.black_scholes import BlackScholes
from src.simulation.gbm import GBMSimulator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env() -> HedgingEnv:
    simulator = GBMSimulator(
        s0=100.0,
        mu=0.05,
        sigma=0.2,
        dt=1 / 252,
        n_steps=63,
        n_paths=1,
    )
    pricer = BlackScholes(r=0.05, sigma=0.2)
    return HedgingEnv(
        simulator=simulator,
        pricer=pricer,
        strike=100.0,
        maturity=63 / 252,
        transaction_cost=0.001,
        risk_aversion=0.1,
    )


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

class TestHedgingEnvStructure:
    def test_reset_returns_obs_and_info(self, env: HedgingEnv) -> None:
        result = env.reset()
        assert isinstance(result, tuple)
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_obs_shape_consistent(self, env: HedgingEnv) -> None:
        obs, _ = env.reset()
        obs_shape = obs.shape
        action = env.action_space.sample()
        obs2, _, _, _, _ = env.step(action)
        assert obs2.shape == obs_shape

    def test_observation_space_contains_obs(self, env: HedgingEnv) -> None:
        obs, _ = env.reset()
        assert env.observation_space.contains(obs.astype(np.float32))

    def test_action_space_shape(self, env: HedgingEnv) -> None:
        assert len(env.action_space.shape) == 1
        assert env.action_space.shape[0] >= 1


# ---------------------------------------------------------------------------
# Episode rollout tests
# ---------------------------------------------------------------------------

class TestHedgingEnvRollout:
    def test_full_episode_terminates(self, env: HedgingEnv) -> None:
        env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            assert steps < 10_000, "Episode did not terminate within 10,000 steps"
        assert steps > 0

    def test_reward_is_scalar(self, env: HedgingEnv) -> None:
        env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert isinstance(float(reward), float)

    def test_info_contains_expected_keys(self, env: HedgingEnv) -> None:
        env.reset()
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        for key in ("step_pnl", "hedge_ratio", "spot_price"):
            assert key in info, f"Missing key in info: {key}"

    def test_reset_between_episodes(self, env: HedgingEnv) -> None:
        """Running two episodes in sequence should not raise."""
        for _ in range(2):
            obs, _ = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    def test_deterministic_with_seed(self, env: HedgingEnv) -> None:
        obs1, _ = env.reset(seed=42)
        rewards1 = []
        done = False
        while not done:
            action = np.array([0.5])  # fixed action
            obs1, r, terminated, truncated, _ = env.step(action)
            rewards1.append(r)
            done = terminated or truncated

        obs2, _ = env.reset(seed=42)
        rewards2 = []
        done = False
        while not done:
            action = np.array([0.5])
            obs2, r, terminated, truncated, _ = env.step(action)
            rewards2.append(r)
            done = terminated or truncated

        np.testing.assert_allclose(rewards1, rewards2, rtol=1e-5)
