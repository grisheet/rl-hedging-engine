"""On-policy rollout buffer used by PPO."""

from __future__ import annotations

from typing import Generator, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """Stores transitions from a single rollout for PPO training.

    Observations, actions, rewards, values, log-probs and done flags are
    accumulated step-by-step.  After the rollout finishes, advantages are
    computed via Generalised Advantage Estimation (GAE) and the buffer
    can be iterated in random mini-batches.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)
        self._ptr = 0
        self._full = False
        self._reset_arrays()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_arrays(self) -> None:
        n = self.capacity
        self.obs = np.zeros((n, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((n, self.act_dim), dtype=np.float32)
        self.rewards = np.zeros(n, dtype=np.float32)
        self.values = np.zeros(n, dtype=np.float32)
        self.log_probs = np.zeros(n, dtype=np.float32)
        self.dones = np.zeros(n, dtype=np.float32)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the buffer and reset the pointer."""
        self._ptr = 0
        self._full = False
        self._reset_arrays()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Store a single transition."""
        i = self._ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.dones[i] = float(done)
        self._ptr += 1
        if self._ptr >= self.capacity:
            self._full = True

    def compute_returns_and_advantages(self, last_value: float = 0.0) -> None:
        """Compute GAE advantages and discounted returns in-place."""
        gae = 0.0
        for t in reversed(range(self._ptr)):
            next_non_terminal = 1.0 - self.dones[t]
            next_value = self.values[t + 1] if t + 1 < self._ptr else last_value
            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns[: self._ptr] = (
            self.advantages[: self._ptr] + self.values[: self._ptr]
        )

    def get(
        self, batch_size: int
    ) -> Generator[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        None,
        None,
    ]:
        """Yield random mini-batches of (obs, actions, log_probs, returns, advantages)."""
        n = self._ptr
        indices = np.random.permutation(n)
        # Normalise advantages
        adv = self.advantages[:n]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield (
                torch.as_tensor(self.obs[idx], device=self.device),
                torch.as_tensor(self.actions[idx], device=self.device),
                torch.as_tensor(self.log_probs[idx], device=self.device),
                torch.as_tensor(self.returns[idx], device=self.device),
                torch.as_tensor(adv[idx], device=self.device),
            )

    @property
    def size(self) -> int:
        return self._ptr
