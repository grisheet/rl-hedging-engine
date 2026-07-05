"""Uniform experience replay buffer used by TD3."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size circular replay buffer for off-policy algorithms.

    Stores (obs, action, reward, next_obs, done) tuples and returns random
    mini-batches as PyTorch tensors on the requested device.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        device: str = "cpu",
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)
        self._ptr = 0
        self._size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Insert a single transition (overwrites oldest when full)."""
        i = self._ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self._ptr = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample a random mini-batch and return tensors on *self.device*."""
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.actions[idx], device=self.device),
            torch.as_tensor(self.rewards[idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.dones[idx], device=self.device),
        )

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        """True once the buffer has at least *capacity* / 10 samples."""
        return self._size >= max(1, self.capacity // 10)
