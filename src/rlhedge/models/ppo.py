"""Proximal Policy Optimisation (PPO) — from-scratch PyTorch implementation.

Reference: Schulman et al. (2017) https://arxiv.org/abs/1707.06347
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rlhedge.models.networks import ActorCritic
from rlhedge.training.rollout_buffer import RolloutBuffer


@dataclass
class PPOConfig:
    """Hyper-parameters for PPO."""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    batch_size: int = 64
    hidden_dims: tuple[int, ...] = (256, 256)
    device: str = "cpu"


class PPO:
    """PPO agent with GAE advantage estimation."""

    def __init__(self, obs_dim: int, act_dim: int, config: Optional[PPOConfig] = None) -> None:
        self.cfg = config or PPOConfig()
        self.device = torch.device(self.cfg.device)
        self.ac = ActorCritic(obs_dim, act_dim, self.cfg.hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.cfg.lr)

    def select_action(self, obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Select action from observation. Returns (action, log_prob, value)."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, _, value = self.ac.get_action_and_value(obs_t)
        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run PPO update epochs on the collected rollout."""
        obs = torch.as_tensor(buffer.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(buffer.actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(buffer.log_probs, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(buffer.advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(buffer.returns, dtype=torch.float32, device=self.device)

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": []}
        n = len(obs)
        for _ in range(self.cfg.n_epochs):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, self.cfg.batch_size):
                mb_idx = idx[start : start + self.cfg.batch_size]
                _, new_log_prob, entropy, new_value = self.ac.get_action_and_value(
                    obs[mb_idx], actions[mb_idx]
                )
                ratio = (new_log_prob - old_log_probs[mb_idx]).exp()
                adv_mb = advantages[mb_idx]
                policy_loss = -torch.min(
                    ratio * adv_mb,
                    torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv_mb,
                ).mean()
                value_loss = nn.functional.mse_loss(new_value, returns[mb_idx])
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"].append(float(policy_loss.item()))
                metrics["value_loss"].append(float(value_loss.item()))
                metrics["entropy"].append(float(entropy.mean().item()))

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def save(self, path: str | Path) -> None:
        torch.save({"ac": self.ac.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["ac"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
