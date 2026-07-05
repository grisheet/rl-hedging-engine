"""Shared neural network building blocks (PyTorch, no external library)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: type[nn.Module] = nn.Tanh,
    output_activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    """Build a multi-layer perceptron.

    Parameters
    ----------
    in_dim:
        Input dimension.
    hidden_dims:
        Sizes of hidden layers.
    out_dim:
        Output dimension.
    activation:
        Non-linearity applied after each hidden layer.
    output_activation:
        Optional non-linearity applied after the output layer.

    Returns
    -------
    nn.Sequential
    """
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic network for PPO.

    Architecture::

        obs --> shared_trunk --> actor_head  --> mean (action)
                             --> critic_head --> value
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.trunk = mlp(obs_dim, list(hidden_dims[:-1]), hidden_dims[-1])
        self.actor_mean = nn.Linear(hidden_dims[-1], act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden_dims[-1], 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (action_mean, action_log_std, value)."""
        h = self.trunk(obs)
        return self.actor_mean(h), self.actor_log_std.expand_as(self.actor_mean(h)), self.critic(h).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action, return (action, log_prob, entropy, value)."""
        h = self.trunk(obs)
        mean = self.actor_mean(h)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(h).squeeze(-1)
        return action, log_prob, entropy, value


class Actor(nn.Module):
    """Deterministic actor network for TD3."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        act_limit: float = 1.0,
    ) -> None:
        super().__init__()
        self.net = mlp(obs_dim, list(hidden_dims), act_dim, output_activation=nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs) * self.act_limit


class TwinCritic(nn.Module):
    """Twin Q-networks for TD3 (separate networks, not shared trunk)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, list(hidden_dims), 1)
        self.q2 = mlp(obs_dim + act_dim, list(hidden_dims), 1)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (Q1, Q2)."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only Q1 (used for actor update)."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1)
