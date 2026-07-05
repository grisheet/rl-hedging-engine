"""Twin Delayed Deep Deterministic Policy Gradient (TD3) agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import ActorNet, CriticNet


@dataclass
class TD3Config:
    """Hyperparameters for TD3."""

    obs_dim: int = 6
    act_dim: int = 1
    act_limit: float = 1.0
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005          # soft update coefficient
    policy_noise: float = 0.2   # noise added to target policy
    noise_clip: float = 0.5
    policy_delay: int = 2       # delay actor / target updates
    batch_size: int = 256
    device: str = "cpu"


class TD3Agent:
    """TD3 agent with twin critics and delayed policy updates."""

    def __init__(self, cfg: TD3Config) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._total_updates = 0

        # Actor
        self.actor = ActorNet(
            cfg.obs_dim, cfg.act_dim, cfg.hidden_sizes, cfg.act_limit
        ).to(self.device)
        self.actor_target = ActorNet(
            cfg.obs_dim, cfg.act_dim, cfg.hidden_sizes, cfg.act_limit
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Twin critics
        self.critic1 = CriticNet(
            cfg.obs_dim + cfg.act_dim, cfg.hidden_sizes
        ).to(self.device)
        self.critic2 = CriticNet(
            cfg.obs_dim + cfg.act_dim, cfg.hidden_sizes
        ).to(self.device)
        self.critic1_target = CriticNet(
            cfg.obs_dim + cfg.act_dim, cfg.hidden_sizes
        ).to(self.device)
        self.critic2_target = CriticNet(
            cfg.obs_dim + cfg.act_dim, cfg.hidden_sizes
        ).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=cfg.critic_lr,
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, noise: float = 0.0) -> np.ndarray:
        """Return a deterministic action, optionally perturbed by Gaussian noise."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action = self.actor(obs_t).cpu().numpy()
        if noise > 0.0:
            action += np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -self.cfg.act_limit, self.cfg.act_limit)

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """One gradient step on a mini-batch sampled from the replay buffer."""
        self._total_updates += 1
        cfg = self.cfg

        with torch.no_grad():
            # Target policy with clipped noise
            noise = (torch.randn_like(actions) * cfg.policy_noise).clamp(
                -cfg.noise_clip, cfg.noise_clip
            )
            next_actions = (self.actor_target(next_obs) + noise).clamp(
                -cfg.act_limit, cfg.act_limit
            )
            next_sa = torch.cat([next_obs, next_actions], dim=-1)
            target_q = torch.min(
                self.critic1_target(next_sa),
                self.critic2_target(next_sa),
            )
            backup = rewards + cfg.gamma * (1.0 - dones) * target_q

        sa = torch.cat([obs, actions], dim=-1)
        q1 = self.critic1(sa)
        q2 = self.critic2(sa)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        metrics: Dict[str, float] = {
            "critic_loss": critic_loss.item(),
            "q1_mean": q1.mean().item(),
        }

        # Delayed actor update
        if self._total_updates % cfg.policy_delay == 0:
            actor_loss = -self.critic1(
                torch.cat([obs, self.actor(obs)], dim=-1)
            ).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

            metrics["actor_loss"] = actor_loss.item()

        return metrics

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _soft_update(self, src: nn.Module, tgt: nn.Module) -> None:
        tau = self.cfg.tau
        for sp, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        # Sync targets
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
