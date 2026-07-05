"""High-level training loops for PPO and TD3 agents."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..envs.hedging_env import HedgingEnv
from ..models.ppo import PPOAgent, PPOConfig
from ..models.td3 import TD3Agent, TD3Config
from .rollout import RolloutBuffer
from .replay import ReplayBuffer
from .logger import MetricLogger


class PPOTrainer:
    """Drives the PPO training loop over a HedgingEnv."""

    def __init__(
        self,
        env: HedgingEnv,
        cfg: PPOConfig,
        log_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
        save_freq: int = 100,
    ) -> None:
        self.env = env
        self.agent = PPOAgent(cfg)
        self.cfg = cfg
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_freq = save_freq
        self.logger = MetricLogger(log_dir=log_dir, print_freq=10)

        obs_dim = cfg.obs_dim
        act_dim = cfg.act_dim
        self.buffer = RolloutBuffer(
            capacity=cfg.n_steps,
            obs_dim=obs_dim,
            act_dim=act_dim,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            device=cfg.device,
        )

    def train(self, total_episodes: int) -> None:
        """Run PPO training for *total_episodes* episodes."""
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        for episode in range(1, total_episodes + 1):
            obs, _ = self.env.reset()
            self.buffer.reset()
            ep_reward = 0.0
            done = False

            while not done:
                action, log_prob, value = self.agent.ac.get_action_and_value(
                    __import__("torch").as_tensor(
                        obs, dtype=__import__("torch").float32,
                        device=self.agent.device
                    )
                )
                action_np = action.cpu().numpy()
                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated

                self.buffer.add(
                    obs=obs,
                    action=action_np,
                    reward=reward,
                    value=value.item(),
                    log_prob=log_prob.item(),
                    done=done,
                )
                obs = next_obs
                ep_reward += reward

            self.buffer.compute_returns_and_advantages(last_value=0.0)
            metrics = self.agent.update(self.buffer)

            self.logger.record("reward", ep_reward)
            self.logger.record_dict(metrics)
            self.logger.dump(step=episode)

            if self.save_dir and episode % self.save_freq == 0:
                self.agent.save(self.save_dir / f"ppo_{episode}.pt")

        self.logger.close()


class TD3Trainer:
    """Drives the TD3 training loop over a HedgingEnv."""

    def __init__(
        self,
        env: HedgingEnv,
        cfg: TD3Config,
        replay_capacity: int = 100_000,
        warmup_steps: int = 1_000,
        exploration_noise: float = 0.1,
        log_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
        save_freq: int = 100,
    ) -> None:
        self.env = env
        self.agent = TD3Agent(cfg)
        self.cfg = cfg
        self.warmup_steps = warmup_steps
        self.exploration_noise = exploration_noise
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_freq = save_freq
        self.logger = MetricLogger(log_dir=log_dir, print_freq=10)

        self.buffer = ReplayBuffer(
            capacity=replay_capacity,
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            device=cfg.device,
        )
        self._total_steps = 0

    def train(self, total_episodes: int) -> None:
        """Run TD3 training for *total_episodes* episodes."""
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        for episode in range(1, total_episodes + 1):
            obs, _ = self.env.reset()
            ep_reward = 0.0
            done = False

            while not done:
                if self._total_steps < self.warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(
                        obs, noise=self.exploration_noise
                    )

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.buffer.add(obs, action, reward, next_obs, done)
                obs = next_obs
                ep_reward += reward
                self._total_steps += 1

                if self.buffer.is_ready:
                    batch = self.buffer.sample(self.cfg.batch_size)
                    metrics = self.agent.update(*batch)
                    self.logger.record_dict(metrics)

            self.logger.record("reward", ep_reward)
            self.logger.dump(step=episode)

            if self.save_dir and episode % self.save_freq == 0:
                self.agent.save(self.save_dir / f"td3_{episode}.pt")

        self.logger.close()
