#!/usr/bin/env python
"""Training entry-point for rl-hedging-engine.

Usage examples::

    # Train PPO with default config
    python scripts/train.py

    # Train TD3 for 2000 episodes
    python scripts/train.py --algorithm td3 --total_episodes 2000

    # Override env and PPO hyperparameters
    python scripts/train.py --vol 0.3 --lr 1e-4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the src layout is importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from rlhedge.envs import HedgingEnv, EnvConfig
from rlhedge.models.ppo import PPOConfig
from rlhedge.models.td3 import TD3Config
from rlhedge.training.trainer import PPOTrainer, TD3Trainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_env(cfg: dict) -> HedgingEnv:
    env_cfg = EnvConfig(**cfg.get("env", {}))
    return HedgingEnv(env_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hedging RL agent")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--algorithm", choices=["ppo", "td3"], default=None)
    parser.add_argument("--total_episodes", type=int, default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.algorithm:
        cfg["training"]["algorithm"] = args.algorithm
    if args.total_episodes:
        cfg["training"]["total_episodes"] = args.total_episodes
    if args.log_dir:
        cfg["training"]["log_dir"] = args.log_dir
    if args.save_dir:
        cfg["training"]["save_dir"] = args.save_dir
    if args.device:
        cfg["ppo"]["device"] = args.device
        cfg["td3"]["device"] = args.device

    algorithm = cfg["training"]["algorithm"]
    total_episodes = cfg["training"]["total_episodes"]
    log_dir = cfg["training"]["log_dir"]
    save_dir = cfg["training"]["save_dir"]
    save_freq = cfg["training"].get("save_freq", 100)

    env = build_env(cfg)

    if algorithm == "ppo":
        agent_cfg = PPOConfig(**cfg["ppo"])
        trainer = PPOTrainer(
            env=env,
            cfg=agent_cfg,
            log_dir=log_dir,
            save_dir=save_dir,
            save_freq=save_freq,
        )
    elif algorithm == "td3":
        agent_cfg = TD3Config(**cfg["td3"])
        trainer = TD3Trainer(
            env=env,
            cfg=agent_cfg,
            replay_capacity=cfg["training"].get("replay_capacity", 100_000),
            warmup_steps=cfg["training"].get("warmup_steps", 1_000),
            exploration_noise=cfg["training"].get("exploration_noise", 0.1),
            log_dir=log_dir,
            save_dir=save_dir,
            save_freq=save_freq,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print(f"Starting {algorithm.upper()} training for {total_episodes} episodes...")
    trainer.train(total_episodes)
    print("Training complete.")


if __name__ == "__main__":
    main()
