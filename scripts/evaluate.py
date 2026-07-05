"""evaluate.py – Evaluation entry-point for rl-hedging-engine.

Loads a trained agent checkpoint and runs a full evaluation episode,
printing a summary of hedging performance metrics.

Usage::

    python scripts/evaluate.py --config configs/default.yaml \\
        --checkpoint checkpoints/td3_final.pt
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Optional

import numpy as np
import torch
import yaml

from src.envs.hedging_env import HedgingEnv
from src.evaluation.metrics import (
    compute_pnl_stats,
    compute_hedge_error,
    compute_transaction_costs,
    compute_sharpe_ratio,
)
from src.evaluation.visualize import plot_episode
from src.models.ppo import PPOAgent
from src.models.td3 import TD3Agent
from src.simulation.gbm import GBMSimulator
from src.pricing.black_scholes import BlackScholes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _build_env(cfg: dict) -> HedgingEnv:
    sim_cfg = cfg["simulation"]
    opt_cfg = cfg["option"]
    env_cfg = cfg["environment"]

    simulator = GBMSimulator(
        s0=sim_cfg["s0"],
        mu=sim_cfg["mu"],
        sigma=sim_cfg["sigma"],
        dt=sim_cfg["dt"],
        n_steps=sim_cfg["n_steps"],
        n_paths=1,
    )
    pricer = BlackScholes(
        r=opt_cfg["r"],
        sigma=sim_cfg["sigma"],
    )
    return HedgingEnv(
        simulator=simulator,
        pricer=pricer,
        strike=opt_cfg["strike"],
        maturity=opt_cfg["maturity"],
        transaction_cost=env_cfg["transaction_cost"],
        risk_aversion=env_cfg["risk_aversion"],
    )


def _load_agent(
    cfg: dict,
    checkpoint: str,
    obs_dim: int,
    act_dim: int,
    device: torch.device,
):
    algorithm = cfg["training"]["algorithm"]
    model_cfg = cfg["model"]

    if algorithm == "ppo":
        from src.models.ppo import PPOConfig
        agent_cfg = PPOConfig(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=model_cfg["hidden_sizes"],
            lr_actor=model_cfg["lr"],
            lr_critic=model_cfg["lr"],
        )
        agent = PPOAgent(agent_cfg, device=device)
    elif algorithm == "td3":
        from src.models.td3 import TD3Config
        agent_cfg = TD3Config(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=model_cfg["hidden_sizes"],
            actor_lr=model_cfg["lr"],
            critic_lr=model_cfg["lr"],
        )
        agent = TD3Agent(agent_cfg, device=device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    state = torch.load(checkpoint, map_location=device)
    agent.load_state_dict(state)
    agent.eval()
    return agent


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    cfg: dict,
    checkpoint: str,
    n_episodes: int = 10,
    render: bool = False,
    save_fig: Optional[str] = None,
    device_str: str = "cpu",
) -> dict:
    """Run *n_episodes* evaluation episodes and return aggregated metrics."""
    device = torch.device(device_str)
    env = _build_env(cfg)
    obs, _ = env.reset()
    obs_dim = obs.shape[0]
    act_dim = env.action_space.shape[0]

    agent = _load_agent(cfg, checkpoint, obs_dim, act_dim, device)

    all_pnls: list[np.ndarray] = []
    all_positions: list[np.ndarray] = []
    all_prices: list[np.ndarray] = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_pnl: list[float] = []
        ep_pos: list[float] = []
        ep_price: list[float] = []

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = agent.act(obs_t)
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy().squeeze()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_pnl.append(float(info.get("step_pnl", reward)))
            ep_pos.append(float(info.get("hedge_ratio", 0.0)))
            ep_price.append(float(info.get("spot_price", 0.0)))

        all_pnls.append(np.array(ep_pnl))
        all_positions.append(np.array(ep_pos))
        all_prices.append(np.array(ep_price))

        if render:
            print(f"  Episode {ep + 1:>3d} | Total PnL: {sum(ep_pnl):+.4f}")

    # ---- Aggregate metrics ------------------------------------------------
    total_pnls = np.array([arr.sum() for arr in all_pnls])
    pnl_stats = compute_pnl_stats(total_pnls)
    hedge_errors = np.array([compute_hedge_error(p, pos) for p, pos in zip(all_prices, all_positions)])
    txn_costs = np.array([compute_transaction_costs(pos, cfg["environment"]["transaction_cost"]) for pos in all_positions])
    sharpe = compute_sharpe_ratio(total_pnls)

    metrics = {
        "mean_pnl": float(pnl_stats["mean"]),
        "std_pnl": float(pnl_stats["std"]),
        "min_pnl": float(pnl_stats["min"]),
        "max_pnl": float(pnl_stats["max"]),
        "sharpe_ratio": float(sharpe),
        "mean_hedge_error": float(hedge_errors.mean()),
        "mean_txn_cost": float(txn_costs.mean()),
    }

    if save_fig:
        plot_episode(
            prices=all_prices[0],
            positions=all_positions[0],
            pnls=all_pnls[0],
            save_path=save_fig,
        )

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained hedging agent.")
    p.add_argument("--config", type=str, default="configs/default.yaml",
                   help="Path to YAML configuration file.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint (.pt file).")
    p.add_argument("--n_episodes", type=int, default=10,
                   help="Number of evaluation episodes.")
    p.add_argument("--device", type=str, default="cpu",
                   help="Torch device ('cpu' or 'cuda').")
    p.add_argument("--save_fig", type=str, default=None,
                   help="If set, save episode plot to this path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.n_episodes}  |  Device: {args.device}")
    print("-" * 50)

    metrics = evaluate(
        cfg=cfg,
        checkpoint=args.checkpoint,
        n_episodes=args.n_episodes,
        render=True,
        save_fig=args.save_fig,
        device_str=args.device,
    )

    print("-" * 50)
    print("Evaluation summary:")
    for k, v in metrics.items():
        print(f"  {k:<25s}: {v:+.6f}")


if __name__ == "__main__":
    main()
