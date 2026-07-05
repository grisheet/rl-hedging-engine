"""Reward functions for the hedging environment."""
from __future__ import annotations

import numpy as np

from rlhedge.envs.config import EnvConfig


def pnl_reward(pnl_change: float, cost: float) -> float:
    """Raw PnL reward: change in portfolio value minus transaction cost."""
    return pnl_change - cost


def cvar_penalised_reward(
    pnl_change: float,
    cost: float,
    episode_pnl_buffer: list[float],
    config: EnvConfig,
) -> float:
    """CVaR-penalised reward (estimated online from episode buffer).

    During training, CVaR is estimated from the PnL history within the
    current episode.  Once we have at least 2 observations the penalty
    kicks in; earlier steps fall back to raw PnL.
    """
    raw = pnl_change - cost
    if len(episode_pnl_buffer) < 2:
        return raw
    arr = np.array(episode_pnl_buffer)
    var_q = np.quantile(arr, 1.0 - config.cvar_alpha)
    tail = arr[arr <= var_q]
    cvar = float(tail.mean()) if len(tail) > 0 else float(var_q)
    penalty = config.risk_lambda * max(-cvar, 0.0)
    return raw - penalty


def compute_reward(
    pnl_change: float,
    cost: float,
    episode_pnl_buffer: list[float],
    config: EnvConfig,
) -> float:
    """Dispatch to the configured reward function."""
    if config.reward_type == "pnl":
        return pnl_reward(pnl_change, cost)
    elif config.reward_type == "cvar":
        return cvar_penalised_reward(pnl_change, cost, episode_pnl_buffer, config)
    else:
        raise ValueError(f"Unknown reward type: {config.reward_type}")
