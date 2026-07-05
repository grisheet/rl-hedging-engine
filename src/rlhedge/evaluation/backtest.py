"""Backtesting utilities for evaluating hedging agents on simulated paths."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..envs.hedging_env import HedgingEnv


class Backtester:
    """Rolls out one or more agents over a fixed set of simulated episodes.

    Parameters
    ----------
    env:
        A *HedgingEnv* instance (or compatible gym environment).
    n_episodes:
        Number of independent episodes to run per agent.
    seed:
        Optional seed for reproducibility.
    """

    def __init__(
        self,
        env: HedgingEnv,
        n_episodes: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        self.env = env
        self.n_episodes = n_episodes
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        policy: Callable[[np.ndarray], np.ndarray],
        label: str = "agent",
    ) -> Dict[str, Any]:
        """Evaluate *policy* for *n_episodes* and collect episode statistics.

        Parameters
        ----------
        policy:
            A callable ``obs -> action`` (deterministic, no grad).
        label:
            Human-readable name stored in the returned dict.

        Returns
        -------
        Dict with keys:
            - ``label``: the agent label
            - ``rewards``: list of total episode rewards
            - ``pnls``: list of final P&L per episode
            - ``hedge_ratios``: list of arrays of hedge ratios per step
        """
        rewards: List[float] = []
        pnls: List[float] = []
        hedge_ratios: List[np.ndarray] = []

        for ep in range(self.n_episodes):
            ep_seed = None if self.seed is None else self.seed + ep
            obs, info = self.env.reset(seed=ep_seed)
            ep_reward = 0.0
            ep_hedges: List[float] = []
            done = False

            while not done:
                action = policy(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
                # Store the scalar hedge ratio (first element of action)
                ep_hedges.append(float(np.asarray(action).flat[0]))

            rewards.append(ep_reward)
            pnls.append(float(info.get("pnl", float("nan"))))
            hedge_ratios.append(np.array(ep_hedges, dtype=np.float32))

        return {
            "label": label,
            "rewards": rewards,
            "pnls": pnls,
            "hedge_ratios": hedge_ratios,
        }

    def run_many(
        self,
        policies: Dict[str, Callable[[np.ndarray], np.ndarray]],
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate multiple policies and return a dict keyed by label."""
        return {label: self.run(policy, label=label) for label, policy in policies.items()}
