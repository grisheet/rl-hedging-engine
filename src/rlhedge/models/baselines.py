"""Classical baseline hedging strategies.

These are non-RL agents that follow deterministic rules:
  - NoHedge: always holds delta = 0
  - BSHedge: continuously rebalances to Black-Scholes delta
"""
from __future__ import annotations

import numpy as np

from rlhedge.envs.config import EnvConfig
from rlhedge.pricing.blackscholes import bs_delta


class BaselineAgent:
    """Abstract base class for deterministic baseline agents."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class NoHedgeAgent(BaselineAgent):
    """Never hedges: always returns delta = 0."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)


class BSHedgeAgent(BaselineAgent):
    """Black-Scholes delta hedge: reads BS delta from observation.

    Assumes include_greeks=True in EnvConfig so that obs[3] == bs_delta.
    If Greeks are not included, falls back to computing delta from obs.
    """

    def __init__(self, config: EnvConfig) -> None:
        self.config = config

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.config.include_greeks and len(obs) >= 7:
            # obs[3] is bs_delta (index 3 of the 7-element obs)
            bs_d = float(obs[3])
        else:
            # Reconstruct from obs: [log(S/K), tau, current_delta, ...]
            log_m = float(obs[0])
            tau = float(obs[1])
            s_over_k = np.exp(log_m)
            # Approximation: compute delta from log-moneyness and tau
            bs_d = float(
                bs_delta(
                    s_over_k * self.config.strike,
                    self.config.strike,
                    tau,
                    self.config.gbm.rate,
                    self.config.gbm.vol,
                    self.config.option_kind,
                )
            )
        return np.array([bs_d], dtype=np.float32)
