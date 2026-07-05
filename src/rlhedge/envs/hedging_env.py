"""Gymnasium-compatible option hedging environment.

State: (S_t/K, tau_t, delta_t, log_moneyness, BS_greeks...)
Action: target delta in [-1, 1] (clipped and rescaled to [0, 1] for calls)
Reward: configured via EnvConfig (pnl or cvar)
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlhedge.envs.config import EnvConfig
from rlhedge.envs.costs import compute_cost, terminal_cost
from rlhedge.envs.ledger import Ledger
from rlhedge.envs.rewards import compute_reward
from rlhedge.pricing.blackscholes import bs_greeks, bs_price
from rlhedge.simulation.gbm import GBMParams, simulate_gbm_paths, remaining_tau


class HedgingEnv(gym.Env):
    """Single-path European option hedging environment.

    At each step the agent chooses a target delta (continuous action).
    The environment simulates one GBM step, rebalances the hedge,
    and returns a reward based on portfolio PnL minus transaction costs.

    Observation vector (when include_greeks=True):
        [log(S/K), tau, current_delta, bs_delta, bs_gamma, bs_vega, bs_theta]
    Otherwise:
        [log(S/K), tau, current_delta]
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        super().__init__()
        self.config: EnvConfig = config or EnvConfig()
        self._gbm: GBMParams = self.config.gbm

        # Observation & action spaces
        n_obs = 7 if self.config.include_greeks else 3
        obs_low = np.full(n_obs, -10.0, dtype=np.float32)
        obs_high = np.full(n_obs, 10.0, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        # Episode state (initialised in reset)
        self._prices: np.ndarray = np.array([])
        self._step: int = 0
        self._ledger: Optional[Ledger] = None
        self._prev_pnl: float = 0.0
        self._pnl_buffer: list[float] = []
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Simulate a single path for this episode
        paths = simulate_gbm_paths(self._gbm, n_paths=1, rng=self._rng, antithetic=False)
        self._prices = paths[0]  # shape (n_steps + 1,)
        self._step = 0

        spot0 = float(self._prices[0])
        tau0 = self._gbm.maturity
        option_price = float(
            bs_price(spot0, self.config.strike, tau0, self._gbm.rate, self._gbm.vol, self.config.option_kind)
        )
        greeks = bs_greeks(spot0, self.config.strike, tau0, self._gbm.rate, self._gbm.vol, self.config.option_kind)
        initial_delta = float(greeks["delta"])

        self._ledger = Ledger(option_price, initial_delta)
        self._prev_pnl = float(self._ledger.mark_to_market(spot0, option_price))
        self._pnl_buffer = [self._prev_pnl]

        obs = self._build_obs(spot0, tau0, initial_delta)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._ledger is not None, "Call reset() before step()."

        new_delta = float(np.clip(action[0], -1.0, 1.0))
        self._step += 1
        spot = float(self._prices[self._step])
        tau = self._gbm.maturity - self._step * self._gbm.dt

        # Transaction cost and rebalancing
        cost = compute_cost(self._ledger.delta, new_delta, spot, self.config)
        self._ledger.rebalance(new_delta, spot, cost)

        # Option value at current step
        option_value = float(
            bs_price(spot, self.config.strike, max(tau, 0.0), self._gbm.rate, self._gbm.vol, self.config.option_kind)
        )
        pnl = self._ledger.mark_to_market(spot, option_value)
        pnl_change = pnl - self._prev_pnl
        self._prev_pnl = pnl
        self._pnl_buffer.append(pnl)

        reward = compute_reward(pnl_change, cost, self._pnl_buffer, self.config)
        terminated = self._step >= self._gbm.n_steps

        if terminated:
            # Unwind hedge at maturity
            term_cost = terminal_cost(self._ledger.delta, spot, self.config)
            reward -= term_cost

        obs = self._build_obs(spot, tau, new_delta)
        info: dict[str, Any] = {
            "pnl": pnl,
            "cost": cost,
            "delta": new_delta,
            "spot": spot,
            "tau": tau,
        }
        return obs, reward, terminated, False, info

    def render(self) -> None:  # type: ignore[override]
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_obs(self, spot: float, tau: float, delta: float) -> np.ndarray:
        """Construct the observation vector."""
        log_moneyness = np.log(spot / self.config.strike)
        tau_safe = max(tau, 1e-8)

        base = np.array([log_moneyness, tau_safe, delta], dtype=np.float32)

        if self.config.include_greeks:
            greeks = bs_greeks(
                spot,
                self.config.strike,
                tau_safe,
                self._gbm.rate,
                self._gbm.vol,
                self.config.option_kind,
            )
            greek_vec = np.array(
                [
                    float(greeks["delta"]),
                    float(greeks["gamma"]) * spot,  # dollar gamma
                    float(greeks["vega"]),
                    float(greeks["theta"]) / 365,  # per-day theta
                ],
                dtype=np.float32,
            )
            obs = np.concatenate([base, greek_vec])
        else:
            obs = base

        if self.config.normalise_obs:
            obs = np.clip(obs, -10.0, 10.0)
        return obs.astype(np.float32)
