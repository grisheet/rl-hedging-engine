"""Geometric Brownian Motion simulation (vectorised NumPy).

Generates batched price paths used by the hedging environment
and by baseline evaluators.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from rlhedge.types import FloatArray


@dataclass(frozen=True)
class GBMParams:
    """Parameters for a GBM process."""

    spot: float = 100.0
    """Initial spot price S0."""
    drift: float = 0.0
    """Risk-neutral drift mu (usually = rate for pricing)."""
    vol: float = 0.2
    """Annualised volatility sigma."""
    rate: float = 0.0
    """Risk-free rate r (used only for discounting / pricing)."""
    maturity: float = 1.0
    """Time to maturity T in years."""
    n_steps: int = 252
    """Number of discrete time steps per path."""

    @property
    def dt(self) -> float:
        """Length of one time step in years."""
        return self.maturity / self.n_steps


def simulate_gbm_paths(
    params: GBMParams,
    n_paths: int,
    rng: Optional[np.random.Generator] = None,
    antithetic: bool = True,
) -> FloatArray:
    """Simulate GBM price paths using the exact log-normal scheme.

    Parameters
    ----------
    params:
        GBM parameter set.
    n_paths:
        Number of independent paths to generate.  If `antithetic` is True
        this must be even; the function will automatically round up.
    rng:
        Optional ``numpy.random.Generator`` for reproducibility.
    antithetic:
        If True, pair each path with its antithetic counterpart
        (reduces variance of Monte-Carlo estimates).

    Returns
    -------
    FloatArray of shape ``(n_paths, n_steps + 1)``.
        prices[:, 0] == params.spot for all paths.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = params.dt
    half_var_dt = 0.5 * params.vol**2 * dt
    vol_sqrt_dt = params.vol * np.sqrt(dt)
    drift_dt = (params.drift - half_var_dt)

    if antithetic:
        n_half = int(np.ceil(n_paths / 2))
        z_half: FloatArray = rng.standard_normal((n_half, params.n_steps))
        z: FloatArray = np.concatenate([z_half, -z_half], axis=0)[:n_paths]
    else:
        z = rng.standard_normal((n_paths, params.n_steps))

    log_increments = drift_dt + vol_sqrt_dt * z  # (n_paths, n_steps)
    log_prices = np.cumsum(log_increments, axis=1)  # cumulative sum

    # Prepend log(S0) and exponentiate
    log_s0 = np.log(params.spot)
    log_path = np.concatenate(
        [np.full((n_paths, 1), log_s0), log_s0 + log_prices], axis=1
    )
    return np.exp(log_path)  # shape (n_paths, n_steps + 1)


def time_grid(params: GBMParams) -> FloatArray:
    """Return the time grid for a GBM simulation: array of shape (n_steps+1,)."""
    return np.linspace(0.0, params.maturity, params.n_steps + 1)


def remaining_tau(params: GBMParams) -> FloatArray:
    """Remaining time to expiry at each step: T - t."""
    return params.maturity - time_grid(params)
