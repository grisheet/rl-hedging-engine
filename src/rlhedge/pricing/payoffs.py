"""Terminal payoff functions for European options."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rlhedge.types import ArrayLike


def call_payoff(spot: ArrayLike, strike: ArrayLike) -> npt.NDArray:
    """European call payoff: max(S - K, 0)."""
    return np.maximum(np.asarray(spot, dtype=float) - np.asarray(strike, dtype=float), 0.0)


def put_payoff(spot: ArrayLike, strike: ArrayLike) -> npt.NDArray:
    """European put payoff: max(K - S, 0)."""
    return np.maximum(np.asarray(strike, dtype=float) - np.asarray(spot, dtype=float), 0.0)


def straddle_payoff(spot: ArrayLike, strike: ArrayLike) -> npt.NDArray:
    """Long straddle payoff: call + put = |S - K|."""
    return np.abs(np.asarray(spot, dtype=float) - np.asarray(strike, dtype=float))


def strangle_payoff(
    spot: ArrayLike,
    call_strike: ArrayLike,
    put_strike: ArrayLike,
) -> npt.NDArray:
    """Long strangle payoff (put_strike < call_strike)."""
    return call_payoff(spot, call_strike) + put_payoff(spot, put_strike)
