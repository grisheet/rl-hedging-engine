"""Transaction cost models."""
from __future__ import annotations

import numpy as np

from rlhedge.envs.config import EnvConfig


def compute_cost(
    delta_old: float,
    delta_new: float,
    spot: float,
    config: EnvConfig,
) -> float:
    """Compute transaction cost for a rebalancing trade.

    Parameters
    ----------
    delta_old:
        Hedge ratio before the trade.
    delta_new:
        Hedge ratio after the trade.
    spot:
        Current spot price.
    config:
        Environment configuration (determines cost model).

    Returns
    -------
    float
        Non-negative transaction cost in dollars.
    """
    trade_size = abs(delta_new - delta_old)
    trade_value = trade_size * spot

    if config.cost_model == "none":
        return 0.0
    elif config.cost_model == "fixed_bps":
        bps_total = config.cost_bps + config.half_spread_bps
        return trade_value * bps_total / 10_000.0
    elif config.cost_model == "proportional":
        return trade_value * config.cost_bps / 10_000.0
    else:
        raise ValueError(f"Unknown cost model: {config.cost_model}")


def terminal_cost(
    delta: float,
    spot: float,
    config: EnvConfig,
) -> float:
    """Cost of unwinding the remaining hedge position at maturity."""
    return compute_cost(delta, 0.0, spot, config)
