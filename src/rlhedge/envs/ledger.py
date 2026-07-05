"""Self-financing portfolio ledger.

Tracks the hedge position and running cash account so that
mark-to-market PnL is a testable accounting identity.
"""
from __future__ import annotations

import numpy as np


class Ledger:
    """Stateful ledger for a single hedging episode.

    Convention:
      - We are SHORT one unit of the option.
      - We HOLD `delta` shares as the hedge.
      - Cash account is funded by option premium at t=0 and
        debited / credited for share purchases / sales and costs.
    """

    def __init__(self, option_price: float, initial_delta: float = 0.0) -> None:
        self.option_price: float = option_price
        """Initial option premium received (positive = received cash)."""
        self.delta: float = initial_delta
        """Current hedge ratio (shares held)."""
        self.cash: float = option_price - initial_delta * 0.0
        """Cash account (initialised with premium)."""
        self.total_cost: float = 0.0
        """Cumulative transaction costs paid."""
        self.pnl_history: list[float] = []
        """Mark-to-market PnL at each step."""

    def rebalance(
        self,
        new_delta: float,
        spot: float,
        cost: float,
    ) -> float:
        """Rebalance the hedge to `new_delta` shares.

        Parameters
        ----------
        new_delta:
            Target hedge ratio after rebalancing.
        spot:
            Current spot price used to value the trade.
        cost:
            Transaction cost for this rebalance (always >= 0).

        Returns
        -------
        float
            Cash flow from this rebalance (negative = paid out).
        """
        trade_size = new_delta - self.delta
        cash_flow = -trade_size * spot - cost  # buy shares => cash decreases
        self.cash += cash_flow
        self.delta = new_delta
        self.total_cost += cost
        return cash_flow

    def mark_to_market(
        self,
        spot: float,
        option_value: float,
    ) -> float:
        """Compute portfolio PnL: cash + share_value - option_liability.

        The option liability is the current mark-to-market value of the
        short option position.

        Returns
        -------
        float
            Portfolio PnL (positive = gain).
        """
        share_value = self.delta * spot
        pnl = self.cash + share_value - option_value
        self.pnl_history.append(pnl)
        return pnl

    def reset(self, option_price: float, initial_delta: float = 0.0) -> None:
        """Reset ledger for a new episode."""
        self.__init__(option_price, initial_delta)
