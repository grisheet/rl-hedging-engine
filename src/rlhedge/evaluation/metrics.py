"""Performance metrics for hedging agents."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def compute_metrics(rewards: Sequence[float], pnls: Sequence[float]) -> Dict[str, float]:
    """Compute a standard set of hedging performance metrics.

    Parameters
    ----------
    rewards:
        Episode total rewards (e.g. cumulative hedging rewards).
    pnls:
        Episode final P&L values.

    Returns
    -------
    Dict containing:
        - ``mean_reward`` / ``std_reward``
        - ``mean_pnl`` / ``std_pnl``
        - ``sharpe``: Sharpe ratio of P&L (annualised with 252 episodes/year)
        - ``var_95``: 95 % Value at Risk (5th percentile of P&L)
        - ``cvar_95``: 95 % Conditional Value at Risk (mean of bottom 5 %)
        - ``max_drawdown``: maximum drawdown of cumulative P&L
    """
    r = np.asarray(rewards, dtype=np.float64)
    p = np.asarray(pnls, dtype=np.float64)

    sharpe = _sharpe(p)
    var_95 = float(np.percentile(p, 5))
    cvar_95 = float(p[p <= var_95].mean()) if (p <= var_95).any() else float("nan")
    max_dd = _max_drawdown(np.cumsum(p))

    return {
        "mean_reward": float(r.mean()),
        "std_reward": float(r.std()),
        "mean_pnl": float(p.mean()),
        "std_pnl": float(p.std()),
        "sharpe": sharpe,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "max_drawdown": max_dd,
    }


def _sharpe(pnl: np.ndarray, periods_per_year: float = 252.0) -> float:
    """Annualised Sharpe ratio (assumes zero risk-free rate)."""
    if pnl.std() == 0:
        return float("nan")
    return float(pnl.mean() / pnl.std() * np.sqrt(periods_per_year))


def _max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown of an equity curve."""
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / max(abs(peak), 1e-9)
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)
