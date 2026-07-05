"""Plotting utilities for visualising hedging agent performance.

All functions return a ``matplotlib.figure.Figure`` so the caller can
save / display as needed.  Matplotlib is imported lazily so the package
can be used without it if plotting is not required.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _get_mpl():
    """Lazy import of matplotlib to avoid hard dependency at import time."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e


def plot_pnl(
    results: Dict[str, Dict[str, Any]],
    title: str = "P&L Distribution by Agent",
    figsize: tuple = (10, 5),
) -> Any:
    """Box-plot comparing P&L distributions across agents.

    Parameters
    ----------
    results:
        Output of :meth:`~rlhedge.evaluation.backtest.Backtester.run_many`.
    title:
        Figure title.
    figsize:
        Matplotlib figure size.

    Returns
    -------
    ``matplotlib.figure.Figure``
    """
    plt = _get_mpl()
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(results.keys())
    data = [results[label]["pnls"] for label in labels]

    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="Break-even")
    ax.set_ylabel("P&L")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_hedge_ratio(
    hedge_ratios: Sequence[np.ndarray],
    label: str = "agent",
    title: Optional[str] = None,
    figsize: tuple = (10, 4),
    n_episodes: int = 5,
) -> Any:
    """Plot hedge ratio trajectories for a sample of episodes.

    Parameters
    ----------
    hedge_ratios:
        List of 1-D arrays (one per episode) from
        :meth:`~rlhedge.evaluation.backtest.Backtester.run`.
    label:
        Agent label used in the legend.
    title:
        Figure title (defaults to *label*).
    figsize:
        Matplotlib figure size.
    n_episodes:
        Number of episodes to overlay (randomly sampled if more available).

    Returns
    -------
    ``matplotlib.figure.Figure``
    """
    plt = _get_mpl()
    fig, ax = plt.subplots(figsize=figsize)

    rng = np.random.default_rng(0)
    indices = rng.choice(
        len(hedge_ratios), size=min(n_episodes, len(hedge_ratios)), replace=False
    )
    for i, idx in enumerate(indices):
        ax.plot(
            hedge_ratios[idx],
            alpha=0.6,
            label=label if i == 0 else None,
        )

    ax.axhline(0, color="black", linestyle=":", linewidth=0.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Hedge ratio")
    ax.set_title(title or label)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_reward_curve(
    results: Dict[str, Dict[str, Any]],
    title: str = "Episode Reward by Agent",
    figsize: tuple = (10, 4),
) -> Any:
    """Line plot of cumulative mean reward across episodes.

    Parameters
    ----------
    results:
        Output of :meth:`~rlhedge.evaluation.backtest.Backtester.run_many`.
    title:
        Figure title.
    figsize:
        Matplotlib figure size.

    Returns
    -------
    ``matplotlib.figure.Figure``
    """
    plt = _get_mpl()
    fig, ax = plt.subplots(figsize=figsize)

    for label, res in results.items():
        rewards = np.asarray(res["rewards"])
        ax.plot(np.cumsum(rewards) / (np.arange(len(rewards)) + 1), label=label)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean reward (cumulative)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
