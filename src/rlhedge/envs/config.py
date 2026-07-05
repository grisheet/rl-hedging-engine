"""Environment configuration dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rlhedge.simulation.gbm import GBMParams
from rlhedge.types import OptionKind


@dataclass
class EnvConfig:
    """Full specification of one hedging-environment instance."""

    # --- market simulation ---
    gbm: GBMParams = field(default_factory=GBMParams)
    """GBM simulation parameters."""

    # --- option specification ---
    strike: float = 100.0
    """Option strike price K."""
    option_kind: OptionKind = "call"
    """Whether we are hedging a short call or short put position."""

    # --- transaction costs ---
    cost_model: Literal["fixed_bps", "proportional", "none"] = "fixed_bps"
    """Transaction cost model to use."""
    cost_bps: float = 10.0
    """Transaction cost in basis points (used by fixed_bps model)."""
    half_spread_bps: float = 0.0
    """Half bid-ask spread in basis points."""

    # --- reward ---
    reward_type: Literal["pnl", "cvar"] = "pnl"
    """Reward signal: raw PnL or CVaR-penalised."""
    cvar_alpha: float = 0.95
    """CVaR confidence level (used when reward_type == 'cvar')."""
    risk_lambda: float = 1.0
    """Risk-aversion weight on CVaR penalty."""

    # --- observation ---
    include_greeks: bool = True
    """Whether to include BS Greeks in the observation vector."""

    # --- normalisation ---
    normalise_obs: bool = True
    """Whether to normalise observations to roughly [-1, 1]."""

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ValueError(f"cvar_alpha must be in (0,1), got {self.cvar_alpha}")
