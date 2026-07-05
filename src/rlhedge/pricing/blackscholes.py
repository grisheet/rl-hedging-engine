"""Closed-form Black-Scholes pricing and Greeks (vectorised NumPy).

All functions accept scalar or array inputs for batch computation.
Time-to-expiry tau is in years.  Volatility vol and rate are continuously
compounded annualised values.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rlhedge.types import ArrayLike, OptionKind

# Avoid division-by-zero at expiry
TAU_FLOOR: float = 1e-8


def _d1_d2(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute d1 and d2 for Black-Scholes formula."""
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    tau = np.maximum(np.asarray(tau, dtype=float), TAU_FLOOR)
    rate = np.asarray(rate, dtype=float)
    vol = np.asarray(vol, dtype=float)

    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * tau) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau
    return d1, d2


def ndtr(x: npt.NDArray) -> npt.NDArray:
    """Standard normal CDF (pure NumPy, avoids scipy dependency)."""
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))


def bs_price(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    kind: OptionKind = "call",
) -> npt.NDArray:
    """Black-Scholes European option price."""
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    tau_arr = np.asarray(tau, dtype=float)
    rate = np.asarray(rate, dtype=float)
    vol = np.asarray(vol, dtype=float)

    d1, d2 = _d1_d2(spot, strike, tau_arr, rate, vol)
    disc = np.exp(-rate * np.maximum(tau_arr, TAU_FLOOR))

    if kind == "call":
        price = spot * ndtr(d1) - strike * disc * ndtr(d2)
    else:
        price = strike * disc * ndtr(-d2) - spot * ndtr(-d1)

    at_expiry = np.asarray(tau_arr, dtype=float) < TAU_FLOOR
    if kind == "call":
        intrinsic = np.maximum(spot - strike, 0.0)
    else:
        intrinsic = np.maximum(strike - spot, 0.0)
    return np.where(at_expiry, intrinsic, price)


def bs_delta(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    kind: OptionKind = "call",
) -> npt.NDArray:
    """Black-Scholes delta dV/dS."""
    d1, _ = _d1_d2(spot, strike, tau, rate, vol)
    tau_arr = np.asarray(tau, dtype=float)
    at_expiry = tau_arr < TAU_FLOOR
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)

    if kind == "call":
        delta = ndtr(d1)
        intrinsic_delta = (spot > strike).astype(float)
    else:
        delta = ndtr(d1) - 1.0
        intrinsic_delta = -(spot < strike).astype(float)
    return np.where(at_expiry, intrinsic_delta, delta)


def bs_gamma(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
) -> npt.NDArray:
    """Black-Scholes gamma d^2V/dS^2 (same for calls and puts)."""
    spot = np.asarray(spot, dtype=float)
    vol = np.asarray(vol, dtype=float)
    tau_arr = np.maximum(np.asarray(tau, dtype=float), TAU_FLOOR)
    d1, _ = _d1_d2(spot, strike, tau_arr, rate, vol)
    phi_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    return phi_d1 / (spot * vol * np.sqrt(tau_arr))


def bs_vega(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
) -> npt.NDArray:
    """Black-Scholes vega dV/dsigma (same for calls and puts)."""
    spot = np.asarray(spot, dtype=float)
    tau_arr = np.maximum(np.asarray(tau, dtype=float), TAU_FLOOR)
    d1, _ = _d1_d2(spot, strike, tau_arr, rate, vol)
    phi_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    return spot * phi_d1 * np.sqrt(tau_arr)


def bs_theta(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    kind: OptionKind = "call",
) -> npt.NDArray:
    """Black-Scholes theta dV/dt (calendar time decay, per year)."""
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    rate = np.asarray(rate, dtype=float)
    vol = np.asarray(vol, dtype=float)
    tau_arr = np.maximum(np.asarray(tau, dtype=float), TAU_FLOOR)

    d1, d2 = _d1_d2(spot, strike, tau_arr, rate, vol)
    phi_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    disc = np.exp(-rate * tau_arr)
    k = strike * disc

    common = -(spot * phi_d1 * vol) / (2.0 * np.sqrt(tau_arr))
    if kind == "call":
        theta = common - rate * k * ndtr(d2)
    else:
        theta = common + rate * k * ndtr(-d2)

    at_expiry = np.asarray(tau, dtype=float) < TAU_FLOOR
    return np.where(at_expiry, 0.0, theta)


def bs_greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    kind: OptionKind = "call",
) -> dict[str, npt.NDArray]:
    """All Greeks in one call (reuses d1 computation once)."""
    return {
        "price": bs_price(spot, strike, tau, rate, vol, kind),
        "delta": bs_delta(spot, strike, tau, rate, vol, kind),
        "gamma": bs_gamma(spot, strike, tau, rate, vol),
        "vega": bs_vega(spot, strike, tau, rate, vol),
        "theta": bs_theta(spot, strike, tau, rate, vol, kind),
    }


def put_call_parity_check(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    vol: float,
    tol: float = 1e-10,
) -> bool:
    """Verify put-call parity: C - P = S - K*exp(-r*tau). Returns True if holds."""
    call = float(bs_price(spot, strike, tau, rate, vol, "call"))
    put = float(bs_price(spot, strike, tau, rate, vol, "put"))
    lhs = call - put
    rhs = spot - strike * np.exp(-rate * tau)
    return abs(lhs - rhs) < tol
