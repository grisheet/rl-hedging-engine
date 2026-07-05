"""tests/test_pricing.py – Unit tests for pricing models."""

from __future__ import annotations

import math
import pytest
import numpy as np

from src.pricing.black_scholes import BlackScholes
from src.pricing.greeks import compute_delta, compute_gamma, compute_vega, compute_theta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bs() -> BlackScholes:
    return BlackScholes(r=0.05, sigma=0.2)


# ---------------------------------------------------------------------------
# Black-Scholes pricing tests
# ---------------------------------------------------------------------------

class TestBlackScholes:
    def test_call_price_positive(self, bs: BlackScholes) -> None:
        price = bs.call_price(S=100.0, K=100.0, T=1.0)
        assert price > 0.0

    def test_put_call_parity(self, bs: BlackScholes) -> None:
        S, K, T = 100.0, 100.0, 1.0
        call = bs.call_price(S=S, K=K, T=T)
        put = bs.put_price(S=S, K=K, T=T)
        # C - P = S - K * exp(-rT)
        lhs = call - put
        rhs = S - K * math.exp(-bs.r * T)
        assert abs(lhs - rhs) < 1e-6

    def test_deep_itm_call_close_to_intrinsic(self, bs: BlackScholes) -> None:
        """Deep ITM call ~ S - K*exp(-rT)."""
        S, K, T = 200.0, 100.0, 1.0
        price = bs.call_price(S=S, K=K, T=T)
        intrinsic = S - K * math.exp(-bs.r * T)
        assert abs(price - intrinsic) < 2.0  # small time-value for deep ITM

    def test_zero_maturity_call(self, bs: BlackScholes) -> None:
        """At expiry, call = max(S-K, 0)."""
        price = bs.call_price(S=110.0, K=100.0, T=1e-8)
        assert abs(price - 10.0) < 0.01

    def test_zero_maturity_otm_call(self, bs: BlackScholes) -> None:
        price = bs.call_price(S=90.0, K=100.0, T=1e-8)
        assert price < 0.01

    def test_call_price_vectorised(self, bs: BlackScholes) -> None:
        S = np.array([90.0, 100.0, 110.0])
        prices = bs.call_price(S=S, K=100.0, T=1.0)
        assert prices.shape == (3,)
        assert np.all(prices > 0)

    def test_put_price_positive(self, bs: BlackScholes) -> None:
        price = bs.put_price(S=100.0, K=100.0, T=1.0)
        assert price > 0.0


# ---------------------------------------------------------------------------
# Greeks tests
# ---------------------------------------------------------------------------

class TestGreeks:
    def test_delta_range(self, bs: BlackScholes) -> None:
        delta = compute_delta(bs, S=100.0, K=100.0, T=1.0)
        assert 0.0 <= delta <= 1.0

    def test_delta_deep_itm(self, bs: BlackScholes) -> None:
        delta = compute_delta(bs, S=200.0, K=100.0, T=1.0)
        assert delta > 0.9

    def test_delta_deep_otm(self, bs: BlackScholes) -> None:
        delta = compute_delta(bs, S=50.0, K=100.0, T=1.0)
        assert delta < 0.1

    def test_gamma_positive(self, bs: BlackScholes) -> None:
        gamma = compute_gamma(bs, S=100.0, K=100.0, T=1.0)
        assert gamma > 0.0

    def test_vega_positive(self, bs: BlackScholes) -> None:
        vega = compute_vega(bs, S=100.0, K=100.0, T=1.0)
        assert vega > 0.0

    def test_theta_negative(self, bs: BlackScholes) -> None:
        """Theta for a long option should be negative (time decay)."""
        theta = compute_theta(bs, S=100.0, K=100.0, T=1.0)
        assert theta < 0.0

    def test_delta_vectorised(self, bs: BlackScholes) -> None:
        S = np.linspace(80, 120, 10)
        deltas = compute_delta(bs, S=S, K=100.0, T=1.0)
        assert deltas.shape == (10,)
        assert np.all(np.diff(deltas) > 0)  # monotonically increasing
