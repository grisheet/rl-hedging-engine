# rl-hedging-engine

> **RL-based Derivatives Hedging Engine** — simulation-first, research-grade Python repo for dynamic option hedging under market frictions.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Overview

This repo answers a concrete research question: **can an RL agent hedge a European option better than discrete Black-Scholes delta hedging once market frictions are present?**

The stack:
- **Gymnasium environment** simulating a short option position under GBM with transaction costs and bid-ask spread
- **Classical baselines** — no-hedge and Black-Scholes delta hedge (closed-form, vectorized)
- **Custom PyTorch implementations** of PPO (on-policy) and TD3 (off-policy) — fully inspectable, no library black-boxes
- **Self-financing ledger accounting** — PnL computed as a testable identity, not ad hoc
- **Evaluation pipeline** — common-random-numbers backtesting, CVaR, drawdown, paired confidence intervals
- **Config-driven** — every experiment reproducible from a single YAML file

This sits in the **deep hedging literature** (Buehler et al. 2019, Kolm & Ritter 2019, Cao-Chen-Hull-Poulos 2021) and borrows their best design patterns.

> **Key insight**: under frictionless GBM with the correct volatility, discrete delta hedging is already near-optimal. RL earns its keep when frictions bite. The headline experiments are **cost sweeps (0, 10, 50 bps)** — the frictionless case is a validation gate, not the research result.

---

## Repository Structure

```
rl-hedging-engine/
├── pyproject.toml          # deps, ruff/mypy/pytest config
├── README.md
├── .gitignore
├── .python-version         # 3.11
│
├── configs/
│   ├── env/
│   │   ├── gbm_call_frictionless.yaml
│   │   ├── gbm_call_10bp.yaml
│   │   └── gbm_call_50bp.yaml
│   ├── agent/
│   │   ├── ppo_default.yaml
│   │   └── td3_default.yaml
│   └── experiment/
│       ├── ppo_vs_baselines.yaml
│       └── cost_sweep.yaml
│
├── src/rlhedge/
│   ├── __init__.py
│   ├── types.py            # shared type aliases
│   ├── pricing/
│   │   ├── __init__.py
│   │   ├── blackscholes.py # closed-form BS price, delta, gamma, vega, theta
│   │   └── payoffs.py      # terminal payoff functions
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── gbm.py          # exact GBM path simulation (P-measure)
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── config.py       # frozen dataclass env config
│   │   ├── ledger.py       # self-financing cash accounting
│   │   ├── costs.py        # transaction cost models
│   │   ├── rewards.py      # reward shaping variants
│   │   └── hedging_env.py  # Gymnasium HedgingEnv
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baselines.py    # NoHedge, BSDelta (Policy protocol)
│   │   ├── networks.py     # shared MLP builder
│   │   ├── ppo.py          # PPO from scratch
│   │   └── td3.py          # TD3 from scratch (DDPG as ablation)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── rollout_buffer.py
│   │   ├── replay_buffer.py
│   │   └── logger.py
│   └── evaluation/
│       ├── __init__.py
│       ├── backtest.py     # CRN backtester
│       ├── metrics.py      # CVaR, drawdown, Sharpe, turnover
│       └── plots.py        # PnL distributions, hedge paths, cost-risk frontier
│
├── scripts/
│   ├── run_env_sanity.py   # gymnasium.utils.env_checker + deterministic checks
│   ├── run_baselines.py    # benchmark no-hedge and delta-hedge
│   ├── train_ppo.py
│   ├── train_td3.py
│   └── run_benchmark.py    # full evaluation suite
│
├── tests/
│   ├── test_blackscholes.py
│   ├── test_gbm.py
│   ├── test_ledger.py
│   ├── test_hedging_env.py
│   ├── test_baselines.py
│   └── test_metrics.py
│
├── notebooks/
│   └── 01_env_sanity_checks.ipynb
│
└── outputs/               # gitignored: figures/, models/, logs/, reports/
```

---

## Design Decisions

### P-measure vs Q-measure separation
The **simulator** (`simulation/gbm.py`) uses the real-world drift μ (P-measure). The **pricer** (`pricing/blackscholes.py`) uses the risk-free rate r (Q-measure). These are kept in separate packages to make it structurally impossible to confuse them — and to enable future misspecification experiments (e.g., BS-pricing agent in a Heston world) without touching either layer.

### Self-financing ledger
At t=0 the agent sells the option and receives the BS premium into cash. Portfolio value:
```
V(t) = cash(t) + h(t)·S(t) - notional·C(t)
```
where `C(t)` is BS mid-price during the episode and intrinsic payoff at maturity. Each step: (1) trade executes, cash pays notional + costs; (2) spot advances, cash accrues at r·dt; (3) option re-marks; (4) step PnL = ΔV. A unit test verifies `V(T) = Σ step_PnL` to machine precision.

### Action space
Default: **trade increment** (hedge adjustment). The executed hedge position is `h(t) = h(t-1) + action·max_trade`. Transaction costs fall out directly from `|action|`. A `target` mode (desired position) is available behind a config flag — useful for comparing to the deep hedging literature which often uses target actions.

### Reward
Default: **mean-variance** proxy
```
r(t) = ΔV(t) - κ · ΔV(t)²   (normalized by S₀·σ·√dt)
```
True episode-level CVaR is computed at evaluation time only — not as a training reward, which would require a distributional critic (deferred to the extension backlog). This matches the Kolm & Ritter (2019) approach.

### Agent implementations
- **PPO**: Gaussian policy (tanh-squashed mean, learned log-std), separate critic, GAE(λ), advantage normalization, clipped surrogate objective
- **TD3**: Deterministic actor, twin critics, delayed policy updates, target-policy smoothing. DDPG falls out as a TD3 config ablation (disable twin critics and smoothing).
- Both implement the `PolicyProtocol` so the backtester is strategy-agnostic.

### Common Random Numbers
All strategies are evaluated on **identical path sets** (fixed eval seed, disjoint from training seeds). This enables paired comparisons with tighter confidence intervals than independent sampling.

---

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Sanity check the environment
python scripts/run_env_sanity.py

# Run Black-Scholes + no-hedge baselines
python scripts/run_baselines.py --config configs/experiment/ppo_vs_baselines.yaml

# Train PPO
python scripts/train_ppo.py --config configs/agent/ppo_default.yaml

# Train TD3
python scripts/train_td3.py --config configs/agent/td3default.yaml

# Full benchmark across all strategies
python scripts/run_benchmark.py --config configs/experiment/cost_sweep.yaml
```

### Run tests
```bash
pytest tests/ -v
pytest tests/test_blackscholes.py tests/test_gbm.py tests/test_hedging_env.py -q
```

---

## Milestones

| # | Deliverable | Risk Retired |
|---|---|---|
| M0 | Scaffold, pricing, GBM simulation | — |
| M1 | Hedging env, ledger, costs, rewards, sanity tests | Financial accounting correctness |
| M2 | No-hedge + delta-hedge baselines, backtest plots | Benchmark validity |
| M3 | PPO training, logging, learned-position-vs-delta diagnostic | RL learnability |
| M4 | TD3 agent, friction stress tests, 5-seed stability | Continuous-control robustness |
| M5 | Full eval suite, cost-risk frontier, README, refactor | Research usability |

### Acceptance criteria at M2 (validates the entire stack before any RL)
- Delta-hedge PnL std < no-hedge PnL std
- Std shrinks by ~√2 per doubling of rebalance frequency (O(√dt) discrete hedging law)
- Delta-hedge mean PnL ≈ 0 even when μ ≠ r (drift immunity)
- No-hedge mean PnL matches closed-form E[payoff] under P

---

## Math Reference

### Black-Scholes pricing
```
d1 = (ln(S/K) + (r + 0.5σ²)τ) / (σ√τ)
d2 = d1 - σ√τ
Call = S·N(d1) - K·e^{-rτ}·N(d2)
Put  = K·e^{-rτ}·N(-d2) - S·N(-d1)
Δ_call = N(d1),   Δ_put = N(d1) - 1
Γ = N'(d1) / (S·σ·√τ)   [same for calls and puts]
```
Numerical edge cases: τ → 0 price → intrinsic, Δ → indicator, Γ/ν → 0. We floor τ at 1e-8 years and document the ATM-at-expiry ambiguity (we pick Δ = 0.5).

### GBM exact scheme
```
S(t+dt) = S(t) · exp((μ - 0.5σ²)dt + σ√dt · Z),  Z ~ N(0,1)
```
Using the exact transition density — no Euler discretization bias. The dt only controls rebalancing frequency, not simulation accuracy.

### CVaR (Expected Shortfall)
```
Loss convention: L = -PnL
VaR_α = α-quantile of L
CVaR_α = E[L | L ≥ VaR_α]
```
Sign convention is explicit in the metrics module to prevent the most common bug in risk code.

---

## Dependencies

**Core**: `torch`, `gymnasium`, `numpy`, `scipy`, `pandas`, `pyyaml`, `pydantic`

**Dev**: `pytest`, `pytest-cov`, `ruff`, `mypy`, `tensorboard`

Deliberately excludes `stable-baselines3` as a core dependency — custom implementations keep the algorithms fully inspectable and modifiable. SB3 is available as an optional cross-check.

---

## Extension Backlog

- [ ] Heston / jump-diffusion dynamics (plug in via `PathSimulatorProtocol`)
- [ ] Leland volatility correction and no-trade-band baselines (Whalley-Wilmott)
- [ ] Terminal expected-shortfall training objective (Buehler OCE representation)
- [ ] Options as hedge instruments (gamma/vega hedging)
- [ ] Recurrent policies for path-dependent products
- [ ] Vol surface / implied vol surface as state feature

---

## References

- Buehler, H. et al. (2019). *Deep Hedging*. Quantitative Finance.
- Kolm, P. & Ritter, G. (2019). *Dynamic Replication and Hedging: A Reinforcement Learning Approach*.
- Cao, J., Chen, J., Hull, J., & Poulos, Z. (2021). *Deep Hedging of Derivatives Using Reinforcement Learning*.
- Whalley, A.E. & Wilmott, P. (1997). *An Asymptotic Analysis of an Optimal Hedging Model for Option Pricing with Transaction Costs*.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*.
- Fujimoto, S. et al. (2018). *Addressing Function Approximation Error in Actor-Critic Methods* (TD3).

---

## License

MIT © Grisheet
