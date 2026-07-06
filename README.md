# rl-hedging-engine

> **RL-based Derivatives Hedging Engine** — a simulation-first, research-grade Python framework for dynamic option hedging under realistic market frictions.

[![CI](https://github.com/grisheet/rl-hedging-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/grisheet/rl-hedging-engine/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Abstract

Discrete Black-Scholes delta hedging is the textbook solution to European option replication — but it assumes frictionless markets, continuous rebalancing, and perfect volatility knowledge. In practice, transaction costs, finite rebalancing frequencies, and model uncertainty erode its effectiveness substantially.

This project investigates whether **deep reinforcement learning agents** (PPO and TD3) can learn superior hedging policies directly from simulated market paths. The environment models a short European call position under Geometric Brownian Motion dynamics with proportional transaction costs, bid-ask spread, and discrete daily rebalancing. Agents observe the option's current delta, moneyness, time-to-expiry, and recent P&L, and output a continuous hedge ratio that minimises a risk-adjusted reward penalising both P&L variance and transaction cost drag.

The codebase is intentionally self-contained — no Stable Baselines3, no Ray RLlib. Every component (GBM simulator, Black-Scholes pricer, actor-critic networks, replay buffer, GAE advantage estimation) is implemented from scratch in PyTorch, making the internal mechanics fully auditable and modifiable by quant researchers and ML practitioners alike.

---

## Overview

The central research question: **can an RL agent hedge a European option better than discrete Black-Scholes delta hedging once market frictions are present?**

The stack:
- **Gymnasium environment** simulating a short option position under GBM with transaction costs and bid-ask spread
- **Classical baselines** — no-hedge and Black-Scholes delta hedge (closed-form, vectorised)
- **Custom PyTorch implementations** of PPO (on-policy) and TD3 (off-policy) — fully inspectable, no library black-boxes
- **Self-financing ledger accounting** — PnL computed as a testable identity, not ad hoc
- **Evaluation pipeline** — common-random-numbers backtesting, CVaR, drawdown, paired confidence intervals
- **Config-driven** — every experiment reproducible from a single YAML file

> **Key insight**: under frictionless GBM with the correct volatility, discrete delta hedging is already near-optimal. RL earns its keep when frictions bite. The headline experiments are **cost sweeps (0, 10, 50 bps)** — the frictionless case is a validation gate, not the research result.

This sits in the **deep hedging literature** (Buehler et al. 2019, Kolm & Ritter 2019, Cao-Chen-Hull-Poulos 2021) and borrows their best design patterns.

---

## Repository Structure

```
rl-hedging-engine/
├── .github/workflows/ci.yml  # GitHub Actions: lint, test, Docker build
├── configs/
│   └── default.yaml            # Single source of truth for all hyperparameters
├── scripts/
│   ├── train.py                # Training entry-point (PPO or TD3)
│   └── evaluate.py             # Evaluation entry-point with metric reporting
├── src/rlhedge/
│   ├── pricing/                # Black-Scholes pricer + Greeks (vectorised)
│   ├── simulation/             # GBM path simulator
│   ├── envs/                   # Gymnasium HedgingEnv
│   ├── models/                 # PPO + TD3 agents (pure PyTorch)
│   ├── training/               # Trainers, replay buffer, rollout collector
│   └── evaluation/             # Metrics, visualisation, backtesting
├── tests/                      # pytest suite (pricing, env, models)
├── Dockerfile                  # Multi-stage production image
├── docker-compose.yml          # Train + evaluate + TensorBoard services
├── pyproject.toml              # Build config, ruff, mypy, pytest settings
├── requirements.txt            # Pinned runtime dependencies
└── LICENSE                     # MIT
```

---

## Quick Start

### Option A — Local (pip)

```bash
# Clone
git clone https://github.com/grisheet/rl-hedging-engine.git
cd rl-hedging-engine

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install (runtime deps)
pip install -r requirements.txt
pip install -e .

# Train TD3 with default config
python scripts/train.py --config configs/default.yaml

# Evaluate a saved checkpoint
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/td3_final.pt \
  --n_episodes 20
```

### Option B — Docker (recommended for production)

```bash
# Build the image
docker build -t rl-hedging-engine:latest .

# Run training (checkpoints persist to ./checkpoints)
docker run --rm \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  rl-hedging-engine:latest

# Or use Docker Compose (train + TensorBoard)
docker compose up train tensorboard
# TensorBoard available at http://localhost:6006
```

### Option C — Development install

```bash
pip install -e ".[dev]"
pytest tests/ -v              # run test suite
ruff check src/ scripts/      # lint
mypy src/                     # type-check
```

---

## Configuration

All experiment parameters live in `configs/default.yaml`. Override any field at the CLI:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --override training.algorithm=ppo \
  --override environment.transaction_cost=0.005
```

Key sections:

| Section | Key parameters |
|---|---|
| `simulation` | `s0`, `mu`, `sigma`, `dt`, `n_steps`, `n_paths` |
| `option` | `strike`, `maturity`, `r` |
| `environment` | `transaction_cost`, `risk_aversion` |
| `training` | `algorithm`, `total_episodes`, `save_freq` |
| `model` | `hidden_sizes`, `lr` |

---

## Technical Design

### Environment

`HedgingEnv` is a `gymnasium.Env` wrapping a discrete-time option hedging problem:
- **State**: `[S/K, time-to-expiry, current_delta, prev_hedge, step_pnl, realised_vol]`
- **Action**: continuous hedge ratio `∈ [0, 1]`
- **Reward**: `step_pnl − λ · transaction_cost − α · variance_penalty`
- **Terminal**: episode ends at option expiry

### Agents

| Agent | Type | Key features |
|---|---|---|
| **PPO** | On-policy actor-critic | GAE advantage, clipped surrogate loss, entropy regularisation |
| **TD3** | Off-policy actor-critic | Twin critics, delayed policy updates, target policy smoothing |

### Evaluation

All metrics computed over common random number (CRN) paths for paired statistical tests:
- Total P&L distribution (mean, std, CVaR-95)
- Hedge error vs. Black-Scholes delta
- Transaction cost breakdown
- Sharpe ratio

---

## Deployment

### Docker

The `Dockerfile` uses a **two-stage build**:
1. **Builder stage** — installs all dependencies into an isolated venv
2. **Runtime stage** — copies only the venv + source, runs as a non-root user

This keeps the final image lean and secure.

### Docker Compose Services

| Service | Description | Port |
|---|---|---|
| `train` | Runs a full training episode, writes checkpoints and logs | — |
| `evaluate` | Loads a checkpoint and prints evaluation metrics | — |
| `tensorboard` | Serves training curves via TensorBoard | `6006` |

### CI/CD (GitHub Actions)

Every push to `main` and every pull request triggers:
1. **Lint** — `ruff` style check + `mypy` type checking
2. **Test** — `pytest` on Python 3.11 and 3.12 with coverage upload
3. **Docker** — builds the image to verify `Dockerfile` integrity

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

MIT © 2026 Grisheet. See [LICENSE](LICENSE) for full terms.
