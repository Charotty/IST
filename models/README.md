# models/ — Adaptive Multi-Model Prediction Layer

## Overview

The `models/` module contains all predictive architectures used by the adaptive trading system.

Unlike traditional trading systems based on a single predictive model, this architecture uses:

```text
dynamic model selection based on market regime
```

Different model families are activated under different market conditions.

The goal is to improve:

- robustness,
- adaptability,
- regime specialization,
- risk-adjusted performance.

---

# Core Architecture

```text
Market Features
    ↓
Regime Detection Layer
    ↓
Model Router
    ↓

Trend Regime
    → GRU/LSTM Model

Mean Reversion Regime
    → XGBoost Model

Volatility Breakout Regime
    → CNN Model

Low Confidence Regime
    → No Trade

    ↓

Meta Filter
    ↓
Position Sizing
    ↓
Execution
```

---

# System Philosophy

The market is non-stationary.

A single model cannot consistently outperform across:

- trending markets,
- ranging markets,
- volatility expansions,
- liquidity shocks.

Therefore, the system dynamically selects specialized models depending on current market structure.

---

# Directory Structure

```text
models/
│
├── regime/
├── router/
├── trend/
├── mean_reversion/
├── volatility/
├── meta/
├── sizing/
│
├── calibration/
├── evaluation/
├── registry/
├── training/
├── inference/
└── utils/
```

---

# 1. regime/

## Purpose

Detect current market conditions.

This module determines:

- trend vs flat,
- bullish vs bearish,
- high volatility vs low volatility.

---

## Primary Model

### LightGBM

Best suited for:
- tabular features,
- regime classification,
- nonlinear feature interactions.

---

## Alternative Models

- Hidden Markov Models
- XGBoost

---

## Outputs

```python
{
    "market_regime": "trend",
    "volatility_regime": "high_vol",
    "trade_allowed": True
}
```

---

# 2. router/

## Purpose

Dynamic model orchestration layer.

This module selects which predictive model should be used based on current regime conditions.

---

## Example

```python
if regime == "trend":
    active_model = GRUModel()

elif regime == "range":
    active_model = XGBoostModel()

elif regime == "breakout":
    active_model = CNNModel()

else:
    skip_trade()
```

---

# 3. trend/

## Purpose

Directional trend continuation prediction.

Activated during:
- trending markets,
- directional momentum regimes.

---

## Primary Models

### GRU / LSTM

Best suited for:
- sequential dependencies,
- momentum persistence,
- temporal pattern learning.

---

## Alternative Models

- Temporal Transformer
- Temporal CNN

---

## Inputs

- RSI
- MACD delta
- EMA slope
- momentum acceleration
- trend persistence

---

## Outputs

```python
{
    "long_probability": 0.74,
    "short_probability": 0.18
}
```

---

# 4. mean_reversion/

## Purpose

Mean reversion prediction in ranging markets.

Activated during:
- flat regimes,
- low trend strength environments.

---

## Primary Models

### XGBoost / CatBoost

Best suited for:
- tabular market structure,
- threshold behavior,
- indicator-based reversals.

---

## Inputs

- RSI extremes
- Bollinger distance
- z-score deviation
- local volatility

---

## Outputs

```python
{
    "reversal_probability": 0.68
}
```

---

# 5. volatility/

## Purpose

Volatility breakout prediction.

This model predicts:
- volatility expansion,
- breakout probability,
- explosive movement conditions.

---

## Primary Models

### CNN

Best suited for:
- local temporal structures,
- volatility clustering,
- compression-expansion detection.

---

## Alternative Models

- LightGBM
- LSTM

---

## Inputs

- ATR compression
- rolling volatility
- volume spikes
- realized volatility

---

## Outputs

```python
{
    "breakout_probability": 0.81
}
```

---

# 6. meta/

## Purpose

Final trade quality filtering.

This layer determines:
- whether the signal should actually be executed.

---

## Primary Models

### Logistic Regression

Chosen for:
- interpretability,
- stability,
- low overfitting risk.

---

## Alternative Models

- LightGBM

---

## Inputs

- model confidence
- spread
- slippage
- volatility state
- regime confidence

---

## Outputs

```python
{
    "take_trade": True
}
```

---

# 7. sizing/

## Purpose

Dynamic position sizing.

Controls:
- exposure,
- leverage,
- capital allocation.

---

## Phase 1

Rule-based sizing.

---

## Phase 2

Bayesian sizing.

---

## Phase 3

RL-based allocation.

---

# 8. calibration/

## Purpose

Probability calibration layer.

Transforms:

```text
raw probabilities → calibrated probabilities
```

---

## Methods

- Platt Scaling
- Isotonic Regression

---

# 9. evaluation/

## Purpose

Evaluate:
- profitability,
- robustness,
- regime specialization.

---

## Key Metrics

### Financial

- Sharpe Ratio
- Profit Factor
- Max Drawdown
- CAGR

### ML

- Precision
- Recall
- Calibration Quality

---

## Advanced Metrics

### Regime Attribution

```text
Which model performs best in which regime?
```

### Stability Analysis

```text
Do models survive changing market conditions?
```

---

# 10. inference/

## Purpose

Real-time inference orchestration.

---

## Pipeline

```text
Features
    ↓
Regime Detection
    ↓
Model Routing
    ↓
Specialized Prediction
    ↓
Meta Filter
    ↓
Sizing
    ↓
Execution
```

---

# Development Roadmap

---

# Phase 1 — Stable Foundation

Implement:
- LightGBM regime model,
- GRU trend model,
- XGBoost mean reversion,
- CNN volatility model,
- logistic meta-filter.

Goal:

```text
validate adaptive architecture
```

---

# Phase 2 — Robustness

Add:
- calibration,
- feature stability,
- drift detection,
- ensemble weighting.

---

# Phase 3 — Advanced Research

Add:
- transformers,
- orderbook modeling,
- reinforcement learning,
- portfolio optimization.

---

# Final Principle

The system is designed as:

```text
an adaptive probabilistic trading architecture
```

NOT:

```text
a single universal prediction model
```

The edge comes from:
- specialization,
- regime adaptation,
- dynamic routing,
- risk control,
- execution discipline.