# Meta-Learning Layer

## Overview

The `meta_learning/` module is the adaptive orchestration and signal aggregation layer of the trading system.

This layer is responsible for:

- ensemble aggregation,
- adaptive model weighting,
- regime-aware signal routing,
- confidence filtering,
- final trade approval.

Unlike classical ensemble systems using static averaging, this architecture dynamically changes model importance depending on current market conditions.

The module acts as:

```text
adaptive probabilistic decision controller
```

between predictive models and execution logic.

---

## Production vs legacy (май 2026)

| Тема | Canonical (`orchestration`) | Legacy (`ist.py` / этот README ниже) |
|------|---------------------------|--------------------------------------|
| Ключи моделей | `lgb`, `gru`, `xgb`, `cnn` | `lgb`, `lstm`, `cnn`, `trans` |
| Режим | `MomentumRegimeDetector` (SMA), `regime_pred` 0/1 | ML + Colab-имена |
| Ансамбль | `DynamicMetaWeighting` + веса из `config/profiles/canonical_4model.yaml` | те же идеи, другие ключи |
| Фильтр сделок | `decision.DecisionPipeline` (`integrated`) | inline `np.where` |
| Online-обучение весов по PnL | **Нет** | `WeightUtils.adaptive_weight_update` — утилита, не в pipeline |

Для ВКР и аудита кода: **`docs/vkr/02-ml-ensemble-decision-features.md`**.

---

# Core Architecture

```text
Market Features
    ↓

Regime Model
    ↓

Model Predictions
(prod: LightGBM + GRU + XGBoost + CNN; legacy doc: LSTM + Transformer)
    ↓

Meta-Learning Layer
    ├── Ensemble Aggregation
    ├── Regime-Adaptive Weighting
    ├── Meta Filtering
    └── Signal Assembly

    ↓

Final Trading Signal
    ↓

Risk Management
    ↓

Execution
```

---

# System Philosophy

Different model architectures perform better under different market conditions.

Examples:

| Market Regime | Preferred Models |
|---|---|
| Trend (prod) | GRU + CNN (+ малый вес LGB/XGB) |
| Range / Flat (prod) | LightGBM + XGBoost (+ CNN) |
| Trend (legacy names) | LSTM + Transformer |
| Low Confidence | No Trade |

The meta-learning layer dynamically adapts model influence depending on detected market structure.

---

# Current Production Logic

The current implementation uses:

- probabilistic ensemble aggregation,
- regime-adaptive weighting,
- confidence-based filtering,
- median adaptive thresholds.

The architecture intentionally avoids:

- overly complex stacking,
- unstable RL routing,
- excessive deep ensemble depth.

Priority is given to:

- robustness,
- interpretability,
- stable walk-forward performance.

---

# Current Status

| Component | Status |
|---|---|
| `DynamicMetaWeighting` (regime-adaptive) | **Production** — via `orchestration` |
| `DecisionPipeline` + thresholds | **Production** — `decision/` |
| `EnsembleAggregator` (simple average) | Baseline / benchmark only |
| `SignalAssembler` | Parallel path; canonical uses `DecisionPipeline` |
| Separate trained MetaFilterModel pickle | **Not** in canonical bundle |
| Full stacking L2 | Research |
| MAML routing | Research |
| RL orchestration of direction | **Not** — see `rl_layer/` (risk multiplier only) |

---

# Module Structure

```text
meta_learning/
│
├── __init__.py
│
├── signal_assembler.py
├── ensemble.py
├── dynamic_meta.py
├── thresholds.py
│
├── calibration/
├── evaluation/
├── configs/
└── utils/
```

---

# 1. signal_assembler.py

## Purpose

Final signal generation layer.

Combines:

- directional prediction,
- meta probability,
- confidence thresholds,
- regime filtering.

---

## Current Logic

```python
final_signal = np.where(
    (meta_prob > meta_threshold)
    & (direction_soft_signal != 0),
    direction_soft_signal,
    0
)
```

---

## Responsibilities

- approve/reject trades,
- reduce low-quality entries,
- enforce confidence filtering.

---

# 2. ensemble.py

## Purpose

Model probability aggregation.

Provides baseline ensemble logic.

---

## Current Ensemble

```python
ensemble_prob = (
    lgb_p +
    lstm_p +
    cnn_p +
    trans_p
) / 4
```

---

## Models Used

| Model | Role |
|---|---|
| LightGBM | Tabular / range logic |
| LSTM | Sequential trend modeling |
| CNN | Volatility structure |
| Transformer | Long-range temporal context |

---

## Purpose of Baseline Ensemble

Used for:

- benchmarking,
- ensemble comparison,
- fallback aggregation.

Production system prioritizes:
- adaptive weighting,
- regime-aware orchestration.

---

# 3. dynamic_meta.py

## Purpose

Dynamic regime-aware ensemble weighting.

This is the CORE adaptive orchestration layer of the system.

The module dynamically changes model importance depending on current market regime.

---

# Regime Routing

| `regime_pred` | Interpretation |
|---|---|
| `1` | Trend / strong directional market |
| `0` | Range / weak directional market |

---

# Current Adaptive Weights

## Trend Regime

```python
{
    'lgb': 0.10,
    'lstm': 0.45,
    'cnn': 0.10,
    'trans': 0.35
}
```

Priority:
- LSTM,
- Transformer.

Because:
- sequential models perform better during momentum persistence.

---

## Range Regime

```python
{
    'lgb': 0.55,
    'lstm': 0.10,
    'cnn': 0.25,
    'trans': 0.10
}
```

Priority:
- LightGBM,
- CNN.

Because:
- tabular and local-pattern models perform better during ranging conditions.

---

# Current Integrated Signal Logic

```python
integrated_signal = np.where(
    (meta_mgmt_prob > integrated_threshold)
    & (direction_soft_signal != 0),
    direction_soft_signal,
    0
)
```

---

# 4. thresholds.py

## Purpose

Centralized threshold management.

---

# Current Thresholds

| Parameter | Value |
|---|---|
| Direction threshold | 0.52 |
| Meta threshold | Median adaptive threshold |
| Ensemble mode | Regime-adaptive |

---

# Current Configuration

```yaml
meta_learning:
  direction_threshold: 0.52

  meta_threshold_mode: "median"

  ensemble:
    default: "regime_adaptive"

    regime_weights:

      trend:
        lgb: 0.10
        lstm: 0.45
        cnn: 0.10
        trans: 0.35

      range:
        lgb: 0.55
        lstm: 0.10
        cnn: 0.25
        trans: 0.10

  dl_window_size: 24
```

---

# Adaptive Architecture Logic

The system dynamically selects dominant model families depending on market structure.

---

# Trend Regime

```text
Trend Market
    ↓
LSTM + Transformer Dominance
```

Because:
- sequential dependencies matter more,
- momentum persistence increases.

---

# Range Regime

```text
Range Market
    ↓
LightGBM + CNN Dominance
```

Because:
- local structures,
- mean reversion,
- tabular thresholds become more important.

---

# MetaFilterModel

## Purpose

Final trade quality controller.

Determines:

```text
Should this trade actually be executed?
```

---

# Current Implementation

## Primary Model

### LightGBM

Used because:
- strong tabular performance,
- stable calibration,
- low-latency inference.

---

# Planned Alternatives

| Model | Status |
|---|---|
| Logistic Regression | Planned |
| CatBoost | Research |
| Stacking Meta-Model | Research |

---

# Walk-Forward Integration

The meta-learning layer is fully integrated into:

```text
run_integrated_wfo
```

Current process:

```text
Train Fold
    ↓
Retrain LGBM
    ↓
Short DL Fine-Tuning
    ↓
Dynamic Meta Aggregation
    ↓
Integrated Signal Generation
    ↓
Backtesting
```

---

# Evaluation Focus

The meta-learning layer is evaluated using:

---

# Financial Metrics

- Profit Factor
- Sharpe Ratio
- Max Drawdown
- Trade Quality
- Turnover Reduction

---

# ML Metrics

- calibration quality,
- confidence stability,
- regime specialization,
- ensemble consistency.

---

# Research Roadmap

---

# Phase 1 — Current Stable Architecture

Implemented:

- regime-adaptive ensemble,
- confidence filtering,
- dynamic weighting,
- integrated signal generation.

Goal:

```text
stable adaptive orchestration
```

---

# Phase 2 — Advanced Meta Learning

Planned:

- stacking meta-model,
- rolling threshold adaptation,
- online ensemble weighting,
- rolling Sharpe-based weight updates.

Goal:

```text
adaptive ensemble optimization
```

---

# Phase 3 — Advanced Research

Future research:

- reinforcement-learning routing,
- Bayesian ensemble optimization,
- online continual adaptation,
- market-state-aware neural weighting.

Goal:

```text
fully adaptive probabilistic orchestration
```

---

# Final Principle

The meta-learning layer is designed as:

```text
adaptive probabilistic orchestration system
```

NOT:

```text
simple static ensemble averaging
```

The primary edge comes from:

- regime adaptation,
- model specialization,
- confidence filtering,
- dynamic weighting,
- robust signal selection.