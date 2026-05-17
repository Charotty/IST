# Orchestration Layer

Unified orchestration for training and inference with explicit pipeline contract.

## Problem Solved

Previously, there were two incompatible orchestration approaches:

### A. models/ branch (Old)
- Regime → Router → ONE active model (GRU / XGB / CNN)
- InferenceEngine.predict() calls one model and reduces to thresholds 0.6/0.4
- ModelRouter registers models by regime keys: 'trend', 'mean_reversion', 'volatility'
- **Issue**: Only single model, no ensemble/meta-layer in live path

### B. meta_learning/ branch (Old)
- All models → weights by regime → meta_mgmt_prob
- DynamicMetaWeighting expects keys: lgb, lstm, cnn, trans
- EnsembleAggregator combines predictions from all models
- **Issue**: Key mismatch with router, no integration with InferenceEngine

### Danger
Building "production" that looks multi-model on paper but is actually:
- Single-model router OR
- Static ensemble without regime routing

## Solution

Unified orchestrator with explicit contract:

```
features[t] → regime[t] → {p_lgb, p_gru, p_xgb, p_cnn, ...}[t] → meta_mgmt[t] → direction_soft[t] → decision → risk
```

### Key Features

1. **Fixed Model Keys in Config**
   - All models registered with consistent keys
   - Config validation ensures keys match across components
   - No more key mismatches

2. **All Models Called Every Time**
   - Old: router → ONE model
   - New: collect predictions from ALL models
   - Enables true regime-aware ensemble

3. **Explicit Pipeline Contract**
   - TrainingOrchestrator: WFO/train→test with full pipeline
   - InferenceOrchestrator: live trading with full pipeline
   - Both follow same contract

4. **Regime-Aware Meta Weighting**
   - DynamicMetaWeighting integrated into pipeline
   - Weights change based on regime
   - All models contribute with appropriate weights

## Usage

### Configuration

```yaml
orchestration:
  model_keys: ['lgb', 'gru', 'xgb', 'cnn']
  regime_keys: ['trend', 'range', 'breakout']
  direction_threshold: 0.52
  meta_threshold: 0.5
  signal_threshold: 0.6
  ensemble_mode: "regime_adaptive"
  trend_weights:
    lgb: 0.10
    gru: 0.45
    xgb: 0.10
    cnn: 0.35
  range_weights:
    lgb: 0.55
    gru: 0.10
    xgb: 0.25
    cnn: 0.10
  breakout_weights:
    lgb: 0.25
    gru: 0.25
    xgb: 0.25
    cnn: 0.25
```

### Training

```python
from orchestration import TrainingOrchestrator, OrchestratorConfig

# Load config
config = OrchestratorConfig.from_yaml('config.yaml')

# Initialize orchestrator
orchestrator = TrainingOrchestrator(config)
orchestrator.initialize(
    models={'lgb': lgb_model, 'gru': gru_model, 'xgb': xgb_model, 'cnn': cnn_model},
    regime_detector=regime_detector,
    meta_weighting=dynamic_meta_weighting,
    decision_engine=decision_engine,
    risk_manager=risk_manager
)

# Walk-forward optimization
results = orchestrator.walk_forward_optimization(features, targets)

# Simple train-test split
train_result, test_result = orchestrator.train_test_split(features, targets)
```

### Inference

```python
from orchestration import InferenceOrchestrator, OrchestratorConfig

# Load config
config = OrchestratorConfig.from_yaml('config.yaml')

# Initialize orchestrator
orchestrator = InferenceOrchestrator(config)
orchestrator.initialize(
    models={'lgb': lgb_model, 'gru': gru_model, 'xgb': xgb_model, 'cnn': cnn_model},
    regime_detector=regime_detector,
    meta_weighting=dynamic_meta_weighting,
    decision_engine=decision_engine,
    risk_manager=risk_manager
)

# Single prediction
result = orchestrator.predict(features)
print(f"Signal: {result.signal}, Confidence: {result.confidence}")
print(f"Meta probability: {result.meta_probability}")
print(f"Model predictions: {result.model_predictions}")
print(f"Meta weights: {result.meta_weights}")

# Batch prediction
results_df = orchestrator.batch_predict(features, window_size=24)
```

### Benchmark on real OHLCV (LightGBM + XGBoost, no mocks)

End-to-end: parquet OHLCV → `FeatureEngine` → train tabular models → leakage-safe WFO backtest. Default demo file: `data/ohlcv/demo_BTC-USDT_1h.parquet`.

```bash
python -m orchestration report-real --max-rows 2200 --json-out docs/e2e_last_metrics.json
```

Integration tests: `pytest tests/test_orchestration_real_models.py -m integration`.

## Migration Guide

### From Old InferenceEngine

**Old:**
```python
inference_engine = InferenceEngine()
inference_engine.initialize(regime_detector, model_router, meta_filter, position_sizer)
result = inference_engine.predict(df)  # Only ONE model called
```

**New:**
```python
orchestrator = InferenceOrchestrator(config)
orchestrator.initialize(
    models={'lgb': lgb_model, 'gru': gru_model, 'xgb': xgb_model, 'cnn': cnn_model},
    regime_detector=regime_detector,
    meta_weighting=dynamic_meta_weighting,
    decision_engine=decision_engine,
    risk_manager=risk_manager
)
result = orchestrator.predict(df)  # ALL models called, meta-weighted
```

### Key Changes

1. **Model Registration**: Register ALL models with fixed keys instead of router by regime
2. **Meta Weighting**: Use DynamicMetaWeighting instead of simple router
3. **Prediction Collection**: All models contribute, not just one
4. **Config-Driven**: All parameters in config, validated at initialization

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OrchestratorConfig                        │
│  - Fixed model_keys: ['lgb', 'gru', 'xgb', 'cnn']           │
│  - Fixed regime_keys: ['trend', 'range', 'breakout']        │
│  - Thresholds & weights                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              TrainingOrchestrator / InferenceOrchestrator    │
│                                                              │
│  Pipeline:                                                  │
│  features → regime → {p_lgb, p_gru, p_xgb, p_cnn} → meta    │
│  → direction → decision → risk                              │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │  Models  │   │  Regime  │   │   Meta   │
        │          │   │ Detector │   │ Weighting│
        │ lgb, gru │   │          │   │          │
        │ xgb, cnn │   │          │   │          │
        └──────────┘   └──────────┘   └──────────┘
```

## Validation

The orchestrator validates at initialization:

1. **Model Keys**: Ensures registered models match config.model_keys
2. **Weight Keys**: Ensures weight dictionaries match model_keys for each regime
3. **Regime Keys**: Ensures regime operations use valid regime_keys

This prevents the key mismatch issues that existed between the old models/ and meta_learning/ branches.
