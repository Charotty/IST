# Features Layer

## Purpose
Synchronize heterogeneous data streams, create sliding windows, and compute immutable features for ML.

## Implemented components

| Component | File | Status | Notes |
|----------|------|--------|-------|
| Base interface | `base.py` | Done | `BaseFeature` (pure, immutable) |
| Synchronizer | `synchronizer.py` | Done | `synchronize_marketdata`, `align_timestamps`, `extract_ohlcv_from_synced` |
| Window generator | `window.py` | Done | `WindowedFeatures`, sliding windows with stride |
| Feature pipeline | `pipeline.py` | Done | Immutable sequence of features |
| Preprocessing | `preprocessing.py` | Done | Missing handling, normalization, resampling |
| Technical indicators | `technical.py` | Done | RSI, MACD, Bollinger, ATR, Stochastic (pure) |
| Order book features | `orderbook.py` | Done | OFI, spread, imbalance, depth, VWAP, density |
| Microstructure | `microstructure.py` | Done | Roll, VPIN, realized volatility, Amihud, Kyle lambda |
| Feature scaler | `scaling.py` | Done | Pure zscore/minmax scaling wrapper |
| Package init | `__init__.py` | Done | Exports all components |

## Data contracts

### Synchronized DataFrame
- Uniform 1s index (`pd.DatetimeIndex`)
- Columns prefixed by source (`price_`, `lob_`, `onchain_`, `sentiment_`)
- Forward-filled or interpolated values (no NaN in output)

### OHLCV bars
- Required columns: `open`, `high`, `low`, `close`, `volume`
- 1-second frequency
- No NaN (filled/dropped)

### Feature output
- Fixed shape: `(n_samples, n_features)`
- Feature names list matches column order
- Pure functions (no side effects)

## Usage example

```python
from its_project.features import (
    synchronize_marketdata,
    extract_ohlcv_from_synced,
    WindowedFeatures,
    FeaturePipeline,
    TechnicalFeatures,
    OrderBookFeatures,
    MicrostructureFeatures,
    FeatureScaler,
)

# 1) Synchronize streams
synced = synchronize_marketdata(price_data, lob_data, onchain_data, sentiment_data)

# 2) Build OHLCV for feature calculation
ohlcv = extract_ohlcv_from_synced(synced)

# 3) Sliding windows
window_gen = WindowedFeatures(window_size=100, stride=10)
windows = list(window_gen.create_windows(ohlcv))

# 4) Feature pipeline (immutable, pure)
pipeline = FeaturePipeline([
    TechnicalFeatures({"indicators": ["rsi", "macd", "bbands", "atr", "stoch"]}),
    OrderBookFeatures({"depth_levels": 5, "imbalance_levels": 10}),
    MicrostructureFeatures({"window": 20}),
])
features = pipeline.fit_transform(ohlcv)  # shape (n_samples, n_features)

# 5) Scaling (pure)
scaler = FeatureScaler({"method": "zscore"})
features_scaled = scaler.fit_transform(features)

# 6) Inverse transform (if needed)
features_orig = scaler.inverse_transform(features_scaled)
```

## Next improvements (without breaking layer)
- Add caching for expensive feature calculations
- Add parallel window processing for large datasets
- Add feature selection utilities
- Add validation for time monotonicity
