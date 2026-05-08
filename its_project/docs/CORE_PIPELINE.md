# ITS Core Pipeline

This document defines the intended single-pipeline structure after removing UI, dashboard, Telegram, and generated experiment artifacts.

## Canonical Flow

```text
data_layer -> preprocessing -> storage -> features -> targets -> models -> metalearning -> decision -> backtesting/execution -> monitoring/mlops
```

The primary local entrypoint is:

```bash
python its_project/run.py check
python its_project/run.py demo
python its_project/run.py test --collect-only
```

## Layers

### `common`

Shared types, logging, rate limiting, error handling, and backend contracts that are still referenced by core modules.

Keep:
- `types.py`: `MarketData`, `MarketDataType`.
- `rate_limiter.py`: reusable data-source throttling.
- `logging.py`, `error_handler.py`, `config.py`.

Review later:
- `backend_contract.py`: keep only if non-UI services still need it.

### `data_layer`

Market data ingestion and validation.

Keep:
- `base.py`: source interface.
- `ccxt_source.py`: exchange abstraction via CCXT.
- `okx_source.py`: OKX-specific realtime/backfill source.
- `real_market_data.py`: historical OHLCV loader.
- `gap_detector.py`, `integrity.py`, `backfill_manager.py`, `data_pipeline.py`.

Review later:
- `reconnection_manager.py`: currently not imported at package import time because it contains a syntax issue; fix before making it part of the default path.
- `auto_downloader.py`, `enhanced_ccxt_source.py`: keep if they are merged into the canonical ingestion path.

### `preprocessing`

Small deterministic cleanup layer restored as part of the refactor.

Keep:
- `cleaner.py`: duplicate, missing value, outlier handling.
- `normalizer.py`: minmax/zscore/log transforms.
- `synchronizer.py`: DataFrame stream alignment.

### `storage`

Raw and aggregated data persistence.

Keep:
- `base.py`, `parquet.py`, `parquet_store.py`, `writer.py`.
- `timescale.py`, `timescale_client.py` if TimescaleDB remains a target storage backend.
- `storage_manager.py`, `raw_parquet_storage.py`, `aggregated_timescale_storage.py`.

Design note:
- Package imports are lazy so Parquet-only workflows do not require DB dependencies at import time.

### `features`

Feature engineering without look-ahead bias.

Keep:
- `base.py`, `pipeline.py`.
- `basic.py`: returns, log returns, lags, volatility.
- `technical.py`: classic indicators.
- `economic_features.py`: returns, volatility, momentum, risk, microstructure proxies.
- `microstructure_features.py`, `orderbook.py`, `synchronizer.py`.

Review later:
- Duplicate concepts between `microstructure.py` and `microstructure_features.py`.
- Ensure every feature explicitly documents whether it can be used online.

### `targets`

Economic labels and regression targets.

Keep:
- `economic_target.py`: future return target and class weights.
- `returns_target.py`: return-target helpers.

### `models`

Model interfaces and implementations.

Keep:
- `base.py`, `registry.py`.
- `boosting_model.py`, `economic_boosting_model.py`, `regression_model.py`.
- `ensemble.py`.
- Neural models (`gru_model.py`, `lstm.py`, `transformer.py`, `cnn_lob_model.py`) as optional model families.

Review later:
- Unify duplicate GRU/CNN files.
- Avoid importing optional torch models in paths that should stay lightweight.

### `metalearning`

Model selection, metrics, hyperparameter search, stacking, weighted ensembles.

Keep:
- `metrics.py`, `cv.py`, `selector.py`, `stacking.py`, `weighted_ensemble.py`.
- `hyperopt.py`, with `optuna` treated as optional until explicitly installed.

### `decision`

Convert predictions into executable intent.

Keep:
- `decision.py`: `Signal`, `Decision`, `Action`.
- `signal_generator.py`.
- `simple.py`, `enhanced_decision.py`, `economic_decision_maker.py`.
- `risk.py`, `risk_manager.py`, `sizer.py`, `sizing.py`, `portfolio.py`, `engine.py`.

Review later:
- Merge duplicated risk/sizing APIs into one public surface.

### `execution`

Paper/live order execution and execution-risk controls.

Keep:
- `base.py`, `paper.py`, `live.py`, `manager.py`, `task.py`.
- `pnl_tracker.py`, `performance_metrics.py`, `kill_switch.py`, `smart_routing.py`.
- `pipeline_latency.py`, `latency_measurement.py`.

Review later:
- Consolidate duplicate `manager.py` and `order_manager.py` responsibilities.

### `backtesting`

Historical simulation and validation.

Keep:
- `base.py`, `simple.py`, `economic_backtester.py`.
- `walkforward.py`, `walkforward_validator.py`.
- `baseline_strategies.py`, `realistic_costs.py`, `metrics.py`, `performance.py`.

### `system`, `mlops`, `monitoring`

Operational support around the core pipeline.

Keep:
- `system/config_system.py`, `system/structured_logging.py`.
- `mlops/drift_detection.py`, `experiment_tracking.py`, `versioning.py`.
- `monitoring/business_metrics.py`.

Review later:
- `monitoring/monitor.py` if it is dashboard-specific.

## Removed From Core Scope

Removed in this cleanup pass:
- root `ui/` React/Vite app.
- `its_project/ui`.
- `its_project/gui`.
- `its_project/web_interface`.
- `its_project/web_dashboard`.
- `its_project/telegram_bot`.
- `its_project/cli`.
- generated reports/caches: `benchmark_results`, `model_test_results`, `okx_reports`, `presentation_charts`, `htmlcov`, `.pytest_cache`, `.coverage`, `__pycache__`, `its_project.egg-info`.

Tests for removed interface/backend surfaces were also removed from the active test tree.

## Current Quality Gate

Use the project virtual environment:

```bash
.venv/Scripts/python.exe -m pytest its_project/tests --collect-only -q -p no:cacheprovider -p no:cov -o addopts=''
```

Current expected result after this pass:

```text
1139 tests collected
```

This confirms import and discovery health. Full test execution is the next quality gate.
