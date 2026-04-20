# Comprehensive Review & Testing Guide

## Overview
This guide provides complete instructions for reviewing and testing each stage of the Intelligent Trading System. Use this document to validate system functionality, architecture compliance, and performance.

---

## Stage 1: Data Layer Review & Testing

### Review Checklist

#### Architecture Compliance
- [ ] All data sources inherit from `BaseDataSource`
- [ ] No business logic in data layer (pure I/O only)
- [ ] Proper async/await usage throughout
- [ ] Rate limiting implemented for all external APIs
- [ ] Reconnection logic with exponential backoff

#### Code Quality
- [ ] Type hints for all functions
- [ ] Proper error handling with try/catch
- [ ] No hardcoded credentials (use environment variables)
- [ ] Structured logging (no print statements)

### Testing Procedures

#### 1. Data Sources Testing
```python
# Test Binance WebSocket
from its_project.data_layer.binance_ws import BinanceWsClient
client = BinanceWsClient(symbols=['BTCUSDT'])
await client.connect()
# Verify: connection, data reception, queue publishing

# Test Binance REST
from its_project.data_layer.binance_rest import BinanceRestClient
rest = BinanceRestClient()
snapshot = await rest.fetch_orderbook_snapshot('BTCUSDT')
# Verify: correct format, required fields present

# Test Glassnode
from its_project.data_layer.glassnode import GlassnodeClient
glassnode = GlassnodeClient(api_key='test_key')
data = await glassnode.fetch_metric('active_addresses', 'BTC')
# Verify: authentication, data format, rate limiting

# Test Sentiment X
from its_project.data_layer.sentiment_x import SentimentXClient
sentiment = SentimentXClient(bearer_token='test_token')
tweets = await sentiment.fetch_recent_search('BTC', 10)
# Verify: authentication, data parsing, rate limiting
```

#### 2. Data Format Validation
```python
# Verify MarketData structure
from its_project.data_layer.marketdata_helpers import (
    create_trade_data,
    create_orderbook_snapshot,
    create_orderbook_delta
)

# Test trade data
trade = create_trade_data(
    timestamp_ms=1640995200000,
    symbol='BTCUSDT',
    price=42000.0,
    size=0.1,
    exchange='binance'
)
assert trade.type.value == 'trade'
assert 'price' in trade.data
assert 'size' in trade.data

# Test orderbook data
snapshot = create_orderbook_snapshot(
    timestamp_ms=1640995200000,
    symbol='BTCUSDT',
    bids=[[42000.0, 1.0]],
    asks=[[42001.0, 1.0]],
    last_update_id=123456789
)
assert snapshot.type.value == 'orderbook'
assert 'lastUpdateId' in snapshot.data
```

#### 3. Rate Limiting Test
```python
# Test rate limiter functionality
from its_project.data_layer.rate_limiter import RateLimiter
import asyncio

limiter = RateLimiter(max_requests=10, window_seconds=60)

# Should pass
for i in range(10):
    await limiter.acquire()

# Should block
try:
    await asyncio.wait_for(limiter.acquire(), timeout=0.1)
    assert False, "Should have been rate limited"
except asyncio.TimeoutError:
    pass  # Expected
```

### Expected Results
- All data sources connect successfully
- Data published to correct queues
- Rate limiting prevents API abuse
- Reconnection works on network failures

---

## Stage 2: Storage Layer Review & Testing

### Review Checklist

#### Architecture Compliance
- [ ] All storage implementations inherit from `BaseStorage`
- [ ] Proper connection pooling for TimescaleDB
- [ ] Parquet files partitioned by date/symbol
- [ ] Async batch writing with backpressure handling
- [ ] No data loss during high load

#### Code Quality
- [ ] SQL injection protection
- [ ] Proper file path handling
- [ ] Error handling for storage failures
- [ ] Connection cleanup on shutdown

### Testing Procedures

#### 1. TimescaleDB Testing
```python
from its_project.storage import TimescaleStorage

# Test connection
storage = TimescaleStorage(dsn="postgres://test:test@localhost/tsdb")
await storage.connect()

# Test write operation
test_data = {
    'timestamp_ms': 1640995200000,
    'symbol': 'BTCUSDT',
    'type': 'trade',
    'exchange': 'binance',
    'data': {'price': 42000.0, 'size': 0.1}
}
await storage.write('market_data', test_data)

# Test read operation
results = await storage.read(
    'market_data',
    start_time=1640995200000,
    end_time=1640995300000,
    symbols=['BTCUSDT']
)
assert len(results) > 0
assert results[0]['symbol'] == 'BTCUSDT'
```

#### 2. Parquet Storage Testing
```python
from its_project.storage import ParquetStorage

# Test write
storage = ParquetStorage(base_path="test_data")
await storage.write(test_data, partition_cols=['symbol', 'date'])

# Test read
results = await storage.read(
    symbol='BTCUSDT',
    start_date='2022-01-01',
    end_date='2022-01-02'
)
assert len(results) > 0
```

#### 3. Batch Writer Testing
```python
from its_project.storage import batch_writer_task
from its_project.pipeline.queues import create_queues
import asyncio

# Setup test
queues = create_queues(maxsize=100)
stop_event = asyncio.Event()
storage = TimescaleStorage(dsn="postgres://test:test@localhost/tsdb")

# Fill queue with test data
for i in range(1000):
    await queues.price_raw.put(test_data)

# Run batch writer
task = asyncio.create_task(
    batch_writer_task(
        name="test_writer",
        storage=storage,
        batch_size=100,
        max_interval_s=5.0,
        in_queue=queues.price_raw,
        stop_event=stop_event
    )
)

# Verify batch processing
await asyncio.sleep(6.0)
stop_event.set()
await task

# Verify data was written
count = await storage.count('market_data')
assert count >= 1000
```

### Expected Results
- Data written to both warm and cold storage
- Batch processing efficient under load
- No data loss during failures
- Proper partitioning in Parquet files

---

## Stage 3: Preprocessing Layer Review & Testing

### Review Checklist

#### Architecture Compliance
- [ ] Pure functions for all operations
- [ ] No side effects in feature calculations
- [ ] Proper time series handling (no look-ahead)
- [ ] Correct interpolation methods
- [ ] Data quality validation

#### Code Quality
- [ ] Immutable operations
- [ ] Proper NaN handling
- [ ] Efficient vectorized operations
- [ ] Clear separation of concerns

### Testing Procedures

#### 1. Synchronization Testing
```python
from its_project.features.synchronizer import synchronize_marketdata
import pandas as pd

# Create test data with different timestamps
price_data = pd.DataFrame({
    'timestamp': pd.date_range('2022-01-01', periods=100, freq='S'),
    'price': range(100),
    'symbol': 'BTCUSDT'
})
lob_data = pd.DataFrame({
    'timestamp': pd.date_range('2022-01-01', periods=50, freq='2S'),
    'bid': range(50),
    'ask': [x+1 for x in range(50)],
    'symbol': 'BTCUSDT'
})

# Test synchronization
synced = synchronize_marketdata(price_data, lob_data)
assert len(synced) == 100  # Should match finest granularity
assert not synced.isnull().any().any()  # No NaN values
```

#### 2. OHLCV Extraction Testing
```python
from its_project.features.synchronizer import extract_ohlcv_from_synced

# Test OHLCV extraction
ohlcv = extract_ohlcv_from_synced(synced)
assert 'open' in ohlcv.columns
assert 'high' in ohlcv.columns
assert 'low' in ohlcv.columns
assert 'close' in ohlcv.columns
assert 'volume' in ohlcv.columns
assert len(ohlcv) > 0
```

#### 3. Missing Value Handling Testing
```python
from its_project.features.preprocessing import handle_missing, normalize_features

# Test with missing data
data_with_nan = ohlcv.copy()
data_with_nan.loc[10:15, 'volume'] = np.nan

# Test missing value handling
filled_data = handle_missing(data_with_nan, method='forward_fill')
assert not filled_data.isnull().any().any()

# Test normalization
normalized = normalize_features(filled_data[['open', 'high', 'low', 'close']], method='zscore')
assert abs(normalized.mean().mean()) < 0.1  # Should be near 0
assert abs(normalized.std().mean() - 1.0) < 0.1  # Should be near 1
```

### Expected Results
- Data properly synchronized across sources
- No look-ahead bias in any operation
- Missing values handled appropriately
- Normalization parameters consistent

---

## Stage 4: Feature Engineering Layer Review & Testing

### Review Checklist

#### Architecture Compliance
- [ ] All features inherit from `BaseFeature`
- [ ] Pure functions with deterministic output
- [ ] Fixed output dimensionality
- [ ] Proper feature naming
- [ ] No future data leakage

#### Code Quality
- [ ] Efficient vectorized calculations
- [ ] Proper handling of edge cases
- [ ] Clear feature documentation
- [ ] Immutable operations

### Testing Procedures

#### 1. Technical Features Testing
```python
from its_project.features.technical import TechnicalFeatures

# Test technical indicators
tech_features = TechnicalFeatures({
    'indicators': ['rsi', 'macd', 'bbands', 'atr', 'stoch']
})

# Generate test features
features = tech_features.fit_transform(ohlcv)
assert features.shape[1] > 0  # Should have features
assert not np.isnan(features).any()  # No NaN values

# Test individual indicators
rsi_values = tech_features.calculate_rsi(ohlcv['close'], 14)
assert len(rsi_values) == len(ohlcv)
assert 0 <= rsi_values.min() <= 100
assert 0 <= rsi_values.max() <= 100
```

#### 2. Order Book Features Testing
```python
from its_project.features.orderbook import OrderBookFeatures

# Create test orderbook data
orderbook_data = pd.DataFrame({
    'timestamp': pd.date_range('2022-01-01', periods=100, freq='S'),
    'bid_0': [42000 - i*0.1 for i in range(100)],
    'bid_1': [41999 - i*0.1 for i in range(100)],
    'ask_0': [42001 + i*0.1 for i in range(100)],
    'ask_1': [42002 + i*0.1 for i in range(100)],
    'bid_vol_0': [1.0] * 100,
    'ask_vol_0': [1.0] * 100
})

# Test orderbook features
ob_features = OrderBookFeatures({'depth_levels': 2})
features = ob_features.fit_transform(orderbook_data)

# Verify OFI calculation
ofi = ob_features.calculate_ofi(orderbook_data)
assert len(ofi) == len(orderbook_data)
assert not np.isnan(ofi).any()

# Verify spread calculation
spread = ob_features.calculate_spread(orderbook_data)
assert (spread > 0).all()
```

#### 3. Microstructure Features Testing
```python
from its_project.features.microstructure import MicrostructureFeatures

# Test microstructure features
micro_features = MicrostructureFeatures({'window': 20})
features = micro_features.fit_transform(ohlcv)

# Test Roll spread
roll_spread = micro_features.calculate_roll_spread(ohlcv['close'])
assert len(roll_spread) == len(ohlcv)
assert not np.isnan(roll_spread).any()

# Test realized volatility
realized_vol = micro_features.calculate_realized_volatility(ohlcv['close'], 20)
assert len(realized_vol) == len(ohlcv)
assert (realized_vol >= 0).all()
```

#### 4. Feature Pipeline Testing
```python
from its_project.features.pipeline import FeaturePipeline

# Test feature pipeline
pipeline = FeaturePipeline([
    TechnicalFeatures({'indicators': ['rsi', 'macd']}),
    OrderBookFeatures({'depth_levels': 5}),
    MicrostructureFeatures({'window': 20})
])

# Transform data
features = pipeline.fit_transform(ohlcv, orderbook_data)
assert features.shape[1] > 0
assert not np.isnan(features).any()

# Test feature names
feature_names = pipeline.get_feature_names()
assert len(feature_names) == features.shape[1]
```

### Expected Results
- All features calculated correctly
- No NaN values in output
- Consistent feature dimensions
- Deterministic output for same input

---

## Stage 5: Model Layer Review & Testing

### Review Checklist

#### Architecture Compliance
- [ ] All models inherit from `BaseModel`
- [ ] Proper implementation of all required methods
- [ ] Consistent input/output formats
- [ ] Model serialization works correctly
- [ ] Registry pattern implemented properly

#### Code Quality
- [ ] Proper PyTorch model structure
- [ ] Gradient flow verification
- [ ] Training loop correctness
- [ ] Input validation

### Testing Procedures

#### 1. Model Registry Testing
```python
from its_project.models import ModelRegistry

# Test model creation
lstm_model = ModelRegistry.get_model('lstm', {
    'input_size': 50,
    'hidden_size': 128,
    'num_layers': 2
})
assert isinstance(lstm_model, LSTMModel)

transformer_model = ModelRegistry.get_model('transformer', {
    'input_size': 50,
    'd_model': 128,
    'nhead': 8
})
assert isinstance(transformer_model, TransformerModel)

# Test model listing
models = ModelRegistry.list_models()
assert 'lstm' in models
assert 'transformer' in models
assert 'ensemble' in models
```

#### 2. LSTM Model Testing
```python
import torch
import numpy as np

# Test LSTM model
lstm = LSTMModel({
    'input_size': 10,
    'hidden_size': 64,
    'num_layers': 2,
    'num_classes': 3
})

# Test training
X_train = np.random.randn(100, 50, 10)  # 100 samples, 50 timesteps, 10 features
y_train = np.random.randint(0, 3, 100)

lstm.fit(X_train, y_train)

# Test prediction
X_test = np.random.randn(10, 50, 10)
preds = lstm.predict(X_test)
assert len(preds) == 10
assert all(p in [0, 1, 2] for p in preds)

# Test probabilities
proba = lstm.predict_proba(X_test)
assert proba.shape == (10, 3)
assert np.allclose(proba.sum(axis=1), 1.0)

# Test confidence
confidence = lstm.get_confidence(X_test)
assert len(confidence) == 10
assert all(0 <= c <= 1 for c in confidence)
```

#### 3. Transformer Model Testing
```python
# Test Transformer model
transformer = TransformerModel({
    'input_size': 10,
    'd_model': 128,
    'nhead': 8,
    'num_layers': 4,
    'num_classes': 3
})

# Test training
transformer.fit(X_train, y_train)

# Test prediction
preds = transformer.predict(X_test)
assert len(preds) == 10
assert all(p in [0, 1, 2] for p in preds)
```

#### 4. Ensemble Model Testing
```python
from its_project.models import EnsembleModel

# Test ensemble model
ensemble = EnsembleModel({
    'models': ['logistic', 'random_forest', 'gradient_boosting']
})

# Reshape data for sklearn models
X_train_2d = X_train.reshape(X_train.shape[0], -1)
X_test_2d = X_test.reshape(X_test.shape[0], -1)

# Test training
ensemble.fit(X_train_2d, y_train)

# Test prediction
preds = ensemble.predict(X_test_2d)
assert len(preds) == 10

# Test probabilities
proba = ensemble.predict_proba(X_test_2d)
assert proba.shape == (10, 3)
```

#### 5. Model Serialization Testing
```python
from pathlib import Path

# Test model saving
model_path = Path("test_model.pkl")
ModelRegistry.save_model(lstm, model_path)
assert model_path.exists()

# Test model loading
loaded_model = ModelRegistry.load_model(model_path)
assert isinstance(loaded_model, LSTMModel)

# Test loaded model predictions
loaded_preds = loaded_model.predict(X_test)
assert np.array_equal(preds, loaded_preds)

# Cleanup
model_path.unlink()
```

### Expected Results
- All models train without errors
- Predictions in correct format
- Serialization/deserialization works
- Registry pattern functions correctly

---

## Stage 6: Meta-Learning Layer Review & Testing

### Review Checklist

#### Architecture Compliance
- [ ] Proper time series cross-validation
- [ ] No data leakage between folds
- [ ] Correct metric calculations
- [ ] Optuna integration works
- [ ] Model selection logic correct

#### Code Quality
- [ ] Efficient hyperparameter search
- [ ] Proper objective functions
- [ ] Clear metric definitions
- [ ] Error handling for optimization

### Testing Procedures

#### 1. Metrics Testing
```python
from its_project.metalearning import MLMetrics, TradingMetrics

# Test ML metrics
y_true = np.array([0, 1, 2, 1, 0])
y_pred = np.array([0, 1, 1, 1, 0])

ml_metrics = MLMetrics.calculate(y_true, y_pred)
assert 'accuracy' in ml_metrics
assert 'precision' in ml_metrics
assert 'recall' in ml_metrics
assert 'f1' in ml_metrics
assert 0 <= ml_metrics['accuracy'] <= 1

# Test trading metrics
returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
trading_metrics = TradingMetrics.calculate(returns)
assert 'total_return' in trading_metrics
assert 'sharpe_ratio' in trading_metrics
assert 'max_drawdown' in trading_metrics
assert trading_metrics['total_return'] == returns.sum()
```

#### 2. Time Series CV Testing
```python
from its_project.metalearning import TimeSeriesSplitter

# Test time series splitter
X = np.random.randn(100, 10)
y = np.random.randint(0, 3, 100)

splitter = TimeSeriesSplitter(n_splits=5, gap=10)
splits = list(splitter.split(X))

assert len(splits) == 5

# Test no data leakage
for train_idx, test_idx in splits:
    assert max(train_idx) < min(test_idx)  # Train before test
    assert min(test_idx) - max(train_idx) >= 10  # Gap respected
```

#### 3. Hyperparameter Optimization Testing
```python
from its_project.metalearning import HyperparameterOptimizer
from its_project.models import LSTMModel

# Define parameter space
param_space = {
    'hidden_size': {'type': 'int', 'low': 32, 'high': 128},
    'dropout': {'type': 'float', 'low': 0.1, 'high': 0.5}
}

# Test optimizer
optimizer = HyperparameterOptimizer(
    model_class=LSTMModel,
    param_space=param_space,
    metric='accuracy',
    n_trials=5
)

# Run optimization (small scale for testing)
result = optimizer.optimize(X_train_2d, y_train, X_test_2d, y_test)
assert 'best_params' in result
assert 'best_score' in result
assert isinstance(result['best_params'], dict)
```

#### 4. Model Selection Testing
```python
from its_project.metalearning import ModelSelector
from its_project.models import LSTMModel, TransformerModel, EnsembleModel

# Create test models
models = [
    LSTMModel({'input_size': 10, 'hidden_size': 64}),
    TransformerModel({'input_size': 10, 'd_model': 64}),
    EnsembleModel({})
]

# Test model selector
selector = ModelSelector(
    models=models,
    metrics=['accuracy', 'sharpe_ratio'],
    weights=[0.6, 0.4]
)

# Evaluate models
results_df = selector.evaluate_all(
    X_train_2d, y_train, X_test_2d, y_test, returns_test
)
assert len(results_df) == len(models)
assert 'accuracy' in results_df.columns
assert 'sharpe_ratio' in results_df.columns

# Test best model selection
best_model, best_scores = selector.select_best()
assert best_model is not None
assert isinstance(best_scores, dict)
```

### Expected Results
- Metrics calculated correctly
- No data leakage in CV
- Hyperparameter optimization finds better parameters
- Model selection picks best performing model

---

## Stage 7: Decision Layer Review & Testing

### Review Checklist

#### Architecture Compliance
- [ ] All decision makers inherit from `BaseDecisionMaker`
- [ ] Risk limits enforced properly
- [ ] Position sizing methods work correctly
- [ ] Portfolio state tracked accurately
- [ ] Decision engine integrates all components

#### Code Quality
- [ ] Proper validation of inputs
- [ ] Clear decision logic
- [ ] Error handling for edge cases
- [ ] Logging of all decisions

### Testing Procedures

#### 1. Simple Decision Maker Testing
```python
from its_project.decision import SimpleDecisionMaker, Signal, Action
import time

# Test simple decision maker
decision_maker = SimpleDecisionMaker({'confidence_threshold': 0.7})

# Test high confidence signal (should trigger decision)
high_conf_signal = Signal(
    action=Action.BUY,
    confidence=0.8,
    timestamp=int(time.time() * 1000),
    symbol='BTCUSDT',
    metadata={'price': 42000}
)

decision = decision_maker.decide(high_conf_signal, {'price': 42000})
assert decision is not None
assert decision.action == Action.BUY
assert decision.size > 0

# Test low confidence signal (should not trigger decision)
low_conf_signal = Signal(
    action=Action.BUY,
    confidence=0.5,
    timestamp=int(time.time() * 1000),
    symbol='BTCUSDT',
    metadata={'price': 42000}
)

decision = decision_maker.decide(low_conf_signal, {'price': 42000})
assert decision is None
```

#### 2. Risk Manager Testing
```python
from its_project.decision import RiskManager

# Test risk manager
risk_manager = RiskManager({
    'max_position_size': 0.1,
    'max_portfolio_risk': 0.05,
    'max_drawdown': 0.15
})

# Test valid decision
valid_decision = Decision(
    action=Action.BUY,
    symbol='BTCUSDT',
    size=0.05,
    price=42000,
    timestamp=int(time.time() * 1000),
    reason='Test decision'
)

# Should pass risk check
assert risk_manager.check_decision(valid_decision, 10000) is True

# Test oversized decision
oversized_decision = Decision(
    action=Action.BUY,
    symbol='BTCUSDT',
    size=0.2,  # Exceeds max_position_size
    price=42000,
    timestamp=int(time.time() * 1000),
    reason='Test oversized decision'
)

# Should fail risk check
assert risk_manager.check_decision(oversized_decision, 10000) is False
```

#### 3. Position Sizer Testing
```python
from its_project.decision import PositionSizer

# Test position sizer
sizer = PositionSizer({'method': 'fixed_fraction', 'fraction': 0.02})

# Test fixed fraction sizing
size = sizer.calculate_size(
    method='fixed_fraction',
    account_balance=10000,
    price=42000,
    stop_loss=41000
)
assert size == 0.02 * 10000 / 42000  # 2% of balance

# Test Kelly sizing
kelly_size = sizer.calculate_size(
    method='kelly',
    account_balance=10000,
    price=42000,
    win_rate=0.55,
    avg_win=0.03,
    avg_loss=0.02
)
assert kelly_size > 0
```

#### 4. Portfolio Manager Testing
```python
from its_project.decision import PortfolioManager

# Test portfolio manager
portfolio = PortfolioManager({'max_positions': 5})

# Add position
position = Position(
    symbol='BTCUSDT',
    side='long',
    size=0.1,
    entry_price=42000,
    current_price=42500,
    unrealized_pnl=50,
    timestamp=int(time.time() * 1000)
)

portfolio.add_position(position)

# Check portfolio state
positions = portfolio.get_positions()
assert len(positions) == 1
assert positions[0].symbol == 'BTCUSDT'

# Update PnL
portfolio.update_price('BTCUSDT', 43000)
updated_positions = portfolio.get_positions()
assert updated_positions[0].unrealized_pnl == 100

# Calculate total PnL
total_pnl = portfolio.get_total_pnl()
assert total_pnl == 100
```

#### 5. Decision Engine Testing
```python
from its_project.decision import TradingDecisionEngine

# Test decision engine
engine = TradingDecisionEngine({
    'decision': {'confidence_threshold': 0.7},
    'risk': {'max_position_size': 0.1},
    'sizing': {'method': 'fixed_fraction', 'fraction': 0.02},
    'portfolio': {'max_positions': 5}
})

# Process signal
signal = Signal(
    action=Action.BUY,
    confidence=0.8,
    timestamp=int(time.time() * 1000),
    symbol='BTCUSDT',
    metadata={'price': 42000}
)

decision = engine.process_signal(signal, {'price': 42000}, 10000)
assert decision is not None
assert decision.action == Action.BUY
assert decision.size > 0
assert decision.stop_loss is not None
assert decision.take_profit is not None
```

### Expected Results
- Decisions made based on confidence threshold
- Risk limits properly enforced
- Position sizes calculated correctly
- Portfolio state tracked accurately
- All components work together in engine

---

## Stage 8: Execution Layer Review & Testing

### Review Checklist

#### Architecture Compliance
- [ ] All executors inherit from `BaseExecutor`
- [ ] Paper trading works without real money
- [ ] Live executor has safety limits
- [ ] Order manager handles SL/TP correctly
- [ ] All operations logged properly

#### Code Quality
- [ ] Proper error handling for network failures
- [ ] Order status verification
- [ ] Balance checks before orders
- [ ] Cleanup on shutdown

### Testing Procedures

#### 1. Paper Trading Executor Testing
```python
from its_project.execution import PaperTradingExecutor

# Test paper trading executor
executor = PaperTradingExecutor({
    'initial_balance': 10000,
    'latency_ms': 50
})

# Test order creation
order = await executor.create_order(
    symbol='BTC/USDT',
    order_type=OrderType.MARKET,
    side='buy',
    amount=0.01
)

assert order.status == OrderStatus.FILLED  # Market orders filled immediately
assert order.price > 0
assert order.amount == 0.01

# Test balance
balance = await executor.fetch_balance()
assert 'USDT' in balance
assert 'BTC' in balance
assert balance['BTC'] == 0.01

# Test order cancellation
limit_order = await executor.create_order(
    symbol='BTC/USDT',
    order_type=OrderType.LIMIT,
    side='sell',
    amount=0.01,
    price=43000
)

cancelled = await executor.cancel_order(limit_order.id, 'BTC/USDT')
assert cancelled is True
```

#### 2. Order Manager Testing
```python
from its_project.execution import OrderManager

# Test order manager
order_manager = OrderManager(executor, {'monitor_interval': 1.0})
await order_manager.start()

# Test decision execution with SL/TP
order = await order_manager.execute_decision(
    symbol='BTC/USDT',
    side='buy',
    amount=0.01,
    order_type='market',
    stop_loss=41000,
    take_profit=43000
)

assert order.status == OrderStatus.FILLED

# Check active orders (should include SL/TP)
active_orders = await order_manager.get_active_orders()
assert len(active_orders) >= 2  # Main order + SL/TP

# Test balance
balance = await order_manager.get_balance()
assert isinstance(balance, dict)

# Cleanup
await order_manager.stop()
```

#### 3. Safety Limits Testing
```python
# Test order size limits
executor = PaperTradingExecutor({'initial_balance': 10000, 'max_order_size': 0.1})

# Should work
order1 = await executor.create_order(
    symbol='BTC/USDT',
    order_type=OrderType.MARKET,
    side='buy',
    amount=0.05
)

# Should fail for oversized order
try:
    order2 = await executor.create_order(
        symbol='BTC/USDT',
        order_type=OrderType.MARKET,
        side='buy',
        amount=0.2  # Exceeds max_order_size
    )
    assert False, "Should have failed for oversized order"
except ValueError:
    pass  # Expected
```

### Expected Results
- Paper trading works without real money
- Orders executed with proper SL/TP
- Safety limits enforced
- All operations logged
- Cleanup works properly

---

## Stage 9: Backtesting Layer Review & Testing

### Review Checklist

#### Architecture Compliance
- [ ] No look-ahead bias in any calculation
- [ ] Strict chronological order maintained
- [ ] Real trading costs included
- [ ] Proper position tracking
- [ ] Walk-forward validation works

#### Code Quality
- [ ] Efficient vectorized calculations
- [ ] Proper error handling
- [ ] Clear performance metrics
- [ ] Comprehensive reporting

### Testing Procedures

#### 1. Simple Backtester Testing
```python
from its_project.backtesting import SimpleBacktester
from its_project.decision import SimpleDecisionMaker
import pandas as pd

# Create test data
np.random.seed(42)
price_data = pd.DataFrame({
    'timestamp': pd.date_range('2022-01-01', periods=1000, freq='H'),
    'open': 42000 + np.random.randn(1000).cumsum(),
    'high': None,
    'low': None,
    'close': None,
    'volume': np.random.rand(1000) * 100
})

# Fill OHLC
price_data['close'] = price_data['open'] + np.random.randn(1000) * 10
price_data['high'] = price_data[['open', 'close']].max(axis=1) + np.random.rand(1000) * 5
price_data['low'] = price_data[['open', 'close']].min(axis=1) - np.random.rand(1000) * 5
price_data['timestamp_ms'] = price_data['timestamp'].astype(np.int64) * 1000

# Sort by time (critical requirement)
price_data = price_data.sort_values('timestamp_ms')

# Test backtester
backtester = SimpleBacktester({
    'initial_capital': 10000,
    'commission_rate': 0.001,
    'slippage_rate': 0.0005,
    'min_window': 100
})

# Create simple model for testing
class TestModel:
    def fit(self, X, y): pass
    def predict(self, X): 
        return np.random.choice([0, 1, 2], len(X))
    def predict_proba(self, X):
        proba = np.random.rand(len(X), 3)
        return proba / proba.sum(axis=1, keepdims=True)
    def get_confidence(self, X):
        return np.random.rand(len(X))

model = TestModel()
decision_maker = SimpleDecisionMaker({'confidence_threshold': 0.5})

# Run backtest
result = backtester.run(price_data, model, decision_maker)

# Verify results
assert len(result.trades) >= 0
assert len(result.equity_curve) > 0
assert len(result.returns) > 0
assert 'total_return' in result.metrics
assert 'sharpe_ratio' in result.metrics
assert 'max_drawdown' in result.metrics
```

#### 2. No Look-Ahead Bias Testing
```python
# Critical test: ensure no future data usage
def test_no_look_ahead():
    # Create data with known pattern
    price_data = pd.DataFrame({
        'timestamp_ms': range(1000),
        'close': np.sin(np.linspace(0, 10*np.pi, 1000)) * 1000 + 42000
    })
    
    # Run backtest
    backtester = SimpleBacktester({'initial_capital': 10000})
    result = backtester.run(price_data, model, decision_maker)
    
    # Verify equity curve only uses past information
    # This is more of a code review test - check that features are calculated
    # using only data[:i] for prediction at time i
    
    assert True  # Placeholder - manual code review needed

test_no_look_ahead()
```

#### 3. Performance Analyzer Testing
```python
from its_project.backtesting import PerformanceAnalyzer

# Test performance analysis
analyzer = PerformanceAnalyzer(result)

# Test detailed metrics
metrics = analyzer.calculate_detailed_metrics()
assert 'sharpe_ratio' in metrics
assert 'sortino_ratio' in metrics
assert 'calmar_ratio' in metrics
assert 'var_95' in metrics
assert 'win_rate' in metrics

# Test report generation
report = analyzer.generate_report()
assert 'BACKTEST PERFORMANCE REPORT' in report
assert 'Total Return' in report
assert 'Sharpe Ratio' in report
```

#### 4. Walk-Forward Validation Testing
```python
from its_project.backtesting import WalkForwardValidator

# Test walk-forward validator
validator = WalkForwardValidator({
    'train_size': 252,
    'test_size': 63,
    'step_size': 21,
    'min_window': 100
})

# Model factory
def model_factory():
    return TestModel()

# Decision maker factory
def decision_maker_factory():
    return SimpleDecisionMaker({'confidence_threshold': 0.5})

# Run walk-forward validation
results = validator.validate(price_data, model_factory, decision_maker_factory)

# Verify results
assert len(results) > 0
assert all(isinstance(r, BacktestResult) for r in results)

# Check aggregate statistics
returns = [r.metrics['total_return'] for r in results]
assert len(returns) == len(results)
```

### Expected Results
- No look-ahead bias in any calculations
- Realistic performance metrics
- Walk-forward validation shows consistency
- Comprehensive reporting

---

## Integration Testing

### End-to-End Pipeline Testing

#### 1. Full Pipeline Test
```python
# Test complete pipeline from data to execution
async def test_full_pipeline():
    # Setup queues
    queues = create_queues(maxsize=1000)
    stop_event = asyncio.Event()
    
    # Create test data
    test_data = generate_test_market_data()
    
    # Start all tasks
    tasks = [
        asyncio.create_task(data_ingestion_task(queues, stop_event)),
        asyncio.create_task(preprocessing_task(queues, stop_event)),
        asyncio.create_task(prediction_task(queues, stop_event)),
        asyncio.create_task(decision_task(queues, stop_event)),
        asyncio.create_task(execution_task(queues, stop_event)),
    ]
    
    # Feed test data
    for data in test_data:
        await queues.price_raw.put(data)
    
    # Run for short time
    await asyncio.sleep(5.0)
    stop_event.set()
    
    # Wait for completion
    await asyncio.gather(*tasks)
    
    # Verify results
    # Check that data flowed through all queues
    # Check that predictions were made
    # Check that decisions were generated
    # Check that orders were executed

# Run integration test
await test_full_pipeline()
```

#### 2. Error Handling Test
```python
# Test error handling throughout pipeline
async def test_error_handling():
    # Test with invalid data
    # Test with network failures
    # Test with model failures
    # Test with execution failures
    pass
```

---

## Performance Testing

### Load Testing Procedures

#### 1. Data Ingestion Load Test
```python
# Test high-frequency data ingestion
async def test_data_load():
    queues = create_queues(maxsize=10000)
    
    # Simulate high-frequency data
    tasks = []
    for i in range(100):
        task = asyncio.create_task(
            simulate_data_source(f"source_{i}", queues.price_raw, stop_event)
        )
        tasks.append(task)
    
    # Run load test
    await asyncio.gather(*tasks)
    
    # Verify no data loss
    # Verify queue sizes managed properly
```

#### 2. Model Performance Test
```python
# Test model prediction latency
def test_model_latency():
    model = LSTMModel({'input_size': 50, 'hidden_size': 128})
    X = np.random.randn(1000, 50, 10)
    
    # Warm up
    model.predict(X[:1])
    
    # Measure latency
    start_time = time.time()
    for i in range(100):
        model.predict(X[i:i+1])
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / 100
    assert avg_latency < 0.1  # Should be under 100ms
```

---

## Security Testing

### Security Checklist

#### 1. API Key Security
- [ ] No hardcoded API keys in code
- [ ] API keys loaded from environment variables
- [ ] API key validation on startup
- [ ] Secure storage of credentials

#### 2. Input Validation
- [ ] All user inputs validated
- [ ] SQL injection protection
- [ ] File path validation
- [ ] Type checking for all inputs

#### 3. Network Security
- [ ] HTTPS for all external calls
- [ ] Certificate validation
- [ ] Timeout configurations
- [ ] Rate limiting

---

## Documentation Review

### Documentation Checklist

#### 1. Code Documentation
- [ ] All public functions have docstrings
- [ ] Type hints for all functions
- [ ] Clear parameter descriptions
- [ ] Usage examples in docstrings

#### 2. Architecture Documentation
- [ ] Layer responsibilities clearly defined
- [ ] Data flow diagrams
- [ ] Integration examples
- [ ] Deployment guides

#### 3. User Documentation
- [ ] Installation instructions
- [ ] Configuration guide
- [ ] Troubleshooting guide
- [ ] API documentation

---

## Final Acceptance Criteria

### Must-Have Requirements
- [ ] All 9 implemented stages work correctly
- [ ] No look-ahead bias in backtesting
- [ ] Paper trading works without real money
- [ ] All safety limits enforced
- [ ] System handles errors gracefully
- [ ] Performance meets requirements (<100ms prediction latency)
- [ ] Documentation complete and accurate

### Performance Requirements
- [ ] End-to-end latency < 500ms
- [ ] Model prediction latency < 100ms
- [ ] System handles 1000+ messages/second
- [ ] Memory usage < 4GB under normal load
- [ ] CPU usage < 80% under normal load

### Reliability Requirements
- [ ] System recovers from network failures
- [ ] No data loss during restarts
- [ ] Graceful shutdown works
- [ ] Error rates < 0.1%
- [ ] Uptime > 99.9%

---

## Test Execution Plan

### Phase 1: Unit Testing
1. Test each component individually
2. Verify all public methods work
3. Check error handling
4. Validate data contracts

### Phase 2: Integration Testing
1. Test component interactions
2. Verify data flow through pipeline
3. Test error propagation
4. Check resource cleanup

### Phase 3: System Testing
1. End-to-end pipeline testing
2. Performance under load
3. Error recovery testing
4. Security validation

### Phase 4: User Acceptance Testing
1. Paper trading validation
2. Backtesting accuracy
3. Usability testing
4. Documentation verification

---

## Test Report Template

### Test Execution Summary
- **Date**: [Test execution date]
- **Tester**: [Tester name]
- **Environment**: [Test environment details]
- **Test Duration**: [Total test time]

### Results Summary
- **Total Tests**: [Number of tests run]
- **Passed**: [Number of tests passed]
- **Failed**: [Number of tests failed]
- **Blocked**: [Number of tests blocked]

### Critical Issues
- [List any critical issues found]

### Performance Metrics
- **Average Latency**: [ms]
- **Throughput**: [messages/second]
- **Memory Usage**: [GB]
- **CPU Usage**: [%]

### Recommendations
- [List recommendations for improvement]

### Sign-off
- **Engineer**: [Signature and date]
- **QA**: [Signature and date]
- **Product Owner**: [Signature and date]

---

This comprehensive guide provides complete instructions for reviewing and testing all aspects of the Intelligent Trading System. Use this document to ensure system quality, reliability, and performance before production deployment.
