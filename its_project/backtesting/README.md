# Backtesting Layer

## Purpose
Simulate trading on historical data, evaluate strategy performance, check for overfitting, and validate strategies before live trading.

## Implemented components

| Component | File | Status | Notes |
|----------|------|--------|-------|
| Base interfaces | `base.py` | Done | Trade, BacktestResult, BaseBacktester |
| Simple Backtester | `simple.py` | Done | No look-ahead, chronological order, cost accounting |
| Performance Analyzer | `performance.py` | Done | Metrics, equity curve, drawdown, visualization |
| Walk Forward Validator | `walkforward.py` | Done | Rolling window validation, multi-asset support |
| Package init | `__init__.py` | Done | Exports all components |

## Data contracts

### Trade
- timestamp: int (ms)
- symbol: str
- side: str ('buy' or 'sell')
- price: float (execution price with slippage)
- size: float
- commission: float
- slippage: float
- pnl: Optional[float] (filled when position closed)

### BacktestResult
- trades: List[Trade]
- equity_curve: np.ndarray (portfolio value over time)
- returns: np.ndarray (period returns)
- metrics: Dict[str, float] (performance statistics)
- positions: pd.DataFrame (detailed position history)

## Critical Rules

### 1. NO LOOK-AHEAD BIAS
**Most important rule: NEVER use future data!**

```python
# WRONG - CATASTROPHIC
normalized = (data - data.mean()) / data.std()  # Uses future!

# CORRECT - Only past data
for i in range(window_size, len(data)):
    train_data = data[:i]
    mean, std = train_data.mean(), train_data.std()
    current_normalized = (data[i] - mean) / std
```

### 2. Strict Chronological Order
- Data MUST be sorted by timestamp
- Train/test split by time (no random splits)
- Features calculated only from past data

### 3. Real Trading Costs
- Commission: 0.1% (configurable)
- Slippage: 0.05% (configurable)
- Bid-ask spread
- Execution delays

## Usage example

```python
from its_project.backtesting import SimpleBacktester, PerformanceAnalyzer, WalkForwardValidator

# 1) Simple backtest
backtester = SimpleBacktester({
    'initial_capital': 10000,
    'commission_rate': 0.001,
    'slippage_rate': 0.0005,
    'min_window': 100,
})

result = backtester.run(historical_data, model, decision_maker)
print(result.summary())

# 2) Performance analysis
analyzer = PerformanceAnalyzer(result)
analyzer.plot_equity_curve()
analyzer.generate_report()

# 3) Walk-forward validation
validator = WalkForwardValidator({
    'train_size': 252,    # 1 year
    'test_size': 63,     # 3 months
    'step_size': 21,     # 1 month
})

results = validator.validate(data, model_factory, decision_maker_factory)
```

## Performance Metrics

### Portfolio Metrics
- Total Return
- Annual Return
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

### Risk Metrics
- Maximum Drawdown
- Average Drawdown Duration
- VaR (95%)
- CVaR (95%)

### Trading Metrics
- Win Rate
- Average Win/Loss
- Profit Factor
- Number of Trades
- Average Trade Duration

## Walk-Forward Validation

### Purpose
- Prevent overfitting
- Test strategy stability
- Validate across different market conditions

### Process
1. Train model on window [t, t+train_size]
2. Test on window [t+train_size, t+train_size+test_size]
3. Shift by step_size and repeat
4. Aggregate results across windows

### Multi-Asset Support
- Validate across multiple assets simultaneously
- Cross-asset performance comparison
- Asset-specific vs universal strategies

## Integration with Pipeline

Backtesting can be integrated with existing components:

```python
# Use existing models
from its_project.models import LSTMModel
from its_project.decision import SimpleDecisionMaker

# Create model factory
def model_factory():
    return LSTMModel({'hidden_size': 64, 'num_layers': 2})

# Create decision maker factory
def decision_maker_factory():
    return SimpleDecisionMaker({'confidence_threshold': 0.7})

# Run walk-forward validation
validator = WalkForwardValidator(config)
results = validator.validate(data, model_factory, decision_maker_factory)
```

## Best Practices

### Data Preparation
- Ensure data is sorted by timestamp
- Handle missing values appropriately
- Use sufficient history for feature calculation

### Model Training
- Train only on data within each window
- No leakage between windows
- Fresh model for each validation window

### Result Interpretation
- Look for consistency across windows
- High variance indicates instability
- Pay attention to drawdown patterns

## Validation Checklist

### Before Running Backtest
- [ ] Data sorted by timestamp
- [ ] No NaN values in critical periods
- [ ] Sufficient history for features
- [ ] Realistic cost assumptions

### During Backtest
- [ ] No future data usage
- [ ] Proper position sizing
- [ ] Commission/slippage applied
- [ ] All trades closed properly

### After Backtest
- [ ] Performance metrics calculated
- [ ] Drawdown periods identified
- [ ] Win rate analyzed
- [ ] Strategy stability assessed

## Next improvements (without breaking layer)
- Add more sophisticated slippage models
- Implement market impact modeling
- Add intraday backtesting capabilities
- Implement portfolio-level backtesting
- Add Monte Carlo simulation
- Implement regime detection integration
