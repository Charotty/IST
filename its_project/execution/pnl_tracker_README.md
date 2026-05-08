# PnL Tracking System

Comprehensive Profit and Loss tracking system for the ITS trading platform, supporting both per-trade and cumulative PnL calculation with advanced performance analytics.

## Features

### Core PnL Tracking (`pnl_tracker.py`)
- **Real-time PnL calculation** for individual trades and positions
- **Position management** with average price calculation
- **Trade history tracking** with detailed metadata
- **Strategy-specific PnL** segregation
- **Database persistence** with SQLite backend
- **Performance metrics** calculation (win rate, profit factor, etc.)
- **Data export** to pandas DataFrames

### Cumulative PnL Tracking (`cumulative_pnl.py`)
- **Equity curve tracking** with time series analysis
- **Drawdown analysis** (max drawdown, duration)
- **Risk-adjusted metrics** (Sharpe, Sortino, Calmar ratios)
- **Advanced performance metrics** (VaR, CVaR, Ulcer Index)
- **Visualization capabilities** with matplotlib/seaborn
- **Performance reporting** with JSON export

## Quick Start

### Basic PnL Tracking

```python
from its_project.execution import PnLTracker, create_pnl_tracker

# Create PnL tracker
pnl_tracker = create_pnl_tracker("data/my_pnl.db")

# Add trades
pnl_tracker.add_trade(
    trade_id="trade_001",
    symbol="BTC/USDT",
    side="buy",
    quantity=1.0,
    price=50000.0,
    commission=10.0,
    strategy_id="momentum_strategy"
)

# Close position
pnl_tracker.add_trade(
    trade_id="trade_002",
    symbol="BTC/USDT",
    side="sell",
    quantity=1.0,
    price=51000.0,
    commission=10.0,
    strategy_id="momentum_strategy"
)

# Get results
print(f"Realized PnL: ${pnl_tracker.get_cumulative_pnl():.2f}")
print(f"Win Rate: {pnl_tracker.get_performance_metrics()['win_rate']:.1f}%")
```

### Cumulative PnL with Performance Analysis

```python
from its_project.execution import PnLTracker, CumulativePnLTracker

# Setup trackers
pnl_tracker = PnLTracker()
cum_tracker = CumulativePnLTracker(pnl_tracker)

# Add trades and update
pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0)
pnl_tracker.add_trade("sell_001", "BTC/USDT", "sell", 1.0, 51000.0)
snapshot = cum_tracker.update()

# Get performance metrics
metrics = cum_tracker._calculate_performance_metrics()
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown:.1%}")

# Generate plots
cum_tracker.plot_equity_curve("equity_curve.png")
cum_tracker.plot_performance_summary("performance.png")
```

## Core Classes

### PnLTracker

Main class for tracking individual trades and positions.

#### Key Methods:
- `add_trade()` - Add new trade with full metadata
- `update_market_price()` - Update current market prices
- `get_cumulative_pnl()` - Get total PnL across all positions
- `get_performance_metrics()` - Calculate comprehensive metrics
- `get_pnl_dataframe()` - Export PnL records to DataFrame
- `get_positions_dataframe()` - Export current positions to DataFrame

### CumulativePnLTracker

Advanced cumulative PnL tracking with performance analysis.

#### Key Methods:
- `update()` - Update cumulative tracking and return snapshot
- `get_equity_curve_dataframe()` - Export equity curve data
- `plot_equity_curve()` - Generate equity curve visualization
- `plot_performance_summary()` - Generate comprehensive performance charts
- `export_performance_report()` - Export detailed performance report

## Data Structures

### TradeRecord
```python
@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    fees: Dict[str, float] = field(default_factory=dict)
    strategy_id: Optional[str] = None
    order_id: Optional[str] = None
```

### PositionSnapshot
```python
@dataclass
class PositionSnapshot:
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    last_price: Optional[float] = None
    last_update: Optional[datetime] = None
    trades_count: int = 0
```

### PerformanceMetrics
```python
@dataclass
class PerformanceMetrics:
    # Return metrics
    total_return: float
    annualized_return: float
    daily_return_mean: float
    daily_return_std: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    var_95: float
    cvar_95: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Trade metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
```

## Performance Metrics

### Basic Metrics
- **Total Return**: Overall profit/loss
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Total profits / Total losses
- **Average Win/Loss**: Mean profit/loss per trade

### Risk Metrics
- **Max Drawdown**: Largest peak-to-trough decline
- **Volatility**: Standard deviation of returns
- **VaR 95%**: Value at Risk at 95% confidence
- **CVaR 95%**: Conditional Value at Risk

### Risk-Adjusted Metrics
- **Sharpe Ratio**: Risk-adjusted return vs. risk-free rate
- **Sortino Ratio**: Return vs. downside deviation
- **Calmar Ratio**: Annualized return / max drawdown
- **Information Ratio**: Excess return vs. tracking error

## Database Schema

The system uses SQLite with the following tables:

### trades
Stores individual trade records with full metadata.

### pnl_records
Stores completed trade PnL calculations.

### positions
Stores current position snapshots.

### daily_pnl
Stores daily PnL aggregation.

### strategy_pnl
Stores strategy-specific performance.

## Integration Examples

### Integration with Order Manager

```python
from its_project.execution import OrderManager, PnLTracker

class IntegratedTradingSystem:
    def __init__(self):
        self.order_manager = OrderManager()
        self.pnl_tracker = PnLTracker()
        
    def on_order_filled(self, order):
        # Convert order to trade record
        self.pnl_tracker.add_trade(
            trade_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_quantity,
            price=order.average_price,
            commission=order.commission,
            order_id=order.id
        )
    
    def on_market_data(self, symbol, price):
        # Update unrealized PnL
        self.pnl_tracker.update_market_price(symbol, price)
```

### Integration with Strategy System

```python
from its_project.execution import PnLTracker, CumulativePnLTracker

class StrategyManager:
    def __init__(self):
        self.pnl_tracker = PnLTracker()
        self.cum_tracker = CumulativePnLTracker(self.pnl_tracker)
        self.strategies = {}
    
    def execute_trade(self, strategy_id, symbol, side, quantity, price):
        # Execute trade
        trade_id = f"{strategy_id}_{datetime.now().timestamp()}"
        
        self.pnl_tracker.add_trade(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            strategy_id=strategy_id
        )
        
        # Update strategy performance
        self.cum_tracker.update()
        strategy_pnl = self.pnl_tracker.get_strategy_pnl(strategy_id)
        
        return trade_id
    
    def get_strategy_performance(self, strategy_id):
        return {
            'pnl': self.pnl_tracker.get_strategy_pnl(strategy_id),
            'trades': len([r for r in self.pnl_tracker.pnl_records 
                          if r.strategy_id == strategy_id]),
            'win_rate': self._calculate_strategy_win_rate(strategy_id)
        }
```

## Visualization Examples

### Equity Curve with Drawdowns
```python
# Generate equity curve plot
cum_tracker.plot_equity_curve("my_equity_curve.png")
```

### Performance Summary Dashboard
```python
# Generate comprehensive performance charts
cum_tracker.plot_performance_summary("performance_dashboard.png")
```

### Custom Analysis
```python
# Export data for custom analysis
equity_df = cum_tracker.get_equity_curve_dataframe()
pnl_df = pnl_tracker.get_pnl_dataframe()

# Custom plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(equity_df.index, equity_df['pnl'])
plt.title('Equity Curve')

plt.subplot(2, 1, 2)
plt.hist(pnl_df['realized_pnl'], bins=50)
plt.title('PnL Distribution')

plt.tight_layout()
plt.savefig('custom_analysis.png')
```

## Testing

Run the comprehensive test suite:

```bash
pytest tests/execution_layer/test_pnl_tracker.py -v
```

Test coverage includes:
- Basic PnL calculation
- Position management
- Cumulative tracking
- Performance metrics
- Data persistence
- Integration scenarios

## Best Practices

### 1. Trade Timing
- Always provide accurate timestamps for trades
- Use consistent timezone handling
- Consider market hours for timestamp validation

### 2. Price Updates
- Update market prices regularly for accurate unrealized PnL
- Handle price gaps and market closures
- Validate price data before updates

### 3. Strategy Segregation
- Use consistent strategy IDs
- Track strategy allocation and exposure
- Monitor strategy-specific risk metrics

### 4. Data Management
- Regular database backups
- Archive historical data periodically
- Monitor database size and performance

### 5. Performance Monitoring
- Track key metrics in real-time
- Set up alerts for drawdown thresholds
- Monitor strategy correlation and diversification

## Troubleshooting

### Common Issues

**Negative Position Quantities**
- Check trade side ('buy' vs 'sell')
- Verify position closing logic
- Ensure proper quantity signs

**Incorrect Average Prices**
- Verify trade quantity calculations
- Check for partial position closes
- Validate commission handling

**Performance Metrics Issues**
- Ensure sufficient trade history
- Check for data gaps
- Verify time series continuity

**Database Issues**
- Check file permissions
- Verify disk space
- Monitor database locks

## Advanced Usage

### Custom Risk Metrics
```python
class CustomRiskMetrics:
    def __init__(self, pnl_tracker):
        self.pnl_tracker = pnl_tracker
    
    def calculate_custom_metric(self):
        # Implement custom risk calculations
        returns = self.pnl_tracker.get_pnl_dataframe()['realized_pnl']
        # Custom logic here
        return custom_value
```

### Real-time Monitoring
```python
import asyncio

async def monitor_pnl():
    pnl_tracker = PnLTracker()
    cum_tracker = CumulativePnLTracker(pnl_tracker)
    
    while True:
        snapshot = cum_tracker.update()
        
        # Check for alerts
        if snapshot.current_drawdown > 0.1:  # 10% drawdown
            print(f"ALERT: Drawdown exceeded 10%: {snapshot.current_drawdown:.1%}")
        
        await asyncio.sleep(60)  # Check every minute
```

### Multi-Asset Portfolio
```python
class PortfolioTracker:
    def __init__(self):
        self.asset_trackers = {}
    
    def add_asset(self, symbol):
        self.asset_trackers[symbol] = PnLTracker()
    
    def get_portfolio_pnl(self):
        return sum(tracker.get_cumulative_pnl() 
                  for tracker in self.asset_trackers.values())
    
    def get_correlation_matrix(self):
        # Calculate asset return correlations
        # Implementation here
        pass
```

## Contributing

When contributing to the PnL tracking system:

1. **Add comprehensive tests** for new features
2. **Update documentation** with examples
3. **Follow existing code style** and patterns
4. **Consider performance impact** of changes
5. **Test edge cases** and error conditions

## License

This PnL tracking system is part of the ITS project and follows the same licensing terms.
