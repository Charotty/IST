# PnL Tracking System - Comprehensive Documentation

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [PnL Tracking](#pnl-tracking)
4. [Performance Metrics](#performance-metrics)
5. [Data Persistence](#data-persistence)
6. [Equity Curve Tracking](#equity-curve-tracking)
7. [Pipeline Latency Tracking](#pipeline-latency-tracking)
8. [Real-time Dashboard](#real-time-dashboard)
9. [Integration Examples](#integration-examples)
10. [API Reference](#api-reference)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The PnL (Profit and Loss) tracking system is a comprehensive solution for trading performance analysis, providing:

- **Real-time PnL calculation** for individual trades and portfolios
- **Advanced performance metrics** with statistical significance testing
- **Data persistence** with database and CSV export capabilities
- **Real-time monitoring** with interactive dashboards
- **Risk analysis** with drawdown and volatility metrics
- **Strategy performance** tracking and comparison

### Key Features

- ✅ **Per-trade PnL tracking** with accurate position management
- ✅ **Cumulative PnL analysis** with equity curve tracking
- ✅ **Risk-adjusted metrics** (Sharpe, Sortino, Calmar ratios)
- ✅ **Drawdown analysis** with duration and recovery metrics
- ✅ **Win rate analysis** with confidence intervals
- ✅ **Database persistence** with automatic backups
- ✅ **CSV export** with compression and metadata
- ✅ **Real-time dashboard** with interactive charts
- ✅ **Performance alerts** with configurable thresholds

---

## Core Components

### 1. PnL Tracker (`pnl_tracker.py`)

Core component for tracking individual trades and calculating PnL.

**Key Classes:**
- `PnLTracker`: Main tracking class
- `TradeRecord`: Individual trade data structure
- `PositionSnapshot`: Current position state
- `PnLRecord`: Completed trade PnL calculation

**Core Functionality:**
- Real-time position management
- Average price calculation for partial closes
- Commission and fee tracking
- Strategy-specific PnL segregation
- Database persistence with SQLite

### 2. Cumulative PnL Tracker (`cumulative_pnl.py`)

Advanced analysis for cumulative performance over time.

**Key Classes:**
- `CumulativePnLTracker`: Time-series analysis
- `CumulativePnLSnapshot`: Performance snapshot
- `PerformanceMetrics`: Comprehensive metrics container

**Core Functionality:**
- Equity curve tracking
- Drawdown analysis and monitoring
- Risk-adjusted performance calculation
- Time-series returns analysis
- Visualization and reporting

### 3. Performance Metrics (`performance_metrics.py`)

Statistical analysis and performance measurement.

**Key Classes:**
- `AdvancedPerformanceMetrics`: Metrics calculator
- `SharpeRatioMetrics`: Sharpe ratio analysis
- `DrawdownMetrics`: Drawdown statistics
- `WinRateMetrics`: Win rate analysis
- `PerformanceDashboard`: Real-time monitoring

**Core Functionality:**
- Multiple Sharpe ratio variants
- Statistical significance testing
- Confidence intervals
- Rolling window analysis
- Alert system integration

### 4. Data Persistence (`data_persistence.py`)

Database and CSV export functionality.

**Key Classes:**
- `DatabaseManager`: SQLite database operations
- `CSVExporter`: CSV export functionality
- `DataPersistenceManager`: Unified persistence interface
- `DatabaseConfig`: Database configuration
- `CSVConfig`: Export configuration

**Core Functionality:**
- Async database operations
- Batch processing for large datasets
- Automatic backups with compression
- CSV export with metadata
- Data validation and error handling

### 5. Performance Dashboard (`performance_dashboard.py`)

Real-time monitoring and visualization.

**Key Classes:**
- `RealTimePerformanceDashboard`: Live dashboard
- `DashboardConfig`: Dashboard configuration

**Core Functionality:**
- Interactive Plotly charts
- Real-time updates
- Alert management
- Data export capabilities
- Custom callbacks

---

## PnL Tracking

### Trade Management

#### Adding Trades

```python
from its_project.execution import PnLTracker, create_pnl_tracker

# Initialize tracker
pnl_tracker = create_pnl_tracker("data/my_trading.db")

# Add a buy trade
pnl_tracker.add_trade(
    trade_id="trade_001",
    symbol="BTC/USDT",
    side="buy",
    quantity=1.0,
    price=50000.0,
    commission=10.0,
    fees={"exchange": 5.0, "network": 2.0},
    strategy_id="momentum_strategy",
    order_id="order_001"
)

# Add a sell trade (close position)
pnl_tracker.add_trade(
    trade_id="trade_002",
    symbol="BTC/USDT",
    side="sell",
    quantity=1.0,
    price=51000.0,
    commission=10.0,
    strategy_id="momentum_strategy",
    order_id="order_002"
)
```

#### Position Management

The system automatically manages positions:

```python
# Get current positions
positions = pnl_tracker.get_all_positions()
for symbol, position in positions.items():
    print(f"{symbol}: {position.quantity} @ {position.avg_price}")
    print(f"Unrealized PnL: ${position.unrealized_pnl:.2f}")
    print(f"Realized PnL: ${position.realized_pnl:.2f}")
    print(f"Total PnL: ${position.total_pnl:.2f}")

# Get specific position
btc_position = pnl_tracker.get_position_pnl("BTC/USDT")
if btc_position:
    print(f"BTC Position: {btc_position.quantity} contracts")
    print(f"Average Price: ${btc_position.avg_price:.2f}")
```

#### Market Price Updates

```python
# Update market prices for unrealized PnL
pnl_tracker.update_market_price("BTC/USDT", 52000.0)
pnl_tracker.update_market_price("ETH/USDT", 3100.0)

# Check updated PnL
current_pnl = pnl_tracker.get_cumulative_pnl()
print(f"Total PnL: ${current_pnl:.2f}")
```

### Performance Analysis

#### Basic Metrics

```python
# Get comprehensive performance metrics
metrics = pnl_tracker.get_performance_metrics()

print(f"Total Trades: {metrics['total_trades']}")
print(f"Win Rate: {metrics['win_rate']:.1%}")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
print(f"Average Win: ${metrics['avg_win']:.2f}")
print(f"Average Loss: ${metrics['avg_loss']:.2f}")
print(f"Largest Win: ${metrics['largest_win']:.2f}")
print(f"Largest Loss: ${metrics['largest_loss']:.2f}")
print(f"Total PnL: ${metrics['total_realized_pnl']:.2f}")
print(f"Net PnL: ${metrics['net_pnl']:.2f}")
```

#### Strategy-Specific Analysis

```python
# Get strategy performance
strategy_pnl = pnl_tracker.get_strategy_pnl("momentum_strategy")
print(f"Momentum Strategy PnL: ${strategy_pnl:.2f}")

# Daily PnL
today_pnl = pnl_tracker.get_daily_pnl()
print(f"Today's PnL: ${today_pnl:.2f}")

# Specific date PnL
date_pnl = pnl_tracker.get_daily_pnl("2024-01-15")
print(f"PnL on 2024-01-15: ${date_pnl:.2f}")
```

#### Data Export

```python
# Export to pandas DataFrames
trades_df = pnl_tracker.get_pnl_dataframe()
positions_df = pnl_tracker.get_positions_dataframe()

# Save to CSV
trades_df.to_csv("trades_export.csv")
positions_df.to_csv("positions_export.csv")

# Analysis with pandas
print(trades_df.describe())
print(f"Average PnL per trade: {trades_df['realized_pnl'].mean():.2f}")
```

---

## Performance Metrics

### Sharpe Ratio Analysis

#### Basic Sharpe Ratio

```python
from its_project.execution import AdvancedPerformanceMetrics

# Calculate returns from trades
returns = [0.01, -0.02, 0.015, 0.008, -0.01]  # Example returns

metrics_calc = AdvancedPerformanceMetrics()
sharpe_metrics = metrics_calc.calculate_sharpe_ratio(returns)

print(f"Simple Sharpe: {sharpe_metrics.sharpe_ratio:.3f}")
print(f"Annualized Sharpe: {sharpe_metrics.sharpe_ratio_annualized:.3f}")
print(f"Sortino Ratio: {sharpe_metrics.sharpe_ratio_sortino:.3f}")
print(f"Information Ratio: {sharpe_metrics.sharpe_ratio_information:.3f}")
print(f"Treynor Ratio: {sharpe_metrics.treynor_ratio:.3f}")
print(f"Jensen's Alpha: {sharpe_metrics.jensen_alpha:.3f}")
```

#### Rolling Sharpe Ratio

```python
# Calculate with rolling window
sharpe_metrics = metrics_calc.calculate_sharpe_ratio(
    returns=returns,
    rolling_window=50,  # 50-period rolling window
    annualization_factor=252  # Daily to annual
)

# Access rolling values
rolling_sharpe = sharpe_metrics.sharpe_ratio_rolling
print(f"Latest Rolling Sharpe: {rolling_sharpe[-1]:.3f}")
print(f"Average Rolling Sharpe: {np.mean(rolling_sharpe):.3f}")
```

#### Statistical Significance

```python
# Confidence interval and p-value
print(f"95% Confidence Interval: {sharpe_metrics.confidence_interval}")
print(f"P-value: {sharpe_metrics.p_value:.4f}")

# Interpret results
if sharpe_metrics.p_value < 0.05:
    print("Sharpe ratio is statistically significant (p < 0.05)")
else:
    print("Sharpe ratio is not statistically significant")
```

### Drawdown Analysis

#### Comprehensive Drawdown Metrics

```python
# Create equity curve
equity_curve = [100000, 105000, 98000, 102000, 95000, 110000]
timestamps = pd.date_range("2024-01-01", periods=6, freq="D")

drawdown_metrics = metrics_calc.calculate_drawdown_metrics(equity_curve, timestamps)

print(f"Max Drawdown: {drawdown_metrics.max_drawdown:.1%}")
print(f"Max Drawdown Duration: {drawdown_metrics.max_drawdown_duration} days")
print(f"Current Drawdown: {drawdown_metrics.current_drawdown:.1%}")
print(f"Average Drawdown: {drawdown_metrics.average_drawdown:.1%}")
print(f"Drawdown Frequency: {drawdown_metrics.drawdown_frequency:.1%}")
print(f"Recovery Time: {drawdown_metrics.recovery_time_avg:.1f} days")
print(f"Time Under Water: {drawdown_metrics.time_under_water:.1%}")
print(f"Pain Index: {drawdown_metrics.pain_index:.4f}")
print(f"Ulcer Index: {drawdown_metrics.ulcer_index:.4f}")
print(f"Martin Ratio: {drawdown_metrics.martin_ratio:.3f}")
```

#### Drawdown Distribution

```python
# Access drawdown distribution
dist = drawdown_metrics.drawdown_distribution
print(f"Drawdown Statistics:")
print(f"  Min: {dist['min']:.1%}")
print(f"  Max: {dist['max']:.1%}")
print(f"  Mean: {dist['mean']:.1%}")
print(f"  Median: {dist['median']:.1%}")
print(f"  Std: {dist['std']:.1%}")
print(f"  25th Percentile: {dist['q25']:.1%}")
print(f"  75th Percentile: {dist['q75']:.1%}")
```

### Win Rate Analysis

#### Detailed Win Rate Metrics

```python
# Trade PnL data
trade_pnls = [1000, -500, 1500, -300, 800, -200, 1200, -600]

win_rate_metrics = metrics_calc.calculate_win_rate_metrics(trade_pnls)

print(f"Win Rate: {win_rate_metrics.win_rate:.1%}")
print(f"Winning Trades: {win_rate_metrics.winning_trades}")
print(f"Losing Trades: {win_rate_metrics.losing_trades}")
print(f"Total Trades: {win_rate_metrics.total_trades}")
print(f"Average Win: ${win_rate_metrics.avg_win:.2f}")
print(f"Average Loss: ${win_rate_metrics.avg_loss:.2f}")
print(f"Largest Win: ${win_rate_metrics.largest_win:.2f}")
print(f"Largest Loss: ${win_rate_metrics.largest_loss:.2f}")
print(f"Profit Factor: {win_rate_metrics.profit_factor:.2f}")
print(f"Expectancy: ${win_rate_metrics.expectancy:.2f}")
```

#### Statistical Analysis

```python
# Confidence intervals and significance
print(f"Standard Error: {win_rate_metrics.standard_error:.4f}")
print(f"95% Confidence Interval: {win_rate_metrics.confidence_interval}")
print(f"Z-score vs 50%: {win_rate_metrics.z_score:.2f}")
print(f"P-value: {win_rate_metrics.p_value:.4f}")

# Trade distribution
dist = win_rate_metrics.trade_distribution
print(f"Trade Distribution:")
print(f"  Skewness: {dist['skewness']:.3f}")
print(f"  Kurtosis: {dist['kurtosis']:.3f}")
print(f"  25th Percentile: {dist['q25']:.2f}")
print(f"  75th Percentile: {dist['q75']:.2f}")
```

#### Consecutive Analysis

```python
# Streak analysis
print(f"Max Consecutive Wins: {win_rate_metrics.consecutive_wins_max}")
print(f"Max Consecutive Losses: {win_rate_metrics.consecutive_losses_max}")
print(f"Average Consecutive Wins: {win_rate_metrics.avg_consecutive_wins:.1f}")
print(f"Average Consecutive Losses: {win_rate_metrics.avg_consecutive_losses:.1f}")
```

### Comprehensive Metrics

```python
# Calculate all metrics at once
comprehensive = metrics_calc.calculate_comprehensive_metrics(
    returns=returns,
    equity_curve=equity_curve,
    trade_pnls=trade_pnls,
    timestamps=timestamps
)

# Access summary
summary = comprehensive['summary']
print(f"Overall Sharpe: {summary['overall_sharpe']:.3f}")
print(f"Max Drawdown: {summary['max_drawdown']:.1%}")
print(f"Win Rate: {summary['win_rate']:.1%}")
print(f"Profit Factor: {summary['profit_factor']:.2f}")
print(f"Expectancy: ${summary['expectancy']:.2f}")
print(f"Ulcer Index: {summary['ulcer_index']:.4f}")
print(f"Martin Ratio: {summary['martin_ratio']:.3f}")
```

---

## Data Persistence

### Database Operations

#### Database Configuration

```python
from its_project.execution import DatabaseManager, DatabaseConfig

# Configure database
config = DatabaseConfig(
    db_path="data/trading_system.db",
    backup_enabled=True,
    backup_interval=3600,  # Every hour
    max_backups=24,      # Keep 24 backups
    connection_timeout=30,
    enable_wal=True,       # WAL mode for performance
    foreign_keys=True
)

# Create database manager
db_manager = DatabaseManager(config)
```

#### Saving Data

```python
# Save single trade
await db_manager.save_trade(trade_record)

# Save batch trades
trades = [trade1, trade2, trade3]
saved_count = await db_manager.save_trades_batch(trades)
print(f"Saved {saved_count} trades")

# Save PnL record
await db_manager.save_pnl_record(pnl_record)

# Save position
await db_manager.save_position(position_snapshot)
```

#### Querying Data

```python
# Get all trades
all_trades = await db_manager.get_trades()

# Filter by symbol
btc_trades = await db_manager.get_trades(symbol="BTC/USDT")

# Filter by strategy
strategy_trades = await db_manager.get_trades(strategy_id="momentum")

# Filter by time range
from datetime import datetime, timedelta
start_time = datetime.now() - timedelta(days=30)
end_time = datetime.now()
recent_trades = await db_manager.get_trades(
    start_time=start_time,
    end_time=end_time
)

# Limit results
recent_100 = await db_manager.get_trades(limit=100)
```

### CSV Export

#### CSV Configuration

```python
from its_project.execution import CSVExporter, CSVConfig

# Configure export
config = CSVConfig(
    export_dir="exports",
    compression=True,        # gzip compression
    date_format="%Y-%m-%d %H:%M:%S",
    include_headers=True,
    encoding="utf-8",
    batch_size=10000
)

# Create exporter
csv_exporter = CSVExporter(config)
```

#### Exporting Trades

```python
# Export trades with metadata
trades = [trade1, trade2, trade3]
filepath = await csv_exporter.export_trades(
    trades=trades,
    filename="trades_export.csv",
    include_metadata=True  # Include calculated fields
)

print(f"Exported trades to: {filepath}")

# Export PnL records
pnl_filepath = await csv_exporter.export_pnl_records(
    pnl_records=pnl_records,
    filename="pnl_records.csv",
    include_metadata=True
)

# Export positions
positions_filepath = await csv_exporter.export_positions(
    positions=positions,
    filename="positions.csv",
    include_metadata=True
)
```

#### Exporting DataFrames

```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'symbol': ['BTC/USDT', 'ETH/USDT'],
    'price': [50000, 3000],
    'quantity': [1.0, 10.0]
})

# Export DataFrame
filepath = await csv_exporter.export_dataframe(
    df=df,
    filename="dataframe_export.csv",
    include_index=True
)

print(f"Exported DataFrame to: {filepath}")
```

### Unified Persistence

#### Persistence Manager Setup

```python
from its_project.execution import DataPersistenceManager, create_persistence_manager

# Create unified persistence manager
persistence = create_persistence_manager(
    db_path="data/trading_system.db",
    export_dir="exports"
)

# Or with custom configuration
persistence = DataPersistenceManager(
    db_config=custom_db_config,
    csv_config=custom_csv_config,
    auto_export_interval=1800  # 30 minutes
)
```

#### Unified Operations

```python
# Save through unified interface
await persistence.save_trade(trade)
await persistence.save_trades_batch(trades)
await persistence.save_pnl_record(pnl_record)
await persistence.save_position(position)

# Export all data
export_results = await persistence.export_all_data(
    include_trades=True,
    include_pnl=True,
    include_positions=True,
    custom_filename="complete_export"
)

print(f"Exported files: {list(export_results.keys())}")
```

---

## Equity Curve Tracking

### Overview

Equity curve tracking provides comprehensive time-series analysis of portfolio performance, enabling detailed visualization and statistical analysis of trading results over time.

**Key Features:**
- ✅ **Real-time equity curve** updates with automatic calculation
- ✅ **Time series analysis** with daily returns and volatility
- ✅ **Drawdown tracking** with duration and recovery analysis
- ✅ **Benchmark comparison** against market indices
- ✅ **Statistical analysis** with VaR, CVaR, and distribution metrics
- ✅ **Interactive visualization** with multiple chart types
- ✅ **Data export** in CSV, JSON, and Excel formats

### Basic Usage

```python
from its_project.execution import EquityCurveTracker, create_equity_curve_tracker

# Create equity curve tracker
equity_tracker = create_equity_curve_tracker(pnl_tracker)

# Update equity curve with current state
equity_point = equity_tracker.update_equity_curve()
print(f"Current Equity: ${equity_point.equity_value:.2f}")
print(f"Cumulative PnL: ${equity_point.cumulative_pnl:.2f}")
```

### Configuration

```python
from its_project.execution import EquityCurveConfig, create_equity_curve_config

# Configure equity curve tracking
config = create_equity_curve_config(
    starting_capital=100000.0,
    include_benchmark=True,
    benchmark_returns=sp500_daily_returns
)

# Create tracker with custom config
equity_tracker = EquityCurveTracker(pnl_tracker, config=config)
```

### Real-time Updates

```python
# Automatic updates with trade processing
for trade in new_trades:
    pnl_tracker.add_trade(**trade)
    
    # Update equity curve after each trade
    equity_point = equity_tracker.update_equity_curve()
    
    print(f"Equity: ${equity_point.equity_value:.2f}")
    print(f"Daily Return: {equity_point.daily_return:.2%}")

# Manual update
equity_tracker.update_equity_curve(timestamp=datetime.now())
```

### Statistical Analysis

```python
# Get comprehensive statistics
stats = equity_tracker.calculate_statistics()

print(f"Total Return: {stats.total_return:.2%}")
print(f"Annualized Return: {stats.annualized_return:.2%}")
print(f"Volatility: {stats.volatility:.2%}")
print(f"Sharpe Ratio: {stats.sharpe_ratio:.3f}")
print(f"Max Drawdown: {stats.max_drawdown:.2%}")
print(f"Calmar Ratio: {stats.calmar_ratio:.3f}")
print(f"Sortino Ratio: {stats.sortino_ratio:.3f}")
print(f"VaR (95%): {stats.var_95:.2%}")
print(f"CVaR (95%): {stats.cvar_95:.2%}")
print(f"Best Day: {stats.best_day:.2%}")
print(f"Worst Day: {stats.worst_day:.2%}")
print(f"Daily Win Rate: {stats.win_rate_daily:.2%}")
```

### Data Export

```python
# Export equity curve data
equity_tracker.export_equity_curve(
    format="csv",
    include_statistics=True,
    save_path="equity_curve_analysis.csv"
)

# Export to JSON for API integration
equity_tracker.export_equity_curve(
    format="json",
    include_benchmark=True
)

# Export to Excel with multiple sheets
equity_tracker.export_equity_curve(
    format="excel",
    include_statistics=True
)
```

### Visualization

#### Equity Curve Chart

```python
# Create interactive equity curve with drawdown
fig = equity_tracker.create_equity_chart(
    include_benchmark=True,
    include_drawdown=True,
    smooth_curve=True,
    save_path="equity_curve.html"
)

# Display in notebook
fig.show()
```

#### Returns Distribution Analysis

```python
# Create comprehensive returns analysis
fig = equity_tracker.create_returns_distribution_chart(
    save_path="returns_analysis.html"
)

# Includes:
# - Returns histogram
# - Daily returns over time
# - Returns heatmap by weekday
# - Q-Q plot for normality
```

#### Performance Dashboard

```python
# Create KPI dashboard
fig = equity_tracker.create_performance_dashboard(
    save_path="performance_dashboard.html"
)

# Includes:
# - Return metrics (total, annualized)
# - Risk metrics (volatility, drawdown)
# - Trade statistics (count, positions)
# - Current status indicators
```

### DataFrames Integration

```python
# Get equity curve as DataFrame
equity_df = equity_tracker.get_equity_dataframe()
print(equity_df.head())

# Get returns as DataFrame
returns_df = equity_tracker.get_returns_dataframe()
print(returns_df.describe())

# Get drawdown as DataFrame
drawdown_df = equity_tracker.get_drawdown_dataframe()
print(drawdown_df.head())

# Analysis with pandas
equity_df['equity_value'].plot(title='Equity Curve')
returns_df['return'].hist(bins=50, title='Returns Distribution')
```

### Benchmark Comparison

```python
# Compare with S&P 500
sp500_returns = get_sp500_daily_returns()
comparison = equity_tracker.compare_with_benchmark(
    benchmark_returns=sp500_returns,
    benchmark_name="S&P 500"
)

print(f"Information Ratio: {comparison['information_ratio']:.3f}")
print(f"Tracking Error: {comparison['tracking_error']:.3f}")
print(f"Beta: {comparison['beta']:.3f}")
print(f"Upside Capture: {comparison['upside_capture']:.2%}")
print(f"Downside Capture: {comparison['downside_capture']:.2%}")
print(f"Outperformance: {comparison['outperformance']:.2%}")
```

### Advanced Features

#### Multi-Strategy Analysis

```python
# Create separate equity curves for each strategy
strategies = ["momentum", "mean_reversion", "arbitrage"]
equity_curves = {}

for strategy in strategies:
    # Filter trades by strategy
    strategy_trades = [t for t in all_trades if t.strategy_id == strategy]
    
    # Create equity curve for strategy
    strategy_pnl = create_pnl_tracker()
    for trade in strategy_trades:
        strategy_pnl.add_trade(**trade)
    
    equity_curves[strategy] = create_equity_curve_tracker(strategy_pnl)

# Compare strategies
for strategy, curve in equity_curves.items():
    stats = curve.calculate_statistics()
    print(f"{strategy}: Sharpe={stats.sharpe_ratio:.3f}, Return={stats.total_return:.2%}")
```

#### Risk Analysis

```python
# Value at Risk calculation
stats = equity_tracker.calculate_statistics()

# Daily VaR at different confidence levels
returns_df = equity_tracker.get_returns_dataframe()
var_90 = returns_df['return'].quantile(0.10)
var_95 = returns_df['return'].quantile(0.05)
var_99 = returns_df['return'].quantile(0.01)

print(f"VaR 90%: {var_90:.2%}")
print(f"VaR 95%: {var_95:.2%}")
print(f"VaR 99%: {var_99:.2%}")

# Expected Shortfall (CVaR)
cvar_95 = returns_df[returns_df['return'] <= var_95]['return'].mean()
print(f"CVaR 95%: {cvar_95:.2%}")
```

#### Performance Attribution

```python
# Analyze contribution of different components
equity_df = equity_tracker.get_equity_dataframe()

# Separate realized vs unrealized PnL
equity_df['realized_contribution'] = equity_df['realized_pnl'].diff()
equity_df['unrealized_contribution'] = equity_df['unrealized_pnl'].diff()

# Calculate contribution percentages
total_contribution = equity_df['realized_contribution'].sum()
equity_df['realized_pct'] = equity_df['realized_contribution'] / total_contribution * 100

print("Performance Attribution:")
print(f"Realized PnL Contribution: {total_contribution:.2f}")
print(f"Average Daily Realized: {equity_df['realized_contribution'].mean():.2f}")
```

### Integration with Trading Systems

#### Live Trading Integration

```python
class LiveTradingSystem:
    def __init__(self):
        self.pnl_tracker = create_pnl_tracker()
        self.equity_tracker = create_equity_curve_tracker(self.pnl_tracker)
        
    async def on_trade_filled(self, trade):
        """Handle trade execution."""
        # Add trade to PnL tracker
        self.pnl_tracker.add_trade(**trade)
        
        # Update equity curve
        equity_point = self.equity_tracker.update_equity_curve()
        
        # Check for alerts
        if equity_point.daily_return < -0.05:  # 5% daily loss
            await send_alert(f"Large daily loss: {equity_point.daily_return:.2%}")
    
    def get_performance_report(self):
        """Generate performance report."""
        stats = self.equity_tracker.calculate_statistics()
        
        return {
            'total_return': stats.total_return,
            'sharpe_ratio': stats.sharpe_ratio,
            'max_drawdown': stats.max_drawdown,
            'current_equity': self.equity_tracker.equity_points[-1].equity_value
        }
```

#### Backtesting Integration

```python
def analyze_backtest(backtest_results):
    """Analyze backtest results with equity curve."""
    # Create equity curve from backtest
    equity_tracker = create_equity_curve_tracker()
    
    # Process all trades from backtest
    for trade in backtest_results.trades:
        equity_tracker.pnl_tracker.add_trade(**trade)
        equity_tracker.update_equity_curve(timestamp=trade.timestamp)
    
    # Generate comprehensive analysis
    stats = equity_tracker.calculate_statistics()
    
    # Create visualizations
    equity_chart = equity_tracker.create_equity_chart()
    returns_chart = equity_tracker.create_returns_distribution_chart()
    
    # Export results
    equity_tracker.export_equity_curve("backtest_analysis.xlsx")
    
    return {
        'statistics': stats,
        'equity_chart': equity_chart,
        'returns_chart': returns_chart
    }
```

---

## Pipeline Latency Tracking

### Overview

The Pipeline Latency Tracking system provides comprehensive monitoring and analysis of pipeline performance, enabling real-time tracking of processing times, error rates, and system health.

### Key Features

- ✅ **Real-time latency measurement** with millisecond precision
- ✅ **Statistical analysis** with percentiles and distributions
- ✅ **Error rate monitoring** with configurable thresholds
- ✅ **Alert system** with customizable warning/critical levels
- ✅ **Historical data management** with automatic cleanup
- ✅ **Interactive visualization** with multiple chart types
- ✅ **Data persistence** with multiple export formats
- ✅ **Performance optimization** with caching and batch operations

### Use Cases

- **Trading Systems**: Monitor order processing, execution latency
- **Data Processing**: Track ETL pipeline performance
- **API Services**: Measure endpoint response times
- **Machine Learning**: Monitor model inference latency
- **Microservices**: Track inter-service communication delays

---

## Real-time Dashboard

### Dashboard Setup

```python
from its_project.execution import RealTimePerformanceDashboard, create_real_time_dashboard

# Create dashboard
dashboard = create_real_time_dashboard(
    pnl_tracker=pnl_tracker,
    config=DashboardConfig(
        update_interval=60,      # Update every minute
        max_history_points=1000,
        chart_width=1200,
        chart_height=800,
        theme="plotly_white",
        alert_enabled=True
    )
)
```

### Starting Dashboard

```python
import asyncio

# Start real-time dashboard
await dashboard.start()

# Run for desired duration
await asyncio.sleep(300)  # 5 minutes

# Stop dashboard
await dashboard.stop()
```

### Dashboard Monitoring

```python
# Get current metrics summary
summary = dashboard.get_current_metrics_summary()

print(f"Last Update: {summary['last_update']}")
print(f"Total PnL: ${summary['total_pnl']:.2f}")
print(f"Total Trades: {summary['total_trades']}")
print(f"Current Positions: {summary['current_positions']}")

# Risk-adjusted returns
risk_metrics = summary['risk_adjusted_returns']
print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
print(f"Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}")
print(f"Information Ratio: {risk_metrics['information_ratio']:.2f}")

# Drawdown analysis
drawdown = summary['drawdown_analysis']
print(f"Max Drawdown: {drawdown['max_drawdown']:.1%}")
print(f"Current Drawdown: {drawdown['current_drawdown']:.1%}")
print(f"Ulcer Index: {drawdown['ulcer_index']:.4f}")

# Win rate analysis
win_rate = summary['win_rate_analysis']
print(f"Win Rate: {win_rate['win_rate']:.1%}")
print(f"Profit Factor: {win_rate['profit_factor']:.2f}")
print(f"Expectancy: ${win_rate['expectancy']:.2f}")

# Recent alerts
alerts = summary['alerts']['recent_alerts']
for alert in alerts:
    print(f"Alert [{alert['level'].upper()}]: {alert['message']}")
```

### Custom Callbacks

```python
# Add custom update callback
async def on_metrics_update(metrics):
    """Custom callback for metrics updates."""
    print(f"Metrics updated: Sharpe={metrics['sharpe_ratio']:.3f}")

dashboard.add_update_callback(on_metrics_update)

# Add custom alert callback
async def on_alert(alert):
    """Custom callback for alerts."""
    if alert['level'] == 'critical':
        # Send notification, email, etc.
        print(f"CRITICAL ALERT: {alert['message']}")

dashboard.add_alert_callback(on_alert)
```

### Chart Export

```python
# Export interactive charts
dashboard.export_charts("dashboard_exports")

# This creates:
# - equity_chart_YYYYMMDD_HHMMSS.html
# - metrics_chart_YYYYMMDD_HHMMSS.html
# - rolling_metrics_YYYYMMDD_HHMMSS.html
```

---

## Integration Examples

### Complete Trading System

```python
from its_project.execution import (
    PnLTracker, AdvancedPerformanceMetrics,
    RealTimePerformanceDashboard, DataPersistenceManager,
    create_pnl_tracker, create_performance_metrics,
    create_real_time_dashboard, create_persistence_manager
)

class TradingSystem:
    def __init__(self):
        # Initialize components
        self.pnl_tracker = create_pnl_tracker("data/trading.db")
        self.metrics_calc = create_performance_metrics()
        self.dashboard = create_real_time_dashboard(self.pnl_tracker)
        self.persistence = create_persistence_manager()
        
        self.is_running = False
    
    async def start(self):
        """Start the trading system."""
        self.is_running = True
        
        # Start dashboard
        await self.dashboard.start()
        
        print("Trading system started")
    
    async def stop(self):
        """Stop the trading system."""
        self.is_running = False
        
        # Stop dashboard
        await self.dashboard.stop()
        
        # Close persistence
        await self.persistence.close()
        
        # Save final data
        self.pnl_tracker.save()
        
        print("Trading system stopped")
    
    async def process_trade(self, trade_data):
        """Process a trade through the system."""
        # Add to PnL tracker
        self.pnl_tracker.add_trade(**trade_data)
        
        # Save to database
        trade = TradeRecord(**trade_data)
        await self.persistence.save_trade(trade)
        
        print(f"Processed trade: {trade_data['trade_id']}")
    
    def get_performance_summary(self):
        """Get comprehensive performance summary."""
        return self.dashboard.get_current_metrics_summary()

# Usage
async def main():
    system = TradingSystem()
    await system.start()
    
    # Process some trades
    await system.process_trade({
        'trade_id': 'demo_001',
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'quantity': 1.0,
        'price': 50000.0,
        'commission': 10.0
    })
    
    # Get performance
    summary = system.get_performance_summary()
    print(f"Current PnL: ${summary['total_pnl']:.2f}")
    
    await asyncio.sleep(60)  # Run for 1 minute
    await system.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Backtesting Integration

```python
# Integrate with backtesting system
class BacktestAnalyzer:
    def __init__(self):
        self.pnl_tracker = create_pnl_tracker("data/backtest.db")
        self.metrics_calc = create_performance_metrics()
    
    def analyze_backtest(self, trades):
        """Analyze backtest results."""
        # Process all trades
        for trade in trades:
            self.pnl_tracker.add_trade(**trade)
        
        # Calculate metrics
        trade_pnls = [record.realized_pnl for record in self.pnl_tracker.pnl_records]
        
        # Create equity curve
        equity_curve = [100000]  # Starting capital
        for pnl in trade_pnls:
            equity_curve.append(equity_curve[-1] + pnl)
        
        # Calculate returns
        returns = []
        for i in range(1, len(equity_curve)):
            returns.append((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1])
        
        # Get comprehensive metrics
        metrics = self.metrics_calc.calculate_comprehensive_metrics(
            returns=returns,
            equity_curve=equity_curve,
            trade_pnls=trade_pnls
        )
        
        return metrics
    
    def export_results(self, metrics, filename="backtest_results"):
        """Export backtest results."""
        # Export to CSV
        trades_df = self.pnl_tracker.get_pnl_dataframe()
        trades_df.to_csv(f"{filename}_trades.csv")
        
        # Export metrics
        import json
        with open(f"{filename}_metrics.json", 'w') as f:
            json.dump(metrics['summary'], f, indent=2)

# Usage
analyzer = BacktestAnalyzer()
backtest_trades = [...]  # Your backtest trades
results = analyzer.analyze_backtest(backtest_trades)
analyzer.export_results(results)
```

### Live Trading Integration

```python
# Integrate with live trading system
class LiveTradingSystem:
    def __init__(self, exchange_api):
        self.exchange = exchange_api
        self.pnl_tracker = create_pnl_tracker("data/live_trading.db")
        self.dashboard = create_real_time_dashboard(self.pnl_tracker)
        self.persistence = create_persistence_manager()
        
        # Setup callbacks
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """Setup exchange callbacks."""
        self.exchange.on_order_filled(self.on_order_filled)
        self.exchange.on_market_data(self.on_market_data)
    
    async def on_order_filled(self, order):
        """Handle order filled event."""
        # Convert order to trade record
        trade_data = {
            'trade_id': order.id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.filled_quantity,
            'price': order.average_price,
            'commission': order.commission,
            'fees': order.fees,
            'strategy_id': order.strategy_id,
            'order_id': order.id
        }
        
        # Process through system
        self.pnl_tracker.add_trade(**trade_data)
        await self.persistence.save_trade(TradeRecord(**trade_data))
        
        print(f"Order filled: {order.id} - PnL updated")
    
    async def on_market_data(self, symbol, price):
        """Handle market data update."""
        # Update market prices for unrealized PnL
        self.pnl_tracker.update_market_price(symbol, price)
        
        print(f"Market data: {symbol} @ {price}")
    
    async def start(self):
        """Start live trading system."""
        await self.dashboard.start()
        await self.exchange.start()
        
        print("Live trading system started")
    
    async def stop(self):
        """Stop live trading system."""
        await self.dashboard.stop()
        await self.exchange.stop()
        await self.persistence.close()
        
        print("Live trading system stopped")

# Usage
# exchange_api = YourExchangeAPI()
# live_system = LiveTradingSystem(exchange_api)
# asyncio.run(live_system.start())
```

---

## API Reference

### PnLTracker

#### Constructor

```python
PnLTracker(
    db_path: str = "data/pnl_tracking.db",
    auto_save: bool = True,
    save_interval: int = 60
)
```

#### Methods

##### add_trade()

```python
add_trade(
    trade_id: str,
    symbol: str,
    side: str,  # 'buy' or 'sell'
    quantity: float,
    price: float,
    timestamp: Optional[datetime] = None,
    commission: float = 0.0,
    fees: Optional[Dict[str, float]] = None,
    strategy_id: Optional[str] = None,
    order_id: Optional[str] = None
) -> None
```

##### update_market_price()

```python
update_market_price(
    symbol: str,
    price: float,
    timestamp: Optional[datetime] = None
) -> None
```

##### get_cumulative_pnl()

```python
get_cumulative_pnl() -> float
```

##### get_performance_metrics()

```python
get_performance_metrics() -> Dict[str, float]
```

##### get_pnl_dataframe()

```python
get_pnl_dataframe() -> pd.DataFrame
```

### AdvancedPerformanceMetrics

#### Constructor

```python
AdvancedPerformanceMetrics(
    risk_free_rate: float = 0.02,
    benchmark_returns: Optional[List[float]] = None,
    confidence_level: float = 0.95
)
```

#### Methods

##### calculate_sharpe_ratio()

```python
calculate_sharpe_ratio(
    returns: List[float],
    method: str = "simple",
    rolling_window: int = 252,
    annualization_factor: int = 252
) -> SharpeRatioMetrics
```

##### calculate_drawdown_metrics()

```python
calculate_drawdown_metrics(
    equity_curve: List[float],
    timestamps: Optional[List[datetime]] = None
) -> DrawdownMetrics
```

##### calculate_win_rate_metrics()

```python
calculate_win_rate_metrics(
    trade_pnls: List[float],
    rolling_window: int = 50
) -> WinRateMetrics
```

### DataPersistenceManager

#### Constructor

```python
DataPersistenceManager(
    db_config: Optional[DatabaseConfig] = None,
    csv_config: Optional[CSVConfig] = None,
    auto_export_interval: int = 3600
)
```

#### Methods

##### save_trade()

```python
async save_trade(trade: TradeRecord) -> bool
```

##### save_trades_batch()

```python
async save_trades_batch(trades: List[TradeRecord]) -> int
```

##### export_all_data()

```python
async export_all_data(
    include_trades: bool = True,
    include_pnl: bool = True,
    include_positions: bool = True,
    custom_filename: Optional[str] = None
) -> Dict[str, str]
```

### RealTimePerformanceDashboard

#### Constructor

```python
RealTimePerformanceDashboard(
    pnl_tracker: PnLTracker,
    config: Optional[DashboardConfig] = None,
    benchmark_returns: Optional[List[float]] = None
)
```

#### Methods

##### start()

```python
async start() -> None
```

##### stop()

```python
async stop() -> None
```

##### get_current_metrics_summary()

```python
get_current_metrics_summary() -> Dict[str, Any]
```

##### export_charts()

```python
export_charts(output_dir: str = "dashboard_charts") -> None
```

---

## Best Practices

### Performance Optimization

#### Database Configuration

```python
# Use WAL mode for better concurrency
config = DatabaseConfig(
    enable_wal=True,
    connection_timeout=30,
    foreign_keys=True
)

# Optimize batch sizes
persistence = DataPersistenceManager(
    auto_export_interval=1800  # 30 minutes
)
```

#### Memory Management

```python
# Limit history size
dashboard_config = DashboardConfig(
    max_history_points=1000,  # Prevent memory issues
    update_interval=60        # Balance responsiveness
)

# Use batch operations
await persistence.save_trades_batch(large_trade_list)
```

#### Data Validation

```python
# Validate trade data before processing
def validate_trade(trade_data):
    required_fields = ['trade_id', 'symbol', 'side', 'quantity', 'price']
    for field in required_fields:
        if field not in trade_data:
            raise ValueError(f"Missing required field: {field}")
    
    if trade_data['quantity'] <= 0:
        raise ValueError("Quantity must be positive")
    
    if trade_data['price'] <= 0:
        raise ValueError("Price must be positive")
    
    if trade_data['side'] not in ['buy', 'sell']:
        raise ValueError("Side must be 'buy' or 'sell'")

# Usage
try:
    validate_trade(trade_data)
    pnl_tracker.add_trade(**trade_data)
except ValueError as e:
    logger.error(f"Invalid trade data: {e}")
```

### Error Handling

#### Robust Error Handling

```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

async def safe_process_trade(trade_data) -> bool:
    """Safely process trade with comprehensive error handling."""
    try:
        # Validate data
        validate_trade(trade_data)
        
        # Process trade
        pnl_tracker.add_trade(**trade_data)
        await persistence.save_trade(TradeRecord(**trade_data))
        
        logger.info(f"Successfully processed trade: {trade_data['trade_id']}")
        return True
        
    except ValueError as e:
        logger.error(f"Validation error for trade {trade_data.get('trade_id', 'unknown')}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing trade {trade_data.get('trade_id', 'unknown')}: {e}")
        # Could send alert, notification, etc.
        return False
```

#### Retry Logic

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def robust_save_trade(trade: TradeRecord) -> bool:
    """Save trade with retry logic."""
    return await persistence.save_trade(trade)
```

### Monitoring and Alerting

#### Performance Alerts

```python
# Configure alert thresholds
alert_thresholds = {
    'min_sharpe': 0.5,        # Alert if Sharpe < 0.5
    'max_drawdown': 0.15,      # Alert if drawdown > 15%
    'min_win_rate': 0.4,      # Alert if win rate < 40%
    'max_consecutive_losses': 5   # Alert if 5+ consecutive losses
}

# Custom alert handler
async def handle_performance_alert(alert):
    """Handle performance alerts with custom logic."""
    if alert['type'] == 'drawdown' and alert['level'] == 'critical':
        # Send email/SMS notification
        await send_critical_notification(alert)
    elif alert['type'] == 'win_rate':
        # Log for strategy review
        await log_strategy_review(alert)

dashboard.add_alert_callback(handle_performance_alert)
```

#### Health Monitoring

```python
async def monitor_system_health():
    """Monitor system health and performance."""
    try:
        # Check database connection
        trades_count = await persistence.db_manager.get_trades(limit=1)
        
        # Check dashboard status
        summary = dashboard.get_current_metrics_summary()
        
        # Check memory usage
        import psutil
        memory_percent = psutil.virtual_memory().percent
        
        health_status = {
            'database': 'healthy' if trades_count is not None else 'error',
            'dashboard': 'healthy' if summary else 'error',
            'memory': f'{memory_percent:.1f}%',
            'timestamp': datetime.now().isoformat()
        }
        
        # Log health status
        logger.info(f"System health: {health_status}")
        
        # Alert on issues
        if any(status == 'error' for status in health_status.values()):
            await send_health_alert(health_status)
            
    except Exception as e:
        logger.error(f"Health monitoring error: {e}")

# Schedule health checks
async def schedule_health_checks():
    while True:
        await monitor_system_health()
        await asyncio.sleep(300)  # Every 5 minutes
```

---

## Troubleshooting

### Common Issues

#### Database Connection Issues

**Problem**: Database connection timeouts or locks

**Solution**:
```python
# Configure connection settings
config = DatabaseConfig(
    connection_timeout=60,  # Increase timeout
    enable_wal=True        # Enable WAL mode
)

# Use connection pooling
db_manager = DatabaseManager(config)

# Handle connection errors
async def safe_database_operation():
    try:
        return await db_manager.get_trades()
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e):
            logger.warning("Database locked, retrying...")
            await asyncio.sleep(1)
            return await safe_database_operation()
        else:
            raise
```

#### Memory Issues

**Problem**: High memory usage with large datasets

**Solution**:
```python
# Limit history size
config = DashboardConfig(
    max_history_points=5000  # Reduce from default 10000
)

# Use batch processing
batch_size = 1000
for i in range(0, len(large_dataset), batch_size):
    batch = large_dataset[i:i+batch_size]
    await persistence.save_trades_batch(batch)
    await asyncio.sleep(0.1)  # Small delay

# Regular cleanup
async def cleanup_old_data():
    # Remove old data beyond certain period
    cutoff_date = datetime.now() - timedelta(days=90)
    await cleanup_old_records(cutoff_date)
```

#### Performance Issues

**Problem**: Slow performance with real-time updates

**Solution**:
```python
# Optimize update intervals
config = DashboardConfig(
    update_interval=120,  # Update every 2 minutes instead of 1
    max_history_points=2000  # Reduce memory usage
)

# Use async operations efficiently
async def batch_process_trades(trades):
    tasks = []
    for trade in trades:
        task = asyncio.create_task(process_single_trade(trade))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

# Optimize database queries
# Use indexes, limit results, and specific queries
recent_trades = await db_manager.get_trades(
    limit=100,  # Limit results
    start_time=datetime.now() - timedelta(days=1)  # Time range
)
```

#### Data Corruption

**Problem**: Database corruption or data inconsistency

**Solution**:
```python
# Enable regular backups
config = DatabaseConfig(
    backup_enabled=True,
    backup_interval=1800,  # Every 30 minutes
    max_backups=48       # Keep 48 hours of backups
)

# Data validation
def validate_database_integrity():
    try:
        # Check total PnL consistency
        total_pnl = pnl_tracker.get_cumulative_pnl()
        calculated_pnl = sum(record.realized_pnl for record in pnl_tracker.pnl_records)
        
        if abs(total_pnl - calculated_pnl) > 0.01:
            logger.error("PnL inconsistency detected")
            return False
        
        # Check position consistency
        positions = pnl_tracker.get_all_positions()
        for symbol, position in positions.items():
            if position.quantity == 0 and abs(position.total_pnl) > 0.01:
                logger.error(f"Zero position with PnL for {symbol}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False

# Regular integrity checks
async def schedule_integrity_checks():
    while True:
        if not validate_database_integrity():
            await send_data_corruption_alert()
        await asyncio.sleep(3600)  # Every hour
```

### Debugging

#### Enable Debug Logging

```python
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Component-specific logging
logger = logging.getLogger('its_project.execution.pnl_tracker')
logger.setLevel(logging.DEBUG)
```

#### Performance Profiling

```python
import cProfile
import pstats

def profile_pnl_tracking():
    """Profile PnL tracking performance."""
    profiler = cProfile.Profile()
    
    # Profile the operation
    profiler.enable()
    
    # Your PnL tracking operations
    for i in range(1000):
        pnl_tracker.add_trade(
            f"test_{i}", "BTC/USDT", "buy",
            1.0, 50000.0 + i
        )
    
    profiler.disable()
    
    # Save profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions

# Usage
profile_pnl_tracking()
```

#### Database Query Analysis

```python
# Enable SQLite query logging
import sqlite3

def setup_query_logging():
    conn = sqlite3.connect("data/trading_system.db")
    conn.set_trace_callback(print)  # Print all queries
    return conn

# Analyze query performance
def analyze_query_performance():
    conn = setup_query_logging()
    
    # Run your queries and analyze output
    cursor = conn.execute("EXPLAIN QUERY PLAN SELECT * FROM trades WHERE symbol = 'BTC/USDT'")
    for row in cursor.fetchall():
        print(row)  # Analyze query plan
```

### Recovery Procedures

#### Database Recovery

```python
async def recover_from_backup():
    """Recover database from most recent backup."""
    try:
        # Find most recent backup
        backup_files = list(Path("data").glob("backup_*.db.gz"))
        if not backup_files:
            logger.error("No backup files found")
            return False
        
        latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
        
        # Extract backup
        import gzip
        with gzip.open(latest_backup, 'rb') as f:
            with open("data/trading_system_recovered.db", 'wb') as out_f:
                out_f.write(f.read())
        
        logger.info(f"Recovered database from {latest_backup}")
        return True
        
    except Exception as e:
        logger.error(f"Recovery failed: {e}")
        return False

# Usage
if not validate_database_integrity():
    await recover_from_backup()
```

#### Data Reconstruction

```python
async def reconstruct_from_exports():
    """Reconstruct database from CSV exports."""
    try:
        # Find recent export
        export_files = list(Path("exports").glob("*_trades.csv"))
        if not export_files:
            logger.error("No export files found")
            return False
        
        latest_export = max(export_files, key=lambda x: x.stat().st_mtime)
        
        # Create new database
        new_db_path = "data/trading_system_reconstructed.db"
        new_persistence = create_persistence_manager(db_path=new_db_path)
        
        # Import data from CSV
        import pandas as pd
        df = pd.read_csv(latest_export)
        
        # Convert to trades and save
        trades = []
        for _, row in df.iterrows():
            trade = TradeRecord(
                trade_id=row['trade_id'],
                symbol=row['symbol'],
                side=row['side'],
                quantity=row['quantity'],
                price=row['price'],
                timestamp=pd.to_datetime(row['timestamp']),
                commission=row.get('commission', 0.0)
            )
            trades.append(trade)
        
        # Batch save
        saved_count = await new_persistence.save_trades_batch(trades)
        logger.info(f"Reconstructed database with {saved_count} trades")
        return True
        
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        return False
```

---

## Conclusion

This PnL tracking system provides a comprehensive solution for trading performance analysis with:

- **Real-time tracking** of trades and positions
- **Advanced performance metrics** with statistical analysis
- **Robust data persistence** with automatic backups
- **Interactive dashboards** for monitoring
- **Flexible integration** options for various use cases

The system is designed for production use with proper error handling, performance optimization, and comprehensive testing coverage.

For additional support or questions, refer to the test files and integration examples provided in the implementation.
