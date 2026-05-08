# Paper Trading Enhancements

## Overview

Enhanced paper trading with realistic transaction costs:
- Real balance tracking with equity curve
- Commission calculation (maker/taker fees)
- Slippage simulation
- Transaction cost analysis

## Configuration

```python
config = {
    "initial_balance": 10000.0,
    "commission": {
        "maker_fee": 0.001,  # 0.1%
        "taker_fee": 0.001,  # 0.1%
        "fee_currency": "quote"
    },
    "slippage": {
        "enabled": True,
        "base_slippage": 0.0005,  # 0.05%
        "max_slippage": 0.01  # 1%
    }
}
```

## Usage

```python
executor = PaperTradingExecutor(config)

# Create order (includes commission and slippage)
order = await executor.create_order(
    symbol="BTC/USDT",
    order_type=OrderType.MARKET,
    side="buy",
    amount=0.1
)

# Get performance summary
summary = executor.get_performance_summary()
print(f"Net Return: {summary['net_return_pct']:.2f}%")
print(f"Total Commissions: ${summary['total_commissions']:.2f}")
print(f"Total Slippage: ${summary['total_slippage']:.2f}")

# Get balance history
history = executor.get_balance_history()
```

## API Reference

### Classes
- `BalanceSnapshot`: Balance state snapshot
- `CommissionConfig`: Commission configuration
- `SlippageConfig`: Slippage configuration

### Methods
- `get_balance_history()`: Get equity curve data
- `get_commission_history()`: Get commission records
- `get_slippage_history()`: Get slippage records
- `get_transaction_costs()`: Get total costs
- `get_performance_summary()`: Get performance metrics
