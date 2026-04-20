# Execution Layer

## Purpose
Execute trading decisions on exchanges with proper order management, stop loss/take profit, and safety controls.

## Implemented components

| Component | File | Status | Notes |
|----------|------|--------|-------|
| Base interfaces | `base.py` | Done | Order, OrderType, OrderStatus, Position, BaseExecutor |
| Paper Trading | `paper.py` | Done | Virtual balance, latency simulation, order tracking |
| Live Trading | `live.py` | Done | ccxt integration, safety limits, testnet support |
| Order Manager | `manager.py` | Done | Order lifecycle, SL/TP, monitoring, error handling |
| Package init | `__init__.py` | Done | Exports all components |

## Data contracts

### Order
- id: str (unique order identifier)
- symbol: str (trading pair)
- type: OrderType (market/limit/stop_loss/take_profit)
- side: str ('buy' or 'sell')
- amount: float (quantity)
- price: Optional[float] (None for market orders)
- status: OrderStatus (pending/open/filled/cancelled/rejected)
- filled: float (how much executed)
- remaining: float (how much left)
- timestamp: int (ms)
- info: dict (exchange-specific data)

### Position
- symbol: str
- side: str ('long' or 'short')
- size: float
- entry_price: float
- current_price: float
- unrealized_pnl: float
- timestamp: int

## Usage example

```python
from its_project.execution import PaperTradingExecutor, OrderManager

# 1) Initialize executor (ALWAYS start with paper)
executor = PaperTradingExecutor({
    "initial_balance": 10000.0,
    "latency_ms": 50,
})

# 2) Initialize order manager
order_manager = OrderManager(executor, {"monitor_interval": 1.0})
await order_manager.start()

# 3) Execute decision with SL/TP
order = await order_manager.execute_decision(
    symbol="BTC/USDT",
    side="buy",
    amount=0.01,
    order_type="market",
    stop_loss=41000.0,
    take_profit=43000.0,
)

# 4) Monitor orders
active_orders = await order_manager.get_active_orders()
balance = await order_manager.get_balance()
positions = await order_manager.get_positions()

# 5) Cleanup
await order_manager.stop()
```

## Safety Rules

### MANDATORY: Start with Paper Trading
- NEVER go live without extensive paper trading validation
- Minimum 1 month of paper trading required
- Validate all strategies in paper first

### Order Safety
- Maximum order size limits
- Daily loss limits
- Balance checks before orders
- Order status verification

### Error Handling
- All order operations logged
- Network error recovery
- Automatic retry with exponential backoff
- Graceful degradation on exchange issues

## Integration in pipeline

- `execution_task` consumes `decisions` queue
- Uses `PaperTradingExecutor` by default (safety first)
- `OrderManager` handles SL/TP and order lifecycle
- Publishes order results to `orders` queue
- All actions logged with timestamps and order IDs

## Live Trading Setup

### Prerequisites
- [ ] 1+ month of profitable paper trading
- [ ] All safety limits tested
- [ ] Monitoring and alerting configured
- [ ] Emergency stop procedures documented

### Configuration
```python
executor = LiveExecutor({
    "exchange": "binance",
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
    "sandbox": True,  # Use testnet first
    "max_order_size": 0.1,
    "daily_loss_limit": 100.0,
})
```

### Safety Checklist
- [ ] API keys stored securely (environment variables)
- [ ] Testnet validation complete
- [ ] Position size limits enforced
- [ ] Circuit breakers configured
- [ ] Real-time monitoring active

## Next improvements (without breaking layer)
- Add more order types (trailing stop, iceberg)
- Implement smart order routing
- Add partial fill handling
- Implement position sizing based on volatility
- Add exchange-specific optimizations
- Implement order book depth-aware execution
