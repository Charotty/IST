# Decision Layer

## Purpose
Make trading decisions based on model signals, manage risk, size positions, and portfolio.

## Implemented components

| Component | File | Status | Notes |
|----------|------|--------|-------|
| Base interfaces | `decision.py` | Done | Action, Signal, Decision, BaseDecisionMaker |
| Simple Decision Maker | `simple.py` | Done | Fixed confidence threshold, basic sizing, SL/TP |
| Risk Manager | `risk.py` | Done | Position size/portfolio risk limits, drawdown control |
| Position Sizer | `sizer.py` | Done | Fixed/fraction/Kelly/risk-based sizing |
| Portfolio Manager | `portfolio.py` | Done | Add/remove positions, PnL, exposure |
| Decision Engine | `engine.py` | Done | Full pipeline: Signal → Decision → Risk → Sizing → Final |
| Package init | `__init__.py` | Done | Exports all components |

## Data contracts

### Signal
- Action: Enum (buy/sell/hold/close_long/close_short)
- Confidence: 0.0 - 1.0
- Timestamp: int (ms)
- Symbol: str
- Metadata: dict (model_type, price, etc.)

### Decision
- Action: Action enum
- Symbol: str
- Size: float (position size)
- Price: Optional[float] (None for market)
- Stop Loss: Optional[float]
- Take Profit: Optional[float]
- Timestamp: int
- Reason: str (why decision made)

### Position
- Symbol: str
- Side: 'long' or 'short'
- Size: float
- Entry/Current Price: float
- Unrealized PnL: float
- Timestamp: int

## Usage example

```python
from its_project.decision import (
    Action,
    Signal,
    Decision,
    TradingDecisionEngine,
)

# 1) Create decision engine
engine = TradingDecisionEngine({
    "decision": {"confidence_threshold": 0.7},
    "risk": {
        "max_position_size": 0.1,
        "max_portfolio_risk": 0.05,
        "max_drawdown": 0.15,
    },
    "sizing": {"method": "risk_based", "stop_loss_pct": 0.02},
    "portfolio": {"max_positions": 5},
})

# 2) Process prediction
signal = Signal(
    action=Action.BUY,
    confidence=0.85,
    timestamp=int(time.time() * 1000),
    symbol="BTCUSDT",
    metadata={"price": 42000},
)

market_state = {"price": 42000}
account_balance = 10000.0

decision = engine.process_signal(signal, market_state, account_balance)

if decision:
    print(f"Decision: {decision.action.value} size={decision.size} SL={decision.stop_loss} TP={decision.take_profit}")
else:
    print("No action")
```

## Integration in pipeline

- `decision_task` consumes `predictions` queue, converts predictions to `Signal`, runs through `TradingDecisionEngine` with configurable risk/sizing/portfolio rules, publishes final decisions to `decisions` queue.
- All components are pure and stateless; configuration is external.
- Decisions include full metadata (reason, SL/TP, model_type) for downstream execution layer.

## Next improvements (without breaking layer)
- Add real-time market data fetching (price, order book depth)
- Add dynamic account balance updates
- Add position lifecycle tracking (open → close)
- Add exposure and correlation analytics
- Add backtesting mode for decision rules validation
