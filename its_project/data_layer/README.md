# Data Layer

## Purpose
Collect raw market data from multiple sources, unify to `MarketData`, and publish to queues. No processing or analysis.

## Implemented components

| Component | File | Status | Notes |
|----------|------|--------|-------|
| Base interface | `base.py` | Done | `BaseDataSource` (connect/disconnect/subscribe/fetch/is_alive) |
| Rate limiter | `rate_limiter.py` | Done | Async, thread-safe via `asyncio.Lock` |
| Reconnect helper | `reconnect.py` | Done | Exponential backoff, max retries |
| MarketData helpers | `marketdata_helpers.py` | Done | Builders for orderbook delta/snapshot and trade |
| Binance WebSocket | `binance_ws.py` | Done | WS streams (trade+depth), publish to queues, snapshot+delta |
| Binance REST | `binance_rest.py` | Done | Async aiohttp, orderbook snapshot, klines for spot/futures |
| Glassnode client | `glassnode.py` | Done | Async polling, configurable metrics |
| X/Twitter client | `sentiment_x.py` | Done | Async recent search, bearer token auth |
| Polling task | `polling.py` | Done | Generic async polling with interval and graceful stop |
| Package init | `__init__.py` | Empty | No exports yet (used via imports) |

## Data contracts

### MarketData envelope
```python
MarketData(
    timestamp_ms: int,
    symbol: str,
    type: MarketDataType,
    exchange: str,
    data: dict[str, Any],
)
```

### Internal markers in `data`
- `_kind`: `"trade" | "delta" | "snapshot"`
- `_ts_recv_ms`: receive timestamp (for lag tracking)

### Important fields to preserve downstream
- Orderbook snapshot: `lastUpdateId`
- Orderbook delta: `U`, `u`, `b`, `a`

## Usage example (from main.py)

```python
from its_project.data_layer.binance_ws import BinanceWsClient, publish_ws_to_queues
from its_project.data_layer.polling import polling_task
# ... create queues, start WS + polling tasks
```

## Next improvements (without breaking layer)
- Health/metrics object for sources
- Better reconnect logic with max backoff cap
- Integration tests for REST snapshot and WS subscription
- Configurable enable/disable flags per source
