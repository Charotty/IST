from __future__ import annotations

import time
from typing import Any, Mapping

from its_project.common.types import MarketData, MarketDataType


def now_ms() -> int:
    return int(time.time() * 1000)


def make_orderbook_snapshot(*, symbol: str, exchange: str, snapshot: Mapping[str, Any]) -> MarketData:
    return MarketData(
        timestamp_ms=now_ms(),
        symbol=symbol.upper(),
        type=MarketDataType.ORDERBOOK,
        exchange=exchange,
        data={"_kind": "snapshot", **dict(snapshot)},
    )


def make_orderbook_delta(*, symbol: str, exchange: str, delta: Mapping[str, Any], ts_event_ms: int, ts_recv_ms: int) -> MarketData:
    enriched = dict(delta)
    enriched["_kind"] = "delta"
    enriched["_ts_recv_ms"] = ts_recv_ms
    return MarketData(
        timestamp_ms=ts_event_ms,
        symbol=symbol.upper(),
        type=MarketDataType.ORDERBOOK,
        exchange=exchange,
        data=enriched,
    )


def make_trade(*, symbol: str, exchange: str, trade: Mapping[str, Any], ts_event_ms: int, ts_recv_ms: int) -> MarketData:
    enriched = dict(trade)
    enriched["_kind"] = "trade"
    enriched["_ts_recv_ms"] = ts_recv_ms
    return MarketData(
        timestamp_ms=ts_event_ms,
        symbol=symbol.upper(),
        type=MarketDataType.TRADE,
        exchange=exchange,
        data=enriched,
    )
