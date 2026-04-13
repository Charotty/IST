from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class MarketDataType(str, Enum):
    TICKER = "ticker"
    TRADE = "trade"
    ORDERBOOK = "orderbook"
    KLINE = "kline"
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"


@dataclass(frozen=True, slots=True)
class MarketData:
    timestamp_ms: int
    symbol: str
    type: MarketDataType
    exchange: str
    data: Mapping[str, Any]
