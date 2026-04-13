from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import aiohttp

from its_project.common.types import MarketData, MarketDataType
from its_project.data_layer.marketdata_helpers import now_ms
from its_project.data_layer.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class XSentimentConfig:
    bearer_token: str
    max_calls: int = 1
    period_s: float = 1.0
    timeout_s: float = 10.0


class XClient:
    def __init__(self, *, session: aiohttp.ClientSession, config: XSentimentConfig) -> None:
        self._session = session
        self._config = config
        self._rl = RateLimiter(max_calls=config.max_calls, period_s=config.period_s)

    async def recent_search(self, *, query: str, max_results: int = 10) -> dict[str, Any]:
        await self._rl.acquire()

        url = "https://api.x.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {self._config.bearer_token}"}
        params = {"query": query, "max_results": int(max_results)}

        timeout = aiohttp.ClientTimeout(total=self._config.timeout_s)
        async with self._session.get(url, params=params, headers=headers, timeout=timeout) as resp:
            resp.raise_for_status()
            data = await resp.json()
            if not isinstance(data, dict):
                raise TypeError("Unexpected X response")
            return data


def build_sentiment_marketdata(*, symbol: str, exchange: str, payload: dict[str, Any]) -> MarketData:
    return MarketData(
        timestamp_ms=now_ms(),
        symbol=symbol.upper(),
        type=MarketDataType.SENTIMENT,
        exchange=exchange,
        data=payload,
    )
