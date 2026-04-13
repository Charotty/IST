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
class GlassnodeConfig:
    api_key: str
    base_url: str = "https://api.glassnode.com"
    max_calls: int = 5
    period_s: float = 1.0
    timeout_s: float = 10.0


class GlassnodeClient:
    def __init__(self, *, session: aiohttp.ClientSession, config: GlassnodeConfig) -> None:
        self._session = session
        self._config = config
        self._rl = RateLimiter(max_calls=config.max_calls, period_s=config.period_s)

    async def fetch_metric(self, *, path: str, params: dict[str, Any]) -> Any:
        await self._rl.acquire()

        url = f"{self._config.base_url}{path}"
        q = dict(params)
        q["api_key"] = self._config.api_key

        timeout = aiohttp.ClientTimeout(total=self._config.timeout_s)
        async with self._session.get(url, params=q, timeout=timeout) as resp:
            resp.raise_for_status()
            return await resp.json()


def build_onchain_marketdata(*, symbol: str, exchange: str, payload: dict[str, Any]) -> MarketData:
    return MarketData(
        timestamp_ms=now_ms(),
        symbol=symbol.upper(),
        type=MarketDataType.ONCHAIN,
        exchange=exchange,
        data=payload,
    )
