from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import aiohttp

from its_project.data_layer.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BinanceRestConfig:
    base_url: str
    max_calls: int = 10
    period_s: float = 1.0
    request_timeout_s: float = 10.0


class BinanceRestClient:
    def __init__(self, *, session: aiohttp.ClientSession, config: BinanceRestConfig) -> None:
        self._session = session
        self._config = config
        self._rl = RateLimiter(max_calls=config.max_calls, period_s=config.period_s)


class BinanceSpotRestClient(BinanceRestClient):
    async def fetch_orderbook_snapshot(self, *, symbol: str, limit: int = 50) -> dict[str, Any]:
        await self._rl.acquire()

        url = f"{self._config.base_url}/api/v3/depth"
        params = {"symbol": symbol.upper(), "limit": int(limit)}

        timeout = aiohttp.ClientTimeout(total=self._config.request_timeout_s)
        async with self._session.get(url, params=params, timeout=timeout) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def fetch_klines(
        self,
        *,
        symbol: str,
        interval: str,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[list[Any]]:
        await self._rl.acquire()

        url = f"{self._config.base_url}/api/v3/klines"
        params: dict[str, Any] = {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)

        timeout = aiohttp.ClientTimeout(total=self._config.request_timeout_s)
        async with self._session.get(url, params=params, timeout=timeout) as resp:
            resp.raise_for_status()
            data = await resp.json()
            if not isinstance(data, list):
                raise TypeError("Unexpected klines response")
            return data


class BinanceFuturesRestClient(BinanceRestClient):
    async def fetch_orderbook_snapshot(self, *, symbol: str, limit: int = 50) -> dict[str, Any]:
        await self._rl.acquire()

        url = f"{self._config.base_url}/fapi/v1/depth"
        params = {"symbol": symbol.upper(), "limit": int(limit)}

        timeout = aiohttp.ClientTimeout(total=self._config.request_timeout_s)
        async with self._session.get(url, params=params, timeout=timeout) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def fetch_klines(
        self,
        *,
        symbol: str,
        interval: str,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1500,
    ) -> list[list[Any]]:
        await self._rl.acquire()

        url = f"{self._config.base_url}/fapi/v1/klines"
        params: dict[str, Any] = {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)

        timeout = aiohttp.ClientTimeout(total=self._config.request_timeout_s)
        async with self._session.get(url, params=params, timeout=timeout) as resp:
            resp.raise_for_status()
            data = await resp.json()
            if not isinstance(data, list):
                raise TypeError("Unexpected futures klines response")
            return data


async def fetch_snapshot_with_retries(
    *,
    client: BinanceRestClient,
    symbol: str,
    limit: int = 50,
    retries: int = 5,
) -> dict[str, Any]:
    for attempt in range(retries):
        try:
            return await client.fetch_orderbook_snapshot(symbol=symbol, limit=limit)
        except Exception as e:
            logger.warning("Snapshot fetch failed (%s) for %s: %s", attempt + 1, symbol, e)
            await asyncio.sleep(2**attempt)

    raise RuntimeError(f"Failed to fetch snapshot for {symbol}")
