from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Iterable

import websockets

from its_project.common.types import MarketData, MarketDataType
from its_project.data_layer.base import BaseDataSource
from its_project.data_layer.binance_rest import fetch_snapshot_with_retries
from its_project.data_layer.marketdata_helpers import make_trade, make_orderbook_delta, now_ms

logger = logging.getLogger(__name__)


class BinanceWsClient(BaseDataSource):
    def __init__(
        self,
        *,
        base_url: str,
        symbols: Iterable[str],
        stream_names: Iterable[str],
        exchange: str,
        reconnect_sleep_s: float = 2.0,
    ) -> None:
        super().__init__()
        self._base_url = base_url
        self._symbols = [s.lower() for s in symbols]
        self._stream_names = list(stream_names)
        self._exchange = exchange
        self._reconnect_sleep_s = reconnect_sleep_s
        self._stop = asyncio.Event()

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def is_alive(self) -> bool:
        return self._connected and not self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def exchange(self) -> str:
        return self._exchange

    def _stream_path(self) -> str:
        streams: list[str] = []
        for sym in self._symbols:
            for stream in self._stream_names:
                streams.append(f"{sym}@{stream}")
        streams_q = "/".join(streams)
        return f"{self._base_url}?streams={streams_q}"

    async def subscribe(self, symbols: list[str] | None = None) -> AsyncIterator[MarketData]:
        if symbols is not None:
            self._symbols = [s.lower() for s in symbols]

        url = self._stream_path()
        logger.info("WS connecting: %s", url)

        while not self._stop.is_set():
            try:
                await self.connect()
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    logger.info("WS connected: %s", self._exchange)
                    async for msg in ws:
                        if self._stop.is_set():
                            break

                        ts_recv = now_ms()
                        try:
                            raw = json.loads(msg)
                        except json.JSONDecodeError:
                            continue

                        data = raw.get("data")
                        stream = raw.get("stream", "")
                        if not isinstance(data, dict):
                            continue

                        symbol = str(data.get("s", ""))
                        if not symbol:
                            symbol = stream.split("@", 1)[0].upper() if "@" in stream else ""

                        event_time = data.get("E")
                        ts_event = int(event_time) if isinstance(event_time, int) else ts_recv

                        md_type = MarketDataType.TRADE
                        if "depth" in stream:
                            md_type = MarketDataType.ORDERBOOK

                        if md_type == MarketDataType.ORDERBOOK:
                            yield make_orderbook_delta(
                                symbol=symbol,
                                exchange=self._exchange,
                                delta=data,
                                ts_event_ms=ts_event,
                                ts_recv_ms=ts_recv,
                            )
                        else:
                            yield make_trade(
                                symbol=symbol,
                                exchange=self._exchange,
                                trade=data,
                                ts_event_ms=ts_event,
                                ts_recv_ms=ts_recv,
                            )
            except Exception:
                logger.exception("WS error (%s). Reconnecting...", self._exchange)
                await self.disconnect()
                await asyncio.sleep(self._reconnect_sleep_s)

        await self.disconnect()

    async def fetch(self, symbol: str, **params) -> MarketData:
        raise NotImplementedError("REST fetch is not implemented for BinanceWsClient")


async def publish_ws_to_queues(
    *,
    ws: BinanceWsClient,
    rest_client=None,
    price_queue: asyncio.Queue[MarketData],
    lob_queue: asyncio.Queue[MarketData],
) -> None:
    if rest_client is not None:
        for sym in ws.symbols:
            symbol = sym.upper()
            snapshot = await fetch_snapshot_with_retries(client=rest_client, symbol=symbol, limit=50)
            await lob_queue.put(
                MarketData(
                    timestamp_ms=now_ms(),
                    symbol=symbol,
                    type=MarketDataType.ORDERBOOK,
                    exchange=ws.exchange,
                    data={"_kind": "snapshot", **snapshot},
                )
            )

    async for md in ws.subscribe():
        if md.type == MarketDataType.TRADE:
            await price_queue.put(md)
        elif md.type == MarketDataType.ORDERBOOK:
            await lob_queue.put(md)
