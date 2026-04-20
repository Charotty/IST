from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections import deque
from typing import AsyncIterator, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import websockets

from its_project.common.types import MarketData, MarketDataType
from its_project.data_layer.base import BaseDataSource
from its_project.data_layer.binance_rest import fetch_snapshot_with_retries
from its_project.data_layer.marketdata_helpers import make_trade, make_orderbook_delta, now_ms

logger = logging.getLogger(__name__)


@dataclasses.dataclass(slots=True)
class _LobQualityStats:
    out_of_order_buffered: int = 0
    out_of_order_dropped: int = 0
    duplicates_dropped: int = 0
    lag_warnings: int = 0


class _BinanceLobReconstructor:
    """Reconstructs Binance order book from snapshot + incremental deltas.

    Notes:
    - Works per-symbol.
    - Buffers out-of-order deltas until a contiguous sequence can be applied.
    - Filters duplicates (u <= last_update_id).
    """

    def __init__(self, *, lag_warn_ms: int = 2_000, max_buffer: int = 2_000) -> None:
        self._lag_warn_ms = lag_warn_ms
        self._max_buffer = max_buffer
        self._books: Dict[str, Dict[str, object]] = {}
        self.stats = _LobQualityStats()

    def _ensure_symbol(self, symbol: str) -> Dict[str, object]:
        if symbol not in self._books:
            self._books[symbol] = {
                "last_update_id": None,
                "bids": {},  # price(str) -> qty(str)
                "asks": {},
                "buffer": {},  # u(int) -> delta payload
            }
        return self._books[symbol]

    def apply_snapshot(self, *, symbol: str, snapshot: Mapping[str, object]) -> None:
        st = self._ensure_symbol(symbol)
        last_id = snapshot.get("lastUpdateId")
        if not isinstance(last_id, int):
            raise ValueError("Snapshot missing lastUpdateId")

        bids: MutableMapping[str, str] = {}
        asks: MutableMapping[str, str] = {}
        raw_bids = snapshot.get("bids", [])
        raw_asks = snapshot.get("asks", [])

        if isinstance(raw_bids, list):
            for lvl in raw_bids:
                if isinstance(lvl, list) and len(lvl) >= 2:
                    bids[str(lvl[0])] = str(lvl[1])
        if isinstance(raw_asks, list):
            for lvl in raw_asks:
                if isinstance(lvl, list) and len(lvl) >= 2:
                    asks[str(lvl[0])] = str(lvl[1])

        st["last_update_id"] = last_id
        st["bids"] = bids
        st["asks"] = asks

        # After setting snapshot, try applying buffered deltas.
        self._drain_buffer(symbol=symbol)

    def _apply_levels(self, *, side: MutableMapping[str, str], levels: object) -> None:
        if not isinstance(levels, list):
            return
        for lvl in levels:
            if not (isinstance(lvl, list) and len(lvl) >= 2):
                continue
            price = str(lvl[0])
            qty = str(lvl[1])
            # Binance sends qty="0" to remove level.
            if qty == "0" or qty == "0.0":
                side.pop(price, None)
            else:
                side[price] = qty

    def apply_delta(
        self,
        *,
        symbol: str,
        delta: Mapping[str, object],
        ts_event_ms: int,
        ts_recv_ms: int,
    ) -> None:
        st = self._ensure_symbol(symbol)
        last_update_id = st["last_update_id"]

        # Lag detection (informational; does not drop by default).
        lag = ts_recv_ms - ts_event_ms
        if lag > self._lag_warn_ms:
            self.stats.lag_warnings += 1
            logger.warning("LOB lag %sms for %s", lag, symbol)

        u = delta.get("u")
        U = delta.get("U")
        if not isinstance(u, int) or not isinstance(U, int):
            # Can't process without update ids.
            return

        # If we don't have snapshot yet, buffer until snapshot arrives.
        if not isinstance(last_update_id, int):
            buf: MutableMapping[int, Mapping[str, object]] = st["buffer"]  # type: ignore[assignment]
            if len(buf) >= self._max_buffer:
                # Drop oldest by u.
                oldest_u = sorted(buf.keys())[0]
                buf.pop(oldest_u, None)
                self.stats.out_of_order_dropped += 1
            buf[u] = dict(delta)
            self.stats.out_of_order_buffered += 1
            return

        # Duplicate / already-applied filtering.
        if u <= last_update_id:
            self.stats.duplicates_dropped += 1
            return

        # If this delta does not connect to our current state, buffer.
        # Binance rule: first event after snapshot must satisfy U <= last_id+1 <= u.
        # For subsequent events we require U == last_id+1 for contiguous application.
        if U > last_update_id + 1:
            buf = st["buffer"]  # type: ignore[assignment]
            if len(buf) >= self._max_buffer:
                oldest_u = sorted(buf.keys())[0]
                buf.pop(oldest_u, None)
                self.stats.out_of_order_dropped += 1
            buf[u] = dict(delta)
            self.stats.out_of_order_buffered += 1
            return

        # Contiguous apply.
        bids: MutableMapping[str, str] = st["bids"]  # type: ignore[assignment]
        asks: MutableMapping[str, str] = st["asks"]  # type: ignore[assignment]
        self._apply_levels(side=bids, levels=delta.get("b"))
        self._apply_levels(side=asks, levels=delta.get("a"))
        st["last_update_id"] = u

        # Drain buffered deltas if they now connect.
        self._drain_buffer(symbol=symbol)

    def _drain_buffer(self, *, symbol: str) -> None:
        st = self._ensure_symbol(symbol)
        last_update_id = st["last_update_id"]
        if not isinstance(last_update_id, int):
            return

        buf: MutableMapping[int, Mapping[str, object]] = st["buffer"]  # type: ignore[assignment]
        if not buf:
            return

        # Try apply smallest u first until we break contiguity.
        bids: MutableMapping[str, str] = st["bids"]  # type: ignore[assignment]
        asks: MutableMapping[str, str] = st["asks"]  # type: ignore[assignment]

        progressed = True
        while progressed and buf:
            progressed = False
            for u in sorted(buf.keys()):
                delta = buf[u]
                U = delta.get("U")
                if not isinstance(U, int):
                    buf.pop(u, None)
                    continue
                # apply if connects
                if U <= last_update_id + 1 <= u:
                    self._apply_levels(side=bids, levels=delta.get("b"))
                    self._apply_levels(side=asks, levels=delta.get("a"))
                    last_update_id = u
                    st["last_update_id"] = u
                    buf.pop(u, None)
                    progressed = True
                elif u <= last_update_id:
                    # stale
                    buf.pop(u, None)
                    self.stats.duplicates_dropped += 1
                    progressed = True
            # loop ends

    def build_snapshot_md(self, *, symbol: str, exchange: str) -> Optional[MarketData]:
        st = self._ensure_symbol(symbol)
        last_update_id = st["last_update_id"]
        if not isinstance(last_update_id, int):
            return None

        bids: Mapping[str, str] = st["bids"]  # type: ignore[assignment]
        asks: Mapping[str, str] = st["asks"]  # type: ignore[assignment]

        # Convert to sorted price levels as lists. Prices/qty remain strings to match Binance payload style.
        bids_lvls = [[p, q] for p, q in sorted(bids.items(), key=lambda kv: float(kv[0]), reverse=True)]
        asks_lvls = [[p, q] for p, q in sorted(asks.items(), key=lambda kv: float(kv[0]))]

        return MarketData(
            timestamp_ms=now_ms(),
            symbol=symbol.upper(),
            type=MarketDataType.ORDERBOOK,
            exchange=exchange,
            data={
                "_kind": "snapshot",
                "_reconstructed": True,
                "lastUpdateId": last_update_id,
                "bids": bids_lvls,
                "asks": asks_lvls,
            },
        )


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
    recon = _BinanceLobReconstructor()
    if rest_client is not None:
        for sym in ws.symbols:
            symbol = sym.upper()
            snapshot = await fetch_snapshot_with_retries(client=rest_client, symbol=symbol, limit=50)
            # Apply snapshot to reconstructor and publish reconstructed snapshot.
            try:
                recon.apply_snapshot(symbol=symbol, snapshot=snapshot)
            except Exception:
                logger.exception("Failed to apply LOB snapshot for %s", symbol)
            md_snapshot = recon.build_snapshot_md(symbol=symbol, exchange=ws.exchange)
            if md_snapshot is not None:
                await lob_queue.put(md_snapshot)

    async for md in ws.subscribe():
        if md.type == MarketDataType.TRADE:
            await price_queue.put(md)
        elif md.type == MarketDataType.ORDERBOOK:
            # Apply delta, then publish reconstructed snapshot (downstream expects bids/asks).
            try:
                ts_event_ms = int(md.timestamp_ms)
                ts_recv_ms = int(md.data.get("_ts_recv_ms", ts_event_ms)) if isinstance(md.data, Mapping) else ts_event_ms
                recon.apply_delta(symbol=md.symbol, delta=md.data, ts_event_ms=ts_event_ms, ts_recv_ms=ts_recv_ms)
                md_snapshot = recon.build_snapshot_md(symbol=md.symbol, exchange=md.exchange)
                if md_snapshot is not None:
                    await lob_queue.put(md_snapshot)
            except Exception:
                logger.exception("LOB reconstruction error for %s", md.symbol)
