from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Optional

import asyncpg

from its_project.storage.base import BaseStorage
from its_project.common.types import MarketData

logger = logging.getLogger(__name__)


class TimescaleStorage(BaseStorage):
    def __init__(self, dsn: str, min_size: int = 5, max_size: int = 20) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
        )
        logger.info("TimescaleDB pool created")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("TimescaleDB pool closed")

    async def write(self, data: MarketData) -> bool:
        return await self.write_batch([data]) == 1

    async def write_batch(self, data: List[MarketData]) -> int:
        if not self._pool:
            raise RuntimeError("TimescaleDB pool not initialized")
        if not data:
            return 0

        records = [
            (
                md.timestamp_ms,
                md.symbol,
                md.type.value,
                md.exchange,
                json.dumps(md.data),
            )
            for md in data
        ]

        query = """
            INSERT INTO market_data (timestamp_ms, symbol, type, exchange, data)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT DO NOTHING
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(query, records)
            logger.debug("TimescaleDB batch write: %d records", len(records))
            return len(records)
        except Exception:
            logger.exception("TimescaleDB batch write error")
            raise

    async def read(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        data_type: Optional[str] = None,
    ) -> List[MarketData]:
        if not self._pool:
            raise RuntimeError("TimescaleDB pool not initialized")

        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        query = """
            SELECT timestamp_ms, symbol, type, exchange, data
            FROM market_data
            WHERE symbol = $1
            AND timestamp_ms >= $2
            AND timestamp_ms <= $3
        """
        params: List[object] = [symbol, start_ms, end_ms]

        if data_type:
            query += " AND type = $4"
            params.append(data_type)

        query += " ORDER BY timestamp_ms ASC"

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            return [
                MarketData(
                    timestamp_ms=row["timestamp_ms"],
                    symbol=row["symbol"],
                    type=row["type"],
                    exchange=row["exchange"],
                    data=json.loads(row["data"]),
                )
                for row in rows
            ]
        except Exception:
            logger.exception("TimescaleDB read error")
            raise

    async def get_latest(self, symbol: str, limit: int = 1) -> List[MarketData]:
        if not self._pool:
            raise RuntimeError("TimescaleDB pool not initialized")

        query = """
            SELECT timestamp_ms, symbol, type, exchange, data
            FROM market_data
            WHERE symbol = $1
            ORDER BY timestamp_ms DESC
            LIMIT $2
        """

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, symbol, limit)

            return [
                MarketData(
                    timestamp_ms=row["timestamp_ms"],
                    symbol=row["symbol"],
                    type=row["type"],
                    exchange=row["exchange"],
                    data=json.loads(row["data"]),
                )
                for row in rows
            ]
        except Exception:
            logger.exception("TimescaleDB get_latest error")
            raise
