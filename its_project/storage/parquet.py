from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq

from its_project.storage.base import BaseStorage
from its_project.common.types import MarketData

logger = logging.getLogger(__name__)


class ParquetStorage(BaseStorage):
    def __init__(self, base_path: str | Path) -> None:
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

    async def write(self, data: MarketData) -> bool:
        return await self.write_batch([data]) == 1

    async def write_batch(self, data: List[MarketData]) -> int:
        if not data:
            return 0

        # Group by date for partitioning
        by_date: dict[date, List[MarketData]] = {}
        for md in data:
            dt = datetime.fromtimestamp(md.timestamp_ms / 1000).date()
            by_date.setdefault(dt, []).append(md)

        written = 0
        for day, records in by_date.items():
            file_path = self._base_path / f"{day.isoformat()}.parquet"

            table = pa.Table.from_pydict({
                "timestamp_ms": [r.timestamp_ms for r in records],
                "symbol": [r.symbol for r in records],
                "type": [r.type.value for r in records],
                "exchange": [r.exchange for r in records],
                "data": [json.dumps(r.data) for r in records],
            })

            # Append mode for incremental writes
            pq.write_to_dataset(
                table,
                root_path=str(file_path),
                format="parquet",
                write_dataset_kwargs={"partitioning": ["symbol"]},
                existing_data_behavior="overwrite_or_ignore",
            )
            written += len(records)
            logger.debug("Parquet batch write: %d records to %s", len(records), file_path)

        return written

    async def read(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        data_type: str | None = None,
    ) -> List[MarketData]:
        # Simple scanning over date range (can be optimized later)
        out: List[MarketData] = []
        current = start.date()
        end_date = end.date()
        while current <= end_date:
            file_path = self._base_path / f"{current.isoformat()}.parquet"
            if file_path.exists():
                try:
                    dataset = pq.ParquetDataset(file_path)
                    table = dataset.read()
                    df = table.to_pandas()
                    # Filter by symbol and time window
                    mask = df["symbol"] == symbol
                    mask &= (df["timestamp_ms"] >= int(start.timestamp() * 1000)) & (
                        df["timestamp_ms"] <= int(end.timestamp() * 1000)
                    )
                    if data_type:
                        mask &= df["type"] == data_type
                    sub = df[mask]
                    for _, row in sub.iterrows():
                        out.append(
                            MarketData(
                                timestamp_ms=int(row["timestamp_ms"]),
                                symbol=str(row["symbol"]),
                                type=row["type"],
                                exchange=str(row["exchange"]),
                                data=json.loads(row["data"]),
                            )
                        )
                except Exception:
                    logger.exception("Parquet read error for %s", file_path)
            current = date.fromordinal(current.toordinal() + 1)

        out.sort(key=lambda md: md.timestamp_ms)
        return out

    async def get_latest(self, symbol: str, limit: int = 1) -> List[MarketData]:
        # Scan recent files (last 7 days) to find latest records
        out: List[MarketData] = []
        today = date.today()
        for offset in range(7):
            d = date.fromordinal(today.toordinal() - offset)
            file_path = self._base_path / f"{d.isoformat()}.parquet"
            if not file_path.exists():
                continue
            try:
                dataset = pq.ParquetDataset(file_path)
                table = dataset.read()
                df = table.to_pandas()
                mask = df["symbol"] == symbol
                sub = df[mask]
                for _, row in sub.iterrows():
                    out.append(
                        MarketData(
                            timestamp_ms=int(row["timestamp_ms"]),
                            symbol=str(row["symbol"]),
                            type=row["type"],
                            exchange=str(row["exchange"]),
                            data=json.loads(row["data"]),
                        )
                    )
            except Exception:
                logger.exception("Parquet get_latest error for %s", file_path)

        out.sort(key=lambda md: md.timestamp_ms, reverse=True)
        return out[:limit]

    async def close(self) -> None:
        # No persistent resources for file-based storage
        pass
