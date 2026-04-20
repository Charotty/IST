from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

from its_project.common.types import MarketData, MarketDataType
from its_project.storage.parquet_store import ParquetStore
from its_project.storage.timescale_client import TimescaleClient

logger = logging.getLogger(__name__)


class DataLoader:
    """Unified data access layer for both Parquet and TimescaleDB storage."""
    
    def __init__(
        self,
        parquet_base_path: str,
        timescale_dsn: str,
        parquet_priority: bool = True  # Try Parquet first for raw data
    ) -> None:
        self.parquet_store = ParquetStore(parquet_base_path)
        self.timescale_client = TimescaleClient(timescale_dsn)
        self.parquet_priority = parquet_priority
        
        self._connected = False
    
    async def connect(self) -> None:
        """Initialize connections to storage backends."""
        await self.timescale_client.connect()
        await self.timescale_client.initialize_schema()
        self._connected = True
        logger.info("DataLoader connected to both storage backends")
    
    async def close(self) -> None:
        """Close storage connections."""
        await self.timescale_client.close()
        self._connected = False
        logger.info("DataLoader connections closed")
    
    async def read_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        data_type: Optional[str] = None,
        use_raw: bool = False,
        limit: Optional[int] = None
    ) -> List[MarketData]:
        """
        Read data with automatic backend selection.
        
        Args:
            symbol: Trading symbol
            start_time: Start time
            end_time: End time
            data_type: Data type filter
            use_raw: If True, prefer raw data from Parquet
            limit: Maximum number of records
        """
        if not self._connected:
            raise RuntimeError("DataLoader not connected")
        
        # Decide which backend to use
        if use_raw or (self.parquet_priority and data_type in ["orderbook", "trade"]):
            # Use Parquet for raw/high-frequency data
            try:
                data = self.parquet_store.read_raw_lob(
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time,
                    filters=[("data_type", "=", data_type)] if data_type else None
                )
                
                if data:
                    logger.debug(f"Read {len(data)} records from Parquet")
                    return data[:limit] if limit else data
                    
            except Exception as e:
                logger.warning(f"Parquet read failed, falling back to TimescaleDB: {e}")
        
        # Use TimescaleDB for aggregated data or fallback
        try:
            data = await self.timescale_client.read_aggregated(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                data_type=data_type,
                limit=limit
            )
            
            logger.debug(f"Read {len(data)} records from TimescaleDB")
            return data
            
        except Exception as e:
            logger.error(f"Failed to read from both backends: {e}")
            return []
    
    async def write_data(
        self,
        data: Union[MarketData, Sequence[MarketData]],
        use_raw: bool = False
    ) -> int:
        """
        Write data with automatic backend selection.
        
        Args:
            data: Single MarketData or sequence
            use_raw: If True, write to Parquet (raw), else TimescaleDB (aggregated)
        """
        if not self._connected:
            raise RuntimeError("DataLoader not connected")
        
        # Convert single item to sequence
        if isinstance(data, MarketData):
            data = [data]
        
        if not data:
            return 0
        
        written = 0
        
        if use_raw:
            # Write raw data to Parquet
            try:
                self.parquet_store.write_batch_raw_lob(data)
                written = len(data)
                logger.debug(f"Wrote {written} records to Parquet")
            except Exception as e:
                logger.error(f"Failed to write to Parquet: {e}")
        else:
            # Write aggregated data to TimescaleDB
            try:
                written = await self.timescale_client.write_batch_aggregated(data)
                logger.debug(f"Wrote {written} records to TimescaleDB")
            except Exception as e:
                logger.error(f"Failed to write to TimescaleDB: {e}")
        
        return written
    
    async def get_latest_data(
        self,
        symbol: str,
        data_type: Optional[str] = None,
        use_raw: bool = False,
        limit: int = 10
    ) -> List[MarketData]:
        """Get latest data for symbol."""
        if not self._connected:
            raise RuntimeError("DataLoader not connected")
        
        if use_raw or (self.parquet_priority and data_type in ["orderbook", "trade"]):
            # Try Parquet first
            try:
                data = self.parquet_store.read_latest_raw_lob(symbol=symbol, limit=limit)
                if data:
                    return data
            except Exception as e:
                logger.warning(f"Parquet latest read failed: {e}")
        
        # Fallback to TimescaleDB
        try:
            return await self.timescale_client.get_latest_aggregated(
                symbol=symbol,
                data_type=data_type,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Failed to get latest data: {e}")
            return []
    
    async def get_ohlcv_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m"
    ) -> List[Dict[str, Any]]:
        """Get OHLCV aggregated data."""
        if not self._connected:
            raise RuntimeError("DataLoader not connected")
        
        return await self.timescale_client.aggregate_ohlcv(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            interval=interval
        )
    
    async def create_dataset_version(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        version_name: str,
        description: str = ""
    ) -> str:
        """Create a versioned dataset snapshot."""
        if not self._connected:
            raise RuntimeError("DataLoader not connected")
        
        # Read all data for the period
        all_data = await self.read_data(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        # Create version metadata
        version_metadata = {
            "version_name": version_name,
            "symbol": symbol,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "record_count": len(all_data),
            "description": description,
            "created_at": datetime.now().isoformat()
        }
        
        # Store version metadata in TimescaleDB
        try:
            async with self.timescale_client.pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS dataset_versions (
                        version_id SERIAL PRIMARY KEY,
                        version_name TEXT UNIQUE NOT NULL,
                        symbol TEXT NOT NULL,
                        start_time TIMESTAMPTZ NOT NULL,
                        end_time TIMESTAMPTZ NOT NULL,
                        record_count INTEGER NOT NULL,
                        description TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                
                await conn.execute("""
                    INSERT INTO dataset_versions 
                    (version_name, symbol, start_time, end_time, record_count, description)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (version_name) DO UPDATE SET
                        record_count = EXCLUDED.record_count,
                        created_at = NOW();
                """,
                    version_name,
                    symbol,
                    start_time,
                    end_time,
                    len(all_data),
                    description
                )
                
                version_id = await conn.fetchval(
                    "SELECT version_id FROM dataset_versions WHERE version_name = $1",
                    version_name
                )
                
                logger.info(f"Created dataset version {version_name} with ID {version_id}")
                return f"{version_id}:{version_name}"
        
        except Exception as e:
            logger.error(f"Failed to create dataset version: {e}")
            raise
    
    async def list_dataset_versions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available dataset versions."""
        if not self._connected:
            raise RuntimeError("DataLoader not connected")
        
        try:
            async with self.timescale_client.pool.acquire() as conn:
                query = """
                    SELECT version_id, version_name, symbol, start_time, end_time, 
                           record_count, description, created_at
                    FROM dataset_versions
                """
                params = []
                
                if symbol:
                    query += " WHERE symbol = $1 ORDER BY created_at DESC"
                    params.append(symbol)
                else:
                    query += " ORDER BY created_at DESC"
                
                rows = await conn.fetch(query, *params)
                
                return [
                    {
                        "version_id": row["version_id"],
                        "version_name": row["version_name"],
                        "symbol": row["symbol"],
                        "start_time": row["start_time"],
                        "end_time": row["end_time"],
                        "record_count": row["record_count"],
                        "description": row["description"],
                        "created_at": row["created_at"]
                    }
                    for row in rows
                ]
        
        except Exception as e:
            logger.error(f"Failed to list dataset versions: {e}")
            return []
    
    async def load_dataset_version(self, version_name: str) -> List[MarketData]:
        """Load data from a specific dataset version."""
        if not self._connected:
            raise RuntimeError("DataLoader not connected")
        
        try:
            async with self.timescale_client.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT symbol, start_time, end_time
                    FROM dataset_versions
                    WHERE version_name = $1
                """, version_name)
                
                if not row:
                    raise ValueError(f"Dataset version {version_name} not found")
                
                return await self.read_data(
                    symbol=row["symbol"],
                    start_time=row["start_time"],
                    end_time=row["end_time"]
                )
        
        except Exception as e:
            logger.error(f"Failed to load dataset version {version_name}: {e}")
            raise
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        stats = {
            "parquet": self.parquet_store.get_statistics(),
            "timescale": None
        }
        
        # Get TimescaleDB stats
        try:
            import asyncio
            if self._connected:
                stats["timescale"] = asyncio.run(self.timescale_client.get_statistics())
        except Exception as e:
            logger.error(f"Failed to get TimescaleDB stats: {e}")
        
        return stats
    
    async def optimize_storage(self) -> None:
        """Optimize both storage backends."""
        logger.info("Starting storage optimization...")
        
        # Optimize Parquet
        try:
            self.parquet_store.optimize_dataset()
            self.parquet_store.cleanup_old_data()
            logger.info("Parquet optimization completed")
        except Exception as e:
            logger.error(f"Parquet optimization failed: {e}")
        
        # TimescaleDB optimization is handled by automatic policies
        logger.info("Storage optimization completed")
