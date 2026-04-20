from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import asyncpg

from its_project.common.types import MarketData, MarketDataType

logger = logging.getLogger(__name__)


class TimescaleClient:
    """Enhanced TimescaleDB client for aggregated data storage."""
    
    def __init__(
        self,
        dsn: str,
        pool_size: int = 10,
        command_timeout: float = 30.0,
        max_inactive_connection_lifetime: float = 300.0
    ) -> None:
        self.dsn = dsn
        self.pool_size = pool_size
        self.command_timeout = command_timeout
        self.max_inactive_connection_lifetime = max_inactive_connection_lifetime
        self.pool: Optional[asyncpg.pool.Pool] = None
    
    async def connect(self) -> None:
        """Establish connection pool."""
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=1,
            max_size=self.pool_size,
            command_timeout=self.command_timeout,
            max_inactive_connection_lifetime=self.max_inactive_connection_lifetime
        )
        logger.info("Connected to TimescaleDB")
    
    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Closed TimescaleDB connection")
    
    async def initialize_schema(self) -> None:
        """Initialize database schema with hypertables."""
        if not self.pool:
            raise RuntimeError("Not connected to database")
        
        async with self.pool.acquire() as conn:
            # Create market_data_aggregated table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data_aggregated (
                    timestamp_ms BIGINT NOT NULL,
                    symbol TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    data JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp_ms, symbol, data_type)
                );
            """)
            
            # Convert to hypertable
            await conn.execute("""
                SELECT create_hypertable('market_data_aggregated', 'timestamp_ms', 
                                        chunk_time_interval => INTERVAL '1 hour');
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_agg_symbol 
                ON market_data_aggregated (symbol, timestamp_ms DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_agg_type 
                ON market_data_aggregated (data_type, timestamp_ms DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_agg_created 
                ON market_data_aggregated (created_at DESC);
            """)
            
            # Set up compression policy
            await conn.execute("""
                ALTER TABLE market_data_aggregated SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol'
                );
            """)
            
            # Add compression policy for old data
            await conn.execute("""
                SELECT add_compression_policy('market_data_aggregated', 
                                            INTERVAL '7 days');
            """)
            
            logger.info("TimescaleDB schema initialized")
    
    async def write_aggregated(self, data: MarketData) -> None:
        """Write aggregated market data."""
        if not self.pool:
            raise RuntimeError("Not connected to database")
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO market_data_aggregated 
                (timestamp_ms, symbol, data_type, exchange, data)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (timestamp_ms, symbol, data_type) 
                DO UPDATE SET 
                    data = EXCLUDED.data,
                    created_at = NOW();
            """,
                data.timestamp_ms,
                data.symbol,
                data.type.value,
                data.exchange,
                dict(data.data)
            )
    
    async def write_batch_aggregated(self, data_list: Sequence[MarketData]) -> int:
        """Write batch of aggregated market data."""
        if not self.pool:
            raise RuntimeError("Not connected to database")
        
        if not data_list:
            return 0
        
        async with self.pool.acquire() as conn:
            records = [
                (
                    md.timestamp_ms,
                    md.symbol,
                    md.type.value,
                    md.exchange,
                    dict(md.data)
                )
                for md in data_list
            ]
            
            await conn.executemany("""
                INSERT INTO market_data_aggregated 
                (timestamp_ms, symbol, data_type, exchange, data)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (timestamp_ms, symbol, data_type) 
                DO UPDATE SET 
                    data = EXCLUDED.data,
                    created_at = NOW();
            """, records)
            
            return len(records)
    
    async def read_aggregated(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        data_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MarketData]:
        """Read aggregated market data."""
        if not self.pool:
            raise RuntimeError("Not connected to database")
        
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        query = """
            SELECT timestamp_ms, symbol, data_type, exchange, data
            FROM market_data_aggregated
            WHERE symbol = $1 
              AND timestamp_ms >= $2 
              AND timestamp_ms <= $3
        """
        params = [symbol, start_ms, end_ms]
        
        if data_type:
            query += " AND data_type = $4"
            params.append(data_type)
        
        query += " ORDER BY timestamp_ms ASC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            results = []
            for row in rows:
                md = MarketData(
                    timestamp_ms=row['timestamp_ms'],
                    symbol=row['symbol'],
                    type=MarketDataType(row['data_type']),
                    exchange=row['exchange'],
                    data=row['data']
                )
                results.append(md)
            
            return results
    
    async def get_latest_aggregated(
        self,
        symbol: str,
        data_type: Optional[str] = None,
        limit: int = 10
    ) -> List[MarketData]:
        """Get latest aggregated data for symbol."""
        if not self.pool:
            raise RuntimeError("Not connected to database")
        
        query = """
            SELECT timestamp_ms, symbol, data_type, exchange, data
            FROM market_data_aggregated
            WHERE symbol = $1
        """
        params = [symbol]
        
        if data_type:
            query += " AND data_type = $2"
            params.append(data_type)
        
        query += " ORDER BY timestamp_ms DESC LIMIT $3"
        params.append(limit)
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            results = []
            for row in rows:
                md = MarketData(
                    timestamp_ms=row['timestamp_ms'],
                    symbol=row['symbol'],
                    type=MarketDataType(row['data_type']),
                    exchange=row['exchange'],
                    data=row['data']
                )
                results.append(md)
            
            return results
    
    async def aggregate_ohlcv(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m"
    ) -> List[Dict[str, Any]]:
        """Aggregate trade data into OHLCV bars."""
        if not self.pool:
            raise RuntimeError("Not connected to database")
        
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        # Map interval to PostgreSQL interval
        interval_map = {
            "1m": "INTERVAL '1 minute'",
            "5m": "INTERVAL '5 minutes'",
            "15m": "INTERVAL '15 minutes'",
            "1h": "INTERVAL '1 hour'",
            "1d": "INTERVAL '1 day'"
        }
        
        pg_interval = interval_map.get(interval, "INTERVAL '1 minute'")
        
        async with self.pool.acquire() as conn:
            query = f"""
                SELECT 
                    time_bucket({pg_interval}, timestamp_ms) AS bucket,
                    MIN((data->>'p')::FLOAT) AS open,
                    MIN((data->>'p')::FLOAT) AS low,
                    MAX((data->>'p')::FLOAT) AS high,
                    MAX((data->>'p')::FLOAT) AS close,
                    SUM((data->>'q')::FLOAT) AS volume
                FROM market_data_aggregated
                WHERE symbol = $1 
                  AND data_type = 'trade'
                  AND timestamp_ms >= $2 
                  AND timestamp_ms <= $3
                GROUP BY bucket
                ORDER BY bucket ASC
            """
            
            rows = await conn.fetch(query, symbol, start_ms, end_ms)
            
            results = []
            for row in rows:
                results.append({
                    'timestamp': row['bucket'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                })
            
            return results
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.pool:
            raise RuntimeError("Not connected to database")
        
        async with self.pool.acquire() as conn:
            # Table size
            size_result = await conn.fetch("""
                SELECT 
                    pg_size_pretty(pg_total_relation_size('market_data_aggregated')) AS total_size,
                    pg_size_pretty(pg_relation_size('market_data_aggregated')) AS table_size
            """)
            
            # Row count
            count_result = await conn.fetch("""
                SELECT COUNT(*) as total_rows FROM market_data_aggregated
            """)
            
            # Oldest and newest records
            time_result = await conn.fetch("""
                SELECT 
                    MIN(timestamp_ms) as oldest_ts,
                    MAX(timestamp_ms) as newest_ts
                FROM market_data_aggregated
            """)
            
            # Unique symbols
            symbols_result = await conn.fetch("""
                SELECT COUNT(DISTINCT symbol) as unique_symbols
                FROM market_data_aggregated
            """)
            
            return {
                'total_size': size_result[0]['total_size'],
                'table_size': size_result[0]['table_size'],
                'total_rows': count_result[0]['total_rows'],
                'oldest_timestamp': time_result[0]['oldest_ts'],
                'newest_timestamp': time_result[0]['newest_ts'],
                'unique_symbols': symbols_result[0]['unique_symbols']
            }
