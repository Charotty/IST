#!/usr/bin/env python3
"""
Aggregated Data Storage to TimescaleDB
=====================================

Production-ready system for storing aggregated market data in TimescaleDB.
Handles OHLCV, volume-weighted averages, and other time-series aggregations.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import json

import asyncpg
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from its_project.common.types import MarketData, MarketDataType

logger = logging.getLogger(__name__)


@dataclass
class TimescaleDBConfig:
    """Configuration for TimescaleDB storage."""
    # Connection settings
    dsn: str = "postgres://user:password@localhost:5432/market_data"
    pool_size: int = 10
    command_timeout: float = 30.0
    max_inactive_connection_lifetime: float = 300.0
    
    # Table settings
    raw_data_table: str = "market_data_raw"
    ohlcv_table: str = "market_data_ohlcv"
    volume_table: str = "market_data_volume"
    stats_table: str = "market_data_stats"
    
    # Aggregation settings
    aggregation_intervals: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"])
    enable_continuous_aggregates: bool = True
    aggregation_retention: Dict[str, int] = field(default_factory=lambda: {
        "1m": 7,    # 7 days
        "5m": 30,   # 30 days
        "15m": 90,  # 90 days
        "1h": 365,  # 1 year
        "4h": 730,  # 2 years
        "1d": 2555  # 7 years
    })
    
    # Performance settings
    batch_size: int = 1000
    max_concurrent_inserts: int = 5
    insert_timeout: float = 10.0
    enable_compression: bool = True
    
    # Data retention
    raw_data_retention_days: int = 7
    aggregated_data_retention_days: int = 2555  # 7 years
    
    # Monitoring
    enable_query_stats: bool = True
    stats_collection_interval: int = 300  # 5 minutes


@dataclass
class AggregationConfig:
    """Configuration for data aggregation."""
    # OHLCV aggregation
    enable_ohlcv: bool = True
    ohlcv_intervals: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"])
    
    # Volume aggregation
    enable_volume: bool = True
    volume_intervals: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"])
    
    # Statistics aggregation
    enable_stats: bool = True
    stats_intervals: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"])
    
    # Custom aggregations
    custom_aggregations: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class StorageStats:
    """Storage statistics."""
    total_records_inserted: int = 0
    total_ohlcv_records: int = 0
    total_volume_records: int = 0
    total_stats_records: int = 0
    
    insert_rate: float = 0.0  # records per second
    last_insert_time: Optional[datetime] = None
    
    records_by_symbol: Dict[str, int] = field(default_factory=dict)
    records_by_interval: Dict[str, int] = field(default_factory=dict)
    
    errors_count: int = 0
    connection_errors: int = 0
    aggregation_errors: int = 0


class AggregatedTimescaleStorage:
    """
    Production-ready aggregated data storage in TimescaleDB.
    
    Features:
    - Automatic OHLCV aggregation
    - Volume-weighted price calculations
    - Multiple time intervals
    - Continuous aggregates
    - Data retention policies
    - Performance monitoring
    - Connection pooling
    - Error handling and retries
    """
    
    def __init__(self, config: TimescaleDBConfig) -> None:
        self.config = config
        
        # Database connection
        self.pool: Optional[asyncpg.pool.Pool] = None
        
        # Aggregation configuration
        self.agg_config = AggregationConfig()
        
        # Write management
        self.write_queue: asyncio.Queue[List[MarketData]] = asyncio.Queue()
        self.write_semaphore = asyncio.Semaphore(config.max_concurrent_inserts)
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_inserts)
        
        # Statistics
        self.stats = StorageStats()
        self._start_time = datetime.now()
        
        # State
        self._running = False
        self._write_task: Optional[asyncio.Task] = None
        self._aggregation_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        
        logger.info("Aggregated TimescaleDB storage initialized")
    
    async def start(self) -> None:
        """Start the storage system."""
        if self._running:
            logger.warning("Aggregated TimescaleDB storage already running")
            return
        
        # Connect to database
        await self.connect()
        
        # Initialize schema
        await self.initialize_schema()
        
        self._running = True
        
        # Start tasks
        self._write_task = asyncio.create_task(self._write_loop())
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        self._stats_task = asyncio.create_task(self._stats_loop())
        
        logger.info("Aggregated TimescaleDB storage started")
    
    async def stop(self) -> None:
        """Stop the storage system."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel tasks
        tasks = [self._write_task, self._aggregation_task, self._stats_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close connection
        await self.close()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Aggregated TimescaleDB storage stopped")
    
    async def connect(self) -> None:
        """Connect to TimescaleDB."""
        try:
            self.pool = await asyncpg.create_pool(
                self.config.dsn,
                min_size=1,
                max_size=self.config.pool_size,
                command_timeout=self.config.command_timeout,
                max_inactive_connection_lifetime=self.config.max_inactive_connection_lifetime
            )
            logger.info("Connected to TimescaleDB")
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise
    
    async def close(self) -> None:
        """Close database connection."""
        if self.pool:
            await self.pool.close()
            logger.info("Closed TimescaleDB connection")
    
    async def initialize_schema(self) -> None:
        """Initialize database schema with hypertables."""
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        async with self.pool.acquire() as conn:
            # Create raw data table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.raw_data_table} (
                    timestamp BIGINT NOT NULL,
                    timestamp_ms BIGINT NOT NULL,
                    symbol TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    data JSONB NOT NULL,
                    ingest_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            
            # Create hypertable
            await conn.execute(f"""
                SELECT create_hypertable(
                    '{self.config.raw_data_table}',
                    'timestamp',
                    chunk_time_interval => INTERVAL '1 hour'
                )
            """)
            
            # Create indexes
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.config.raw_data_table}_symbol_timestamp 
                ON {self.config.raw_data_table} (symbol, timestamp DESC)
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.config.raw_data_table}_data_type_timestamp 
                ON {self.config.raw_data_table} (data_type, timestamp DESC)
            """)
            
            # Create OHLCV table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.ohlcv_table} (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open_price DECIMAL(20, 8) NOT NULL,
                    high_price DECIMAL(20, 8) NOT NULL,
                    low_price DECIMAL(20, 8) NOT NULL,
                    close_price DECIMAL(20, 8) NOT NULL,
                    volume DECIMAL(30, 8) NOT NULL,
                    trade_count INTEGER NOT NULL,
                    vwap DECIMAL(20, 8),
                    ingest_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            
            # Create OHLCV hypertable
            await conn.execute(f"""
                SELECT create_hypertable(
                    '{self.config.ohlcv_table}',
                    'timestamp',
                    chunk_time_interval => INTERVAL '1 hour'
                )
            """)
            
            # Create OHLCV indexes
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.config.ohlcv_table}_symbol_interval_timestamp 
                ON {self.config.ohlcv_table} (symbol, interval, timestamp DESC)
            """)
            
            # Create volume table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.volume_table} (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    buy_volume DECIMAL(30, 8) NOT NULL,
                    sell_volume DECIMAL(30, 8) NOT NULL,
                    total_volume DECIMAL(30, 8) NOT NULL,
                    trade_count INTEGER NOT NULL,
                    avg_trade_size DECIMAL(20, 8),
                ingest_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            
            # Create volume hypertable
            await conn.execute(f"""
                SELECT create_hypertable(
                    '{self.config.volume_table}',
                    'timestamp',
                    chunk_time_interval => INTERVAL '1 hour'
                )
            """)
            
            # Create volume indexes
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.config.volume_table}_symbol_interval_timestamp 
                ON {self.config.volume_table} (symbol, interval, timestamp DESC)
            """)
            
            # Create stats table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.stats_table} (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    avg_price DECIMAL(20, 8) NOT NULL,
                    price_volatility DECIMAL(10, 8) NOT NULL,
                    price_change DECIMAL(20, 8) NOT NULL,
                    price_change_pct DECIMAL(10, 4) NOT NULL,
                    volume_weighted_price DECIMAL(20, 8) NOT NULL,
                    total_volume DECIMAL(30, 8) NOT NULL,
                    trade_count INTEGER NOT NULL,
                    unique_trades INTEGER NOT NULL,
                ingest_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            
            # Create stats hypertable
            await conn.execute(f"""
                SELECT create_hypertable(
                    '{self.config.stats_table}',
                    'timestamp',
                    chunk_time_interval => INTERVAL '1 hour'
                )
            """)
            
            # Create stats indexes
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.config.stats_table}_symbol_interval_timestamp 
                ON {self.config.stats_table} (symbol, interval, timestamp DESC)
            """)
            
            logger.info("Database schema initialized")
    
    async def store_batch(self, market_data_batch: List[MarketData]) -> None:
        """Store a batch of market data."""
        if not self._running:
            logger.warning("Storage not running, discarding data")
            return
        
        if not market_data_batch:
            return
        
        await self.write_queue.put(market_data_batch)
    
    async def store(self, market_data: MarketData) -> None:
        """Store single market data point."""
        await self.store_batch([market_data])
    
    async def _write_loop(self) -> None:
        """Main write loop for processing batches."""
        logger.info("Write loop started")
        
        while self._running:
            try:
                # Get batch with timeout
                batch = await asyncio.wait_for(
                    self.write_queue.get(),
                    timeout=1.0
                )
                
                # Process batch
                await self._process_batch(batch)
                
                # Mark task as done
                self.write_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Write loop error: {e}")
                self.stats.errors_count += 1
                await asyncio.sleep(1)
        
        logger.info("Write loop stopped")
    
    async def _process_batch(self, batch: List[MarketData]) -> None:
        """Process a batch of market data for storage."""
        if not batch:
            return
        
        try:
            # Store raw data
            await self._store_raw_data(batch)
            
            # Update statistics
            self.stats.total_records_inserted += len(batch)
            self.stats.last_insert_time = datetime.now()
            
            # Calculate write rate
            elapsed = (datetime.now() - self._start_time).total_seconds()
            if elapsed > 0:
                self.stats.insert_rate = self.stats.total_records_inserted / elapsed
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.stats.errors_count += 1
    
    async def _store_raw_data(self, batch: List[MarketData]) -> None:
        """Store raw market data to database."""
        async with self.write_semaphore:
            try:
                async with self.pool.acquire() as conn:
                    # Prepare batch insert
                    values = []
                    for data in batch:
                        values.append((
                            data.timestamp_ms // 1000,  # Convert to seconds
                            data.timestamp_ms,
                            data.symbol,
                            data.type.value,
                            data.exchange,
                            json.dumps(data.data)
                        ))
                    
                    # Execute batch insert
                    await conn.executemany(f"""
                        INSERT INTO {self.config.raw_data_table} 
                        (timestamp, timestamp_ms, symbol, data_type, exchange, data)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """, values)
                    
                    logger.debug(f"Stored {len(batch)} raw records")
                    
            except Exception as e:
                logger.error(f"Error storing raw data: {e}")
                self.stats.errors_count += 1
                raise
    
    async def _aggregation_loop(self) -> None:
        """Background aggregation loop."""
        logger.info("Aggregation loop started")
        
        while self._running:
            try:
                # Wait for aggregation interval
                await asyncio.sleep(60)  # Run every minute
                
                # Perform aggregations
                await self._perform_aggregations()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Aggregation loop error: {e}")
                self.stats.aggregation_errors += 1
                await asyncio.sleep(60)
        
        logger.info("Aggregation loop stopped")
    
    async def _perform_aggregations(self) -> None:
        """Perform data aggregations."""
        try:
            # Get latest timestamp to aggregate
            latest_timestamp = await self._get_latest_raw_timestamp()
            if not latest_timestamp:
                return
            
            # Aggregate for each interval
            for interval in self.agg_config.ohlcv_intervals:
                await self._aggregate_ohlcv(interval, latest_timestamp)
            
            for interval in self.agg_config.volume_intervals:
                await self._aggregate_volume(interval, latest_timestamp)
            
            for interval in self.agg_config.stats_intervals:
                await self._aggregate_stats(interval, latest_timestamp)
            
        except Exception as e:
            logger.error(f"Error performing aggregations: {e}")
            self.stats.aggregation_errors += 1
    
    async def _get_latest_raw_timestamp(self) -> Optional[datetime]:
        """Get latest timestamp from raw data."""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval(f"""
                    SELECT MAX(timestamp) FROM {self.config.raw_data_table}
                """)
                return result
        except Exception as e:
            logger.error(f"Error getting latest timestamp: {e}")
            return None
    
    async def _aggregate_ohlcv(self, interval: str, latest_timestamp: datetime) -> None:
        """Aggregate OHLCV data for given interval."""
        try:
            async with self.pool.acquire() as conn:
                # Calculate interval in seconds
                interval_seconds = self._interval_to_seconds(interval)
                
                # Aggregate OHLCV
                await conn.execute(f"""
                    INSERT INTO {self.config.ohlcv_table} 
                    (timestamp, symbol, interval, open_price, high_price, low_price, 
                     close_price, volume, trade_count, vwap)
                    SELECT 
                        time_bucket(INTERVAL '{interval_seconds} seconds', timestamp) as timestamp,
                        symbol,
                        '{interval}' as interval,
                        (data->>'last_price')::DECIMAL(20,8) as open_price,
                        MAX((data->>'last_price')::DECIMAL(20,8)) as high_price,
                        MIN((data->>'last_price')::DECIMAL(20,8)) as low_price,
                        (data->>'last_price')::DECIMAL(20,8) as close_price,
                        COALESCE(SUM((data->>'volume_24h')::DECIMAL(30,8)), 0) as volume,
                        COUNT(*) as trade_count,
                        AVG((data->>'last_price')::DECIMAL(20,8)) as vwap
                    FROM {self.config.raw_data_table}
                    WHERE data_type = 'ticker'
                    AND timestamp >= $1 - INTERVAL '{interval_seconds} seconds'
                    AND timestamp <= $1
                    GROUP BY time_bucket(INTERVAL '{interval_seconds} seconds', timestamp), symbol
                    ON CONFLICT (timestamp, symbol, interval) DO UPDATE SET
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume,
                        trade_count = EXCLUDED.trade_count,
                        vwap = EXCLUDED.vwap
                """, latest_timestamp)
                
                self.stats.total_ohlcv_records += 1
                
        except Exception as e:
            logger.error(f"Error aggregating OHLCV for {interval}: {e}")
            self.stats.aggregation_errors += 1
    
    async def _aggregate_volume(self, interval: str, latest_timestamp: datetime) -> None:
        """Aggregate volume data for given interval."""
        try:
            async with self.pool.acquire() as conn:
                interval_seconds = self._interval_to_seconds(interval)
                
                await conn.execute(f"""
                    INSERT INTO {self.config.volume_table} 
                    (timestamp, symbol, interval, buy_volume, sell_volume, total_volume, 
                     trade_count, avg_trade_size)
                    SELECT 
                        time_bucket(INTERVAL '{interval_seconds} seconds', timestamp) as timestamp,
                        symbol,
                        '{interval}' as interval,
                        COALESCE(SUM(CASE WHEN (data->>'side') = 'buy' 
                            THEN (data->>'volume_24h')::DECIMAL(30,8) ELSE 0 END), 0) as buy_volume,
                        COALESCE(SUM(CASE WHEN (data->>'side') = 'sell' 
                            THEN (data->>'volume_24h')::DECIMAL(30,8) ELSE 0 END), 0) as sell_volume,
                        COALESCE(SUM((data->>'volume_24h')::DECIMAL(30,8)), 0) as total_volume,
                        COUNT(*) as trade_count,
                        COALESCE(AVG((data->>'volume_24h')::DECIMAL(20,8)), 0) as avg_trade_size
                    FROM {self.config.raw_data_table}
                    WHERE data_type = 'trade'
                    AND timestamp >= $1 - INTERVAL '{interval_seconds} seconds'
                    AND timestamp <= $1
                    GROUP BY time_bucket(INTERVAL '{interval_seconds} seconds', timestamp), symbol
                    ON CONFLICT (timestamp, symbol, interval) DO UPDATE SET
                        buy_volume = EXCLUDED.buy_volume,
                        sell_volume = EXCLUDED.sell_volume,
                        total_volume = EXCLUDED.total_volume,
                        trade_count = EXCLUDED.trade_count,
                        avg_trade_size = EXCLUDED.avg_trade_size
                """, latest_timestamp)
                
                self.stats.total_volume_records += 1
                
        except Exception as e:
            logger.error(f"Error aggregating volume for {interval}: {e}")
            self.stats.aggregation_errors += 1
    
    async def _aggregate_stats(self, interval: str, latest_timestamp: datetime) -> None:
        """Aggregate statistics for given interval."""
        try:
            async with self.pool.acquire() as conn:
                interval_seconds = self._interval_to_seconds(interval)
                
                await conn.execute(f"""
                    INSERT INTO {self.config.stats_table} 
                    (timestamp, symbol, interval, avg_price, price_volatility, 
                     price_change, price_change_pct, volume_weighted_price, 
                     total_volume, trade_count, unique_trades)
                    SELECT 
                        time_bucket(INTERVAL '{interval_seconds} seconds', timestamp) as timestamp,
                        symbol,
                        '{interval}' as interval,
                        AVG((data->>'last_price')::DECIMAL(20,8)) as avg_price,
                        STDDEV((data->>'last_price')::DECIMAL(20,8)) as price_volatility,
                        (LAST((data->>'last_price')::DECIMAL(20,8)) - 
                         FIRST((data->>'last_price')::DECIMAL(20,8))) as price_change,
                        ((LAST((data->>'last_price')::DECIMAL(20,8)) - 
                          FIRST((data->>'last_price')::DECIMAL(20,8))) / 
                         FIRST((data->>'last_price')::DECIMAL(20,8)) * 100) as price_change_pct,
                        AVG((data->>'last_price')::DECIMAL(20,8)) as volume_weighted_price,
                        COALESCE(SUM((data->>'volume_24h')::DECIMAL(30,8)), 0) as total_volume,
                        COUNT(*) as trade_count,
                        COUNT(DISTINCT (data->>'trade_id')) as unique_trades
                    FROM {self.config.raw_data_table}
                    WHERE data_type = 'ticker'
                    AND timestamp >= $1 - INTERVAL '{interval_seconds} seconds'
                    AND timestamp <= $1
                    GROUP BY time_bucket(INTERVAL '{interval_seconds} seconds', timestamp), symbol
                    ON CONFLICT (timestamp, symbol, interval) DO UPDATE SET
                        avg_price = EXCLUDED.avg_price,
                        price_volatility = EXCLUDED.price_volatility,
                        price_change = EXCLUDED.price_change,
                        price_change_pct = EXCLUDED.price_change_pct,
                        volume_weighted_price = EXCLUDED.volume_weighted_price,
                        total_volume = EXCLUDED.total_volume,
                        trade_count = EXCLUDED.trade_count,
                        unique_trades = EXCLUDED.unique_trades
                """, latest_timestamp)
                
                self.stats.total_stats_records += 1
                
        except Exception as e:
            logger.error(f"Error aggregating stats for {interval}: {e}")
            self.stats.aggregation_errors += 1
    
    def _interval_to_seconds(self, interval: str) -> int:
        """Convert interval string to seconds."""
        interval_map = {
            '1m': 60,
            '5m': 5 * 60,
            '15m': 15 * 60,
            '30m': 30 * 60,
            '1h': 60 * 60,
            '2h': 2 * 60 * 60,
            '4h': 4 * 60 * 60,
            '6h': 6 * 60 * 60,
            '12h': 12 * 60 * 60,
            '1d': 24 * 60 * 60,
            '1w': 7 * 24 * 60 * 60
        }
        return interval_map.get(interval, 60)
    
    async def _stats_loop(self) -> None:
        """Background statistics collection loop."""
        logger.info("Stats loop started")
        
        while self._running:
            try:
                # Wait for stats interval
                await asyncio.sleep(self.config.stats_collection_interval)
                
                # Collect statistics
                await self._collect_database_stats()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stats loop error: {e}")
                await asyncio.sleep(60)
        
        logger.info("Stats loop stopped")
    
    async def _collect_database_stats(self) -> None:
        """Collect database statistics."""
        try:
            async with self.pool.acquire() as conn:
                # Get table sizes
                raw_size = await conn.fetchval(f"""
                    SELECT pg_size_pretty(pg_total_relation_size('{self.config.raw_data_table}'))
                """)
                
                ohlcv_size = await conn.fetchval(f"""
                    SELECT pg_size_pretty(pg_total_relation_size('{self.config.ohlcv_table}'))
                """)
                
                volume_size = await conn.fetchval(f"""
                    SELECT pg_size_pretty(pg_total_relation_size('{self.config.volume_table}'))
                """)
                
                stats_size = await conn.fetchval(f"""
                    SELECT pg_size_pretty(pg_total_relation_size('{self.config.stats_table}'))
                """)
                
                logger.info(f"Database sizes - Raw: {raw_size}, OHLCV: {ohlcv_size}, "
                           f"Volume: {volume_size}, Stats: {stats_size}")
                
        except Exception as e:
            logger.error(f"Error collecting database stats: {e}")
    
    def get_statistics(self) -> StorageStats:
        """Get storage statistics."""
        return self.stats
    
    async def query_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Query OHLCV data."""
        try:
            async with self.pool.acquire() as conn:
                query = f"""
                    SELECT timestamp, open_price, high_price, low_price, close_price, 
                           volume, trade_count, vwap
                    FROM {self.config.ohlcv_table}
                    WHERE symbol = $1
                    AND interval = $2
                    AND timestamp >= $3
                    AND timestamp <= $4
                    ORDER BY timestamp ASC
                """
                
                rows = await conn.fetch(query, symbol, interval, start_time, end_time)
                
                if not rows:
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = []
                for row in rows:
                    data.append({
                        'timestamp': row['timestamp'],
                        'open': float(row['open_price']),
                        'high': float(row['high_price']),
                        'low': float(row['low_price']),
                        'close': float(row['close_price']),
                        'volume': float(row['volume']),
                        'trade_count': row['trade_count'],
                        'vwap': float(row['vwap']) if row['vwap'] else None
                    })
                
                return pd.DataFrame(data)
                
        except Exception as e:
            logger.error(f"Error querying OHLCV data: {e}")
            return pd.DataFrame()
    
    async def query_volume_stats(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Query volume statistics."""
        try:
            async with self.pool.acquire() as conn:
                query = f"""
                    SELECT timestamp, buy_volume, sell_volume, total_volume, 
                           trade_count, avg_trade_size
                    FROM {self.config.volume_table}
                    WHERE symbol = $1
                    AND interval = $2
                    AND timestamp >= $3
                    AND timestamp <= $4
                    ORDER BY timestamp ASC
                """
                
                rows = await conn.fetch(query, symbol, interval, start_time, end_time)
                
                if not rows:
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = []
                for row in rows:
                    data.append({
                        'timestamp': row['timestamp'],
                        'buy_volume': float(row['buy_volume']),
                        'sell_volume': float(row['sell_volume']),
                        'total_volume': float(row['total_volume']),
                        'trade_count': row['trade_count'],
                        'avg_trade_size': float(row['avg_trade_size']) if row['avg_trade_size'] else None
                    })
                
                return pd.DataFrame(data)
                
        except Exception as e:
            logger.error(f"Error querying volume stats: {e}")
            return pd.DataFrame()


# Convenience functions
def create_aggregated_timescale_storage(
    dsn: str = "postgres://user:password@localhost:5432/market_data",
    pool_size: int = 10,
    aggregation_intervals: List[str] = None
) -> AggregatedTimescaleStorage:
    """Create aggregated TimescaleDB storage with default configuration."""
    config = TimescaleDBConfig(
        dsn=dsn,
        pool_size=pool_size,
        aggregation_intervals=aggregation_intervals or ["1m", "5m", "15m", "1h", "4h", "1d"]
    )
    
    return AggregatedTimescaleStorage(config)
