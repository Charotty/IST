#!/usr/bin/env python3
"""
Raw Data Storage to Parquet
===========================

Production-ready system for storing raw market data in Parquet format.
Handles high-frequency data with efficient partitioning and compression.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

from its_project.common.types import MarketData, MarketDataType

logger = logging.getLogger(__name__)


@dataclass
class ParquetStorageConfig:
    """Configuration for Parquet storage."""
    # Storage paths
    base_path: str = "data/parquet/raw"
    temp_path: str = "data/parquet/temp"
    
    # Partitioning
    partition_cols: List[str] = field(default_factory=lambda: ["symbol", "date", "data_type"])
    partition_format: str = "year={year}/month={month}/day={day}"
    
    # File settings
    file_format: str = "parquet"
    compression: str = "snappy"  # snappy, gzip, brotli, lz4
    row_group_size: int = 100000
    max_file_size_mb: int = 100
    
    # Performance settings
    write_batch_size: int = 10000
    max_concurrent_writes: int = 4
    write_timeout: float = 30.0
    
    # Data retention
    retention_days: int = 365
    cleanup_interval_hours: int = 24
    
    # Schema evolution
    enable_schema_evolution: bool = True
    schema_validation: bool = True


@dataclass
class StorageStats:
    """Storage statistics."""
    total_files_written: int = 0
    total_records_written: int = 0
    total_size_mb: float = 0.0
    write_rate: float = 0.0  # records per second
    last_write_time: Optional[datetime] = None
    files_by_symbol: Dict[str, int] = field(default_factory=dict)
    files_by_data_type: Dict[str, int] = field(default_factory=dict)
    errors_count: int = 0


class RawParquetStorage:
    """
    High-performance raw data storage system using Parquet format.
    
    Features:
    - Automatic partitioning by symbol, date, and data type
    - Efficient compression with Snappy
    - Schema validation and evolution
    - Concurrent writing with backpressure control
    - Automatic cleanup and retention
    - Performance monitoring and statistics
    """
    
    def __init__(self, config: ParquetStorageConfig) -> None:
        self.config = config
        
        # Storage paths
        self.base_path = Path(config.base_path)
        self.temp_path = Path(config.temp_path)
        
        # Ensure directories exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)
        
        # Schema definition
        self.schema = self._create_schema()
        
        # Write management
        self.write_queue: asyncio.Queue[List[MarketData]] = asyncio.Queue()
        self.write_semaphore = asyncio.Semaphore(config.max_concurrent_writes)
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_writes)
        
        # Statistics
        self.stats = StorageStats()
        self._start_time = datetime.now()
        
        # State
        self._running = False
        self._write_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"Raw Parquet storage initialized at {config.base_path}")
    
    def _create_schema(self) -> pa.Schema:
        """Create Arrow schema for market data."""
        return pa.schema([
            pa.field("timestamp_ms", pa.int64(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("data_type", pa.string(), nullable=False),
            pa.field("exchange", pa.string(), nullable=False),
            pa.field("data", pa.struct([
                # Common fields
                pa.field("_kind", pa.string()),
                pa.field("_ts_recv_ms", pa.int64()),
                
                # Ticker fields
                pa.field("last_price", pa.float64()),
                pa.field("best_bid", pa.float64()),
                pa.field("best_ask", pa.float64()),
                pa.field("bid_size", pa.float64()),
                pa.field("ask_size", pa.float64()),
                pa.field("volume_24h", pa.float64()),
                pa.field("high_24h", pa.float64()),
                pa.field("low_24h", pa.float64()),
                pa.field("open_24h", pa.float64()),
                pa.field("change_24h", pa.float64()),
                pa.field("change_pct_24h", pa.float64()),
                
                # Orderbook fields
                pa.field("bids", pa.list_(pa.list_(pa.string()))),
                pa.field("asks", pa.list_(pa.list_(pa.string()))),
                pa.field("lastUpdateId", pa.int64()),
                pa.field("U", pa.int64()),
                pa.field("u", pa.int64()),
                
                # Trade fields
                pa.field("trades", pa.list_(pa.struct([
                    pa.field("trade_id", pa.string()),
                    pa.field("price", pa.float64()),
                    pa.field("size", pa.float64()),
                    pa.field("side", pa.string()),
                    pa.field("timestamp", pa.int64())
                ]))),
            
            # Metadata
            pa.field("ingest_time", pa.timestamp('us'), nullable=False),
            pa.field("partition_date", pa.date32(), nullable=False)
        ])
    
    async def start(self) -> None:
        """Start the storage system."""
        if self._running:
            logger.warning("Raw Parquet storage already running")
            return
        
        self._running = True
        
        # Start write task
        self._write_task = asyncio.create_task(self._write_loop())
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Raw Parquet storage started")
    
    async def stop(self) -> None:
        """Stop the storage system."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel tasks
        if self._write_task and not self._write_task.done():
            self._write_task.cancel()
            try:
                await self._write_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Raw Parquet storage stopped")
    
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
        
        # Group by symbol and data type for partitioning
        grouped_data = self._group_by_partition(batch)
        
        # Write each group concurrently
        write_tasks = []
        for (symbol, data_type), data_list in grouped_data.items():
            task = asyncio.create_task(
                self._write_partition(symbol, data_type, data_list)
            )
            write_tasks.append(task)
        
        # Wait for all writes to complete
        await asyncio.gather(*write_tasks, return_exceptions=True)
    
    def _group_by_partition(self, batch: List[MarketData]) -> Dict[tuple, List[MarketData]]:
        """Group data by partition key."""
        grouped = {}
        
        for data in batch:
            partition_key = (data.symbol, data.type.value)
            if partition_key not in grouped:
                grouped[partition_key] = []
            grouped[partition_key].append(data)
        
        return grouped
    
    async def _write_partition(self, symbol: str, data_type: str, data_list: List[MarketData]) -> None:
        """Write data for a specific partition."""
        async with self.write_semaphore:
            try:
                # Convert to DataFrame
                df = self._convert_to_dataframe(data_list, symbol, data_type)
                
                if df.empty:
                    return
                
                # Write to Parquet
                file_path, file_size = await self._write_dataframe(df, symbol, data_type)
                
                # Update statistics
                self._update_stats(len(data_list), file_size, symbol, data_type)
                
                logger.debug(f"Wrote {len(data_list)} records to {file_path}")
                
            except Exception as e:
                logger.error(f"Error writing partition {symbol}/{data_type}: {e}")
                self.stats.errors_count += 1
    
    def _convert_to_dataframe(self, data_list: List[MarketData], symbol: str, data_type: str) -> pd.DataFrame:
        """Convert market data to pandas DataFrame."""
        rows = []
        
        for data in data_list:
            # Extract common fields
            row = {
                "timestamp_ms": data.timestamp_ms,
                "symbol": data.symbol,
                "data_type": data.type.value,
                "exchange": data.exchange,
                "ingest_time": datetime.now(),
                "partition_date": datetime.fromtimestamp(data.timestamp_ms / 1000).date()
            }
            
            # Extract data fields based on type
            if data.type == MarketDataType.TICKER:
                row.update({
                    "data.last_price": data.data.get("last_price"),
                    "data.best_bid": data.data.get("best_bid"),
                    "data.best_ask": data.data.get("best_ask"),
                    "data.bid_size": data.data.get("bid_size"),
                    "data.ask_size": data.data.get("ask_size"),
                    "data.volume_24h": data.data.get("volume_24h"),
                    "data.high_24h": data.data.get("high_24h"),
                    "data.low_24h": data.data.get("low_24h"),
                    "data.open_24h": data.data.get("open_24h"),
                    "data.change_24h": data.data.get("change_24h"),
                    "data.change_pct_24h": data.data.get("change_pct_24h"),
                    "data._kind": data.data.get("_kind", "ticker"),
                    "data._ts_recv_ms": data.data.get("_ts_recv_ms")
                })
            
            elif data.type == MarketDataType.ORDERBOOK:
                row.update({
                    "data.bids": json.dumps(data.data.get("bids", [])),
                    "data.asks": json.dumps(data.data.get("asks", [])),
                    "data.lastUpdateId": data.data.get("lastUpdateId"),
                    "data.U": data.data.get("U"),
                    "data.u": data.data.get("u"),
                    "data._kind": data.data.get("_kind", "orderbook"),
                    "data._ts_recv_ms": data.data.get("_ts_recv_ms")
                })
            
            elif data.type == MarketDataType.TRADE:
                row.update({
                    "data.trades": json.dumps(data.data.get("trades", [])),
                    "data._kind": data.data.get("_kind", "trade"),
                    "data._ts_recv_ms": data.data.get("_ts_recv_ms")
                })
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    async def _write_dataframe(self, df: pd.DataFrame, symbol: str, data_type: str) -> tuple[Path, int]:
        """Write DataFrame to Parquet file."""
        # Convert to Arrow Table
        table = pa.Table.from_pandas(df, schema=self.schema, safe=True)
        
        # Determine partition path
        date = df["partition_date"].iloc[0]
        partition_path = self._get_partition_path(symbol, data_type, date)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{symbol}_{data_type}_{timestamp}.parquet"
        file_path = partition_path / filename
        
        # Write to temporary file first
        temp_file = self.temp_path / f"temp_{filename}"
        
        def write_parquet():
            pq.write_table(
                table,
                temp_file,
                compression=self.config.compression,
                row_group_size=self.config.row_group_size,
                write_statistics=True
            )
        
        # Write in thread pool
        await asyncio.get_event_loop().run_in_executor(
            self.executor, write_parquet
        )
        
        # Move to final location
        temp_file.rename(file_path)
        
        # Get file size
        file_size = file_path.stat().st_size
        
        return file_path, file_size
    
    def _get_partition_path(self, symbol: str, data_type: str, date: datetime) -> Path:
        """Get partition path for given symbol, data type, and date."""
        year = date.year
        month = date.month
        day = date.day
        
        partition_dir = self.base_path / f"symbol={symbol}" / f"date={year:04d}-{month:02d}-{day:02d}" / f"type={data_type}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        
        return partition_dir
    
    def _update_stats(self, record_count: int, file_size: int, symbol: str, data_type: str) -> None:
        """Update storage statistics."""
        self.stats.total_files_written += 1
        self.stats.total_records_written += record_count
        self.stats.total_size_mb += file_size / (1024 * 1024)
        self.stats.last_write_time = datetime.now()
        
        # Update by-symbol and by-type stats
        self.stats.files_by_symbol[symbol] = self.stats.files_by_symbol.get(symbol, 0) + 1
        self.stats.files_by_data_type[data_type] = self.stats.files_by_data_type.get(data_type, 0) + 1
        
        # Calculate write rate
        elapsed = (datetime.now() - self._start_time).total_seconds()
        if elapsed > 0:
            self.stats.write_rate = self.stats.total_records_written / elapsed
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for old data."""
        logger.info("Cleanup loop started")
        
        while self._running:
            try:
                # Wait for cleanup interval
                await asyncio.sleep(self.config.cleanup_interval_hours * 3600)
                
                # Perform cleanup
                await self._cleanup_old_data()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
        
        logger.info("Cleanup loop stopped")
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old data based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        
        deleted_files = 0
        deleted_size = 0
        
        # Walk through all files
        for file_path in self.base_path.rglob("*.parquet"):
            try:
                # Extract date from path
                date_str = file_path.parent.parent.name  # date=YYYY-MM-DD
                if date_str.startswith("date="):
                    date_part = date_str[5:]  # Remove "date=" prefix
                    file_date = datetime.strptime(date_part, "%Y-%m-%d")
                    
                    if file_date < cutoff_date:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        deleted_files += 1
                        deleted_size += file_size
                        
            except Exception as e:
                logger.error(f"Error cleaning up file {file_path}: {e}")
        
        if deleted_files > 0:
            logger.info(f"Cleaned up {deleted_files} files, freed {deleted_size / (1024*1024):.1f} MB")
    
    def get_statistics(self) -> StorageStats:
        """Get storage statistics."""
        return self.stats
    
    def get_partition_info(self) -> Dict[str, Any]:
        """Get information about partitions."""
        partitions = {}
        
        # Scan partition directories
        for symbol_dir in self.base_path.glob("symbol=*"):
            symbol = symbol_dir.name.split("=")[1]
            
            for date_dir in symbol_dir.glob("date=*"):
                date_str = date_dir.name.split("=")[1]
                
                for type_dir in date_dir.glob("type=*"):
                    data_type = type_dir.name.split("=")[1]
                    
                    # Count files
                    file_count = len(list(type_dir.glob("*.parquet")))
                    
                    partition_key = f"{symbol}/{date_str}/{data_type}"
                    partitions[partition_key] = {
                        "symbol": symbol,
                        "date": date_str,
                        "data_type": data_type,
                        "file_count": file_count
                    }
        
        return partitions
    
    async def query_data(
        self,
        symbol: Optional[str] = None,
        data_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Query stored data."""
        # Build dataset path
        dataset_path = self.base_path
        
        # Build filters
        filters = []
        
        if symbol:
            filters.append(("symbol", "=", symbol))
        
        if data_type:
            filters.append(("type", "=", data_type))
        
        if start_date or end_date:
            # This would need more sophisticated filtering
            pass
        
        # Read dataset
        try:
            dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")
            
            # Apply filters
            if filters:
                table = dataset.to_table(filter=ds.field("symbol") == symbol if symbol else None)
            else:
                table = dataset.to_table()
            
            # Convert to DataFrame
            df = table.to_pandas()
            
            # Apply date filtering if needed
            if start_date:
                df = df[df["timestamp_ms"] >= int(start_date.timestamp() * 1000)]
            
            if end_date:
                df = df[df["timestamp_ms"] <= int(end_date.timestamp() * 1000)]
            
            return df
            
        except Exception as e:
            logger.error(f"Error querying data: {e}")
            return pd.DataFrame()


# Convenience functions
def create_raw_parquet_storage(
    base_path: str = "data/parquet/raw",
    compression: str = "snappy",
    retention_days: int = 365
) -> RawParquetStorage:
    """Create raw Parquet storage with default configuration."""
    config = ParquetStorageConfig(
        base_path=base_path,
        compression=compression,
        retention_days=retention_days
    )
    
    return RawParquetStorage(config)
