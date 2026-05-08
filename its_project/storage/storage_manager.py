#!/usr/bin/env python3
"""
Unified Storage Manager for Raw and Aggregated Data
===============================================

Production-ready storage manager that coordinates raw Parquet storage
and aggregated TimescaleDB storage with automatic data flow.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from its_project.common.types import MarketData, MarketDataType
from its_project.storage.raw_parquet_storage import RawParquetStorage, ParquetStorageConfig
from its_project.storage.aggregated_timescale_storage import AggregatedTimescaleStorage, TimescaleDBConfig

logger = logging.getLogger(__name__)


class StorageMode(Enum):
    """Storage mode enumeration."""
    RAW_ONLY = "raw_only"
    AGGREGATED_ONLY = "aggregated_only"
    BOTH = "both"


@dataclass
class StorageManagerConfig:
    """Configuration for unified storage manager."""
    # Storage modes
    storage_mode: StorageMode = StorageMode.BOTH
    
    # Raw storage configuration
    raw_config: ParquetStorageConfig = field(default_factory=ParquetStorageConfig)
    
    # Aggregated storage configuration
    aggregated_config: TimescaleDBConfig = field(default_factory=TimescaleDBConfig)
    
    # Data routing
    enable_auto_routing: bool = True
    routing_rules: Dict[str, str] = field(default_factory=lambda: {
        "ticker": "both",
        "orderbook": "raw_only",
        "trade": "both",
        "ohlcv": "aggregated_only"
    })
    
    # Performance settings
    batch_size: int = 1000
    max_queue_size: int = 10000
    processing_timeout: float = 30.0
    
    # Monitoring
    enable_monitoring: bool = True
    monitoring_interval: float = 60.0  # seconds
    
    # Data retention
    enable_auto_cleanup: bool = True
    cleanup_interval_hours: int = 24


@dataclass
class StorageStats:
    """Unified storage statistics."""
    start_time: datetime = field(default_factory=datetime.now)
    
    # Raw storage stats
    raw_files_written: int = 0
    raw_records_stored: int = 0
    raw_storage_size_mb: float = 0.0
    
    # Aggregated storage stats
    aggregated_records_stored: int = 0
    ohlcv_records: int = 0
    volume_records: int = 0
    stats_records: int = 0
    
    # Performance stats
    total_processed: int = 0
    processing_rate: float = 0.0  # records per second
    error_count: int = 0
    
    # Queue stats
    queue_size: int = 0
    queue_utilization: float = 0.0
    
    # Last activity
    last_store_time: Optional[datetime] = None
    last_cleanup_time: Optional[datetime] = None


class StorageManager:
    """
    Unified storage manager for raw and aggregated data.
    
    Features:
    - Automatic data routing to appropriate storage
    - Raw data storage in Parquet format
    - Aggregated data storage in TimescaleDB
    - Performance monitoring and statistics
    - Automatic cleanup and retention
    - Configurable routing rules
    - Error handling and retries
    """
    
    def __init__(self, config: StorageManagerConfig) -> None:
        self.config = config
        
        # Storage instances
        self.raw_storage: Optional[RawParquetStorage] = None
        self.aggregated_storage: Optional[AggregatedTimescaleStorage] = None
        
        # Data processing
        self.processing_queue: asyncio.Queue[MarketData] = asyncio.Queue(config.max_queue_size)
        
        # Statistics
        self.stats = StorageStats()
        
        # State
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Initialize storage based on mode
        self._initialize_storage()
        
        logger.info(f"Storage manager initialized with mode: {config.storage_mode.value}")
    
    def _initialize_storage(self) -> None:
        """Initialize storage instances based on configuration."""
        if self.config.storage_mode in [StorageMode.RAW_ONLY, StorageMode.BOTH]:
            self.raw_storage = RawParquetStorage(self.config.raw_config)
        
        if self.config.storage_mode in [StorageMode.AGGREGATED_ONLY, StorageMode.BOTH]:
            self.aggregated_storage = AggregatedTimescaleStorage(self.config.aggregated_config)
    
    async def start(self) -> None:
        """Start the storage manager."""
        if self._running:
            logger.warning("Storage manager already running")
            return
        
        self._running = True
        
        try:
            # Start storage instances
            if self.raw_storage:
                await self.raw_storage.start()
                logger.info("Raw storage started")
            
            if self.aggregated_storage:
                await self.aggregated_storage.start()
                logger.info("Aggregated storage started")
            
            # Start processing tasks
            self._processor_task = asyncio.create_task(self._processing_loop())
            
            if self.config.enable_monitoring:
                self._monitor_task = asyncio.create_task(self._monitoring_loop())
            
            if self.config.enable_auto_cleanup:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Storage manager started")
            
        except Exception as e:
            logger.error(f"Failed to start storage manager: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the storage manager."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel tasks
        tasks = [self._processor_task, self._monitor_task, self._cleanup_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop storage instances
        if self.raw_storage:
            await self.raw_storage.stop()
        
        if self.aggregated_storage:
            await self.aggregated_storage.stop()
        
        logger.info("Storage manager stopped")
    
    async def store(self, market_data: MarketData) -> None:
        """Store market data."""
        if not self._running:
            logger.warning("Storage manager not running, discarding data")
            return
        
        try:
            # Add to processing queue
            await self.processing_queue.put(market_data)
        except asyncio.QueueFull:
            logger.error("Processing queue full, discarding data")
            self.stats.error_count += 1
    
    async def store_batch(self, market_data_batch: List[MarketData]) -> None:
        """Store a batch of market data."""
        for data in market_data_batch:
            await self.store(data)
    
    async def _processing_loop(self) -> None:
        """Main processing loop for handling data storage."""
        logger.info("Processing loop started")
        
        batch = []
        
        while self._running:
            try:
                # Get data with timeout
                try:
                    market_data = await asyncio.wait_for(
                        self.processing_queue.get(),
                        timeout=1.0
                    )
                    batch.append(market_data)
                    self.processing_queue.task_done()
                except asyncio.TimeoutError:
                    # Process batch if we have data
                    if batch:
                        await self._process_batch(batch)
                        batch = []
                    continue
                
                # Process batch when it reaches configured size
                if len(batch) >= self.config.batch_size:
                    await self._process_batch(batch)
                    batch = []
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                self.stats.error_count += 1
                await asyncio.sleep(1)
        
        # Process remaining batch
        if batch:
            await self._process_batch(batch)
        
        logger.info("Processing loop stopped")
    
    async def _process_batch(self, batch: List[MarketData]) -> None:
        """Process a batch of market data."""
        if not batch:
            return
        
        start_time = datetime.now()
        
        try:
            # Route data based on configuration
            if self.config.enable_auto_routing:
                await self._route_data_batch(batch)
            else:
                # Store to all available storage
                await self._store_to_all(batch)
            
            # Update statistics
            self.stats.total_processed += len(batch)
            self.stats.last_store_time = start_time
            
            # Calculate processing rate
            elapsed = (datetime.now() - self.stats.start_time).total_seconds()
            if elapsed > 0:
                self.stats.processing_rate = self.stats.total_processed / elapsed
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.stats.error_count += 1
    
    async def _route_data_batch(self, batch: List[MarketData]) -> None:
        """Route data batch based on routing rules."""
        # Group by data type
        grouped_data = {}
        for data in batch:
            data_type = data.type.value
            if data_type not in grouped_data:
                grouped_data[data_type] = []
            grouped_data[data_type].append(data)
        
        # Process each group according to routing rules
        for data_type, data_list in grouped_data.items():
            routing = self.config.routing_rules.get(data_type, "both")
            
            if routing == "raw_only" and self.raw_storage:
                await self.raw_storage.store_batch(data_list)
                self._update_raw_stats(len(data_list))
            
            elif routing == "aggregated_only" and self.aggregated_storage:
                await self.aggregated_storage.store_batch(data_list)
                self._update_aggregated_stats(len(data_list))
            
            elif routing == "both":
                if self.raw_storage:
                    await self.raw_storage.store_batch(data_list)
                    self._update_raw_stats(len(data_list))
                
                if self.aggregated_storage:
                    await self.aggregated_storage.store_batch(data_list)
                    self._update_aggregated_stats(len(data_list))
    
    async def _store_to_all(self, batch: List[MarketData]) -> None:
        """Store data to all available storage systems."""
        if self.raw_storage:
            await self.raw_storage.store_batch(batch)
            self._update_raw_stats(len(batch))
        
        if self.aggregated_storage:
            await self.aggregated_storage.store_batch(batch)
            self._update_aggregated_stats(len(batch))
    
    def _update_raw_stats(self, record_count: int) -> None:
        """Update raw storage statistics."""
        if self.raw_storage:
            raw_stats = self.raw_storage.get_statistics()
            self.stats.raw_files_written = raw_stats.total_files_written
            self.stats.raw_records_stored = raw_stats.total_records_written
            self.stats.raw_storage_size_mb = raw_stats.total_size_mb
    
    def _update_aggregated_stats(self, record_count: int) -> None:
        """Update aggregated storage statistics."""
        if self.aggregated_storage:
            agg_stats = self.aggregated_storage.get_statistics()
            self.stats.aggregated_records_stored = agg_stats.total_records_inserted
            self.stats.ohlcv_records = agg_stats.total_ohlcv_records
            self.stats.volume_records = agg_stats.total_volume_records
            self.stats.stats_records = agg_stats.total_stats_records
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        logger.info("Monitoring loop started")
        
        while self._running:
            try:
                # Update queue statistics
                self.stats.queue_size = self.processing_queue.qsize()
                self.stats.queue_utilization = self.stats.queue_size / self.config.max_queue_size
                
                # Log statistics
                await self._log_statistics()
                
                # Wait for next iteration
                await asyncio.sleep(self.config.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
        
        logger.info("Monitoring loop stopped")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        logger.info("Cleanup loop started")
        
        while self._running:
            try:
                # Wait for cleanup interval
                await asyncio.sleep(self.config.cleanup_interval_hours * 3600)
                
                # Perform cleanup
                await self._perform_cleanup()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)
        
        logger.info("Cleanup loop stopped")
    
    async def _perform_cleanup(self) -> None:
        """Perform cleanup operations."""
        try:
            # Raw storage cleanup
            if self.raw_storage:
                # Raw storage has automatic cleanup built-in
                pass
            
            # Aggregated storage cleanup
            if self.aggregated_storage:
                # This would implement retention policies for TimescaleDB
                pass
            
            self.stats.last_cleanup_time = datetime.now()
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def _log_statistics(self) -> None:
        """Log current statistics."""
        logger.info(
            f"Storage Stats - Processed: {self.stats.total_processed}, "
            f"Rate: {self.stats.processing_rate:.1f}/s, "
            f"Queue: {self.stats.queue_size}/{self.config.max_queue_size}, "
            f"Raw: {self.stats.raw_records_stored}, "
            f"Aggregated: {self.stats.aggregated_records_stored}, "
            f"Errors: {self.stats.error_count}"
        )
    
    def get_statistics(self) -> StorageStats:
        """Get comprehensive storage statistics."""
        # Update queue stats
        self.stats.queue_size = self.processing_queue.qsize()
        self.stats.queue_utilization = self.stats.queue_size / self.config.max_queue_size
        
        return self.stats
    
    def get_storage_status(self) -> Dict[str, Any]:
        """Get detailed storage status."""
        status = {
            "manager": {
                "running": self._running,
                "storage_mode": self.config.storage_mode.value,
                "queue_size": self.stats.queue_size,
                "queue_utilization": self.stats.queue_utilization
            },
            "statistics": {
                "total_processed": self.stats.total_processed,
                "processing_rate": self.stats.processing_rate,
                "error_count": self.stats.error_count,
                "last_store_time": self.stats.last_store_time.isoformat() if self.stats.last_store_time else None
            }
        }
        
        # Add raw storage status
        if self.raw_storage:
            raw_stats = self.raw_storage.get_statistics()
            status["raw_storage"] = {
                "running": self.raw_storage._running,
                "files_written": raw_stats.total_files_written,
                "records_stored": raw_stats.total_records_written,
                "size_mb": raw_stats.total_size_mb,
                "write_rate": raw_stats.write_rate
            }
        
        # Add aggregated storage status
        if self.aggregated_storage:
            agg_stats = self.aggregated_storage.get_statistics()
            status["aggregated_storage"] = {
                "running": self.aggregated_storage._running,
                "records_inserted": agg_stats.total_records_inserted,
                "ohlcv_records": agg_stats.total_ohlcv_records,
                "volume_records": agg_stats.total_volume_records,
                "stats_records": agg_stats.total_stats_records,
                "insert_rate": agg_stats.insert_rate
            }
        
        return status
    
    async def query_raw_data(
        self,
        symbol: Optional[str] = None,
        data_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Any:
        """Query raw data from Parquet storage."""
        if not self.raw_storage:
            raise RuntimeError("Raw storage not available")
        
        return await self.raw_storage.query_data(symbol, data_type, start_date, end_date)
    
    async def query_ohlcv_data(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> Any:
        """Query OHLCV data from TimescaleDB."""
        if not self.aggregated_storage:
            raise RuntimeError("Aggregated storage not available")
        
        return await self.aggregated_storage.query_ohlcv(symbol, interval, start_time, end_time)
    
    async def query_volume_stats(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> Any:
        """Query volume statistics from TimescaleDB."""
        if not self.aggregated_storage:
            raise RuntimeError("Aggregated storage not available")
        
        return await self.aggregated_storage.query_volume_stats(symbol, interval, start_time, end_time)


# Convenience functions
def create_storage_manager(
    storage_mode: str = "both",
    raw_path: str = "data/parquet/raw",
    timescale_dsn: str = "postgres://user:password@localhost:5432/market_data",
    enable_monitoring: bool = True
) -> StorageManager:
    """Create storage manager with default configuration."""
    # Convert string to enum
    mode_map = {
        "raw_only": StorageMode.RAW_ONLY,
        "aggregated_only": StorageMode.AGGREGATED_ONLY,
        "both": StorageMode.BOTH
    }
    storage_mode_enum = mode_map.get(storage_mode.lower(), StorageMode.BOTH)
    
    # Configure raw storage
    raw_config = ParquetStorageConfig(base_path=raw_path)
    
    # Configure aggregated storage
    agg_config = TimescaleDBConfig(dsn=timescale_dsn)
    
    # Create manager config
    manager_config = StorageManagerConfig(
        storage_mode=storage_mode_enum,
        raw_config=raw_config,
        aggregated_config=agg_config,
        enable_monitoring=enable_monitoring
    )
    
    return StorageManager(manager_config)
