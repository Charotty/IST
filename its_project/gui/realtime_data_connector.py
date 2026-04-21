from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
try:
    import psycopg2
    from psycopg2.extras import execute_values
    import asyncpg
except ImportError:
    logging.warning("Database modules not available, using fallback")
    psycopg2 = None
    execute_values = None
    asyncpg = None

# Backend imports
try:
    from storage.parquet_store import ParquetStore
    from storage.timescale_client import TimescaleClient
    from common.types import MarketData, MarketDataType
    from async_integration import AsyncEventLoopManager, AsyncDataStream, get_async_manager
except ImportError:
    logging.warning("Backend modules not available, using fallback")
    ParquetStore = None
    TimescaleClient = None
    MarketData = None
    MarketDataType = None
    AsyncEventLoopManager = None
    AsyncDataStream = None
    get_async_manager = None

logger = logging.getLogger(__name__)


@dataclass
class DataChange:
    """Data change notification."""
    source: str  # "parquet" or "timescale"
    table: str
    operation: str  # "insert", "update", "delete"
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None
    primary_key: Optional[str] = None


@dataclass
class RealTimeConfig:
    """Configuration for real-time data connector."""
    # Parquet configuration
    parquet_path: str = "data/parquet"
    parquet_watch_interval: float = 1.0  # seconds
    
    # TimescaleDB configuration
    timescale_dsn: str = "postgresql://user:pass@localhost/its"
    timescale_listen_channel: str = "data_changes"
    
    # Data processing
    batch_size: int = 100
    max_memory_mb: int = 512
    buffer_timeout: float = 0.5  # seconds
    
    # Change detection
    enable_change_detection: bool = True
    change_detection_interval: float = 2.0  # seconds
    
    # Performance
    max_concurrent_readers: int = 4
    prefetch_size: int = 1000


class ParquetRealTimeReader:
    """Real-time Parquet data reader with change detection."""
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.parquet_store = ParquetStore(config.parquet_path)
        
        # File watching
        self.file_mtimes: Dict[str, float] = {}
        self.last_scan_time = datetime.now()
        
        # Data buffers
        self.data_buffers: Dict[str, List[MarketData]] = {}
        self.buffer_locks: Dict[str, threading.Lock] = {}
        
        # Statistics
        self.stats = {
            "files_scanned": 0,
            "changes_detected": 0,
            "records_processed": 0,
            "errors": 0
        }
        
        logger.info("ParquetRealTimeReader initialized")
    
    async def start_monitoring(self, data_stream: AsyncDataStream):
        """Start monitoring Parquet files for changes."""
        logger.info("Starting Parquet file monitoring")
        
        try:
            # Initial scan
            await self._scan_parquet_files(data_stream)
            
            # Continuous monitoring
            while True:
                await asyncio.sleep(self.config.parquet_watch_interval)
                await self._scan_parquet_files(data_stream)
                
        except asyncio.CancelledError:
            logger.info("Parquet monitoring stopped")
        except Exception as e:
            logger.error(f"Error in Parquet monitoring: {e}")
            raise
    
    async def _scan_parquet_files(self, data_stream: AsyncDataStream):
        """Scan Parquet files for changes."""
        try:
            parquet_path = Path(self.config.parquet_path)
            
            if not parquet_path.exists():
                return
            
            # Get all Parquet files
            parquet_files = list(parquet_path.rglob("*.parquet"))
            
            self.stats["files_scanned"] += len(parquet_files)
            
            for file_path in parquet_files:
                await self._check_file_changes(file_path, data_stream)
                
        except Exception as e:
            logger.error(f"Error scanning Parquet files: {e}")
            self.stats["errors"] += 1
    
    async def _check_file_changes(self, file_path: Path, data_stream: AsyncDataStream):
        """Check for changes in a specific Parquet file."""
        try:
            # Get file modification time
            mtime = file_path.stat().st_mtime
            
            file_key = str(file_path.relative_to(self.config.parquet_path))
            
            # Check if file changed
            if file_key not in self.file_mtimes or mtime > self.file_mtimes[file_key]:
                self.file_mtimes[file_key] = mtime
                
                # Read new data
                new_data = await self._read_parquet_file(file_path)
                
                if new_data:
                    # Extract symbol from path
                    symbol = self._extract_symbol_from_path(file_path)
                    
                    # Send to data stream
                    for market_data in new_data:
                        await data_stream.send_data("parquet_data", {
                            "source": "parquet",
                            "symbol": symbol,
                            "data": market_data,
                            "file_path": str(file_path),
                            "timestamp": datetime.now()
                        })
                    
                    self.stats["changes_detected"] += 1
                    self.stats["records_processed"] += len(new_data)
                    
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")
            self.stats["errors"] += 1
    
    async def _read_parquet_file(self, file_path: Path) -> List[MarketData]:
        """Read Parquet file and convert to MarketData objects."""
        try:
            # Read Parquet file
            table = pq.read_table(file_path)
            df = table.to_pandas()
            
            market_data_list = []
            
            for _, row in df.iterrows():
                try:
                    # Create MarketData object
                    market_data = MarketData(
                        timestamp_ms=int(row.get("timestamp_ms", 0)),
                        symbol=row.get("symbol", ""),
                        type=MarketDataType(row.get("type", "trade")),
                        exchange=row.get("exchange", "unknown"),
                        data=row.get("data", {})
                    )
                    market_data_list.append(market_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing row in {file_path}: {e}")
            
            return market_data_list
            
        except Exception as e:
            logger.error(f"Error reading Parquet file {file_path}: {e}")
            return []
    
    def _extract_symbol_from_path(self, file_path: Path) -> str:
        """Extract symbol from file path."""
        # Try to extract symbol from path components
        parts = file_path.parts
        
        for part in parts:
            # Look for common symbol patterns
            if any(symbol in part.upper() for symbol in ["BTC", "ETH", "BNB", "USDT", "USD"]):
                return part.replace(".parquet", "").upper()
        
        # Fallback to filename
        return file_path.stem.upper()
    
    async def read_symbol_data(self, symbol: str, start_time: datetime, end_time: datetime) -> List[MarketData]:
        """Read data for specific symbol and time range."""
        try:
            # Use ParquetStore for efficient reading
            data = await self.parquet_store.read_data(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error reading symbol data for {symbol}: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reading statistics."""
        return self.stats.copy()


class TimescaleRealTimeListener:
    """Real-time TimescaleDB listener with notifications."""
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.timescale_client = TimescaleClient(config.timescale_dsn)
        
        # Connection management
        self.connection: Optional[asyncpg.Connection] = None
        self.is_listening = False
        
        # Data processing
        self.notification_buffer: List[Dict[str, Any]] = []
        self.buffer_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "notifications_received": 0,
            "records_processed": 0,
            "connection_errors": 0,
            "processing_errors": 0
        }
        
        logger.info("TimescaleRealTimeListener initialized")
    
    async def start_listening(self, data_stream: AsyncDataStream):
        """Start listening to TimescaleDB changes."""
        logger.info("Starting TimescaleDB listening")
        
        try:
            # Connect to database
            await self.timescale_client.connect()
            self.connection = await asyncpg.connect(self.config.timescale_dsn)
            
            # Listen to notifications
            await self.connection.add_listener(self.config.timescale_listen_channel, self._handle_notification)
            
            self.is_listening = True
            
            # Start processing loop
            await self._process_notifications(data_stream)
            
        except Exception as e:
            logger.error(f"Error starting TimescaleDB listening: {e}")
            self.stats["connection_errors"] += 1
            raise
    
    async def stop_listening(self):
        """Stop listening to TimescaleDB changes."""
        self.is_listening = False
        
        if self.connection:
            await self.connection.close()
            self.connection = None
        
        await self.timescale_client.close()
        
        logger.info("TimescaleDB listening stopped")
    
    async def _process_notifications(self, data_stream: AsyncDataStream):
        """Process buffered notifications."""
        while self.is_listening:
            try:
                # Get notifications from buffer
                notifications = []
                
                with self.buffer_lock:
                    if self.notification_buffer:
                        notifications = self.notification_buffer.copy()
                        self.notification_buffer.clear()
                
                # Process notifications
                for notification in notifications:
                    await self._process_notification(notification, data_stream)
                
                # Wait for next batch
                await asyncio.sleep(self.config.buffer_timeout)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing notifications: {e}")
                self.stats["processing_errors"] += 1
                await asyncio.sleep(1.0)
    
    def _handle_notification(self, connection, pid, channel, payload):
        """Handle database notification."""
        try:
            # Parse notification payload
            notification_data = json.loads(payload)
            
            with self.buffer_lock:
                self.notification_buffer.append(notification_data)
            
            self.stats["notifications_received"] += 1
            
        except Exception as e:
            logger.error(f"Error handling notification: {e}")
            self.stats["processing_errors"] += 1
    
    async def _process_notification(self, notification: Dict[str, Any], data_stream: AsyncDataStream):
        """Process individual notification."""
        try:
            # Extract data from notification
            operation = notification.get("operation", "insert")
            table = notification.get("table", "")
            data = notification.get("data", {})
            
            # Convert to MarketData if applicable
            if table in ["market_data", "trades", "quotes"]:
                market_data = self._notification_to_market_data(data, operation)
                
                if market_data:
                    await data_stream.send_data("timescale_data", {
                        "source": "timescale",
                        "operation": operation,
                        "table": table,
                        "data": market_data,
                        "timestamp": datetime.now()
                    })
                    
                    self.stats["records_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing notification: {e}")
            self.stats["processing_errors"] += 1
    
    def _notification_to_market_data(self, data: Dict[str, Any], operation: str) -> Optional[MarketData]:
        """Convert notification data to MarketData object."""
        try:
            return MarketData(
                timestamp_ms=int(data.get("timestamp_ms", 0)),
                symbol=data.get("symbol", ""),
                type=MarketDataType(data.get("type", "trade")),
                exchange=data.get("exchange", "unknown"),
                data=data.get("data", {})
            )
        except Exception as e:
            logger.warning(f"Error converting notification to MarketData: {e}")
            return None
    
    async def query_real_time_data(self, query: str, params: Optional[List] = None) -> List[Dict[str, Any]]:
        """Execute real-time query on TimescaleDB."""
        try:
            if not self.connection:
                raise RuntimeError("Not connected to TimescaleDB")
            
            rows = await self.connection.fetch(query, *params or [])
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error executing real-time query: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get listening statistics."""
        return self.stats.copy()


class UnifiedRealTimeConnector:
    """Unified real-time data connector for multiple storage systems."""
    
    def __init__(self, config: Optional[RealTimeConfig] = None):
        self.config = config or RealTimeConfig()
        
        # Components
        self.parquet_reader = ParquetRealTimeReader(self.config)
        self.timescale_listener = TimescaleRealTimeListener(self.config)
        
        # Async system
        self.async_manager: Optional[AsyncEventLoopManager] = None
        self.data_stream: Optional[AsyncDataStream] = None
        
        # Monitoring tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Data processing
        self.data_processors: Dict[str, Callable] = {}
        self.change_handlers: List[Callable] = []
        
        # Statistics
        self.total_stats = {
            "start_time": datetime.now(),
            "total_records": 0,
            "parquet_records": 0,
            "timescale_records": 0,
            "errors": 0,
            "last_update": None
        }
        
        logger.info("UnifiedRealTimeConnector initialized")
    
    async def start(self, async_manager: AsyncEventLoopManager):
        """Start real-time data monitoring."""
        self.async_manager = async_manager
        self.data_stream = get_data_stream()
        
        logger.info("Starting unified real-time data connector")
        
        try:
            # Create data streams
            await self.data_stream.create_stream("parquet_data", max_size=1000)
            await self.data_stream.create_stream("timescale_data", max_size=1000)
            await self.data_stream.create_stream("unified_data", max_size=2000)
            
            # Start monitoring tasks
            await self._start_monitoring()
            
            # Start data processing
            await self._start_data_processing()
            
            logger.info("Unified real-time data connector started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start real-time connector: {e}")
            raise
    
    async def _start_monitoring(self):
        """Start monitoring tasks for all storage systems."""
        # Start Parquet monitoring
        parquet_task = asyncio.create_task(
            self.parquet_reader.start_monitoring(self.data_stream)
        )
        self.monitoring_tasks.append(parquet_task)
        
        # Start TimescaleDB listening
        timescale_task = asyncio.create_task(
            self.timescale_listener.start_listening(self.data_stream)
        )
        self.monitoring_tasks.append(timescale_task)
        
        logger.info("Started monitoring tasks for all storage systems")
    
    async def _start_data_processing(self):
        """Start data processing streams."""
        # Process Parquet data
        self.data_stream.start_stream_processor(
            "parquet_data",
            self._process_parquet_data,
            error_handler=self._handle_processing_error
        )
        
        # Process TimescaleDB data
        self.data_stream.start_stream_processor(
            "timescale_data",
            self._process_timescale_data,
            error_handler=self._handle_processing_error
        )
        
        # Process unified data
        self.data_stream.start_stream_processor(
            "unified_data",
            self._process_unified_data,
            error_handler=self._handle_processing_error
        )
        
        logger.info("Started data processing streams")
    
    def _process_parquet_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process Parquet data."""
        try:
            # Add processing timestamp
            data["processed_at"] = datetime.now()
            data["source_system"] = "parquet"
            
            # Apply custom processors
            processor = self.data_processors.get("parquet")
            if processor:
                data = processor(data)
            
            # Update statistics
            self.total_stats["parquet_records"] += 1
            self.total_stats["total_records"] += 1
            self.total_stats["last_update"] = datetime.now()
            
            # Send to unified stream
            asyncio.create_task(
                self.data_stream.send_data("unified_data", data)
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing Parquet data: {e}")
            self.total_stats["errors"] += 1
            return None
    
    def _process_timescale_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process TimescaleDB data."""
        try:
            # Add processing timestamp
            data["processed_at"] = datetime.now()
            data["source_system"] = "timescale"
            
            # Apply custom processors
            processor = self.data_processors.get("timescale")
            if processor:
                data = processor(data)
            
            # Update statistics
            self.total_stats["timescale_records"] += 1
            self.total_stats["total_records"] += 1
            self.total_stats["last_update"] = datetime.now()
            
            # Send to unified stream
            asyncio.create_task(
                self.data_stream.send_data("unified_data", data)
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing TimescaleDB data: {e}")
            self.total_stats["errors"] += 1
            return None
    
    def _process_unified_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process unified data."""
        try:
            # Apply change handlers
            for handler in self.change_handlers:
                try:
                    handler(data)
                except Exception as e:
                    logger.warning(f"Error in change handler: {e}")
            
            # Add final processing timestamp
            data["final_processed_at"] = datetime.now()
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing unified data: {e}")
            self.total_stats["errors"] += 1
            return None
    
    def _handle_processing_error(self, error: Exception):
        """Handle data processing errors."""
        logger.error(f"Data processing error: {error}")
        self.total_stats["errors"] += 1
    
    def add_data_processor(self, source: str, processor: Callable):
        """Add custom data processor for a source."""
        self.data_processors[source] = processor
        logger.info(f"Added data processor for {source}")
    
    def add_change_handler(self, handler: Callable):
        """Add change notification handler."""
        self.change_handlers.append(handler)
        logger.info("Added change notification handler")
    
    async def query_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        source: str = "auto"
    ) -> List[MarketData]:
        """Query historical data from appropriate source."""
        try:
            if source == "auto" or source == "parquet":
                # Try Parquet first
                data = await self.parquet_reader.read_symbol_data(symbol, start_time, end_time)
                if data:
                    return data
            
            if source == "auto" or source == "timescale":
                # Fallback to TimescaleDB
                query = """
                SELECT * FROM market_data 
                WHERE symbol = $1 
                AND timestamp >= $2 
                AND timestamp <= $3
                ORDER BY timestamp
                """
                
                rows = await self.timescale_listener.query_real_time_data(
                    query, [symbol, start_time, end_time]
                )
                
                # Convert to MarketData objects
                data = []
                for row in rows:
                    market_data = MarketData(
                        timestamp_ms=int(row["timestamp_ms"]),
                        symbol=row["symbol"],
                        type=MarketDataType(row["type"]),
                        exchange=row["exchange"],
                        data=row["data"]
                    )
                    data.append(market_data)
                
                return data
            
            return []
            
        except Exception as e:
            logger.error(f"Error querying historical data: {e}")
            return []
    
    async def get_latest_data(self, symbol: str, limit: int = 100) -> List[MarketData]:
        """Get latest data for a symbol."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)  # Last hour
        
        return await self.query_historical_data(symbol, start_time, end_time)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = self.total_stats.copy()
        
        # Add component statistics
        stats["parquet_stats"] = self.parquet_reader.get_statistics()
        stats["timescale_stats"] = self.timescale_listener.get_statistics()
        
        # Calculate uptime
        stats["uptime_seconds"] = (datetime.now() - stats["start_time"]).total_seconds()
        
        # Calculate rates
        if stats["uptime_seconds"] > 0:
            stats["records_per_second"] = stats["total_records"] / stats["uptime_seconds"]
        else:
            stats["records_per_second"] = 0
        
        return stats
    
    async def stop(self):
        """Stop real-time data monitoring."""
        logger.info("Stopping unified real-time data connector")
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.monitoring_tasks.clear()
        
        # Stop TimescaleDB listening
        await self.timescale_listener.stop_listening()
        
        # Stop data streams
        if self.data_stream:
            for stream_id in ["parquet_data", "timescale_data", "unified_data"]:
                self.data_stream.stop_stream(stream_id)
        
        logger.info("Unified real-time data connector stopped")
    
    def is_healthy(self) -> bool:
        """Check if the connector is healthy."""
        # Check if monitoring tasks are running
        active_tasks = sum(1 for task in self.monitoring_tasks if not task.done())
        
        # Check error rate
        if self.total_stats["total_records"] > 0:
            error_rate = self.total_stats["errors"] / self.total_stats["total_records"]
            if error_rate > 0.1:  # More than 10% errors
                return False
        
        return active_tasks > 0


# Factory function
def create_realtime_connector(config: Optional[RealTimeConfig] = None) -> UnifiedRealTimeConnector:
    """Create real-time data connector."""
    return UnifiedRealTimeConnector(config)


# Example usage and testing
async def test_realtime_connector():
    """Test the real-time connector."""
    config = RealTimeConfig(
        parquet_path="data/parquet",
        timescale_dsn="postgresql://user:pass@localhost/its",
        parquet_watch_interval=2.0,
        enable_change_detection=True
    )
    
    connector = create_realtime_connector(config)
    
    # Mock async manager for testing
    class MockAsyncManager:
        def submit_task(self, coro, **kwargs):
            return "mock_task_id"
    
    mock_manager = MockAsyncManager()
    
    try:
        # Start connector
        await connector.start(mock_manager)
        
        # Monitor for a while
        await asyncio.sleep(10)
        
        # Get statistics
        stats = connector.get_statistics()
        print(f"Statistics: {stats}")
        
        # Test query
        latest_data = await connector.get_latest_data("BTCUSDT", 10)
        print(f"Latest data count: {len(latest_data)}")
        
    finally:
        # Stop connector
        await connector.stop()


if __name__ == "__main__":
    # Run test
    asyncio.run(test_realtime_connector())
