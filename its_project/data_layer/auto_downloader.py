#!/usr/bin/env python3
"""
Auto-Download System for Market Data
===================================

Automated system for downloading and maintaining historical market data.
Supports scheduling, incremental updates, and data validation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
from pathlib import Path
import pandas as pd
import hashlib

from its_project.data_layer.okx_source import OKXDataSource, OKXConfig
from its_project.data_layer.backfill_manager import BackfillManager, BackfillConfig

logger = logging.getLogger(__name__)


class DownloadStatus(Enum):
    """Download operation status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATING = "validating"


@dataclass
class DownloadTask:
    """Download task definition."""
    id: str
    symbol: str
    exchange: str
    data_type: str
    timeframe: Optional[str]
    priority: int = 1
    status: DownloadStatus = DownloadStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    file_size: int = 0
    records_count: int = 0


@dataclass
class AutoDownloadConfig:
    """Auto-download system configuration."""
    # General settings
    enabled: bool = True
    max_concurrent_downloads: int = 2
    download_timeout: int = 1800  # 30 minutes
    retry_delay: int = 300  # 5 minutes
    
    # Scheduling
    enable_scheduling: bool = True
    schedule_interval: int = 3600  # 1 hour
    incremental_updates: bool = True
    update_window_hours: int = 24
    
    # Data settings
    default_timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "1d"])
    max_history_days: int = 365
    batch_size: int = 1000
    
    # Storage settings
    base_storage_path: str = "data/market_data"
    enable_compression: bool = True
    file_format: str = "parquet"  # "csv" or "parquet"
    
    # Validation settings
    enable_validation: bool = True
    validate_completeness: bool = True
    validate_continuity: bool = True
    max_gap_minutes: int = 60
    
    # Database
    database_path: str = "data/auto_download.db"


class AutoDownloader:
    """
    Automated market data downloader with scheduling and validation.
    
    Features:
    - Scheduled automatic downloads
    - Incremental updates
    - Data validation and integrity checking
    - Multiple exchange support
    - Configurable timeframes and data types
    - Progress tracking and monitoring
    - Error handling and retries
    """
    
    def __init__(self, config: AutoDownloadConfig) -> None:
        self.config = config
        
        # Data sources
        self.data_sources: Dict[str, Any] = {}
        
        # Task management
        self.task_queue: asyncio.Queue[DownloadTask] = asyncio.Queue()
        self.active_tasks: Dict[str, DownloadTask] = {}
        self.completed_tasks: List[DownloadTask] = []
        
        # Background tasks
        self.download_tasks: List[asyncio.Task] = []
        self.scheduler_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Statistics
        self.stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'data_points_downloaded': 0,
            'total_size_mb': 0.0,
            'last_download': None,
            'last_validation': None
        }
        
        # Initialize database
        self._init_database()
        
        logger.info("Auto-downloader initialized")
    
    def _init_database(self) -> None:
        """Initialize auto-download database."""
        Path(self.config.database_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.config.database_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS download_tasks (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    timeframe TEXT,
                    priority INTEGER DEFAULT 1,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    file_path TEXT,
                    file_hash TEXT,
                    file_size INTEGER DEFAULT 0,
                    records_count INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS download_schedule (
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    timeframe TEXT,
                    last_download TEXT,
                    next_download TEXT,
                    priority INTEGER DEFAULT 1,
                    PRIMARY KEY (symbol, exchange, data_type, timeframe)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS download_stats (
                    date TEXT PRIMARY KEY,
                    downloads_completed INTEGER DEFAULT 0,
                    downloads_failed INTEGER DEFAULT 0,
                    data_points_downloaded INTEGER DEFAULT 0,
                    total_size_mb REAL DEFAULT 0.0,
                    total_time_ms INTEGER DEFAULT 0
                )
            """)
            
            conn.commit()
    
    def add_data_source(self, name: str, source: Any) -> None:
        """Add a data source for downloads."""
        self.data_sources[name] = source
        logger.info(f"Added data source: {name}")
    
    async def start(self) -> None:
        """Start the auto-downloader."""
        if self._running:
            logger.warning("Auto-downloader already running")
            return
        
        self._running = True
        
        # Load scheduled tasks
        await self._load_scheduled_tasks()
        
        # Start download workers
        for i in range(self.config.max_concurrent_downloads):
            task = asyncio.create_task(self._download_worker(f"worker-{i}"))
            self.download_tasks.append(task)
        
        # Start scheduler if enabled
        if self.config.enable_scheduling:
            self.scheduler_task = asyncio.create_task(self._scheduler())
        
        logger.info(f"Auto-downloader started with {len(self.download_tasks)} workers")
    
    async def stop(self) -> None:
        """Stop the auto-downloader."""
        self._running = False
        
        # Cancel worker tasks
        for task in self.download_tasks:
            task.cancel()
        
        # Cancel scheduler
        if self.scheduler_task:
            self.scheduler_task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.download_tasks, return_exceptions=True)
        
        if self.scheduler_task:
            await self.scheduler_task
        
        logger.info("Auto-downloader stopped")
    
    async def schedule_download(
        self,
        symbol: str,
        exchange: str,
        data_type: str,
        timeframe: Optional[str] = None,
        priority: int = 1
    ) -> str:
        """Schedule a download task."""
        task_id = f"{exchange}_{symbol}_{data_type}_{timeframe or 'default'}_{int(datetime.now().timestamp())}"
        
        task = DownloadTask(
            id=task_id,
            symbol=symbol,
            exchange=exchange,
            data_type=data_type,
            timeframe=timeframe,
            priority=priority
        )
        
        # Save to database
        await self._save_task(task)
        
        # Add to queue
        await self.task_queue.put(task)
        
        # Update schedule
        await self._update_schedule(symbol, exchange, data_type, timeframe)
        
        self.stats['total_downloads'] += 1
        
        logger.info(f"Scheduled download: {task_id}")
        return task_id
    
    async def _download_worker(self, worker_name: str) -> None:
        """Worker task for processing downloads."""
        logger.info(f"Download worker {worker_name} started")
        
        while self._running:
            try:
                # Get task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Process task
                await self._process_download(task)
                
                # Mark as completed
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Download worker {worker_name} error: {e}")
        
        logger.info(f"Download worker {worker_name} stopped")
    
    async def _process_download(self, task: DownloadTask) -> None:
        """Process a download task."""
        task.status = DownloadStatus.RUNNING
        task.started_at = datetime.now()
        self.active_tasks[task.id] = task
        
        try:
            # Get data source
            source = self.data_sources.get(task.exchange)
            if not source:
                raise RuntimeError(f"Data source not found: {task.exchange}")
            
            # Determine download range
            start_time, end_time = await self._get_download_range(task)
            
            # Download data
            if task.data_type == "ohlcv" and task.timeframe:
                data = await self._download_ohlcv(source, task, start_time, end_time)
            else:
                data = await self._download_market_data(source, task, start_time, end_time)
            
            # Validate data
            if self.config.enable_validation:
                task.status = DownloadStatus.VALIDATING
                await self._validate_data(task, data)
            
            # Save data
            file_path = await self._save_data(task, data)
            task.file_path = file_path
            task.file_size = Path(file_path).stat().st_size
            task.records_count = len(data)
            
            # Calculate file hash
            task.file_hash = self._calculate_file_hash(file_path)
            
            # Mark as completed
            task.status = DownloadStatus.COMPLETED
            task.completed_at = datetime.now()
            
            self.stats['successful_downloads'] += 1
            self.stats['data_points_downloaded'] += len(data)
            self.stats['total_size_mb'] += task.file_size / (1024 * 1024)
            
            logger.info(f"Completed download: {task.id} ({len(data)} records)")
        
        except Exception as e:
            task.status = DownloadStatus.FAILED
            task.error_message = str(e)
            task.retry_count += 1
            
            self.stats['failed_downloads'] += 1
            
            logger.error(f"Failed download {task.id}: {e}")
            
            # Retry if within limits
            if task.retry_count < task.max_retries:
                task.status = DownloadStatus.PENDING
                task.error_message = None
                await asyncio.sleep(self.config.retry_delay)
                await self.task_queue.put(task)
        
        finally:
            # Remove from active tasks
            self.active_tasks.pop(task.id, None)
            self.completed_tasks.append(task)
            
            # Update database
            await self._save_task(task)
        
        self.stats['last_download'] = datetime.now()
    
    async def _get_download_range(self, task: DownloadTask) -> tuple[datetime, datetime]:
        """Get download time range for task."""
        end_time = datetime.now()
        
        if self.config.incremental_updates:
            # Get last download time
            last_download = await self._get_last_download(task.symbol, task.exchange, task.data_type, task.timeframe)
            if last_download:
                start_time = last_download
            else:
                start_time = end_time - timedelta(days=self.config.max_history_days)
        else:
            start_time = end_time - timedelta(days=self.config.max_history_days)
        
        return start_time, end_time
    
    async def _download_ohlcv(self, source: Any, task: DownloadTask, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Download OHLCV data."""
        all_data = []
        
        # Download in batches
        current_time = start_time
        batch_duration = self.config.batch_size * self._get_timeframe_seconds(task.timeframe)
        
        while current_time < end_time:
            batch_end = min(current_time + timedelta(seconds=batch_duration), end_time)
            
            # Use data source to fetch data
            if hasattr(source, 'fetch_ohlcv'):
                ohlcv_data = await source.fetch_ohlcv(
                    symbol=task.symbol,
                    timeframe=task.timeframe,
                    since=int(current_time.timestamp() * 1000),
                    limit=self.config.batch_size
                )
                
                # Convert to DataFrame
                batch_df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                batch_df['timestamp'] = pd.to_datetime(batch_df['timestamp'], unit='ms')
                batch_df.set_index('timestamp', inplace=True)
                
                all_data.append(batch_df)
            
            current_time = batch_end
            await asyncio.sleep(0.1)  # Rate limiting
        
        # Combine all batches
        if all_data:
            return pd.concat(all_data).sort_index()
        else:
            return pd.DataFrame()
    
    async def _download_market_data(self, source: Any, task: DownloadTask, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Download other market data types."""
        # Implementation for ticker, orderbook, trades
        return pd.DataFrame()
    
    def _get_timeframe_seconds(self, timeframe: str) -> int:
        """Get timeframe in seconds."""
        timeframe_map = {
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
        return timeframe_map.get(timeframe, 60)
    
    async def _validate_data(self, task: DownloadTask, data: pd.DataFrame) -> None:
        """Validate downloaded data."""
        if data.empty:
            raise ValueError("Downloaded data is empty")
        
        # Check completeness
        if self.config.validate_completeness:
            expected_records = int((task.started_at - task.started_at.replace(hour=0, minute=0, second=0)).total_seconds() / self._get_timeframe_seconds(task.timeframe))
            if len(data) < expected_records * 0.95:  # Allow 5% tolerance
                logger.warning(f"Incomplete data for {task.symbol}: expected {expected_records}, got {len(data)}")
        
        # Check continuity
        if self.config.validate_continuity:
            time_diffs = data.index.to_series().diff().dropna()
            expected_diff = pd.Timedelta(seconds=self._get_timeframe_seconds(task.timeframe))
            
            gaps = time_diffs[time_diffs > expected_diff * 2]  # Allow 2x tolerance
            if not gaps.empty:
                logger.warning(f"Found {len(gaps)} gaps in data for {task.symbol}")
        
        self.stats['last_validation'] = datetime.now()
    
    async def _save_data(self, task: DownloadTask, data: pd.DataFrame) -> str:
        """Save data to file."""
        # Create directory structure
        symbol_dir = Path(self.config.base_storage_path) / task.exchange / task.symbol / task.data_type
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if task.timeframe:
            filename = f"{task.timeframe}_{timestamp}.{self.config.file_format}"
        else:
            filename = f"{task.data_type}_{timestamp}.{self.config.file_format}"
        
        file_path = symbol_dir / filename
        
        # Save data
        if self.config.file_format == "parquet":
            data.to_parquet(file_path, compression='snappy' if self.config.enable_compression else None)
        else:  # CSV
            data.to_csv(file_path, compression='gzip' if self.config.enable_compression else None)
        
        logger.debug(f"Saved data to {file_path}")
        return str(file_path)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file."""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    async def _scheduler(self) -> None:
        """Background scheduler for automatic downloads."""
        logger.info("Auto-download scheduler started")
        
        while self._running:
            try:
                # Check scheduled downloads
                await self._process_scheduled_downloads()
                
                await asyncio.sleep(self.config.schedule_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)
        
        logger.info("Auto-download scheduler stopped")
    
    async def _process_scheduled_downloads(self) -> None:
        """Process scheduled downloads."""
        with sqlite3.connect(self.config.database_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM download_schedule 
                WHERE next_download <= ?
                ORDER BY priority DESC, next_download ASC
            """, (datetime.now().isoformat(),))
            
            for row in cursor.fetchall():
                symbol, exchange, data_type, timeframe = row[0], row[1], row[2], row[3]
                
                await self.schedule_download(symbol, exchange, data_type, timeframe)
                
                # Update next download time
                next_download = datetime.now() + timedelta(hours=self.config.update_window_hours)
                conn.execute("""
                    UPDATE download_schedule 
                    SET last_download = ?, next_download = ?
                    WHERE symbol = ? AND exchange = ? AND data_type = ? AND timeframe = ?
                """, (datetime.now().isoformat(), next_download.isoformat(), symbol, exchange, data_type, timeframe))
                
                conn.commit()
    
    async def _load_scheduled_tasks(self) -> None:
        """Load scheduled tasks from database."""
        with sqlite3.connect(self.config.database_path) as conn:
            cursor = conn.execute("SELECT * FROM download_schedule")
            
            for row in cursor.fetchall():
                symbol, exchange, data_type, timeframe = row[0], row[1], row[2], row[3]
                next_download = datetime.fromisoformat(row[5]) if row[5] else None
                
                if next_download and next_download <= datetime.now():
                    await self.schedule_download(symbol, exchange, data_type, timeframe)
        
        logger.info(f"Loaded {self.task_queue.qsize()} scheduled tasks")
    
    async def _update_schedule(self, symbol: str, exchange: str, data_type: str, timeframe: Optional[str]) -> None:
        """Update download schedule."""
        next_download = datetime.now() + timedelta(hours=self.config.update_window_hours)
        
        with sqlite3.connect(self.config.database_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO download_schedule 
                (symbol, exchange, data_type, timeframe, last_download, next_download)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, exchange, data_type, timeframe, datetime.now().isoformat(), next_download.isoformat()))
            conn.commit()
    
    async def _get_last_download(self, symbol: str, exchange: str, data_type: str, timeframe: Optional[str]) -> Optional[datetime]:
        """Get last download time."""
        with sqlite3.connect(self.config.database_path) as conn:
            cursor = conn.execute("""
                SELECT last_download FROM download_schedule 
                WHERE symbol = ? AND exchange = ? AND data_type = ? AND timeframe = ?
            """, (symbol, exchange, data_type, timeframe))
            
            row = cursor.fetchone()
            if row and row[0]:
                return datetime.fromisoformat(row[0])
        
        return None
    
    async def _save_task(self, task: DownloadTask) -> None:
        """Save task to database."""
        with sqlite3.connect(self.config.database_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO download_tasks (
                    id, symbol, exchange, data_type, timeframe, priority, status,
                    created_at, started_at, completed_at, error_message,
                    retry_count, max_retries, file_path, file_hash, file_size, records_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.id, task.symbol, task.exchange, task.data_type, task.timeframe,
                task.priority, task.status.value, task.created_at.isoformat(),
                task.started_at.isoformat() if task.started_at else None,
                task.completed_at.isoformat() if task.completed_at else None,
                task.error_message, task.retry_count, task.max_retries,
                task.file_path, task.file_hash, task.file_size, task.records_count
            ))
            conn.commit()
    
    def get_download_status(self, task_id: str) -> Optional[DownloadTask]:
        """Get download task status."""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        
        for task in self.completed_tasks:
            if task.id == task_id:
                return task
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get download statistics."""
        return {
            **self.stats,
            'queue_size': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'running': self._running
        }
    
    async def cleanup_old_tasks(self, older_than_days: int = 30) -> int:
        """Clean up old completed tasks."""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        # Remove from memory
        self.completed_tasks = [
            task for task in self.completed_tasks
            if task.completed_at and task.completed_at > cutoff_date
        ]
        
        # Remove from database
        with sqlite3.connect(self.config.database_path) as conn:
            cursor = conn.execute("""
                DELETE FROM download_tasks 
                WHERE status IN ('completed', 'failed') 
                AND completed_at < ?
            """, (cutoff_date.isoformat(),))
            
            deleted_count = cursor.rowcount
            conn.commit()
        
        logger.info(f"Cleaned up {deleted_count} old download tasks")
        return deleted_count


# Convenience functions
def create_auto_downloader(
    max_concurrent_downloads: int = 2,
    enable_scheduling: bool = True,
    storage_path: str = "data/market_data"
) -> AutoDownloader:
    """Create auto-downloader with default configuration."""
    config = AutoDownloadConfig(
        max_concurrent_downloads=max_concurrent_downloads,
        enable_scheduling=enable_scheduling,
        base_storage_path=storage_path
    )
    
    return AutoDownloader(config)
