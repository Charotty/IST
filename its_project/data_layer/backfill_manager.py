#!/usr/bin/env python3
"""
Backfill Manager for Historical Data
===================================

Comprehensive backfill system for filling gaps in historical data.
Supports multiple exchanges, data types, and scheduling.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
from pathlib import Path

from its_project.data_layer.okx_source import OKXDataSource, OKXConfig, BackfillRequest

logger = logging.getLogger(__name__)


class BackfillStatus(Enum):
    """Backfill operation status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackfillJob:
    """Backfill job definition."""
    id: str
    symbol: str
    exchange: str
    data_type: str
    timeframe: Optional[str]
    start_time: datetime
    end_time: datetime
    priority: int = 1
    status: BackfillStatus = BackfillStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    progress: float = 0.0  # 0.0 to 1.0


@dataclass
class BackfillConfig:
    """Backfill manager configuration."""
    max_concurrent_jobs: int = 3
    job_timeout: int = 3600  # 1 hour
    retry_delay: int = 60  # 1 minute
    database_path: str = "data/backfill.db"
    enable_scheduling: bool = True
    schedule_interval: int = 3600  # 1 hour
    gap_threshold: int = 300  # 5 minutes
    batch_size: int = 1000
    enable_compression: bool = True


class BackfillManager:
    """
    Comprehensive backfill manager for historical data.
    
    Features:
    - Job queue management with priority
    - Gap detection and automatic backfill
    - Progress tracking and monitoring
    - Database persistence
    - Scheduling and automation
    - Error handling and retries
    """
    
    def __init__(self, config: BackfillConfig) -> None:
        self.config = config
        
        # Data sources
        self.data_sources: Dict[str, Any] = {}
        
        # Job management
        self.job_queue: asyncio.Queue[BackfillJob] = asyncio.Queue()
        self.active_jobs: Dict[str, BackfillJob] = {}
        self.completed_jobs: List[BackfillJob] = []
        
        # Background tasks
        self.worker_tasks: List[asyncio.Task] = []
        self.scheduler_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Statistics
        self.stats = {
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'active_jobs': 0,
            'data_points_filled': 0,
            'last_activity': None
        }
        
        # Initialize database
        self._init_database()
        
        logger.info("Backfill manager initialized")
    
    def _init_database(self) -> None:
        """Initialize backfill database."""
        Path(self.config.database_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.config.database_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backfill_jobs (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    timeframe TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    priority INTEGER DEFAULT 1,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    progress REAL DEFAULT 0.0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backfill_stats (
                    date TEXT PRIMARY KEY,
                    jobs_completed INTEGER DEFAULT 0,
                    jobs_failed INTEGER DEFAULT 0,
                    data_points_filled INTEGER DEFAULT 0,
                    total_time_ms INTEGER DEFAULT 0
                )
            """)
            
            conn.commit()
    
    def add_data_source(self, name: str, source: Any) -> None:
        """Add a data source for backfill operations."""
        self.data_sources[name] = source
        logger.info(f"Added data source: {name}")
    
    async def start(self) -> None:
        """Start the backfill manager."""
        if self._running:
            logger.warning("Backfill manager already running")
            return
        
        self._running = True
        
        # Load pending jobs from database
        await self._load_pending_jobs()
        
        # Start worker tasks
        for i in range(self.config.max_concurrent_jobs):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(task)
        
        # Start scheduler if enabled
        if self.config.enable_scheduling:
            self.scheduler_task = asyncio.create_task(self._scheduler())
        
        logger.info(f"Backfill manager started with {len(self.worker_tasks)} workers")
    
    async def stop(self) -> None:
        """Stop the backfill manager."""
        self._running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Cancel scheduler
        if self.scheduler_task:
            self.scheduler_task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        if self.scheduler_task:
            await self.scheduler_task
        
        logger.info("Backfill manager stopped")
    
    async def create_backfill_job(
        self,
        symbol: str,
        exchange: str,
        data_type: str,
        start_time: datetime,
        end_time: datetime,
        timeframe: Optional[str] = None,
        priority: int = 1
    ) -> str:
        """Create a new backfill job."""
        job_id = f"{exchange}_{symbol}_{data_type}_{int(start_time.timestamp())}_{int(end_time.timestamp())}"
        
        job = BackfillJob(
            id=job_id,
            symbol=symbol,
            exchange=exchange,
            data_type=data_type,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            priority=priority
        )
        
        # Save to database
        await self._save_job(job)
        
        # Add to queue
        await self.job_queue.put(job)
        
        self.stats['total_jobs'] += 1
        
        logger.info(f"Created backfill job: {job_id}")
        return job_id
    
    async def detect_gaps(self, symbol: str, exchange: str, data_type: str) -> List[BackfillJob]:
        """Detect gaps in data and create backfill jobs."""
        gaps = []
        
        # Get data source
        source = self.data_sources.get(exchange)
        if not source:
            logger.error(f"Data source not found: {exchange}")
            return gaps
        
        try:
            # Get existing data range
            existing_data = await self._get_existing_data_range(symbol, exchange, data_type)
            
            if not existing_data:
                # No existing data, download full history
                end_time = datetime.now()
                start_time = end_time - timedelta(days=365)  # 1 year of data
                
                job_id = await self.create_backfill_job(
                    symbol=symbol,
                    exchange=exchange,
                    data_type=data_type,
                    start_time=start_time,
                    end_time=end_time,
                    priority=2  # Higher priority for initial download
                )
                
                gaps.append(job_id)
            else:
                # Check for gaps within existing data
                gap_periods = await self._find_gap_periods(
                    symbol, exchange, data_type, existing_data
                )
                
                for gap_start, gap_end in gap_periods:
                    job_id = await self.create_backfill_job(
                        symbol=symbol,
                        exchange=exchange,
                        data_type=data_type,
                        start_time=gap_start,
                        end_time=gap_end,
                        priority=3  # Lower priority for gaps
                    )
                    
                    gaps.append(job_id)
        
        except Exception as e:
            logger.error(f"Error detecting gaps for {symbol}: {e}")
        
        return gaps
    
    async def _worker(self, worker_name: str) -> None:
        """Worker task for processing backfill jobs."""
        logger.info(f"Worker {worker_name} started")
        
        while self._running:
            try:
                # Get job from queue
                job = await asyncio.wait_for(self.job_queue.get(), timeout=1.0)
                
                # Process job
                await self._process_job(job)
                
                # Mark as completed
                self.job_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _process_job(self, job: BackfillJob) -> None:
        """Process a backfill job."""
        job.status = BackfillStatus.RUNNING
        job.started_at = datetime.now()
        self.active_jobs[job.id] = job
        
        try:
            # Get data source
            source = self.data_sources.get(job.exchange)
            if not source:
                raise RuntimeError(f"Data source not found: {job.exchange}")
            
            # Process based on data type
            if job.data_type == "ohlcv" and job.timeframe:
                await self._backfill_ohlcv(source, job)
            else:
                await self._backfill_market_data(source, job)
            
            # Mark as completed
            job.status = BackfillStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress = 1.0
            
            self.stats['completed_jobs'] += 1
            self.stats['data_points_filled'] += int(job.end_time.timestamp() - job.start_time.timestamp())
            
            logger.info(f"Completed backfill job: {job.id}")
        
        except Exception as e:
            job.status = BackfillStatus.FAILED
            job.error_message = str(e)
            job.retry_count += 1
            
            self.stats['failed_jobs'] += 1
            
            logger.error(f"Failed backfill job {job.id}: {e}")
            
            # Retry if within limits
            if job.retry_count < job.max_retries:
                job.status = BackfillStatus.PENDING
                job.error_message = None
                await asyncio.sleep(self.config.retry_delay)
                await self.job_queue.put(job)
        
        finally:
            # Remove from active jobs
            self.active_jobs.pop(job.id, None)
            self.completed_jobs.append(job)
            
            # Update database
            await self._save_job(job)
        
        self.stats['last_activity'] = datetime.now()
    
    async def _backfill_ohlcv(self, source: Any, job: BackfillJob) -> None:
        """Backfill OHLCV data."""
        total_duration = (job.end_time - job.start_time).total_seconds()
        batch_duration = self.config.batch_size * self._get_timeframe_seconds(job.timeframe)
        
        current_time = job.start_time
        
        while current_time < job.end_time and job.status == BackfillStatus.RUNNING:
            batch_end = min(current_time + timedelta(seconds=batch_duration), job.end_time)
            
            # Request backfill from data source
            if hasattr(source, 'request_backfill'):
                source.request_backfill(
                    symbol=job.symbol,
                    data_type=job.data_type,
                    start_time=current_time,
                    end_time=batch_end,
                    timeframe=job.timeframe,
                    reason=f"backfill_job_{job.id}"
                )
            
            # Update progress
            elapsed = (current_time - job.start_time).total_seconds()
            job.progress = min(elapsed / total_duration, 1.0)
            
            # Save job state
            await self._save_job(job)
            
            current_time = batch_end
            await asyncio.sleep(0.1)  # Rate limiting
    
    async def _backfill_market_data(self, source: Any, job: BackfillJob) -> None:
        """Backfill other market data types."""
        # Implementation for ticker, orderbook, trades backfill
        pass
    
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
    
    async def _scheduler(self) -> None:
        """Background scheduler for automatic gap detection."""
        logger.info("Backfill scheduler started")
        
        while self._running:
            try:
                # Detect gaps for all symbols
                for exchange_name, source in self.data_sources.items():
                    if hasattr(source, 'config') and hasattr(source.config, 'symbols'):
                        for symbol in source.config.symbols:
                            for data_type in source.config.data_types:
                                await self.detect_gaps(symbol, exchange_name, data_type)
                
                await asyncio.sleep(self.config.schedule_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)
        
        logger.info("Backfill scheduler stopped")
    
    async def _load_pending_jobs(self) -> None:
        """Load pending jobs from database."""
        with sqlite3.connect(self.config.database_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM backfill_jobs 
                WHERE status = 'pending' 
                ORDER BY priority DESC, created_at ASC
            """)
            
            for row in cursor.fetchall():
                job = BackfillJob(
                    id=row[0],
                    symbol=row[1],
                    exchange=row[2],
                    data_type=row[3],
                    timeframe=row[4],
                    start_time=datetime.fromisoformat(row[5]),
                    end_time=datetime.fromisoformat(row[6]),
                    priority=row[7],
                    status=BackfillStatus(row[8]),
                    created_at=datetime.fromisoformat(row[9]),
                    started_at=datetime.fromisoformat(row[10]) if row[10] else None,
                    completed_at=datetime.fromisoformat(row[11]) if row[11] else None,
                    error_message=row[12],
                    retry_count=row[13],
                    max_retries=row[14],
                    progress=row[15]
                )
                
                await self.job_queue.put(job)
        
        logger.info(f"Loaded {self.job_queue.qsize()} pending jobs from database")
    
    async def _save_job(self, job: BackfillJob) -> None:
        """Save job to database."""
        with sqlite3.connect(self.config.database_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO backfill_jobs (
                    id, symbol, exchange, data_type, timeframe, start_time, end_time,
                    priority, status, created_at, started_at, completed_at,
                    error_message, retry_count, max_retries, progress
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.id, job.symbol, job.exchange, job.data_type, job.timeframe,
                job.start_time.isoformat(), job.end_time.isoformat(), job.priority,
                job.status.value, job.created_at.isoformat(),
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.error_message, job.retry_count, job.max_retries, job.progress
            ))
            conn.commit()
    
    async def _get_existing_data_range(self, symbol: str, exchange: str, data_type: str) -> Optional[tuple]:
        """Get existing data range for symbol."""
        # This would check existing data files or database
        # For now, return None to indicate no existing data
        return None
    
    async def _find_gap_periods(self, symbol: str, exchange: str, data_type: str, existing_data: tuple) -> List[tuple]:
        """Find gap periods in existing data."""
        # This would analyze existing data for gaps
        # For now, return empty list
        return []
    
    def get_job_status(self, job_id: str) -> Optional[BackfillJob]:
        """Get job status."""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        for job in self.completed_jobs:
            if job.id == job_id:
                return job
        
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status and statistics."""
        return {
            'queue_size': self.job_queue.qsize(),
            'active_jobs': len(self.active_jobs),
            'completed_jobs': len(self.completed_jobs),
            'stats': self.stats.copy(),
            'running': self._running
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a backfill job."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = BackfillStatus.CANCELLED
            return True
        
        return False
    
    async def clear_completed_jobs(self, older_than_days: int = 7) -> int:
        """Clear completed jobs older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        # Remove from memory
        self.completed_jobs = [
            job for job in self.completed_jobs
            if job.completed_at and job.completed_at > cutoff_date
        ]
        
        # Remove from database
        with sqlite3.connect(self.config.database_path) as conn:
            cursor = conn.execute("""
                DELETE FROM backfill_jobs 
                WHERE status IN ('completed', 'failed', 'cancelled') 
                AND completed_at < ?
            """, (cutoff_date.isoformat(),))
            
            deleted_count = cursor.rowcount
            conn.commit()
        
        logger.info(f"Cleared {deleted_count} old completed jobs")
        return deleted_count


# Convenience functions
def create_backfill_manager(
    max_concurrent_jobs: int = 3,
    enable_scheduling: bool = True,
    database_path: str = "data/backfill.db"
) -> BackfillManager:
    """Create backfill manager with default configuration."""
    config = BackfillConfig(
        max_concurrent_jobs=max_concurrent_jobs,
        enable_scheduling=enable_scheduling,
        database_path=database_path
    )
    
    return BackfillManager(config)
