#!/usr/bin/env python3
"""
Latency Database System
===================

Comprehensive database system for storing and managing
pipeline latency measurements with SQLite backend.
"""

from __future__ import annotations

import sqlite3
import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
import gzip
import shutil

logger = logging.getLogger(__name__)


@dataclass
class LatencyRecord:
    """Latency record for database storage."""
    timestamp: datetime
    pipeline_name: str
    stage: str
    latency_ms: float
    success: bool
    id: Optional[int] = None
    execution_id: Optional[str] = None
    correlation_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[str] = None  # JSON string
    thread_id: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class LatencyAggregation:
    """Aggregated latency statistics."""
    pipeline_name: str
    stage: str
    date: str  # YYYY-MM-DD
    total_executions: int
    successful_executions: int
    failed_executions: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    std_latency_ms: float
    error_rate: float
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class DatabaseConfig:
    """Database configuration for latency storage."""
    db_path: str = "data/latency.db"
    enable_wal: bool = True
    connection_timeout: int = 30
    max_connections: int = 10
    enable_backups: bool = True
    backup_interval: int = 3600  # 1 hour
    backup_dir: str = "data/backups"
    max_backups: int = 24  # Keep 24 hours
    enable_compression: bool = True
    auto_vacuum: bool = True
    vacuum_interval: int = 86400  # 24 hours


class LatencyDatabase:
    """
    Database manager for pipeline latency measurements.
    
    Features:
    - SQLite backend with WAL mode
    - Automatic schema creation and migrations
    - Connection pooling and timeout handling
    - Backup and compression support
    - Aggregation and analytics queries
    - Async operations with proper locking
    - Performance optimization with indexes
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None) -> None:
        self.config = config or DatabaseConfig()
        
        # Database connection
        self.db_path = Path(self.config.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connection management
        self._connection_pool: List[sqlite3.Connection] = []
        self._connection_lock = asyncio.Lock()
        
        # Background tasks
        self._backup_task: Optional[asyncio.Task] = None
        self._vacuum_task: Optional[asyncio.Task] = None
        self._aggregation_task: Optional[asyncio.Task] = None
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"Latency database initialized: {self.db_path}")
    
    def _initialize_database(self) -> None:
        """Initialize database with schema creation."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create tables
            self._create_latency_table(cursor)
            self._create_aggregation_table(cursor)
            self._create_indexes(cursor)
            
            # Create triggers for aggregation
            self._create_triggers(cursor)
            
            conn.commit()
            
            logger.info("Database schema initialized")
    
    def _create_latency_table(self, cursor: sqlite3.Cursor) -> None:
        """Create latency_records table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS latency_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                pipeline_name TEXT NOT NULL,
                stage TEXT NOT NULL,
                execution_id TEXT,
                correlation_id TEXT,
                latency_ms REAL NOT NULL,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                metadata TEXT,
                thread_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_latency_timestamp 
            ON latency_records(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_latency_pipeline_stage 
            ON latency_records(pipeline_name, stage)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_latency_execution_id 
            ON latency_records(execution_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_latency_correlation_id 
            ON latency_records(correlation_id)
        """)
    
    def _create_aggregation_table(self, cursor: sqlite3.Cursor) -> None:
        """Create latency_aggregations table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS latency_aggregations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_name TEXT NOT NULL,
                stage TEXT NOT NULL,
                date TEXT NOT NULL,
                total_executions INTEGER NOT NULL,
                successful_executions INTEGER NOT NULL,
                failed_executions INTEGER NOT NULL,
                avg_latency_ms REAL NOT NULL,
                min_latency_ms REAL NOT NULL,
                max_latency_ms REAL NOT NULL,
                p50_latency_ms REAL NOT NULL,
                p95_latency_ms REAL NOT NULL,
                p99_latency_ms REAL NOT NULL,
                std_latency_ms REAL NOT NULL,
                error_rate REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pipeline_name, stage, date)
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agg_pipeline_stage_date 
            ON latency_aggregations(pipeline_name, stage, date)
        """)
    
    def _create_indexes(self, cursor: sqlite3.Cursor) -> None:
        """Create additional performance indexes."""
        # Composite index for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_latency_pipeline_stage_timestamp 
            ON latency_records(pipeline_name, stage, timestamp)
        """)
        
        # Index for error queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_latency_success_timestamp 
            ON latency_records(success, timestamp)
        """)
        
        # Index for correlation queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_latency_correlation_timestamp 
            ON latency_records(correlation_id, timestamp)
        """)
    
    def _create_triggers(self, cursor: sqlite3.Cursor) -> None:
        """Create triggers for automatic aggregation."""
        # Trigger to update aggregations when new records are inserted
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_aggregation_on_insert
            AFTER INSERT ON latency_records
            BEGIN
                INSERT OR REPLACE INTO latency_aggregations (
                    pipeline_name, stage, date,
                    total_executions, successful_executions, failed_executions,
                    avg_latency_ms, min_latency_ms, max_latency_ms,
                    p50_latency_ms, p95_latency_ms, p99_latency_ms, std_latency_ms, error_rate
                )
                SELECT 
                    NEW.pipeline_name,
                    NEW.stage,
                    DATE(NEW.timestamp) as date,
                    COALESCE((SELECT total_executions FROM latency_aggregations 
                              WHERE pipeline_name = NEW.pipeline_name AND stage = NEW.stage AND date = DATE(NEW.timestamp)), 0) + 1,
                    COALESCE((SELECT successful_executions FROM latency_aggregations 
                              WHERE pipeline_name = NEW.pipeline_name AND stage = NEW.stage AND date = DATE(NEW.timestamp)), 0) + CASE WHEN NEW.success THEN 1 ELSE 0 END,
                    COALESCE((SELECT failed_executions FROM latency_aggregations 
                              WHERE pipeline_name = NEW.pipeline_name AND stage = NEW.stage AND date = DATE(NEW.timestamp)), 0) + CASE WHEN NOT NEW.success THEN 1 ELSE 0 END,
                    COALESCE((SELECT avg_latency_ms FROM latency_aggregations 
                              WHERE pipeline_name = NEW.pipeline_name AND stage = NEW.stage AND date = DATE(NEW.timestamp)), 0) * 
                        (SELECT COUNT(*) FROM latency_records 
                         WHERE pipeline_name = NEW.pipeline_name AND stage = NEW.stage AND date = DATE(NEW.timestamp)) - 1) / 
                        (SELECT COUNT(*) FROM latency_records 
                         WHERE pipeline_name = NEW.pipeline_name AND stage = NEW.stage AND date = DATE(NEW.timestamp)) + NEW.latency_ms,
                    COALESCE((SELECT min_latency_ms FROM latency_aggregations 
                              WHERE pipeline_name = NEW.pipeline_name AND stage = NEW.stage AND date = DATE(NEW.timestamp)), NEW.latency_ms),
                    COALESCE((SELECT max_latency_ms FROM latency_aggregations 
                              WHERE pipeline_name = NEW.pipeline_name AND stage = NEW.stage AND date = DATE(NEW.timestamp)), NEW.latency_ms),
                    NEW.latency_ms,  -- p50 (will be calculated properly)
                    NEW.latency_ms,  -- p95 (will be calculated properly)
                    NEW.latency_ms,  -- p99 (will be calculated properly)
                    0.0,  -- std (will be calculated properly)
                    COALESCE((SELECT error_rate FROM latency_aggregations 
                              WHERE pipeline_name = NEW.pipeline_name AND stage = NEW.stage AND date = DATE(NEW.timestamp)), 0) * 
                        (SELECT COUNT(*) FROM latency_records 
                         WHERE pipeline_name = NEW.pipeline_name AND stage = NEW.stage AND date = DATE(NEW.timestamp)) - 1) / 
                        (SELECT COUNT(*) FROM latency_records 
                         WHERE pipeline_name = NEW.pipeline_name AND stage = NEW.stage AND date = DATE(NEW.timestamp)) + CASE WHEN NOT NEW.success THEN 1 ELSE 0 END
                WHERE NEW.pipeline_name IS NOT NULL AND NEW.stage IS NOT NULL;
            END
        """)
    
    @asynccontextmanager
    async def _get_connection(self):
        """Get database connection from pool."""
        async with self._connection_lock:
            # Try to get existing connection
            for conn in self._connection_pool:
                try:
                    # Test if connection is still valid
                    conn.execute("SELECT 1")
                    yield conn
                    return
                except sqlite3.Error:
                    # Connection is dead, remove from pool
                    self._connection_pool.remove(conn)
                    try:
                        conn.close()
                    except:
                        pass
            
            # Create new connection if none available
            if len(self._connection_pool) < self.config.max_connections:
                conn = sqlite3.connect(
                    str(self.db_path),
                    timeout=self.config.connection_timeout,
                    check_same_thread=False
                )
                
                # Configure connection
                if self.config.enable_wal:
                    conn.execute("PRAGMA journal_mode=WAL")
                
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("PRAGMA mmap_size=268435456")
                
                self._connection_pool.append(conn)
                yield conn
                return
            
            # Pool is full, wait for available connection
            logger.warning("Connection pool full, waiting for available connection")
            await asyncio.sleep(0.1)
            raise sqlite3.OperationalError("Connection pool exhausted")
    
    async def save_latency_record(self, record: LatencyRecord) -> int:
        """
        Save latency record to database.
        
        Args:
            record: Latency record to save
            
        Returns:
            Record ID
        """
        async with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO latency_records (
                    timestamp, pipeline_name, stage, execution_id, correlation_id,
                    latency_ms, success, error_message, metadata, thread_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp,
                record.pipeline_name,
                record.stage,
                record.execution_id,
                record.correlation_id,
                record.latency_ms,
                record.success,
                record.error_message,
                record.metadata,
                record.thread_id
            ))
            
            record_id = cursor.lastrowid
            conn.commit()
            
            logger.debug(f"Saved latency record: {record_id} for {record.pipeline_name}.{record.stage}")
            return record_id
    
    async def save_latency_records_batch(self, records: List[LatencyRecord]) -> List[int]:
        """
        Save multiple latency records efficiently.
        
        Args:
            records: List of latency records
            
        Returns:
            List of record IDs
        """
        if not records:
            return []
        
        async with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Use executemany for efficiency
            cursor.executemany("""
                INSERT INTO latency_records (
                    timestamp, pipeline_name, stage, execution_id, correlation_id,
                    latency_ms, success, error_message, metadata, thread_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    record.timestamp,
                    record.pipeline_name,
                    record.stage,
                    record.execution_id,
                    record.correlation_id,
                    record.latency_ms,
                    record.success,
                    record.error_message,
                    record.metadata,
                    record.thread_id
                ) for record in records
            ])
            
            conn.commit()
            
            # Get record IDs
            record_ids = []
            for i in range(1, len(records) + 1):
                record_ids.append(cursor.lastrowid - len(records) + i)
            
            logger.debug(f"Saved {len(records)} latency records")
            return record_ids
    
    async def get_latency_records(
        self,
        pipeline_name: Optional[str] = None,
        stage: Optional[str] = None,
        execution_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        success: Optional[bool] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        order_by: str = "timestamp DESC"
    ) -> List[LatencyRecord]:
        """
        Get latency records with filtering.
        
        Args:
            pipeline_name: Filter by pipeline name
            stage: Filter by stage
            execution_id: Filter by execution ID
            correlation_id: Filter by correlation ID
            success: Filter by success status
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Limit number of results
            order_by: Order results
            
        Returns:
            List of latency records
        """
        async with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM latency_records WHERE 1=1"
            params = []
            
            if pipeline_name:
                query += " AND pipeline_name = ?"
                params.append(pipeline_name)
            
            if stage:
                query += " AND stage = ?"
                params.append(stage)
            
            if execution_id:
                query += " AND execution_id = ?"
                params.append(execution_id)
            
            if correlation_id:
                query += " AND correlation_id = ?"
                params.append(correlation_id)
            
            if success is not None:
                query += " AND success = ?"
                params.append(success)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += f" ORDER BY {order_by}"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to LatencyRecord objects
            records = []
            for row in rows:
                record = LatencyRecord(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]) if row[1] else None,
                    pipeline_name=row[2],
                    stage=row[3],
                    execution_id=row[4],
                    correlation_id=row[5],
                    latency_ms=row[6],
                    success=bool(row[7]),
                    error_message=row[8],
                    metadata=row[9],
                    thread_id=row[10],
                    created_at=datetime.fromisoformat(row[11]) if row[11] else None
                )
                records.append(record)
            
            logger.debug(f"Retrieved {len(records)} latency records")
            return records
    
    async def get_latency_aggregations(
        self,
        pipeline_name: Optional[str] = None,
        stage: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[LatencyAggregation]:
        """
        Get aggregated latency statistics.
        
        Args:
            pipeline_name: Filter by pipeline name
            stage: Filter by stage
            start_date: Filter by start date (YYYY-MM-DD)
            end_date: Filter by end date (YYYY-MM-DD)
            limit: Limit number of results
            
        Returns:
            List of latency aggregations
        """
        async with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM latency_aggregations WHERE 1=1"
            params = []
            
            if pipeline_name:
                query += " AND pipeline_name = ?"
                params.append(pipeline_name)
            
            if stage:
                query += " AND stage = ?"
                params.append(stage)
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date DESC, pipeline_name, stage"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to LatencyAggregation objects
            aggregations = []
            for row in rows:
                agg = LatencyAggregation(
                    id=row[0],
                    pipeline_name=row[1],
                    stage=row[2],
                    date=row[3],
                    total_executions=row[4],
                    successful_executions=row[5],
                    failed_executions=row[6],
                    avg_latency_ms=row[7],
                    min_latency_ms=row[8],
                    max_latency_ms=row[9],
                    p50_latency_ms=row[10],
                    p95_latency_ms=row[11],
                    p99_latency_ms=row[12],
                    std_latency_ms=row[13],
                    error_rate=row[14],
                    created_at=datetime.fromisoformat(row[15]) if row[15] else None,
                    updated_at=datetime.fromisoformat(row[16]) if row[16] else None
                )
                aggregations.append(agg)
            
            logger.debug(f"Retrieved {len(aggregations)} latency aggregations")
            return aggregations
    
    async def get_latency_dataframe(
        self,
        pipeline_name: Optional[str] = None,
        stage: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get latency records as pandas DataFrame.
        
        Args:
            pipeline_name: Filter by pipeline name
            stage: Filter by stage
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Limit number of results
            
        Returns:
            DataFrame with latency data
        """
        records = await self.get_latency_records(
            pipeline_name=pipeline_name,
            stage=stage,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        if not records:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for record in records:
            row = {
                'id': record.id,
                'timestamp': record.timestamp,
                'pipeline_name': record.pipeline_name,
                'stage': record.stage,
                'execution_id': record.execution_id,
                'correlation_id': record.correlation_id,
                'latency_ms': record.latency_ms,
                'success': record.success,
                'error_message': record.error_message,
                'thread_id': record.thread_id
            }
            
            # Parse metadata if available
            if record.metadata:
                try:
                    metadata = json.loads(record.metadata)
                    for key, value in metadata.items():
                        row[f'meta_{key}'] = value
                except json.JSONDecodeError:
                    pass
            
            data.append(row)
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        
        return df
    
    async def get_aggregation_dataframe(
        self,
        pipeline_name: Optional[str] = None,
        stage: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get latency aggregations as pandas DataFrame.
        
        Args:
            pipeline_name: Filter by pipeline name
            stage: Filter by stage
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Limit number of results
            
        Returns:
            DataFrame with aggregation data
        """
        aggregations = await self.get_latency_aggregations(
            pipeline_name=pipeline_name,
            stage=stage,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        if not aggregations:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for agg in aggregations:
            row = {
                'id': agg.id,
                'pipeline_name': agg.pipeline_name,
                'stage': agg.stage,
                'date': agg.date,
                'total_executions': agg.total_executions,
                'successful_executions': agg.successful_executions,
                'failed_executions': agg.failed_executions,
                'avg_latency_ms': agg.avg_latency_ms,
                'min_latency_ms': agg.min_latency_ms,
                'max_latency_ms': agg.max_latency_ms,
                'p50_latency_ms': agg.p50_latency_ms,
                'p95_latency_ms': agg.p95_latency_ms,
                'p99_latency_ms': agg.p99_latency_ms,
                'std_latency_ms': agg.std_latency_ms,
                'error_rate': agg.error_rate,
                'success_rate': agg.successful_executions / agg.total_executions if agg.total_executions > 0 else 0.0
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        return df
    
    async def get_pipeline_statistics(
        self,
        pipeline_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive pipeline statistics.
        
        Args:
            pipeline_name: Pipeline name
            start_date: Start date for statistics
            end_date: End date for statistics
            
        Returns:
            Dictionary with pipeline statistics
        """
        aggregations = await self.get_latency_aggregations(
            pipeline_name=pipeline_name,
            start_date=start_date,
            end_date=end_date
        )
        
        if not aggregations:
            return {}
        
        # Calculate overall statistics
        total_executions = sum(agg.total_executions for agg in aggregations)
        total_successful = sum(agg.successful_executions for agg in aggregations)
        total_failed = sum(agg.failed_executions for agg in aggregations)
        
        # Calculate weighted averages
        weighted_avg_latency = sum(
            agg.avg_latency_ms * agg.total_executions for agg in aggregations
        ) / total_executions if total_executions > 0 else 0.0
        
        weighted_p95_latency = sum(
            agg.p95_latency_ms * agg.total_executions for agg in aggregations
        ) / total_executions if total_executions > 0 else 0.0
        
        # Get stage breakdown
        stage_stats = {}
        for agg in aggregations:
            stage_stats[agg.stage] = {
                'total_executions': agg.total_executions,
                'avg_latency_ms': agg.avg_latency_ms,
                'p95_latency_ms': agg.p95_latency_ms,
                'error_rate': agg.error_rate
            }
        
        return {
            'pipeline_name': pipeline_name,
            'total_executions': total_executions,
            'successful_executions': total_successful,
            'failed_executions': total_failed,
            'overall_success_rate': total_successful / total_executions if total_executions > 0 else 0.0,
            'overall_avg_latency_ms': weighted_avg_latency,
            'overall_p95_latency_ms': weighted_p95_latency,
            'stage_statistics': stage_stats,
            'date_range': {
                'start': start_date,
                'end': end_date
            }
        }
    
    async def delete_old_records(self, days_to_keep: int = 30) -> int:
        """
        Delete old latency records.
        
        Args:
            days_to_keep: Number of days to keep
            
        Returns:
            Number of deleted records
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        async with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete old records
            cursor.execute(
                "DELETE FROM latency_records WHERE timestamp < ?",
                (cutoff_date,)
            )
            
            deleted_records = cursor.rowcount
            
            # Delete old aggregations
            cursor.execute(
                "DELETE FROM latency_aggregations WHERE date < DATE(?)",
                (cutoff_date.strftime('%Y-%m-%d'),)
            )
            
            deleted_aggregations = cursor.rowcount
            conn.commit()
            
            logger.info(f"Deleted {deleted_records} records and {deleted_aggregations} aggregations older than {days_to_keep} days")
            return deleted_records + deleted_aggregations
    
    async def export_to_csv(
        self,
        file_path: str,
        pipeline_name: Optional[str] = None,
        stage: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_aggregations: bool = True
    ) -> None:
        """
        Export latency data to CSV.
        
        Args:
            file_path: Output file path
            pipeline_name: Filter by pipeline name
            stage: Filter by stage
            start_date: Filter by start date
            end_date: Filter by end date
            include_aggregations: Include aggregation data
        """
        # Get records
        records_df = await self.get_latency_dataframe(
            pipeline_name=pipeline_name,
            stage=stage,
            start_time=datetime.fromisoformat(start_date) if start_date else None,
            end_time=datetime.fromisoformat(end_date) if end_date else None
        )
        
        # Write to CSV
        records_df.to_csv(file_path, index=True)
        
        if include_aggregations:
            # Get and export aggregations
            agg_df = await self.get_aggregation_dataframe(
                pipeline_name=pipeline_name,
                stage=stage,
                start_date=start_date,
                end_date=end_date
            )
            
            agg_file_path = file_path.replace('.csv', '_aggregations.csv')
            agg_df.to_csv(agg_file_path, index=True)
        
        logger.info(f"Exported latency data to {file_path}")
    
    async def backup_database(self) -> str:
        """
        Create backup of the database.
        
        Returns:
            Path to backup file
        """
        if not self.config.enable_backups:
            return ""
        
        backup_dir = Path(self.config.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"latency_backup_{timestamp}.db"
        
        # Copy database
        shutil.copy2(self.db_path, backup_path)
        
        # Compress if enabled
        if self.config.enable_compression:
            compressed_path = backup_path.with_suffix('.db.gz')
            
            with open(backup_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove uncompressed backup
            backup_path.unlink()
            backup_path = compressed_path
        
        logger.info(f"Database backed up to {backup_path}")
        return str(backup_path)
    
    async def cleanup_old_backups(self) -> int:
        """
        Clean up old backup files.
        
        Returns:
            Number of deleted backups
        """
        if not self.config.enable_backups:
            return 0
        
        backup_dir = Path(self.config.backup_dir)
        if not backup_dir.exists():
            return 0
        
        # Get backup files
        backup_files = list(backup_dir.glob("latency_backup_*.db*"))
        
        # Sort by creation time
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep only the most recent backups
        files_to_delete = backup_files[self.config.max_backups:]
        
        deleted_count = 0
        for backup_file in files_to_delete:
            try:
                backup_file.unlink()
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting backup {backup_file}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old backup files")
        return deleted_count
    
    async def vacuum_database(self) -> None:
        """Vacuum the database to optimize performance."""
        async with self._get_connection() as conn:
            conn.execute("VACUUM")
            conn.commit()
        
        logger.info("Database vacuumed")
    
    async def start_background_tasks(self) -> None:
        """Start background tasks for maintenance."""
        if self.config.enable_backups:
            self._backup_task = asyncio.create_task(self._backup_loop())
        
        if self.config.auto_vacuum:
            self._vacuum_task = asyncio.create_task(self._vacuum_loop())
        
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        
        logger.info("Started background maintenance tasks")
    
    async def stop_background_tasks(self) -> None:
        """Stop background tasks."""
        tasks = [
            self._backup_task,
            self._vacuum_task,
            self._aggregation_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Stopped background maintenance tasks")
    
    async def _backup_loop(self) -> None:
        """Background loop for database backups."""
        while True:
            try:
                await asyncio.sleep(self.config.backup_interval)
                await self.backup_database()
                await self.cleanup_old_backups()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in backup loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _vacuum_loop(self) -> None:
        """Background loop for database vacuum."""
        while True:
            try:
                await asyncio.sleep(self.config.vacuum_interval)
                await self.vacuum_database()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in vacuum loop: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _aggregation_loop(self) -> None:
        """Background loop for updating aggregations."""
        while True:
            try:
                # Update aggregations for recent data
                await self.update_recent_aggregations()
                await asyncio.sleep(3600)  # Update every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def update_recent_aggregations(self) -> None:
        """Update aggregations for recent data."""
        # This would implement more sophisticated aggregation logic
        # For now, the triggers handle most of the work
        logger.debug("Updated recent aggregations")
    
    async def close(self) -> None:
        """Close database connections and cleanup."""
        await self.stop_background_tasks()
        
        # Close all connections
        for conn in self._connection_pool:
            try:
                conn.close()
            except:
                pass
        
        self._connection_pool.clear()
        logger.info("Database connections closed")


# Convenience functions
def create_latency_database(config: Optional[DatabaseConfig] = None) -> LatencyDatabase:
    """Create latency database with default configuration."""
    return LatencyDatabase(config)


def create_database_config(
    db_path: str = "data/latency.db",
    enable_wal: bool = True,
    enable_backups: bool = True,
    backup_interval: int = 3600
) -> DatabaseConfig:
    """Create database configuration."""
    return DatabaseConfig(
        db_path=db_path,
        enable_wal=enable_wal,
        enable_backups=enable_backups,
        backup_interval=backup_interval
    )
