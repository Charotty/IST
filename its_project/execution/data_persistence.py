#!/usr/bin/env python3
"""
Data Persistence for Trading System
================================

Comprehensive data persistence supporting database and CSV storage
for trades, PnL records, and performance metrics.
"""

from __future__ import annotations

import sqlite3
import csv
import json
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gzip
import shutil

from .pnl_tracker import TradeRecord, PnLRecord, PositionSnapshot

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Configuration for database persistence."""
    db_path: str = "data/trading_system.db"
    backup_enabled: bool = True
    backup_interval: int = 3600  # seconds
    max_backups: int = 10
    connection_timeout: int = 30
    enable_wal: bool = True
    foreign_keys: bool = True


@dataclass
class CSVConfig:
    """Configuration for CSV export."""
    export_dir: str = "exports"
    compression: bool = True
    date_format: str = "%Y-%m-%d %H:%M:%S"
    include_headers: bool = True
    encoding: str = "utf-8"
    batch_size: int = 10000


class DatabaseManager:
    """
    Advanced database manager with SQLite backend.
    
    Features:
    - Multiple table management
    - Transaction support
    - Backup functionality
    - Connection pooling
    - Query optimization
    """
    
    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config
        self.db_path = Path(config.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connection management
        self._connection: Optional[sqlite3.Connection] = None
        self._connection_lock = asyncio.Lock()
        
        # Backup management
        self._backup_task: Optional[asyncio.Task] = None
        self._backup_running = False
        
        # Initialize database
        self._init_database()
        
        # Start backup task if enabled
        if config.backup_enabled:
            self._start_backup_task()
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            # Enable optimizations
            if self.config.enable_wal:
                conn.execute("PRAGMA journal_mode=WAL")
            if self.config.foreign_keys:
                conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            
            # Create tables
            self._create_tables(conn)
            
            # Create indexes
            self._create_indexes(conn)
    
    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Create all database tables."""
        
        # Trades table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
                quantity REAL NOT NULL CHECK (quantity > 0),
                price REAL NOT NULL CHECK (price > 0),
                timestamp INTEGER NOT NULL,
                commission REAL DEFAULT 0.0 CHECK (commission >= 0),
                fees TEXT DEFAULT '[]',
                strategy_id TEXT,
                order_id TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        # PnL records table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pnl_records (
                record_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                entry_time INTEGER NOT NULL,
                exit_time INTEGER,
                entry_price REAL NOT NULL CHECK (entry_price > 0),
                exit_price REAL CHECK (exit_price > 0),
                quantity REAL NOT NULL CHECK (quantity > 0),
                side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
                realized_pnl REAL NOT NULL,
                commission REAL DEFAULT 0.0 CHECK (commission >= 0),
                fees TEXT DEFAULT '[]',
                strategy_id TEXT,
                trade_duration INTEGER,
                pnl_percentage REAL,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        # Positions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                quantity REAL NOT NULL,
                avg_price REAL NOT NULL CHECK (avg_price > 0),
                unrealized_pnl REAL DEFAULT 0.0,
                realized_pnl REAL DEFAULT 0.0,
                total_pnl REAL DEFAULT 0.0,
                last_price REAL CHECK (last_price > 0),
                last_update INTEGER,
                trades_count INTEGER DEFAULT 0 CHECK (trades_count >= 0),
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        # Performance metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                metric_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metadata TEXT DEFAULT '{}',
                strategy_id TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        # Daily summary table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary (
                date TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                realized_pnl REAL DEFAULT 0.0,
                unrealized_pnl REAL DEFAULT 0.0,
                max_drawdown REAL DEFAULT 0.0,
                volume REAL DEFAULT 0.0,
                commissions REAL DEFAULT 0.0,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        # Strategy performance table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                strategy_id TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                realized_pnl REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.0,
                profit_factor REAL DEFAULT 0.0,
                avg_win REAL DEFAULT 0.0,
                avg_loss REAL DEFAULT 0.0,
                largest_win REAL DEFAULT 0.0,
                largest_loss REAL DEFAULT 0.0,
                sharpe_ratio REAL DEFAULT 0.0,
                max_drawdown REAL DEFAULT 0.0,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        # System metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)
    
    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create database indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id)",
            "CREATE INDEX IF NOT EXISTS idx_trades_side ON trades(side)",
            "CREATE INDEX IF NOT EXISTS idx_pnl_symbol ON pnl_records(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_pnl_entry_time ON pnl_records(entry_time)",
            "CREATE INDEX IF NOT EXISTS idx_pnl_strategy ON pnl_records(strategy_id)",
            "CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_type ON performance_metrics(metric_type)",
            "CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_summary(date)",
            "CREATE INDEX IF NOT EXISTS idx_strategy_id ON strategy_performance(strategy_id)"
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper configuration."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.config.connection_timeout,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn
    
    async def save_trade(self, trade: TradeRecord) -> bool:
        """Save trade record to database."""
        try:
            async with self._connection_lock:
                with self._get_connection() as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO trades VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                    """, (
                        trade.trade_id, trade.symbol, trade.side, trade.quantity,
                        trade.price, trade.timestamp.timestamp(), trade.commission,
                        json.dumps(trade.fees), trade.strategy_id, trade.order_id,
                        int(datetime.now().timestamp()), int(datetime.now().timestamp())
                    ))
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving trade {trade.trade_id}: {e}")
            return False
    
    async def save_trades_batch(self, trades: List[TradeRecord]) -> int:
        """Save multiple trades in a batch."""
        if not trades:
            return 0
        
        try:
            async with self._connection_lock:
                with self._get_connection() as conn:
                    data = []
                    for trade in trades:
                        data.append((
                            trade.trade_id, trade.symbol, trade.side, trade.quantity,
                            trade.price, trade.timestamp.timestamp(), trade.commission,
                            json.dumps(trade.fees), trade.strategy_id, trade.order_id,
                            int(datetime.now().timestamp()), int(datetime.now().timestamp())
                        ))
                    
                    conn.executemany("""
                        INSERT OR REPLACE INTO trades VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                    """, data)
                    conn.commit()
            return len(trades)
        except Exception as e:
            logger.error(f"Error saving batch trades: {e}")
            return 0
    
    async def save_pnl_record(self, record: PnLRecord) -> bool:
        """Save PnL record to database."""
        try:
            async with self._connection_lock:
                with self._get_connection() as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO pnl_records VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                    """, (
                        record.record_id, record.symbol, record.entry_time.timestamp(),
                        record.exit_time.timestamp() if record.exit_time else None,
                        record.entry_price, record.exit_price, record.quantity,
                        record.side, record.realized_pnl, record.commission,
                        json.dumps(record.fees), record.strategy_id,
                        record.trade_duration.total_seconds() if record.trade_duration else None,
                        record.pnl_percentage, int(datetime.now().timestamp()),
                        int(datetime.now().timestamp())
                    ))
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving PnL record {record.record_id}: {e}")
            return False
    
    async def save_position(self, position: PositionSnapshot) -> bool:
        """Save position snapshot to database."""
        try:
            async with self._connection_lock:
                with self._get_connection() as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO positions VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                    """, (
                        position.symbol, position.quantity, position.avg_price,
                        position.unrealized_pnl, position.realized_pnl,
                        position.total_pnl, position.last_price,
                        position.last_update.timestamp() if position.last_update else None,
                        position.trades_count, int(datetime.now().timestamp()),
                        int(datetime.now().timestamp())
                    ))
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving position {position.symbol}: {e}")
            return False
    
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve trades with optional filtering."""
        try:
            with self._get_connection() as conn:
                query = "SELECT * FROM trades WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                if strategy_id:
                    query += " AND strategy_id = ?"
                    params.append(strategy_id)
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.timestamp())
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.timestamp())
                
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error retrieving trades: {e}")
            return []
    
    async def get_pnl_records(
        self,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve PnL records with optional filtering."""
        try:
            with self._get_connection() as conn:
                query = "SELECT * FROM pnl_records WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                if strategy_id:
                    query += " AND strategy_id = ?"
                    params.append(strategy_id)
                
                if start_time:
                    query += " AND entry_time >= ?"
                    params.append(start_time.timestamp())
                
                if end_time:
                    query += " AND entry_time <= ?"
                    params.append(end_time.timestamp())
                
                query += " ORDER BY entry_time DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error retrieving PnL records: {e}")
            return []
    
    async def _start_backup_task(self) -> None:
        """Start automatic backup task."""
        if not self._backup_running:
            self._backup_running = True
            self._backup_task = asyncio.create_task(self._backup_loop())
    
    async def _backup_loop(self) -> None:
        """Backup loop for automatic database backups."""
        while self._backup_running:
            try:
                await self._create_backup()
                await asyncio.sleep(self.config.backup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in backup loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _create_backup(self) -> bool:
        """Create database backup."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.db_path.parent / f"backup_{timestamp}.db"
            
            # Create backup
            shutil.copy2(self.db_path, backup_path)
            
            # Compress backup
            if backup_path.exists():
                with open(backup_path, 'rb') as f_in:
                    with gzip.open(f"{backup_path}.gz", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                backup_path.unlink()  # Remove uncompressed backup
            
            # Clean old backups
            await self._cleanup_old_backups()
            
            logger.info(f"Database backup created: {backup_path}.gz")
            return True
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    async def _cleanup_old_backups(self) -> None:
        """Clean up old backup files."""
        try:
            backup_files = list(self.db_path.parent.glob("backup_*.db.gz"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if len(backup_files) > self.config.max_backups:
                for old_backup in backup_files[self.config.max_backups:]:
                    old_backup.unlink()
                    logger.info(f"Removed old backup: {old_backup}")
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
    
    async def close(self) -> None:
        """Close database manager and cleanup."""
        self._backup_running = False
        if self._backup_task:
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
        
        # Create final backup
        if self.config.backup_enabled:
            await self._create_backup()


class CSVExporter:
    """
    Advanced CSV export functionality for trading data.
    
    Features:
    - Batch export support
    - Compression options
    - Custom formatting
    - Data validation
    - Progress tracking
    """
    
    def __init__(self, config: CSVConfig) -> None:
        self.config = config
        self.export_dir = Path(config.export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    async def export_trades(
        self,
        trades: List[Union[TradeRecord, Dict[str, Any]]],
        filename: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Export trades to CSV format.
        
        Args:
            trades: List of trade records
            filename: Output filename (auto-generated if None)
            include_metadata: Include additional metadata columns
            
        Returns:
            Path to exported file
        """
        if not trades:
            logger.warning("No trades to export")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trades_{timestamp}.csv"
        
        filepath = self.export_dir / filename
        
        # Prepare data
        if isinstance(trades[0], TradeRecord):
            data = []
            for trade in trades:
                row = {
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'timestamp': trade.timestamp.strftime(self.config.date_format),
                    'commission': trade.commission,
                    'fees': json.dumps(trade.fees),
                    'strategy_id': trade.strategy_id or '',
                    'order_id': trade.order_id or ''
                }
                
                if include_metadata:
                    row.update({
                        'notional': trade.quantity * trade.price,
                        'fee_total': trade.commission + sum(trade.fees.values()),
                        'timestamp_epoch': trade.timestamp.timestamp()
                    })
                
                data.append(row)
        else:
            # Assume dict format
            data = trades
        
        # Export to CSV
        try:
            await self._write_csv(data, filepath, self.config.include_headers)
            logger.info(f"Exported {len(trades)} trades to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error exporting trades: {e}")
            return ""
    
    async def export_pnl_records(
        self,
        pnl_records: List[Union[PnLRecord, Dict[str, Any]]],
        filename: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Export PnL records to CSV format.
        
        Args:
            pnl_records: List of PnL records
            filename: Output filename (auto-generated if None)
            include_metadata: Include additional metadata columns
            
        Returns:
            Path to exported file
        """
        if not pnl_records:
            logger.warning("No PnL records to export")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pnl_records_{timestamp}.csv"
        
        filepath = self.export_dir / filename
        
        # Prepare data
        if isinstance(pnl_records[0], PnLRecord):
            data = []
            for record in pnl_records:
                row = {
                    'record_id': record.record_id,
                    'symbol': record.symbol,
                    'entry_time': record.entry_time.strftime(self.config.date_format),
                    'exit_time': record.exit_time.strftime(self.config.date_format) if record.exit_time else '',
                    'entry_price': record.entry_price,
                    'exit_price': record.exit_price if record.exit_price else '',
                    'quantity': record.quantity,
                    'side': record.side,
                    'realized_pnl': record.realized_pnl,
                    'commission': record.commission,
                    'fees': json.dumps(record.fees),
                    'strategy_id': record.strategy_id or '',
                    'trade_duration': record.trade_duration.total_seconds() if record.trade_duration else '',
                    'pnl_percentage': record.pnl_percentage or ''
                }
                
                if include_metadata:
                    row.update({
                        'net_pnl': record.realized_pnl - record.commission,
                        'notional': record.quantity * record.entry_price,
                        'price_change': (record.exit_price - record.entry_price) if record.exit_price else '',
                        'price_change_pct': ((record.exit_price - record.entry_price) / record.entry_price * 100) if record.exit_price else ''
                    })
                
                data.append(row)
        else:
            # Assume dict format
            data = pnl_records
        
        # Export to CSV
        try:
            await self._write_csv(data, filepath, self.config.include_headers)
            logger.info(f"Exported {len(pnl_records)} PnL records to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error exporting PnL records: {e}")
            return ""
    
    async def export_positions(
        self,
        positions: List[Union[PositionSnapshot, Dict[str, Any]]],
        filename: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Export positions to CSV format.
        
        Args:
            positions: List of position snapshots
            filename: Output filename (auto-generated if None)
            include_metadata: Include additional metadata columns
            
        Returns:
            Path to exported file
        """
        if not positions:
            logger.warning("No positions to export")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"positions_{timestamp}.csv"
        
        filepath = self.export_dir / filename
        
        # Prepare data
        if isinstance(positions[0], PositionSnapshot):
            data = []
            for position in positions:
                row = {
                    'symbol': position.symbol,
                    'quantity': position.quantity,
                    'avg_price': position.avg_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl,
                    'total_pnl': position.total_pnl,
                    'last_price': position.last_price if position.last_price else '',
                    'last_update': position.last_update.strftime(self.config.date_format) if position.last_update else '',
                    'trades_count': position.trades_count
                }
                
                if include_metadata:
                    row.update({
                        'notional': abs(position.quantity * position.avg_price),
                        'pnl_percentage': (position.total_pnl / abs(position.quantity * position.avg_price) * 100) if position.quantity != 0 else 0,
                        'price_change': (position.last_price - position.avg_price) if position.last_price else '',
                        'price_change_pct': ((position.last_price - position.avg_price) / position.avg_price * 100) if position.last_price and position.avg_price != 0 else ''
                    })
                
                data.append(row)
        else:
            # Assume dict format
            data = positions
        
        # Export to CSV
        try:
            await self._write_csv(data, filepath, self.config.include_headers)
            logger.info(f"Exported {len(positions)} positions to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error exporting positions: {e}")
            return ""
    
    async def export_dataframe(
        self,
        df: pd.DataFrame,
        filename: str,
        include_index: bool = False
    ) -> str:
        """
        Export pandas DataFrame to CSV.
        
        Args:
            df: DataFrame to export
            filename: Output filename
            include_index: Include DataFrame index
            
        Returns:
            Path to exported file
        """
        filepath = self.export_dir / filename
        
        try:
            # Use pandas for efficient export
            df.to_csv(
                filepath,
                index=include_index,
                encoding=self.config.encoding,
                date_format=self.config.date_format,
                chunksize=self.config.batch_size if len(df) > self.config.batch_size else None
            )
            
            # Compress if enabled
            if self.config.compression:
                compressed_path = filepath.with_suffix('.csv.gz')
                with open(filepath, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                filepath.unlink()  # Remove uncompressed
                filepath = compressed_path
            
            logger.info(f"Exported DataFrame with {len(df)} rows to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error exporting DataFrame: {e}")
            return ""
    
    async def _write_csv(
        self,
        data: List[Dict[str, Any]],
        filepath: Path,
        include_headers: bool
    ) -> None:
        """Write data to CSV file."""
        # Use ThreadPoolExecutor for file I/O
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                self._write_csv_sync,
                data, filepath, include_headers
            )
    
    def _write_csv_sync(
        self,
        data: List[Dict[str, Any]],
        filepath: Path,
        include_headers: bool
    ) -> None:
        """Synchronous CSV writing."""
        if not data:
            return
        
        fieldnames = list(data[0].keys())
        
        with open(filepath, 'w', newline='', encoding=self.config.encoding) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if include_headers:
                writer.writeheader()
            
            writer.writerows(data)
        
        # Compress if enabled
        if self.config.compression:
            compressed_path = filepath.with_suffix('.csv.gz')
            with open(filepath, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            filepath.unlink()  # Remove uncompressed


class DataPersistenceManager:
    """
    Unified data persistence manager combining database and CSV export.
    
    Features:
    - Automatic database operations
    - Scheduled CSV exports
    - Data validation
    - Backup management
    - Performance monitoring
    """
    
    def __init__(
        self,
        db_config: Optional[DatabaseConfig] = None,
        csv_config: Optional[CSVConfig] = None,
        auto_export_interval: int = 3600  # seconds
    ) -> None:
        self.db_config = db_config or DatabaseConfig()
        self.csv_config = csv_config or CSVConfig()
        self.auto_export_interval = auto_export_interval
        
        # Initialize components
        self.db_manager = DatabaseManager(self.db_config)
        self.csv_exporter = CSVExporter(self.csv_config)
        
        # Export management
        self._export_task: Optional[asyncio.Task] = None
        self._export_running = False
        
        # Start auto-export if enabled
        if self.auto_export_interval > 0:
            self._start_auto_export()
    
    async def save_trade(self, trade: TradeRecord) -> bool:
        """Save trade to database."""
        return await self.db_manager.save_trade(trade)
    
    async def save_trades_batch(self, trades: List[TradeRecord]) -> int:
        """Save multiple trades to database."""
        return await self.db_manager.save_trades_batch(trades)
    
    async def save_pnl_record(self, record: PnLRecord) -> bool:
        """Save PnL record to database."""
        return await self.db_manager.save_pnl_record(record)
    
    async def save_position(self, position: PositionSnapshot) -> bool:
        """Save position to database."""
        return await self.db_manager.save_position(position)
    
    async def export_all_data(
        self,
        include_trades: bool = True,
        include_pnl: bool = True,
        include_positions: bool = True,
        custom_filename: Optional[str] = None
    ) -> Dict[str, str]:
        """Export all data to CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {}
        
        if include_trades:
            trades_data = await self.db_manager.get_trades(limit=10000)
            if trades_data:
                filename = custom_filename or f"trades_{timestamp}.csv"
                filepath = await self.csv_exporter.export_trades(trades_data, filename)
                if filepath:
                    results['trades'] = filepath
        
        if include_pnl:
            pnl_data = await self.db_manager.get_pnl_records(limit=10000)
            if pnl_data:
                filename = custom_filename or f"pnl_records_{timestamp}.csv"
                filepath = await self.csv_exporter.export_pnl_records(pnl_data, filename)
                if filepath:
                    results['pnl_records'] = filepath
        
        if include_positions:
            # Get current positions (would need to implement this method)
            positions_data = []  # Would get from database
            if positions_data:
                filename = custom_filename or f"positions_{timestamp}.csv"
                filepath = await self.csv_exporter.export_positions(positions_data, filename)
                if filepath:
                    results['positions'] = filepath
        
        return results
    
    def _start_auto_export(self) -> None:
        """Start automatic CSV export task."""
        if not self._export_running:
            self._export_running = True
            self._export_task = asyncio.create_task(self._auto_export_loop())
    
    async def _auto_export_loop(self) -> None:
        """Automatic export loop."""
        while self._export_running:
            try:
                await self.export_all_data()
                await asyncio.sleep(self.auto_export_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-export loop: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def close(self) -> None:
        """Close persistence manager and cleanup."""
        self._export_running = False
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass
        
        # Final export
        await self.export_all_data()
        
        # Close database manager
        await self.db_manager.close()


# Convenience functions
def create_database_manager(db_path: str = "data/trading_system.db") -> DatabaseManager:
    """Create DatabaseManager with default settings."""
    config = DatabaseConfig(db_path=db_path)
    return DatabaseManager(config)


def create_csv_exporter(export_dir: str = "exports") -> CSVExporter:
    """Create CSVExporter with default settings."""
    config = CSVConfig(export_dir=export_dir)
    return CSVExporter(config)


def create_persistence_manager(
    db_path: str = "data/trading_system.db",
    export_dir: str = "exports"
) -> DataPersistenceManager:
    """Create DataPersistenceManager with default settings."""
    db_config = DatabaseConfig(db_path=db_path)
    csv_config = CSVConfig(export_dir=export_dir)
    return DataPersistenceManager(db_config, csv_config)
