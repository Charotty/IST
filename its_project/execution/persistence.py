#!/usr/bin/env python3
"""
Order State Persistence
======================

Handles persistence of order states, positions, and execution history.
Provides both in-memory and disk-based storage options.
"""

from __future__ import annotations

import json
import sqlite3
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

try:
    from its_project.execution.base import Order, OrderStatus, Position
except ImportError:
    from .base import Order, OrderStatus, Position

logger = logging.getLogger(__name__)


@dataclass
class OrderRecord:
    """Order record for database storage."""
    id: str
    symbol: str
    type: str
    side: str
    amount: float
    price: Optional[float]
    status: str
    filled: float
    remaining: float
    timestamp: int
    info: Dict[str, Any]
    created_at: str
    updated_at: str


@dataclass
class PositionRecord:
    """Position record for database storage."""
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: int
    created_at: str
    updated_at: str


class ExecutionPersistence:
    """
    Handles persistence of execution data.
    
    Features:
    - SQLite database for order and position history
    - In-memory cache for fast access
    - Automatic cleanup of old records
    - Export functionality for analysis
    """
    
    def __init__(self, db_path: str, config: Dict[str, Any]) -> None:
        self.db_path = Path(db_path)
        self.config = config
        
        # In-memory caches
        self.order_cache: Dict[str, Order] = {}
        self.position_cache: List[Position] = []
        
        # Database settings
        self.max_history_days = config.get("max_history_days", 30)
        self.backup_enabled = config.get("backup_enabled", True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                # Orders table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        type TEXT NOT NULL,
                        side TEXT NOT NULL,
                        amount REAL NOT NULL,
                        price REAL,
                        status TEXT NOT NULL,
                        filled REAL NOT NULL,
                        remaining REAL NOT NULL,
                        timestamp INTEGER NOT NULL,
                        info TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                
                # Positions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        symbol TEXT PRIMARY KEY,
                        side TEXT NOT NULL,
                        size REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        current_price REAL NOT NULL,
                        unrealized_pnl REAL NOT NULL,
                        timestamp INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                
                # Execution results table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS execution_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        decision_id TEXT NOT NULL,
                        order_id TEXT,
                        success BOOLEAN NOT NULL,
                        error TEXT,
                        execution_time INTEGER NOT NULL,
                        decision_data TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)
                
                # Create indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_timestamp ON orders(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
                
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
                
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def save_order(self, order: Order) -> None:
        """Save order to database and cache."""
        try:
            # Update cache
            self.order_cache[order.id] = order
            
            # Save to database
            order_record = OrderRecord(
                id=order.id,
                symbol=order.symbol,
                type=order.type.value,
                side=order.side,
                amount=order.amount,
                price=order.price,
                status=order.status.value,
                filled=order.filled,
                remaining=order.remaining,
                timestamp=order.timestamp,
                info=json.dumps(order.info),
                created_at=datetime.fromtimestamp(order.timestamp/1000).isoformat(),
                updated_at=datetime.now().isoformat()
            )
            
            await asyncio.get_event_loop().run_in_executor(
                None, self._save_order_sync, order_record
            )
            
        except Exception as e:
            logger.error(f"Failed to save order {order.id}: {e}")
    
    def _save_order_sync(self, order_record: OrderRecord) -> None:
        """Synchronous order save for database operations."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO orders 
                (id, symbol, type, side, amount, price, status, filled, remaining, 
                 timestamp, info, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_record.id, order_record.symbol, order_record.type,
                order_record.side, order_record.amount, order_record.price,
                order_record.status, order_record.filled, order_record.remaining,
                order_record.timestamp, order_record.info, order_record.created_at,
                order_record.updated_at
            ))
            conn.commit()
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order from cache or database."""
        # Check cache first
        if order_id in self.order_cache:
            return self.order_cache[order_id]
        
        # Load from database
        try:
            order = await asyncio.get_event_loop().run_in_executor(
                None, self._get_order_sync, order_id
            )
            
            if order:
                self.order_cache[order_id] = order
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None
    
    def _get_order_sync(self, order_id: str) -> Optional[Order]:
        """Synchronous order retrieval."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM orders WHERE id = ?", (order_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return Order(
                    id=row[0],
                    symbol=row[1],
                    type=row[2],  # Store as string
                    side=row[3],
                    amount=row[4],
                    price=row[5],
                    status=OrderStatus(row[6]),
                    filled=row[7],
                    remaining=row[8],
                    timestamp=row[9],
                    info=json.loads(row[10]),
                )
        
        return None
    
    async def save_position(self, position: Position) -> None:
        """Save position to database and cache."""
        try:
            # Update cache
            existing = next((p for p in self.position_cache if p.symbol == position.symbol), None)
            if existing:
                self.position_cache.remove(existing)
            self.position_cache.append(position)
            
            # Save to database
            position_record = PositionRecord(
                symbol=position.symbol,
                side=position.side,
                size=position.size,
                entry_price=position.entry_price,
                current_price=position.current_price,
                unrealized_pnl=position.unrealized_pnl,
                timestamp=position.timestamp,
                created_at=datetime.fromtimestamp(position.timestamp/1000).isoformat(),
                updated_at=datetime.now().isoformat()
            )
            
            await asyncio.get_event_loop().run_in_executor(
                None, self._save_position_sync, position_record
            )
            
        except Exception as e:
            logger.error(f"Failed to save position {position.symbol}: {e}")
    
    def _save_position_sync(self, position_record: PositionRecord) -> None:
        """Synchronous position save."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO positions 
                (symbol, side, size, entry_price, current_price, unrealized_pnl, 
                 timestamp, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position_record.symbol, position_record.side, position_record.size,
                position_record.entry_price, position_record.current_price,
                position_record.unrealized_pnl, position_record.timestamp,
                position_record.created_at, position_record.updated_at
            ))
            conn.commit()
    
    async def get_positions(self) -> List[Position]:
        """Get all positions from cache or database."""
        if self.position_cache:
            return self.position_cache.copy()
        
        try:
            positions = await asyncio.get_event_loop().run_in_executor(
                None, self._get_positions_sync
            )
            self.position_cache = positions
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def _get_positions_sync(self) -> List[Position]:
        """Synchronous positions retrieval."""
        positions = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM positions")
            for row in cursor.fetchall():
                positions.append(Position(
                    symbol=row[0],
                    side=row[1],
                    size=row[2],
                    entry_price=row[3],
                    current_price=row[4],
                    unrealized_pnl=row[5],
                    timestamp=row[6],
                ))
        
        return positions
    
    async def save_execution_result(self, result: Dict[str, Any]) -> None:
        """Save execution result."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._save_execution_result_sync, result
            )
        except Exception as e:
            logger.error(f"Failed to save execution result: {e}")
    
    def _save_execution_result_sync(self, result: Dict[str, Any]) -> None:
        """Synchronous execution result save."""
        with sqlite3.connect(self.db_path) as conn:
            # Extract order_id if it's an Order object
            order_id = result.get("order_id")
            if hasattr(order_id, 'id'):  # It's an Order object
                order_id = order_id.id
            elif order_id is None:
                order_id = None
            
            conn.execute("""
                INSERT INTO execution_results 
                (decision_id, order_id, success, error, execution_time, decision_data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                result.get("decision_id"),
                order_id,
                result.get("success"),
                result.get("error"),
                result.get("execution_time"),
                json.dumps(result.get("decision_data", {})),
                datetime.now().isoformat()
            ))
            conn.commit()
    
    async def get_order_history(
        self, 
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        limit: int = 100
    ) -> List[Order]:
        """Get order history with optional filters."""
        try:
            orders = await asyncio.get_event_loop().run_in_executor(
                None, self._get_order_history_sync, symbol, status, limit
            )
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get order history: {e}")
            return []
    
    def _get_order_history_sync(
        self, 
        symbol: Optional[str], 
        status: Optional[OrderStatus], 
        limit: int
    ) -> List[Order]:
        """Synchronous order history retrieval."""
        orders = []
        
        query = "SELECT * FROM orders"
        params = []
        
        if symbol or status:
            conditions = []
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            if status:
                conditions.append("status = ?")
                params.append(status.value)
            
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                orders.append(Order(
                    id=row[0],
                    symbol=row[1],
                    type=row[2],  # Store as string
                    side=row[3],
                    amount=row[4],
                    price=row[5],
                    status=OrderStatus(row[6]),
                    filled=row[7],
                    remaining=row[8],
                    timestamp=row[9],
                    info=json.loads(row[10]),
                ))
        
        return orders
    
    async def cleanup_old_records(self) -> None:
        """Clean up old records based on retention policy."""
        try:
            cutoff_date = datetime.now().timestamp() - (self.max_history_days * 24 * 3600 * 1000)
            
            await asyncio.get_event_loop().run_in_executor(
                None, self._cleanup_old_records_sync, cutoff_date
            )
            
            logger.info(f"Cleaned up records older than {self.max_history_days} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
    
    def _cleanup_old_records_sync(self, cutoff_timestamp: float) -> None:
        """Synchronous cleanup of old records."""
        with sqlite3.connect(self.db_path) as conn:
            # Clean old orders
            conn.execute(
                "DELETE FROM orders WHERE timestamp < ?", (cutoff_timestamp,)
            )
            
            # Clean old execution results
            cutoff_date = datetime.fromtimestamp(cutoff_timestamp/1000).isoformat()
            conn.execute(
                "DELETE FROM execution_results WHERE created_at < ?", (cutoff_date,)
            )
            
            conn.commit()
    
    async def export_data(self, export_path: str) -> None:
        """Export all data to JSON file."""
        try:
            export_data = {
                "orders": [],
                "positions": [],
                "execution_results": [],
                "export_timestamp": datetime.now().isoformat()
            }
            
            # Export orders
            orders = await self.get_order_history(limit=10000)
            export_data["orders"] = [asdict(order) for order in orders]
            
            # Export positions
            positions = await self.get_positions()
            export_data["positions"] = [asdict(position) for position in positions]
            
            # Export execution results
            execution_results = await asyncio.get_event_loop().run_in_executor(
                None, self._get_execution_results_sync
            )
            export_data["execution_results"] = execution_results
            
            # Save to file
            export_path_obj = Path(export_path)
            export_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Data exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
    
    def _get_execution_results_sync(self) -> List[Dict[str, Any]]:
        """Synchronous execution results retrieval."""
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM execution_results ORDER BY created_at DESC")
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "decision_id": row[1],
                    "order_id": row[2],
                    "success": bool(row[3]),
                    "error": row[4],
                    "execution_time": row[5],
                    "decision_data": json.loads(row[6]),
                    "created_at": row[7],
                })
        
        return results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # Count orders
                cursor = conn.execute("SELECT COUNT(*) FROM orders")
                stats["total_orders"] = cursor.fetchone()[0]
                
                # Count positions
                cursor = conn.execute("SELECT COUNT(*) FROM positions")
                stats["total_positions"] = cursor.fetchone()[0]
                
                # Count execution results
                cursor = conn.execute("SELECT COUNT(*) FROM execution_results")
                stats["total_execution_results"] = cursor.fetchone()[0]
                
                # Database size
                stats["database_size_bytes"] = self.db_path.stat().st_size
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
