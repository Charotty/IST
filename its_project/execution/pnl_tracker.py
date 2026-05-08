#!/usr/bin/env python3
"""
PnL Tracking System
==================

Comprehensive Profit and Loss tracking for trading systems.
Supports per-trade PnL calculation and cumulative PnL tracking.
"""

from __future__ import annotations

import json
import sqlite3
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Individual trade record for PnL calculation."""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    fees: Dict[str, float] = field(default_factory=dict)
    strategy_id: Optional[str] = None
    order_id: Optional[str] = None


@dataclass
class PositionSnapshot:
    """Position snapshot for PnL tracking."""
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    last_price: Optional[float] = None
    last_update: Optional[datetime] = None
    trades_count: int = 0


@dataclass
class PnLRecord:
    """PnL record for a completed trade or period."""
    record_id: str
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str
    realized_pnl: float
    commission: float
    fees: Dict[str, float]
    strategy_id: Optional[str]
    trade_duration: Optional[timedelta] = None
    pnl_percentage: Optional[float] = None


class PnLTracker:
    """
    Advanced PnL tracking system for trading operations.
    
    Features:
    - Real-time PnL calculation
    - Per-trade PnL tracking
    - Cumulative PnL aggregation
    - Position management
    - Performance metrics
    - Database persistence
    """
    
    def __init__(
        self,
        db_path: str = "data/pnl_tracking.db",
        auto_save: bool = True,
        save_interval: int = 60
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Position tracking
        self.positions: Dict[str, PositionSnapshot] = {}
        
        # Trade history
        self.trades: List[TradeRecord] = []
        self.pnl_records: List[PnLRecord] = []
        
        # Cumulative metrics
        self.cumulative_pnl = 0.0
        self.daily_pnl: Dict[str, float] = {}
        self.strategy_pnl: Dict[str, float] = {}
        
        # Configuration
        self.auto_save = auto_save
        self.save_interval = save_interval
        self._last_save = datetime.now()
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_positions()
        self._load_pnl_records()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for PnL tracking."""
        with sqlite3.connect(self.db_path) as conn:
            # Trades table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    quantity REAL,
                    price REAL,
                    timestamp INTEGER,
                    commission REAL,
                    fees TEXT,
                    strategy_id TEXT,
                    order_id TEXT
                )
            """)
            
            # PnL records table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pnl_records (
                    record_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    entry_time INTEGER,
                    exit_time INTEGER,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    side TEXT,
                    realized_pnl REAL,
                    commission REAL,
                    fees TEXT,
                    strategy_id TEXT,
                    trade_duration INTEGER,
                    pnl_percentage REAL
                )
            """)
            
            # Positions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL,
                    avg_price REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    total_pnl REAL,
                    last_price REAL,
                    last_update INTEGER,
                    trades_count INTEGER
                )
            """)
            
            # Daily PnL table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_pnl (
                    date TEXT PRIMARY KEY,
                    pnl REAL,
                    trades_count INTEGER,
                    volume REAL
                )
            """)
            
            # Strategy PnL table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_pnl (
                    strategy_id TEXT PRIMARY KEY,
                    total_pnl REAL,
                    trades_count INTEGER,
                    win_rate REAL,
                    avg_win REAL,
                    avg_loss REAL
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pnl_symbol ON pnl_records(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pnl_strategy ON pnl_records(strategy_id)")
    
    def add_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: Optional[datetime] = None,
        commission: float = 0.0,
        fees: Optional[Dict[str, float]] = None,
        strategy_id: Optional[str] = None,
        order_id: Optional[str] = None
    ) -> None:
        """
        Add a new trade to the PnL tracker.
        
        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Trade quantity
            price: Trade price
            timestamp: Trade timestamp (default: now)
            commission: Trade commission
            fees: Additional fees
            strategy_id: Strategy identifier
            order_id: Order identifier
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        trade = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=side.lower(),
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            commission=commission,
            fees=fees or {},
            strategy_id=strategy_id,
            order_id=order_id
        )
        
        self.trades.append(trade)
        self._update_position(trade)
        self._save_trade(trade)
        
        # Auto-save if enabled
        if self.auto_save and (datetime.now() - self._last_save).seconds >= self.save_interval:
            self._save_positions()
            self._last_save = datetime.now()
        
        logger.debug(f"Trade added: {trade_id} {symbol} {side} {quantity}@{price}")
    
    def _update_position(self, trade: TradeRecord) -> None:
        """Update position based on new trade."""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            # New position
            quantity = trade.quantity if trade.side == 'buy' else -trade.quantity
            self.positions[symbol] = PositionSnapshot(
                symbol=symbol,
                quantity=quantity,
                avg_price=trade.price,
                trades_count=1
            )
        else:
            # Update existing position
            position = self.positions[symbol]
            old_quantity = position.quantity
            old_avg_price = position.avg_price
            
            # Calculate new quantity and average price
            if trade.side == 'buy':
                new_quantity = old_quantity + trade.quantity
                if new_quantity != 0:
                    new_avg_price = (old_avg_price * old_quantity + trade.price * trade.quantity) / new_quantity
                else:
                    new_avg_price = trade.price
            else:  # sell
                new_quantity = old_quantity - trade.quantity
                if new_quantity != 0:
                    new_avg_price = position.avg_price
                else:
                    new_avg_price = trade.price
            
            # Calculate realized PnL if position is reduced or closed
            if old_quantity != 0 and new_quantity != old_quantity:
                # Determine if this is a closing trade
                if (old_quantity > 0 and trade.side == 'sell') or (old_quantity < 0 and trade.side == 'buy'):
                    # Calculate realized PnL
                    if old_quantity > 0:  # Long position
                        realized_quantity = min(trade.quantity, abs(old_quantity))
                        realized_pnl = (trade.price - old_avg_price) * realized_quantity
                    else:  # Short position
                        realized_quantity = min(trade.quantity, abs(old_quantity))
                        realized_pnl = (old_avg_price - trade.price) * realized_quantity
                    
                    position.realized_pnl += realized_pnl
                    position.total_pnl = position.realized_pnl + position.unrealized_pnl
                    
                    # Create PnL record
                    self._create_pnl_record(trade, position, realized_pnl)
                    
                    # Update strategy PnL
                    if trade.strategy_id:
                        self.strategy_pnl[trade.strategy_id] = self.strategy_pnl.get(trade.strategy_id, 0.0) + realized_pnl
            
            # Update position
            position.quantity = new_quantity
            position.avg_price = new_avg_price
            position.trades_count += 1
            
            # Remove position if closed
            if abs(new_quantity) < 1e-10:  # Essentially zero
                del self.positions[symbol]
        
        # Update daily PnL
        date_str = trade.timestamp.strftime('%Y-%m-%d')
        if date_str not in self.daily_pnl:
            self.daily_pnl[date_str] = 0.0
        
        # Update cumulative PnL
        self._update_cumulative_pnl()
    
    def _create_pnl_record(
        self,
        trade: TradeRecord,
        position: PositionSnapshot,
        realized_pnl: float
    ) -> None:
        """Create a PnL record for a completed trade."""
        record_id = f"pnl_{trade.trade_id}_{datetime.now().timestamp()}"
        
        # Find entry trade (simplified - in practice, you'd track entry/exit pairs)
        entry_price = position.avg_price
        entry_time = trade.timestamp  # Simplified
        
        pnl_record = PnLRecord(
            record_id=record_id,
            symbol=trade.symbol,
            entry_time=entry_time,
            exit_time=trade.timestamp,
            entry_price=entry_price,
            exit_price=trade.price,
            quantity=trade.quantity,
            side=trade.side,
            realized_pnl=realized_pnl,
            commission=trade.commission,
            fees=trade.fees,
            strategy_id=trade.strategy_id,
            trade_duration=timedelta(0),  # Simplified
            pnl_percentage=(realized_pnl / (entry_price * trade.quantity)) * 100 if entry_price * trade.quantity != 0 else 0.0
        )
        
        self.pnl_records.append(pnl_record)
        self._save_pnl_record(pnl_record)
    
    def update_market_price(self, symbol: str, price: float, timestamp: Optional[datetime] = None) -> None:
        """
        Update market price for unrealized PnL calculation.
        
        Args:
            symbol: Trading symbol
            price: Current market price
            timestamp: Update timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if symbol in self.positions:
            position = self.positions[symbol]
            position.last_price = price
            position.last_update = timestamp
            
            # Calculate unrealized PnL
            if position.quantity != 0:
                if position.quantity > 0:  # Long position
                    position.unrealized_pnl = (price - position.avg_price) * position.quantity
                else:  # Short position
                    position.unrealized_pnl = (position.avg_price - price) * abs(position.quantity)
                
                position.total_pnl = position.realized_pnl + position.unrealized_pnl
            
            logger.debug(f"Updated market price for {symbol}: {price}")
    
    def _update_cumulative_pnl(self) -> None:
        """Update cumulative PnL across all positions."""
        total_realized = sum(record.realized_pnl for record in self.pnl_records)
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        self.cumulative_pnl = total_realized + total_unrealized
    
    def get_position_pnl(self, symbol: str) -> Optional[PositionSnapshot]:
        """Get PnL information for a specific position."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, PositionSnapshot]:
        """Get all current positions."""
        return self.positions.copy()
    
    def get_cumulative_pnl(self) -> float:
        """Get cumulative PnL across all positions."""
        self._update_cumulative_pnl()
        return self.cumulative_pnl
    
    def get_daily_pnl(self, date: Optional[str] = None) -> float:
        """Get PnL for a specific date."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        return self.daily_pnl.get(date, 0.0)
    
    def get_strategy_pnl(self, strategy_id: str) -> float:
        """Get PnL for a specific strategy."""
        return self.strategy_pnl.get(strategy_id, 0.0)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self.pnl_records:
            return {}
        
        realized_pnls = [record.realized_pnl for record in self.pnl_records]
        winning_trades = [pnl for pnl in realized_pnls if pnl > 0]
        losing_trades = [pnl for pnl in realized_pnls if pnl < 0]
        
        metrics = {
            'total_trades': len(self.pnl_records),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.pnl_records) * 100 if self.pnl_records else 0,
            'total_realized_pnl': sum(realized_pnls),
            'total_unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
            'total_pnl': self.get_cumulative_pnl(),
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'largest_win': max(winning_trades) if winning_trades else 0,
            'largest_loss': min(losing_trades) if losing_trades else 0,
            'profit_factor': sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf'),
            'total_commission': sum(record.commission for record in self.pnl_records),
            'net_pnl': sum(record.realized_pnl - record.commission for record in self.pnl_records)
        }
        
        return metrics
    
    def get_pnl_dataframe(self) -> pd.DataFrame:
        """Get PnL records as a pandas DataFrame."""
        if not self.pnl_records:
            return pd.DataFrame()
        
        data = []
        for record in self.pnl_records:
            data.append({
                'record_id': record.record_id,
                'symbol': record.symbol,
                'entry_time': record.entry_time,
                'exit_time': record.exit_time,
                'entry_price': record.entry_price,
                'exit_price': record.exit_price,
                'quantity': record.quantity,
                'side': record.side,
                'realized_pnl': record.realized_pnl,
                'commission': record.commission,
                'net_pnl': record.realized_pnl - record.commission,
                'pnl_percentage': record.pnl_percentage,
                'strategy_id': record.strategy_id,
                'trade_duration': record.trade_duration
            })
        
        return pd.DataFrame(data)
    
    def get_positions_dataframe(self) -> pd.DataFrame:
        """Get current positions as a pandas DataFrame."""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for symbol, position in self.positions.items():
            data.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'total_pnl': position.total_pnl,
                'last_price': position.last_price,
                'last_update': position.last_update,
                'trades_count': position.trades_count
            })
        
        return pd.DataFrame(data)
    
    def _save_trade(self, trade: TradeRecord) -> None:
        """Save trade to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trades VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                trade.trade_id, trade.symbol, trade.side, trade.quantity,
                trade.price, trade.timestamp.timestamp(), trade.commission,
                json.dumps(trade.fees), trade.strategy_id, trade.order_id
            ))
    
    def _save_pnl_record(self, record: PnLRecord) -> None:
        """Save PnL record to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO pnl_records VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                record.record_id, record.symbol, record.entry_time.timestamp(),
                record.exit_time.timestamp() if record.exit_time else None,
                record.entry_price, record.exit_price, record.quantity,
                record.side, record.realized_pnl, record.commission,
                json.dumps(record.fees), record.strategy_id,
                record.trade_duration.total_seconds() if record.trade_duration else None,
                record.pnl_percentage
            ))
    
    def _save_positions(self) -> None:
        """Save current positions to database."""
        with sqlite3.connect(self.db_path) as conn:
            # Clear existing positions
            conn.execute("DELETE FROM positions")
            
            # Insert current positions
            for symbol, position in self.positions.items():
                conn.execute("""
                    INSERT INTO positions VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    symbol, position.quantity, position.avg_price,
                    position.unrealized_pnl, position.realized_pnl,
                    position.total_pnl, position.last_price,
                    position.last_update.timestamp() if position.last_update else None,
                    position.trades_count
                ))
    
    def _load_positions(self) -> None:
        """Load positions from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM positions")
                for row in cursor.fetchall():
                    self.positions[row[0]] = PositionSnapshot(
                        symbol=row[0],
                        quantity=row[1],
                        avg_price=row[2],
                        unrealized_pnl=row[3],
                        realized_pnl=row[4],
                        total_pnl=row[5],
                        last_price=row[6],
                        last_update=datetime.fromtimestamp(row[7]) if row[7] else None,
                        trades_count=row[8]
                    )
        except Exception as e:
            logger.warning(f"Failed to load positions: {e}")
    
    def _load_pnl_records(self) -> None:
        """Load PnL records from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM pnl_records")
                for row in cursor.fetchall():
                    self.pnl_records.append(PnLRecord(
                        record_id=row[0],
                        symbol=row[1],
                        entry_time=datetime.fromtimestamp(row[2]),
                        exit_time=datetime.fromtimestamp(row[3]) if row[3] else None,
                        entry_price=row[4],
                        exit_price=row[5],
                        quantity=row[6],
                        side=row[7],
                        realized_pnl=row[8],
                        commission=row[9],
                        fees=json.loads(row[10]) if row[10] else {},
                        strategy_id=row[11],
                        trade_duration=timedelta(seconds=row[12]) if row[12] else None,
                        pnl_percentage=row[13]
                    ))
        except Exception as e:
            logger.warning(f"Failed to load PnL records: {e}")
    
    def save(self) -> None:
        """Force save all data to database."""
        self._save_positions()
        self._last_save = datetime.now()
        logger.info("PnL data saved to database")
    
    def reset(self) -> None:
        """Reset all PnL tracking data."""
        self.positions.clear()
        self.trades.clear()
        self.pnl_records.clear()
        self.cumulative_pnl = 0.0
        self.daily_pnl.clear()
        self.strategy_pnl.clear()
        
        # Clear database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM trades")
            conn.execute("DELETE FROM pnl_records")
            conn.execute("DELETE FROM positions")
            conn.execute("DELETE FROM daily_pnl")
            conn.execute("DELETE FROM strategy_pnl")
        
        logger.info("PnL tracking data reset")


# Convenience functions
def create_pnl_tracker(db_path: str = "data/pnl_tracking.db") -> PnLTracker:
    """Create PnL tracker with default settings."""
    return PnLTracker(db_path)


def calculate_simple_pnl(
    entry_price: float,
    exit_price: float,
    quantity: float,
    side: str,
    commission: float = 0.0
) -> float:
    """
    Calculate simple PnL for a single trade.
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        quantity: Trade quantity
        side: 'long' or 'short'
        commission: Trade commission
        
    Returns:
        PnL amount
    """
    if side.lower() == 'long':
        pnl = (exit_price - entry_price) * quantity
    else:  # short
        pnl = (entry_price - exit_price) * quantity
    
    return pnl - commission
