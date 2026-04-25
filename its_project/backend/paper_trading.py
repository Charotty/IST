"""
Paper Trading Engine
====================

Implements paper trading functionality for MVP:
- Portfolio state management
- Order execution simulation
- PnL and metrics calculation
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from common.backend_contract import (
    PaperTradeUpdateEvent,
    MetricsUpdateEvent,
    TradingMode,
)

logger = logging.getLogger(__name__)


class PaperPosition:
    """Paper trading position."""
    
    def __init__(self, symbol: str, side: str, qty: float, entry_price: float):
        self.symbol = symbol
        self.side = side  # 'buy' or 'sell'
        self.qty = qty
        self.entry_price = entry_price
        self.entry_time = int(time.time() * 1000)
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[int] = None
        self.fee = 0.0
        self.pnl: Optional[float] = None
    
    def close(self, exit_price: float, fee: float = 0.0):
        """Close position and calculate PnL."""
        self.exit_price = exit_price
        self.exit_time = int(time.time() * 1000)
        self.fee = fee
        
        if self.side == 'buy':
            self.pnl = ((exit_price - self.entry_price) / self.entry_price) * 100
        else:
            self.pnl = ((self.entry_price - exit_price) / self.entry_price) * 100
    
    def value(self) -> float:
        """Get current position value."""
        current_price = self.exit_price if self.exit_price else self.entry_price
        return self.qty * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL at current price."""
        if self.side == 'buy':
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100


class PaperPortfolio:
    """Paper trading portfolio."""
    
    def __init__(self, initial_balance: float = 10000.0, commission_rate: float = 0.001):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission_rate = commission_rate
        self.positions: Dict[str, PaperPosition] = {}  # symbol -> position
        self.closed_trades: List[Dict] = []
        self.trades_count = 0
        self.wins = 0
        self.losses = 0
    
    def execute_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float
    ) -> Optional[PaperTradeUpdateEvent]:
        """
        Execute a paper order.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            qty: Quantity
            price: Execution price
        
        Returns:
            Trade event or None
        """
        # Calculate fee
        notional = qty * price
        fee = notional * self.commission_rate
        
        # Check if we have enough balance for buy
        if side == 'buy':
            required = notional + fee
            if self.balance < required:
                logger.warning(f"Insufficient balance for buy order: {required} > {self.balance}")
                return None
            
            self.balance -= required
            
            # Create position
            position = PaperPosition(symbol, side, qty, price)
            position.fee = fee
            self.positions[symbol] = position
            
        elif side == 'sell':
            # Check if we have position to close
            if symbol not in self.positions:
                logger.warning(f"No position to close for {symbol}")
                return None
            
            position = self.positions[symbol]
            
            # Close position
            position.close(price, fee)
            
            # Calculate PnL
            pnl_amount = (position.pnl / 100) * (qty * position.entry_price)
            self.balance += (qty * price) - fee + pnl_amount
            
            # Track trade
            self.closed_trades.append({
                'symbol': symbol,
                'side': side,
                'qty': qty,
                'entry_price': position.entry_price,
                'exit_price': price,
                'pnl': position.pnl,
                'fee': fee,
                'entry_time': position.entry_time,
                'exit_time': position.exit_time,
            })
            
            # Update stats
            self.trades_count += 1
            if position.pnl > 0:
                self.wins += 1
            else:
                self.losses += 1
            
            # Remove position
            del self.positions[symbol]
        
        # Create trade event
        trade_id = f"paper_{int(time.time() * 1000)}"
        event = PaperTradeUpdateEvent(
            id=trade_id,
            symbol=symbol,
            side=side.upper(),
            qty=qty,
            price=price,
            fee=fee,
            status="CLOSED" if side == 'sell' else "OPEN",
            ts=int(time.time() * 1000),
            exit_price=price if side == 'sell' else None,
            pnl=self.positions[symbol].pnl if symbol in self.positions else None,
        )
        
        logger.info(f"Executed paper order: {side} {qty} {symbol} @ {price}")
        return event
    
    def get_metrics(self) -> MetricsUpdateEvent:
        """Calculate current portfolio metrics."""
        # Calculate total PnL
        total_pnl = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        # Calculate win rate
        win_rate = (self.wins / self.trades_count * 100) if self.trades_count > 0 else 0.0
        
        # Calculate exposure (sum of open position values)
        exposure = sum(pos.value() for pos in self.positions.values())
        
        # Calculate max drawdown (simplified)
        max_drawdown = 0.0
        if self.closed_trades:
            min_balance = min(
                self.initial_balance,
                *[t['exit_price'] * t['qty'] for t in self.closed_trades]
            )
            max_drawdown = ((self.initial_balance - min_balance) / self.initial_balance) * 100
        
        # Calculate profit factor
        profit_factor = 0.0
        if self.losses > 0:
            avg_win = sum(t['pnl'] for t in self.closed_trades if t['pnl'] > 0) / self.wins if self.wins > 0 else 0
            avg_loss = abs(sum(t['pnl'] for t in self.closed_trades if t['pnl'] < 0) / self.losses) if self.losses > 0 else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        return MetricsUpdateEvent(
            pnl=total_pnl,
            drawdown=max_drawdown,
            exposure=exposure,
            trades_count=self.trades_count,
            win_rate=win_rate,
            ts=int(time.time() * 1000),
            sharpe=None,  # TODO: Calculate Sharpe ratio
            profit_factor=profit_factor if profit_factor > 0 else None,
        )
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.balance = self.initial_balance
        self.positions.clear()
        self.closed_trades.clear()
        self.trades_count = 0
        self.wins = 0
        self.losses = 0
        logger.info("Paper portfolio reset")


class PaperTradingEngine:
    """Paper trading engine with signal-based trading."""
    
    def __init__(self, portfolio: Optional[PaperPortfolio] = None):
        self.portfolio = portfolio or PaperPortfolio()
        self.running = False
        self.callbacks: List[callable] = []
    
    def on_trade(self, callback: callable):
        """Register callback for trade events."""
        self.callbacks.append(callback)
    
    def execute_signal(
        self,
        symbol: str,
        action: str,
        confidence: float,
        current_price: float
    ) -> Optional[PaperTradeUpdateEvent]:
        """
        Execute a trading signal.
        
        Args:
            symbol: Trading symbol
            action: 'BUY', 'SELL', 'HOLD'
            confidence: Signal confidence (0-1)
            current_price: Current market price
        
        Returns:
            Trade event or None
        """
        if action == 'HOLD':
            return None
        
        if action not in ['BUY', 'SELL']:
            logger.warning(f"Invalid action: {action}")
            return None
        
        # Determine position size based on confidence
        # Simple strategy: 10% of balance per trade, scaled by confidence
        base_size = self.portfolio.balance * 0.1
        size = base_size * confidence
        qty = size / current_price
        
        # Execute order
        side = action.lower()
        event = self.portfolio.execute_order(symbol, side, qty, current_price)
        
        if event:
            # Emit to callbacks
            for callback in self.callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in trade callback: {e}")
        
        return event
    
    def get_metrics(self) -> MetricsUpdateEvent:
        """Get current metrics."""
        return self.portfolio.get_metrics()
    
    def reset(self):
        """Reset the engine."""
        self.portfolio.reset()
        logger.info("Paper trading engine reset")
