#!/usr/bin/env python3
"""
Business Metrics Monitoring
=========================

Production-ready business metrics for trading system:
- PnL tracking and analysis
- Drawdown monitoring
- Hit rate and win/loss statistics
- Latency and performance metrics
- Real-time dashboards
"""

from __future__ import annotations

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import json
from collections import deque, defaultdict

from its_project.execution.base import Position, Order, OrderStatus
from its_project.decision.decision import Decision, Action

logger = logging.getLogger(__name__)


class MetricPeriod(Enum):
    """Metric aggregation periods."""
    REALTIME = "realtime"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class TradeMetrics:
    """Individual trade metrics."""
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: int
    exit_time: int
    pnl: float
    pnl_percentage: float
    fees: float
    slippage: float
    duration_ms: int
    latency_ms: Optional[float] = None
    strategy: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Performance summary metrics."""
    total_pnl: float
    total_pnl_pct: float
    realized_pnl: float
    unrealized_pnl: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_duration_ms: float
    avg_latency_ms: float


@dataclass
class LatencyMetrics:
    """Latency and performance metrics."""
    signal_to_order_ms: float
    order_to_execution_ms: float
    end_to_end_ms: float
    order_ack_rate: float
    fill_rate: float
    rejection_rate: float
    timeout_rate: float


class BusinessMetricsCollector:
    """
    Comprehensive business metrics collector.
    
    Features:
    - Real-time PnL tracking
    - Performance analytics
    - Latency monitoring
    - Hit rate statistics
    - Risk metrics
    - Historical data storage
    """
    
    def __init__(
        self,
        max_history_size: int = 10000,
        aggregation_periods: List[MetricPeriod] = None
    ) -> None:
        self.max_history_size = max_history_size
        self.aggregation_periods = aggregation_periods or [
            MetricPeriod.REALTIME,
            MetricPeriod.MINUTE,
            MetricPeriod.HOUR,
            MetricPeriod.DAY
        ]
        
        # Trade history
        self.trades: List[TradeMetrics] = []
        self.open_positions: Dict[str, Position] = {}
        
        # Time-series data
        self.pnl_history: deque = deque(maxlen=max_history_size)
        self.equity_curve: deque = deque(maxlen=max_history_size)
        self.drawdown_history: deque = deque(maxlen=max_history_size)
        self.latency_history: deque = deque(maxlen=max_history_size)
        
        # Aggregated metrics by period
        self.aggregated_metrics: Dict[MetricPeriod, Dict[str, Any]] = {
            period: {} for period in self.aggregation_periods
        }
        
        # Current state
        self.current_equity: float = 0.0
        self.peak_equity: float = 0.0
        self.daily_pnl: float = 0.0
        self.last_reset_date: str = datetime.now().date().isoformat()
        
        # Performance cache
        self._performance_cache: Optional[PerformanceMetrics] = None
        self._cache_timestamp: int = 0
        self._cache_ttl_seconds: int = 60
        
        # Statistics
        self.stats = {
            'total_trades_recorded': 0,
            'total_pnl_recorded': 0.0,
            'max_drawdown_recorded': 0.0,
            'best_day': 0.0,
            'worst_day': 0.0,
            'current_win_streak': 0,
            'current_loss_streak': 0,
            'max_win_streak': 0,
            'max_loss_streak': 0
        }
        
        # Callbacks
        self.metric_callbacks: List[callable] = []
    
    def record_trade(
        self,
        trade_data: Dict[str, Any]
    ) -> None:
        """
        Record a completed trade.
        
        Args:
            trade_data: Trade information including entry/exit details
        """
        trade = TradeMetrics(
            trade_id=trade_data.get('trade_id', ''),
            symbol=trade_data.get('symbol', ''),
            side=trade_data.get('side', ''),
            entry_price=trade_data.get('entry_price', 0.0),
            exit_price=trade_data.get('exit_price', 0.0),
            size=trade_data.get('size', 0.0),
            entry_time=trade_data.get('entry_time', 0),
            exit_time=trade_data.get('exit_time', 0),
            pnl=trade_data.get('pnl', 0.0),
            pnl_percentage=trade_data.get('pnl_percentage', 0.0),
            fees=trade_data.get('fees', 0.0),
            slippage=trade_data.get('slippage', 0.0),
            duration_ms=trade_data.get('duration_ms', 0),
            latency_ms=trade_data.get('latency_ms'),
            strategy=trade_data.get('strategy'),
            confidence=trade_data.get('confidence')
        )
        
        self.trades.append(trade)
        self.stats['total_trades_recorded'] += 1
        self.stats['total_pnl_recorded'] += trade.pnl
        
        # Update streaks
        if trade.pnl > 0:
            self.stats['current_win_streak'] += 1
            self.stats['current_loss_streak'] = 0
            self.stats['max_win_streak'] = max(self.stats['max_win_streak'], self.stats['current_win_streak'])
        else:
            self.stats['current_loss_streak'] += 1
            self.stats['current_win_streak'] = 0
            self.stats['max_loss_streak'] = max(self.stats['max_loss_streak'], self.stats['current_loss_streak'])
        
        # Update PnL history
        self.pnl_history.append({
            'timestamp': trade.exit_time,
            'pnl': trade.pnl,
            'cumulative_pnl': self.stats['total_pnl_recorded'],
            'symbol': trade.symbol
        })
        
        # Invalidate cache
        self._performance_cache = None
        
        # Trigger callbacks
        self._trigger_metric_callbacks('trade_recorded', trade)
        
        logger.debug(f"Trade recorded: {trade.symbol} PnL: ${trade.pnl:.2f}")
    
    def update_positions(
        self,
        positions: Dict[str, Position]
    ) -> None:
        """
        Update current positions and calculate unrealized PnL.
        
        Args:
            positions: Current open positions
        """
        self.open_positions = positions
        
        # Calculate unrealized PnL
        unrealized_pnl = sum(pos.unrealized_pnl for pos in positions.values())
        
        # Update equity
        realized_pnl = sum(t.pnl for t in self.trades)
        self.current_equity = realized_pnl + unrealized_pnl
        
        # Update peak equity
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        
        # Calculate current drawdown
        current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        
        # Update history
        current_time = int(time.time() * 1000)
        self.equity_curve.append({
            'timestamp': current_time,
            'equity': self.current_equity,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl
        })
        
        self.drawdown_history.append({
            'timestamp': current_time,
            'drawdown': current_drawdown,
            'peak_equity': self.peak_equity
        })
        
        # Update max drawdown
        if current_drawdown > self.stats['max_drawdown_recorded']:
            self.stats['max_drawdown_recorded'] = current_drawdown
        
        # Invalidate cache
        self._performance_cache = None
        
        # Trigger callbacks
        self._trigger_metric_callbacks('positions_updated', {
            'equity': self.current_equity,
            'drawdown': current_drawdown,
            'positions': len(positions)
        })
    
    def record_latency(
        self,
        latency_data: Dict[str, float]
    ) -> None:
        """
        Record latency metrics.
        
        Args:
            latency_data: Dictionary with various latency measurements
        """
        current_time = int(time.time() * 1000)
        
        self.latency_history.append({
            'timestamp': current_time,
            **latency_data
        })
        
        # Trigger callbacks
        self._trigger_metric_callbacks('latency_recorded', latency_data)
    
    def get_performance_metrics(
        self,
        period: MetricPeriod = MetricPeriod.REALTIME,
        force_refresh: bool = False
    ) -> PerformanceMetrics:
        """
        Get comprehensive performance metrics.
        
        Args:
            period: Time period for metrics
            force_refresh: Force cache refresh
            
        Returns:
            PerformanceMetrics with all calculations
        """
        current_time = int(time.time() * 1000)
        
        # Check cache
        if (not force_refresh and 
            self._performance_cache and 
            current_time - self._cache_timestamp < self._cache_ttl_seconds * 1000):
            return self._performance_cache
        
        # Filter trades by period
        filtered_trades = self._filter_trades_by_period(period)
        
        if not filtered_trades:
            return PerformanceMetrics(
                total_pnl=0.0, total_pnl_pct=0.0, realized_pnl=0.0,
                unrealized_pnl=0.0, max_drawdown=0.0, current_drawdown=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, win_rate=0.0,
                profit_factor=0.0, avg_win=0.0, avg_loss=0.0,
                largest_win=0.0, largest_loss=0.0, total_trades=0,
                winning_trades=0, losing_trades=0, avg_trade_duration_ms=0.0,
                avg_latency_ms=0.0
            )
        
        # Calculate basic metrics
        total_pnl = sum(t.pnl for t in filtered_trades)
        realized_pnl = total_pnl
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.open_positions.values())
        
        # Calculate drawdown
        equity_curve = self._calculate_equity_curve(filtered_trades)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        current_drawdown = self._calculate_current_drawdown()
        
        # Calculate trade statistics
        winning_trades = [t for t in filtered_trades if t.pnl > 0]
        losing_trades = [t for t in filtered_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(filtered_trades) if filtered_trades else 0.0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        largest_win = max([t.pnl for t in winning_trades], default=0.0)
        largest_loss = min([t.pnl for t in losing_trades], default=0.0)
        
        # Calculate profit factor
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Calculate risk-adjusted returns
        returns = [t.pnl_percentage for t in filtered_trades]
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        
        # Calculate averages
        avg_trade_duration = np.mean([t.duration_ms for t in filtered_trades]) if filtered_trades else 0.0
        
        # Calculate average latency
        latency_values = [t.latency_ms for t in filtered_trades if t.latency_ms is not None]
        avg_latency = np.mean(latency_values) if latency_values else 0.0
        
        # Create metrics object
        metrics = PerformanceMetrics(
            total_pnl=total_pnl,
            total_pnl_pct=(total_pnl / self.peak_equity * 100) if self.peak_equity > 0 else 0.0,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            total_trades=len(filtered_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_trade_duration_ms=avg_trade_duration,
            avg_latency_ms=avg_latency
        )
        
        # Update cache
        self._performance_cache = metrics
        self._cache_timestamp = current_time
        
        return metrics
    
    def get_latency_metrics(
        self,
        period: MetricPeriod = MetricPeriod.REALTIME
    ) -> LatencyMetrics:
        """
        Get latency metrics for specified period.
        
        Args:
            period: Time period for metrics
            
        Returns:
            LatencyMetrics with calculated values
        """
        filtered_latency = self._filter_latency_by_period(period)
        
        if not filtered_latency:
            return LatencyMetrics(
                signal_to_order_ms=0.0,
                order_to_execution_ms=0.0,
                end_to_end_ms=0.0,
                order_ack_rate=0.0,
                fill_rate=0.0,
                rejection_rate=0.0,
                timeout_rate=0.0
            )
        
        # Calculate averages
        signal_to_order = np.mean([l.get('signal_to_order_ms', 0) for l in filtered_latency])
        order_to_execution = np.mean([l.get('order_to_execution_ms', 0) for l in filtered_latency])
        end_to_end = np.mean([l.get('end_to_end_ms', 0) for l in filtered_latency])
        
        # Calculate rates
        total_orders = len(filtered_latency)
        ack_orders = sum(1 for l in filtered_latency if l.get('acknowledged', False))
        filled_orders = sum(1 for l in filtered_latency if l.get('filled', False))
        rejected_orders = sum(1 for l in filtered_latency if l.get('rejected', False))
        timeout_orders = sum(1 for l in filtered_latency if l.get('timeout', False))
        
        order_ack_rate = ack_orders / total_orders if total_orders > 0 else 0.0
        fill_rate = filled_orders / total_orders if total_orders > 0 else 0.0
        rejection_rate = rejected_orders / total_orders if total_orders > 0 else 0.0
        timeout_rate = timeout_orders / total_orders if total_orders > 0 else 0.0
        
        return LatencyMetrics(
            signal_to_order_ms=signal_to_order,
            order_to_execution_ms=order_to_execution,
            end_to_end_ms=end_to_end,
            order_ack_rate=order_ack_rate,
            fill_rate=fill_rate,
            rejection_rate=rejection_rate,
            timeout_rate=timeout_rate
        )
    
    def get_symbol_metrics(
        self,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get metrics by symbol.
        
        Args:
            symbol: Specific symbol or None for all symbols
            
        Returns:
            Dictionary with symbol-specific metrics
        """
        symbol_trades = [t for t in self.trades if symbol is None or t.symbol == symbol]
        
        if not symbol_trades:
            return {}
        
        # Group by symbol
        symbol_groups = defaultdict(list)
        for trade in symbol_trades:
            symbol_groups[trade.symbol].append(trade)
        
        metrics = {}
        for sym, trades in symbol_groups.items():
            total_pnl = sum(t.pnl for t in trades)
            win_rate = len([t for t in trades if t.pnl > 0]) / len(trades)
            avg_duration = np.mean([t.duration_ms for t in trades])
            
            metrics[sym] = {
                'total_trades': len(trades),
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'avg_duration_ms': avg_duration,
                'avg_pnl': total_pnl / len(trades),
                'first_trade': min(t.entry_time for t in trades),
                'last_trade': max(t.exit_time for t in trades)
            }
        
        return metrics
    
    def get_daily_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get daily performance summary.
        
        Args:
            date: Date in YYYY-MM-DD format or None for today
            
        Returns:
            Daily summary metrics
        """
        if date is None:
            date = datetime.now().date().isoformat()
        
        # Filter trades by date
        target_date = datetime.fromisoformat(date).date()
        daily_trades = [
            t for t in self.trades 
            if datetime.fromtimestamp(t.exit_time / 1000).date() == target_date
        ]
        
        if not daily_trades:
            return {
                'date': date,
                'trades': 0,
                'pnl': 0.0,
                'win_rate': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0
            }
        
        total_pnl = sum(t.pnl for t in daily_trades)
        win_rate = len([t for t in daily_trades if t.pnl > 0]) / len(daily_trades)
        best_trade = max([t.pnl for t in daily_trades])
        worst_trade = min([t.pnl for t in daily_trades])
        
        return {
            'date': date,
            'trades': len(daily_trades),
            'pnl': total_pnl,
            'win_rate': win_rate,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_trade': total_pnl / len(daily_trades),
            'total_fees': sum(t.fees for t in daily_trades),
            'total_slippage': sum(t.slippage for t in daily_trades)
        }
    
    def export_metrics(
        self,
        format: str = "json",
        period: MetricPeriod = MetricPeriod.DAY
    ) -> str:
        """
        Export metrics in specified format.
        
        Args:
            format: Export format (json, csv)
            period: Time period for data
            
        Returns:
            Formatted metrics string
        """
        performance = self.get_performance_metrics(period)
        latency = self.get_latency_metrics(period)
        symbol_metrics = self.get_symbol_metrics()
        
        data = {
            'timestamp': int(time.time() * 1000),
            'period': period.value,
            'performance': {
                'total_pnl': performance.total_pnl,
                'total_pnl_pct': performance.total_pnl_pct,
                'realized_pnl': performance.realized_pnl,
                'unrealized_pnl': performance.unrealized_pnl,
                'max_drawdown': performance.max_drawdown,
                'current_drawdown': performance.current_drawdown,
                'sharpe_ratio': performance.sharpe_ratio,
                'sortino_ratio': performance.sortino_ratio,
                'win_rate': performance.win_rate,
                'profit_factor': performance.profit_factor,
                'avg_win': performance.avg_win,
                'avg_loss': performance.avg_loss,
                'largest_win': performance.largest_win,
                'largest_loss': performance.largest_loss,
                'total_trades': performance.total_trades,
                'winning_trades': performance.winning_trades,
                'losing_trades': performance.losing_trades,
                'avg_trade_duration_ms': performance.avg_trade_duration_ms,
                'avg_latency_ms': performance.avg_latency_ms
            },
            'latency': {
                'signal_to_order_ms': latency.signal_to_order_ms,
                'order_to_execution_ms': latency.order_to_execution_ms,
                'end_to_end_ms': latency.end_to_end_ms,
                'order_ack_rate': latency.order_ack_rate,
                'fill_rate': latency.fill_rate,
                'rejection_rate': latency.rejection_rate,
                'timeout_rate': latency.timeout_rate
            },
            'symbols': symbol_metrics,
            'statistics': self.stats
        }
        
        if format.lower() == "json":
            return json.dumps(data, indent=2, default=str)
        elif format.lower() == "csv":
            # Simplified CSV export
            lines = ["metric,value"]
            for key, value in data['performance'].items():
                lines.append(f"{key},{value}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def add_metric_callback(self, callback: callable) -> None:
        """Add callback for metric updates."""
        self.metric_callbacks.append(callback)
    
    def _filter_trades_by_period(self, period: MetricPeriod) -> List[TradeMetrics]:
        """Filter trades by time period."""
        if period == MetricPeriod.REALTIME:
            return self.trades[-100:]  # Last 100 trades
        
        current_time = int(time.time() * 1000)
        
        if period == MetricPeriod.MINUTE:
            cutoff = current_time - 60 * 1000
        elif period == MetricPeriod.HOUR:
            cutoff = current_time - 3600 * 1000
        elif period == MetricPeriod.DAY:
            cutoff = current_time - 24 * 3600 * 1000
        elif period == MetricPeriod.WEEK:
            cutoff = current_time - 7 * 24 * 3600 * 1000
        elif period == MetricPeriod.MONTH:
            cutoff = current_time - 30 * 24 * 3600 * 1000
        else:
            return self.trades
        
        return [t for t in self.trades if t.exit_time >= cutoff]
    
    def _filter_latency_by_period(self, period: MetricPeriod) -> List[Dict[str, float]]:
        """Filter latency data by time period."""
        if period == MetricPeriod.REALTIME:
            return list(self.latency_history)[-100:]  # Last 100 entries
        
        current_time = int(time.time() * 1000)
        
        if period == MetricPeriod.MINUTE:
            cutoff = current_time - 60 * 1000
        elif period == MetricPeriod.HOUR:
            cutoff = current_time - 3600 * 1000
        elif period == MetricPeriod.DAY:
            cutoff = current_time - 24 * 3600 * 1000
        else:
            return list(self.latency_history)
        
        return [l for l in self.latency_history if l['timestamp'] >= cutoff]
    
    def _calculate_equity_curve(self, trades: List[TradeMetrics]) -> List[float]:
        """Calculate equity curve from trades."""
        if not trades:
            return []
        
        equity_curve = []
        running_pnl = 0.0
        
        for trade in sorted(trades, key=lambda t: t.exit_time):
            running_pnl += trade.pnl
            equity_curve.append(running_pnl)
        
        return equity_curve
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]
        max_drawdown = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown."""
        if self.peak_equity == 0:
            return 0.0
        
        return (self.peak_equity - self.current_equity) / self.peak_equity
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
    
    def _trigger_metric_callbacks(self, event_type: str, data: Any) -> None:
        """Trigger metric update callbacks."""
        for callback in self.metric_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in metric callback: {e}")


# Convenience functions
def create_business_metrics_collector() -> BusinessMetricsCollector:
    """Create business metrics collector with default settings."""
    return BusinessMetricsCollector(
        max_history_size=50000,
        aggregation_periods=[
            MetricPeriod.REALTIME,
            MetricPeriod.MINUTE,
            MetricPeriod.HOUR,
            MetricPeriod.DAY
        ]
    )


if __name__ == "__main__":
    # Test business metrics collector
    logging.basicConfig(level=logging.INFO)
    
    collector = create_business_metrics_collector()
    
    # Simulate some trades
    test_trades = [
        {
            'trade_id': '001',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'entry_price': 42000.0,
            'exit_price': 42500.0,
            'size': 0.1,
            'entry_time': int(time.time() * 1000) - 3600000,
            'exit_time': int(time.time() * 1000) - 1800000,
            'pnl': 50.0,
            'pnl_percentage': 0.0119,
            'fees': 2.0,
            'slippage': 1.0,
            'duration_ms': 1800000,
            'latency_ms': 150.0
        },
        {
            'trade_id': '002',
            'symbol': 'ETH/USDT',
            'side': 'sell',
            'entry_price': 3000.0,
            'exit_price': 2950.0,
            'size': 1.0,
            'entry_time': int(time.time() * 1000) - 1800000,
            'exit_time': int(time.time() * 1000) - 900000,
            'pnl': 50.0,
            'pnl_percentage': 0.0167,
            'fees': 1.5,
            'slippage': 0.5,
            'duration_ms': 900000,
            'latency_ms': 120.0
        }
    ]
    
    # Record trades
    for trade_data in test_trades:
        collector.record_trade(trade_data)
    
    # Get performance metrics
    performance = collector.get_performance_metrics()
    print(f"Performance metrics:")
    print(f"  Total PnL: ${performance.total_pnl:.2f}")
    print(f"  Win rate: {performance.win_rate:.2%}")
    print(f"  Sharpe ratio: {performance.sharpe_ratio:.2f}")
    print(f"  Max drawdown: {performance.max_drawdown:.2%}")
    
    # Get latency metrics
    latency = collector.get_latency_metrics()
    print(f"\nLatency metrics:")
    print(f"  Signal to order: {latency.signal_to_order_ms:.1f}ms")
    print(f"  Order to execution: {latency.order_to_execution_ms:.1f}ms")
    print(f"  Fill rate: {latency.fill_rate:.2%}")
    
    # Export metrics
    json_export = collector.export_metrics("json")
    print(f"\nJSON export length: {len(json_export)} characters")
