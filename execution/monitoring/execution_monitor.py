"""
Execution monitor for ITS.

Monitors execution quality, tracks performance metrics, and provides alerts.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..brokers.base_broker import Order, OrderStatus
from ..execution_manager import ExecutionManager


@dataclass
class ExecutionMetrics:
    """Execution quality metrics."""
    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    fill_rate: float = 0.0
    avg_fill_time: float = 0.0
    total_slippage: float = 0.0
    total_fees: float = 0.0
    execution_latency: float = 0.0


@dataclass
class Alert:
    """Alert data structure."""
    timestamp: float
    level: str  # info, warning, error, critical
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionMonitor:
    """
    Monitors execution quality and performance.
    
    Tracks order execution metrics, latency, slippage, and generates alerts.
    Provides reconciliation with backtesting assumptions.
    """
    
    def __init__(self, execution_manager: ExecutionManager):
        """
        Initialize execution monitor.
        
        Args:
            execution_manager: Execution manager to monitor
        """
        self.execution_manager = execution_manager
        self.logger = logging.getLogger(__name__)
        
        # Metrics tracking
        self.metrics = ExecutionMetrics()
        self.order_latencies: Dict[str, float] = {}
        self.order_fill_times: Dict[str, float] = {}
        
        # Alerts
        self.alerts: List[Alert] = []
        self.max_alerts = 1000
        
        # Reconciliation data
        self.backtest_assumptions = {
            'commission_rate': 0.0006,  # 0.06%
            'slippage_rate': 0.0002  # 0.02%
        }
    
    def update_metrics(self):
        """Update execution metrics from order manager."""
        order_stats = self.execution_manager.order_manager.get_order_statistics()
        
        self.metrics.total_orders = order_stats['total_orders']
        self.metrics.filled_orders = order_stats['filled_orders']
        self.metrics.fill_rate = order_stats['fill_rate']
        self.metrics.total_fees = order_stats['total_fees']
        
        # Count cancelled and rejected orders
        status_counts = order_stats.get('status_counts', {})
        self.metrics.cancelled_orders = status_counts.get('cancelled', 0)
        self.metrics.rejected_orders = status_counts.get('rejected', 0)
    
    def track_order_latency(self, order_id: str, latency: float):
        """
        Track order submission latency.
        
        Args:
            order_id: Order ID
            latency: Latency in seconds
        """
        self.order_latencies[order_id] = latency
        
        # Alert on high latency
        if latency > 1.0:  # 1 second threshold
            self.alert(
                level='warning',
                message=f"High order latency: {latency:.3f}s for order {order_id}",
                metadata={'order_id': order_id, 'latency': latency}
            )
    
    def track_order_fill_time(self, order_id: str, fill_time: float):
        """
        Track order fill time.
        
        Args:
            order_id: Order ID
            fill_time: Time to fill in seconds
        """
        self.order_fill_times[order_id] = fill_time
        
        # Update average fill time
        if self.order_fill_times:
            self.metrics.avg_fill_time = sum(self.order_fill_times.values()) / len(self.order_fill_times)
    
    def calculate_slippage(self, order: Order, expected_price: float) -> float:
        """
        Calculate slippage for an order.
        
        Args:
            order: Executed order
            expected_price: Expected execution price
            
        Returns:
            Slippage as percentage
        """
        if order.status != OrderStatus.FILLED or order.filled_price is None:
            return 0.0
        
        slippage = abs(order.filled_price - expected_price) / expected_price
        return slippage
    
    def alert(self, level: str, message: str, metadata: Dict[str, Any] = None):
        """
        Generate an alert.
        
        Args:
            level: Alert level (info, warning, error, critical)
            message: Alert message
            metadata: Additional metadata
        """
        alert = Alert(
            timestamp=time.time(),
            level=level,
            message=message,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        # Log alert
        log_method = {
            'info': self.logger.info,
            'warning': self.logger.warning,
            'error': self.logger.error,
            'critical': self.logger.critical
        }.get(level, self.logger.info)
        
        log_method(f"[{level.upper()}] {message}")
    
    def get_recent_alerts(self, limit: int = 10) -> List[Alert]:
        """
        Get recent alerts.
        
        Args:
            limit: Number of alerts to return
            
        Returns:
            List of recent alerts
        """
        return self.alerts[-limit:]
    
    def get_alerts_by_level(self, level: str) -> List[Alert]:
        """
        Get alerts by level.
        
        Args:
            level: Alert level
            
        Returns:
            List of alerts with specified level
        """
        return [alert for alert in self.alerts if alert.level == level]
    
    def reconcile_with_backtest(self, backtest_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconcile execution with backtesting assumptions.
        
        Args:
            backtest_metrics: Metrics from backtesting
            
        Returns:
            Reconciliation report
        """
        self.update_metrics()
        
        # Compare fill rates
        backtest_fill_rate = backtest_metrics.get('fill_rate', 1.0)
        fill_rate_diff = abs(self.metrics.fill_rate - backtest_fill_rate)
        
        # Compare costs
        backtest_commission = backtest_metrics.get('total_commission', 0)
        commission_diff = abs(self.metrics.total_fees - backtest_commission)
        
        # Generate alerts for significant deviations
        if fill_rate_diff > 0.05:  # 5% threshold
            self.alert(
                level='warning',
                message=f"Fill rate deviation: {fill_rate_diff:.2%} vs backtest",
                metadata={
                    'execution_fill_rate': self.metrics.fill_rate,
                    'backtest_fill_rate': backtest_fill_rate
                }
            )
        
        if commission_diff > backtest_commission * 0.1:  # 10% threshold
            self.alert(
                level='warning',
                message=f"Commission deviation: {commission_diff:.2f} vs backtest",
                metadata={
                    'execution_commission': self.metrics.total_fees,
                    'backtest_commission': backtest_commission
                }
            )
        
        return {
            'fill_rate': {
                'execution': self.metrics.fill_rate,
                'backtest': backtest_fill_rate,
                'difference': fill_rate_diff
            },
            'commission': {
                'execution': self.metrics.total_fees,
                'backtest': backtest_commission,
                'difference': commission_diff
            },
            'orders': {
                'execution': self.metrics.total_orders,
                'backtest': backtest_metrics.get('total_orders', 0)
            }
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.
        
        Returns:
            Performance report dictionary
        """
        self.update_metrics()
        
        # Calculate average latency
        avg_latency = 0.0
        if self.order_latencies:
            avg_latency = sum(self.order_latencies.values()) / len(self.order_latencies)
        
        # Alert statistics
        alert_counts = {}
        for alert in self.alerts:
            alert_counts[alert.level] = alert_counts.get(alert.level, 0) + 1
        
        return {
            'execution_metrics': {
                'total_orders': self.metrics.total_orders,
                'filled_orders': self.metrics.filled_orders,
                'cancelled_orders': self.metrics.cancelled_orders,
                'rejected_orders': self.metrics.rejected_orders,
                'fill_rate': self.metrics.fill_rate,
                'avg_fill_time': self.metrics.avg_fill_time,
                'total_fees': self.metrics.total_fees
            },
            'latency_metrics': {
                'avg_order_latency': avg_latency,
                'total_orders_tracked': len(self.order_latencies)
            },
            'alert_summary': alert_counts,
            'recent_alerts': self.get_recent_alerts(5)
        }
    
    def check_health(self) -> bool:
        """
        Check execution health status.
        
        Returns:
            True if healthy, False otherwise
        """
        self.update_metrics()
        
        # Check for critical issues
        critical_alerts = self.get_alerts_by_level('critical')
        if critical_alerts:
            return False
        
        # Check fill rate
        if self.metrics.total_orders > 10 and self.metrics.fill_rate < 0.8:
            self.alert(
                level='error',
                message=f"Low fill rate: {self.metrics.fill_rate:.2%}",
                metadata={'fill_rate': self.metrics.fill_rate}
            )
            return False
        
        # Check rejection rate
        if self.metrics.total_orders > 10:
            rejection_rate = self.metrics.rejected_orders / self.metrics.total_orders
            if rejection_rate > 0.1:  # 10% threshold
                self.alert(
                    level='error',
                    message=f"High rejection rate: {rejection_rate:.2%}",
                    metadata={'rejection_rate': rejection_rate}
                )
                return False
        
        return True
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = ExecutionMetrics()
        self.order_latencies.clear()
        self.order_fill_times.clear()
        self.alerts.clear()
