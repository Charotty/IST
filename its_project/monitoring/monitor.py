#!/usr/bin/env python3
"""
Advanced Monitoring System
===========================

Comprehensive monitoring with metrics collection, health checks, and alerting.
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import threading

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Metric data point."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert notification."""
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    timestamp: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores system metrics."""
    
    def __init__(self, max_history: int = 1000) -> None:
        self.metrics: Dict[str, deque] = {}
        self.max_history = max_history
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=self.max_history)
            
            metric = Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {}
            )
            
            self.metrics[name].append(metric)
    
    def get_metric(self, name: str, limit: int = 100) -> List[Metric]:
        """Get recent metric values."""
        with self._lock:
            if name not in self.metrics:
                return []
            return list(self.metrics[name])[-limit:]
    
    def get_all_metrics(self) -> Dict[str, List[Metric]]:
        """Get all metrics."""
        with self._lock:
            return {name: list(values) for name, values in self.metrics.items()}
    
    def get_metric_summary(self, name: str) -> Optional[Dict[str, float]]:
        """Get metric summary statistics."""
        metrics = self.get_metric(name)
        if not metrics:
            return None
        
        values = [m.value for m in metrics]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1]
        }


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self) -> None:
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable] = []
        self.alert_rules: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def add_alert_handler(self, handler: Callable) -> None:
        """Add alert handler callback."""
        self.alert_handlers.append(handler)
    
    def add_alert_rule(self, rule: Dict[str, Any]) -> None:
        """Add alert rule."""
        self.alert_rules.append(rule)
    
    def trigger_alert(
        self,
        severity: str,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Trigger an alert."""
        alert = Alert(
            severity=severity,
            message=message,
            timestamp=time.time(),
            source=source,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.alerts.append(alert)
            # Keep last 1000 alerts
            if len(self.alerts) > 1000:
                self.alerts.pop(0)
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def check_alert_rules(self, metrics: Dict[str, List[Metric]]) -> None:
        """Check if any alert rules are triggered."""
        for rule in self.alert_rules:
            try:
                metric_name = rule.get("metric")
                if metric_name not in metrics:
                    continue
                
                latest_value = metrics[metric_name][-1].value if metrics[metric_name] else None
                if latest_value is None:
                    continue
                
                # Check threshold
                threshold = rule.get("threshold")
                operator = rule.get("operator", ">")
                severity = rule.get("severity", "WARNING")
                
                triggered = False
                if operator == ">" and latest_value > threshold:
                    triggered = True
                elif operator == "<" and latest_value < threshold:
                    triggered = True
                elif operator == "==" and latest_value == threshold:
                    triggered = True
                
                if triggered:
                    self.trigger_alert(
                        severity=severity,
                        message=f"{metric_name} {operator} {threshold}: {latest_value}",
                        source="alert_manager",
                        metadata={"metric": metric_name, "value": latest_value}
                    )
                    
            except Exception as e:
                logger.error(f"Error checking alert rule: {e}")
    
    def get_recent_alerts(self, limit: int = 50) -> List[Alert]:
        """Get recent alerts."""
        with self._lock:
            return self.alerts[-limit:]


class SystemMonitor:
    """Comprehensive system monitoring."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.metrics = MetricsCollector(max_history=config.get("max_history", 1000))
        self.alerts = AlertManager()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self.start_time = time.time()
        
        # Setup default alert rules
        self._setup_default_alert_rules()
    
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules."""
        self.alerts.add_alert_rule({
            "metric": "cpu_usage_percent",
            "threshold": 90.0,
            "operator": ">",
            "severity": "WARNING"
        })
        
        self.alerts.add_alert_rule({
            "metric": "memory_usage_percent",
            "threshold": 85.0,
            "operator": ">",
            "severity": "WARNING"
        })
        
        self.alerts.add_alert_rule({
            "metric": "disk_usage_percent",
            "threshold": 90.0,
            "operator": ">",
            "severity": "WARNING"
        })
    
    async def start(self, interval: float = 5.0) -> None:
        """Start monitoring."""
        if self._running:
            logger.warning("Monitor already running")
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info("System monitoring started")
    
    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")
    
    async def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_metrics(self) -> None:
        """Collect system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.metrics.record_metric("cpu_usage_percent", cpu_percent)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics.record_metric("memory_usage_percent", memory.percent)
        self.metrics.record_metric("memory_available_gb", memory.available / (1024**3))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metrics.record_metric("disk_usage_percent", disk.percent)
        self.metrics.record_metric("disk_free_gb", disk.free / (1024**3))
        
        # Network metrics
        net_io = psutil.net_io_counters()
        self.metrics.record_metric("network_bytes_sent", net_io.bytes_sent)
        self.metrics.record_metric("network_bytes_recv", net_io.bytes_recv)
        
        # System uptime
        uptime = time.time() - self.start_time
        self.metrics.record_metric("system_uptime_seconds", uptime)
        
        # Check alert rules
        all_metrics = self.metrics.get_all_metrics()
        self.alerts.check_alert_rules(all_metrics)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        uptime = time.time() - self.start_time
        
        cpu_summary = self.metrics.get_metric_summary("cpu_usage_percent")
        memory_summary = self.metrics.get_metric_summary("memory_usage_percent")
        disk_summary = self.metrics.get_metric_summary("disk_usage_percent")
        
        # Determine overall health
        health_status = "healthy"
        if cpu_summary and cpu_summary["avg"] > 80:
            health_status = "degraded"
        if memory_summary and memory_summary["avg"] > 85:
            health_status = "degraded"
        if disk_summary and disk_summary["avg"] > 90:
            health_status = "critical"
        
        recent_alerts = self.alerts.get_recent_alerts(10)
        critical_alerts = [a for a in recent_alerts if a.severity == "CRITICAL"]
        if critical_alerts:
            health_status = "critical"
        
        return {
            "status": health_status,
            "uptime_seconds": uptime,
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            "cpu": cpu_summary,
            "memory": memory_summary,
            "disk": disk_summary,
            "recent_alerts": len(recent_alerts),
            "critical_alerts": len(critical_alerts),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_metrics_report(self) -> Dict[str, Any]:
        """Get comprehensive metrics report."""
        return {
            "system_health": self.get_system_health(),
            "all_metrics": {name: self.metrics.get_metric_summary(name) 
                          for name in self.metrics.metrics.keys()},
            "recent_alerts": [
                {
                    "severity": a.severity,
                    "message": a.message,
                    "timestamp": datetime.fromtimestamp(a.timestamp).isoformat(),
                    "source": a.source
                }
                for a in self.alerts.get_recent_alerts(20)
            ]
        }
    
    def record_custom_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record custom application metric."""
        self.metrics.record_metric(name, value, labels)
    
    def record_trade_metric(self, symbol: str, action: str, amount: float, price: float) -> None:
        """Record trade execution metric."""
        labels = {"symbol": symbol, "action": action}
        self.metrics.record_metric("trade_count", 1.0, labels)
        self.metrics.record_metric("trade_volume", amount * price, labels)
    
    def record_model_metric(self, model_name: str, prediction_time: float, confidence: float) -> None:
        """Record model inference metric."""
        labels = {"model": model_name}
        self.metrics.record_metric("model_prediction_time_ms", prediction_time * 1000, labels)
        self.metrics.record_metric("model_confidence", confidence, labels)


# Singleton instance
_monitor_instance: Optional[SystemMonitor] = None


def get_monitor() -> SystemMonitor:
    """Get global monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SystemMonitor()
    return _monitor_instance


async def test_monitoring():
    """Test monitoring system."""
    monitor = SystemMonitor()
    
    # Add console alert handler
    def console_handler(alert: Alert):
        print(f"[{alert.severity}] {alert.message}")
    
    monitor.alerts.add_alert_handler(console_handler)
    
    # Start monitoring
    await monitor.start(interval=2.0)
    
    # Record some custom metrics
    for i in range(5):
        monitor.record_custom_metric("test_metric", float(i * 10))
        monitor.record_trade_metric("BTC/USDT", "buy", 0.001, 42000.0)
        await asyncio.sleep(1)
    
    # Get health status
    health = monitor.get_system_health()
    print(f"System Health: {health['status']}")
    print(f"Uptime: {health['uptime_formatted']}")
    
    # Get metrics report
    report = monitor.get_metrics_report()
    print(f"Metrics: {len(report['all_metrics'])} metrics collected")
    print(f"Alerts: {len(report['recent_alerts'])} recent alerts")
    
    # Stop monitoring
    await monitor.stop()
    
    print("Monitoring test completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_monitoring())
