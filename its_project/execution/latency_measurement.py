#!/usr/bin/env python3
"""
Latency Measurement System
==========================

Production-ready latency measurement:
- End-to-end latency tracking
- Component-level latency measurement
- Real-time monitoring
- Performance analytics
"""

from __future__ import annotations

import time
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import sqlite3
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


class LatencyStage(Enum):
    """Latency measurement stages."""
    DATA_INGESTION = "data_ingestion"
    FEATURE_PROCESSING = "feature_processing"
    MODEL_INFERENCE = "model_inference"
    SIGNAL_GENERATION = "signal_generation"
    DECISION_MAKING = "decision_making"
    ORDER_ROUTING = "order_routing"
    EXECUTION = "execution"
    CONFIRMATION = "confirmation"


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    measurement_id: str
    stage: LatencyStage
    start_time: int
    end_time: int
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class EndToEndLatency:
    """End-to-end latency measurement."""
    request_id: str
    start_time: int
    end_time: Optional[int] = None
    total_duration_ms: Optional[float] = None
    stages: List[LatencyMeasurement] = field(default_factory=list)
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Latency statistics."""
    stage: LatencyStage
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    recent_avg_ms: float


class LatencyMeasurementSystem:
    """
    Production-ready latency measurement system.
    
    Features:
    - Multi-stage latency tracking
    - End-to-end latency measurement
    - Real-time statistics
    - Performance alerts
    - Historical analysis
    """
    
    def __init__(
        self,
        base_path: str = "latency_tracking",
        window_size: int = 1000,
        alert_thresholds: Optional[Dict[str, float]] = None
    ) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.window_size = window_size
        self.alert_thresholds = alert_thresholds or {
            'data_ingestion': 10.0,      # 10ms
            'feature_processing': 50.0,   # 50ms
            'model_inference': 100.0,      # 100ms
            'signal_generation': 20.0,     # 20ms
            'decision_making': 30.0,      # 30ms
            'order_routing': 50.0,        # 50ms
            'execution': 100.0,           # 100ms
            'end_to_end': 500.0           # 500ms
        }
        
        # Database
        self.db_path = self.base_path / "latency.db"
        self._init_database()
        
        # Active measurements
        self.active_requests: Dict[str, EndToEndLatency] = {}
        self.stage_timers: Dict[str, Dict[LatencyStage, int]] = {}
        
        # Latency history
        self.latency_history: Dict[LatencyStage, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.end_to_end_history: deque = deque(maxlen=window_size)
        
        # Statistics cache
        self.stats_cache: Dict[LatencyStage, LatencyStats] = {}
        self.cache_timestamp: float = 0
        self.cache_ttl_seconds = 5
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Background monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Performance metrics
        self.metrics = {
            'total_measurements': 0,
            'alerts_triggered': 0,
            'avg_end_to_end_ms': 0.0,
            'max_end_to_end_ms': 0.0,
            'min_end_to_end_ms': float('inf')
        }
    
    def _init_database(self) -> None:
        """Initialize SQLite database for latency tracking."""
        with sqlite3.connect(self.db_path) as conn:
            # Latency measurements table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS latency_measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    measurement_id TEXT,
                    stage TEXT,
                    start_time INTEGER,
                    end_time INTEGER,
                    duration_ms REAL,
                    timestamp INTEGER,
                    metadata TEXT
                )
            """)
            
            # End-to-end measurements table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS end_to_end_measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT,
                    start_time INTEGER,
                    end_time INTEGER,
                    total_duration_ms REAL,
                    stages TEXT,
                    status TEXT,
                    metadata TEXT
                )
            """)
            
            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_measurements_stage ON latency_measurements(stage)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_measurements_timestamp ON latency_measurements(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ee_request_id ON end_to_end_measurements(request_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ee_timestamp ON end_to_end_measurements(start_time)")
    
    def start_measurement(self, request_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Start end-to-end latency measurement.
        
        Args:
            request_id: Unique request identifier
            metadata: Additional metadata
        """
        start_time = int(time.time() * 1000)
        
        measurement = EndToEndLatency(
            request_id=request_id,
            start_time=start_time,
            metadata=metadata or {}
        )
        
        self.active_requests[request_id] = measurement
        self.stage_timers[request_id] = {}
        
        logger.debug(f"Started latency measurement: {request_id}")
    
    def start_stage(self, request_id: str, stage: LatencyStage, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Start timing for a specific stage.
        
        Args:
            request_id: Request identifier
            stage: Latency stage
            metadata: Stage metadata
        """
        if request_id not in self.active_requests:
            logger.warning(f"No active measurement for request: {request_id}")
            return
        
        start_time = int(time.time() * 1000)
        self.stage_timers[request_id][stage] = start_time
        
        logger.debug(f"Started stage {stage.value} for request: {request_id}")
    
    def end_stage(self, request_id: str, stage: LatencyStage, metadata: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """
        End timing for a specific stage.
        
        Args:
            request_id: Request identifier
            stage: Latency stage
            metadata: Stage metadata
            
        Returns:
            Stage duration in milliseconds
        """
        if request_id not in self.active_requests:
            logger.warning(f"No active measurement for request: {request_id}")
            return None
        
        if request_id not in self.stage_timers or stage not in self.stage_timers[request_id]:
            logger.warning(f"No start time for stage {stage.value} in request: {request_id}")
            return None
        
        end_time = int(time.time() * 1000)
        start_time = self.stage_timers[request_id][stage]
        duration_ms = (end_time - start_time)
        
        # Create measurement
        measurement = LatencyMeasurement(
            measurement_id=f"{request_id}_{stage.value}_{int(time.time())}",
            stage=stage,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
        
        # Add to active request
        self.active_requests[request_id].stages.append(measurement)
        
        # Add to history
        self.latency_history[stage].append(duration_ms)
        
        # Save to database
        self._save_measurement(measurement)
        
        # Check for alerts
        self._check_stage_alerts(stage, duration_ms)
        
        # Clean up stage timer
        del self.stage_timers[request_id][stage]
        
        logger.debug(f"Ended stage {stage.value} for request: {request_id} ({duration_ms:.2f}ms)")
        return duration_ms
    
    def end_measurement(self, request_id: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """
        End end-to-end latency measurement.
        
        Args:
            request_id: Request identifier
            metadata: Additional metadata
            
        Returns:
            Total duration in milliseconds
        """
        if request_id not in self.active_requests:
            logger.warning(f"No active measurement for request: {request_id}")
            return None
        
        end_time = int(time.time() * 1000)
        measurement = self.active_requests[request_id]
        measurement.end_time = end_time
        measurement.total_duration_ms = (end_time - measurement.start_time)
        measurement.status = "completed"
        
        if metadata:
            measurement.metadata.update(metadata)
        
        # Add to history
        self.end_to_end_history.append(measurement.total_duration_ms)
        
        # Save to database
        self._save_end_to_end_measurement(measurement)
        
        # Update metrics
        self._update_metrics(measurement.total_duration_ms)
        
        # Check for alerts
        self._check_end_to_end_alerts(measurement.total_duration_ms)
        
        # Clean up
        del self.active_requests[request_id]
        if request_id in self.stage_timers:
            del self.stage_timers[request_id]
        
        logger.debug(f"Ended latency measurement: {request_id} ({measurement.total_duration_ms:.2f}ms)")
        return measurement.total_duration_ms
    
    def measure_function(self, request_id: str, stage: LatencyStage):
        """
        Decorator for measuring function execution time.
        
        Args:
            request_id: Request identifier
            stage: Latency stage
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                self.start_stage(request_id, stage)
                try:
                    result = func(*args, **kwargs)
                    self.end_stage(request_id, stage)
                    return result
                except Exception as e:
                    self.end_stage(request_id, stage, {'error': str(e)})
                    raise
            return wrapper
        return decorator
    
    def get_stage_stats(self, stage: LatencyStage, force_refresh: bool = False) -> LatencyStats:
        """
        Get statistics for a specific stage.
        
        Args:
            stage: Latency stage
            force_refresh: Force refresh of cached stats
            
        Returns:
            Stage statistics
        """
        current_time = time.time()
        
        # Check cache
        if (not force_refresh and 
            stage in self.stats_cache and 
            current_time - self.cache_timestamp < self.cache_ttl_seconds):
            return self.stats_cache[stage]
        
        # Get measurements
        measurements = list(self.latency_history[stage])
        
        if not measurements:
            return LatencyStats(
                stage=stage,
                count=0,
                mean_ms=0.0,
                median_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                std_ms=0.0,
                recent_avg_ms=0.0
            )
        
        # Calculate statistics
        measurements_array = np.array(measurements)
        
        stats = LatencyStats(
            stage=stage,
            count=len(measurements),
            mean_ms=float(np.mean(measurements_array)),
            median_ms=float(np.median(measurements_array)),
            p95_ms=float(np.percentile(measurements_array, 95)),
            p99_ms=float(np.percentile(measurements_array, 99)),
            min_ms=float(np.min(measurements_array)),
            max_ms=float(np.max(measurements_array)),
            std_ms=float(np.std(measurements_array)),
            recent_avg_ms=float(np.mean(measurements[-100:])) if len(measurements) >= 100 else float(np.mean(measurements_array))
        )
        
        # Update cache
        self.stats_cache[stage] = stats
        self.cache_timestamp = current_time
        
        return stats
    
    def get_end_to_end_stats(self) -> LatencyStats:
        """Get end-to-end latency statistics."""
        measurements = list(self.end_to_end_history)
        
        if not measurements:
            return LatencyStats(
                stage=LatencyStage.DATA_INGESTION,  # Placeholder
                count=0,
                mean_ms=0.0,
                median_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                std_ms=0.0,
                recent_avg_ms=0.0
            )
        
        measurements_array = np.array(measurements)
        
        return LatencyStats(
            stage=LatencyStage.DATA_INGESTION,  # Placeholder
            count=len(measurements),
            mean_ms=float(np.mean(measurements_array)),
            median_ms=float(np.median(measurements_array)),
            p95_ms=float(np.percentile(measurements_array, 95)),
            p99_ms=float(np.percentile(measurements_array, 99)),
            min_ms=float(np.min(measurements_array)),
            max_ms=float(np.max(measurements_array)),
            std_ms=float(np.std(measurements_array)),
            recent_avg_ms=float(np.mean(measurements[-100:])) if len(measurements) >= 100 else float(np.mean(measurements_array))
        )
    
    def get_recent_measurements(
        self,
        stage: Optional[LatencyStage] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent latency measurements.
        
        Args:
            stage: Specific stage (None for all)
            limit: Maximum number of measurements
            
        Returns:
            List of measurements
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM latency_measurements"
                params = []
                
                if stage:
                    query += " WHERE stage = ?"
                    params.append(stage.value)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [
                    {
                        'id': row[0],
                        'measurement_id': row[1],
                        'stage': row[2],
                        'start_time': row[3],
                        'end_time': row[4],
                        'duration_ms': row[5],
                        'timestamp': row[6],
                        'metadata': json.loads(row[7]) if row[7] else {}
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Error getting recent measurements: {e}")
            return []
    
    def get_latency_breakdown(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get latency breakdown for a specific request.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Latency breakdown
        """
        # Check active requests first
        if request_id in self.active_requests:
            measurement = self.active_requests[request_id]
        else:
            # Load from database
            measurement = self._load_end_to_end_measurement(request_id)
        
        if not measurement:
            return None
        
        breakdown = {
            'request_id': request_id,
            'total_duration_ms': measurement.total_duration_ms,
            'status': measurement.status,
            'stages': []
        }
        
        for stage_measurement in measurement.stages:
            breakdown['stages'].append({
                'stage': stage_measurement.stage.value,
                'duration_ms': stage_measurement.duration_ms,
                'percentage': (stage_measurement.duration_ms / measurement.total_duration_ms * 100) if measurement.total_duration_ms else 0,
                'metadata': stage_measurement.metadata
            })
        
        return breakdown
    
    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Latency monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Latency monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.running:
            try:
                # Update statistics cache
                for stage in LatencyStage:
                    self.get_stage_stats(stage, force_refresh=True)
                
                # Sleep for next update
                threading.Event().wait(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                threading.Event().wait(5)
    
    def _save_measurement(self, measurement: LatencyMeasurement) -> None:
        """Save measurement to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO latency_measurements VALUES (
                    NULL, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                measurement.measurement_id, measurement.stage.value,
                measurement.start_time, measurement.end_time,
                measurement.duration_ms, measurement.timestamp,
                json.dumps(measurement.metadata)
            ))
        
        self.metrics['total_measurements'] += 1
    
    def _save_end_to_end_measurement(self, measurement: EndToEndLatency) -> None:
        """Save end-to-end measurement to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO end_to_end_measurements VALUES (
                    NULL, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                measurement.request_id, measurement.start_time,
                measurement.end_time, measurement.total_duration_ms,
                json.dumps([asdict(s) for s in measurement.stages]),
                measurement.status, json.dumps(measurement.metadata)
            ))
    
    def _load_end_to_end_measurement(self, request_id: str) -> Optional[EndToEndLatency]:
        """Load end-to-end measurement from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM end_to_end_measurements WHERE request_id = ?
                """, (request_id,))
                
                row = cursor.fetchone()
                if row:
                    stages_data = json.loads(row[4]) if row[4] else []
                    stages = [
                        LatencyMeasurement(
                            measurement_id=s['measurement_id'],
                            stage=LatencyStage(s['stage']),
                            start_time=s['start_time'],
                            end_time=s['end_time'],
                            duration_ms=s['duration_ms'],
                            metadata=s.get('metadata', {})
                        )
                        for s in stages_data
                    ]
                    
                    return EndToEndLatency(
                        request_id=row[1],
                        start_time=row[2],
                        end_time=row[3],
                        total_duration_ms=row[4],
                        stages=stages,
                        status=row[5],
                        metadata=json.loads(row[6]) if row[6] else {}
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading measurement {request_id}: {e}")
            return None
    
    def _update_metrics(self, duration_ms: float) -> None:
        """Update performance metrics."""
        self.metrics['avg_end_to_end_ms'] = (
            (self.metrics['avg_end_to_end_ms'] * (self.metrics['total_measurements'] - 1) + duration_ms) /
            self.metrics['total_measurements']
        )
        
        self.metrics['max_end_to_end_ms'] = max(self.metrics['max_end_to_end_ms'], duration_ms)
        self.metrics['min_end_to_end_ms'] = min(self.metrics['min_end_to_end_ms'], duration_ms)
    
    def _check_stage_alerts(self, stage: LatencyStage, duration_ms: float) -> None:
        """Check for stage latency alerts."""
        threshold = self.alert_thresholds.get(stage.value)
        if threshold and duration_ms > threshold:
            self._trigger_alert(stage, duration_ms, threshold)
    
    def _check_end_to_end_alerts(self, duration_ms: float) -> None:
        """Check for end-to-end latency alerts."""
        threshold = self.alert_thresholds.get('end_to_end')
        if threshold and duration_ms > threshold:
            self._trigger_alert('end_to_end', duration_ms, threshold)
    
    def _trigger_alert(self, component: str, value: float, threshold: float) -> None:
        """Trigger latency alert."""
        self.metrics['alerts_triggered'] += 1
        
        alert_data = {
            'component': component,
            'value': value,
            'threshold': threshold,
            'timestamp': int(time.time() * 1000),
            'severity': 'high' if value > threshold * 2 else 'medium'
        }
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Latency alert: {component} = {value:.2f}ms (threshold: {threshold:.2f}ms)")
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert callback."""
        self.alert_callbacks.append(callback)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all latency statistics."""
        stats = {
            'stage_stats': {},
            'end_to_end_stats': self.get_end_to_end_stats(),
            'metrics': self.metrics.copy(),
            'active_requests': len(self.active_requests),
            'alert_thresholds': self.alert_thresholds.copy()
        }
        
        for stage in LatencyStage:
            stats['stage_stats'][stage.value] = self.get_stage_stats(stage)
        
        return stats


# Convenience functions
def create_latency_system(
    base_path: str = "latency_tracking",
    window_size: int = 1000
) -> LatencyMeasurementSystem:
    """Create latency measurement system with default settings."""
    return LatencyMeasurementSystem(
        base_path=base_path,
        window_size=window_size
    )


def create_high_frequency_latency_system() -> LatencyMeasurementSystem:
    """Create latency system for high-frequency trading."""
    alert_thresholds = {
        'data_ingestion': 1.0,       # 1ms
        'feature_processing': 5.0,    # 5ms
        'model_inference': 10.0,      # 10ms
        'signal_generation': 2.0,    # 2ms
        'decision_making': 3.0,      # 3ms
        'order_routing': 5.0,        # 5ms
        'execution': 10.0,           # 10ms
        'end_to_end': 50.0           # 50ms
    }
    
    return LatencyMeasurementSystem(
        base_path="hft_latency",
        window_size=5000,
        alert_thresholds=alert_thresholds
    )


if __name__ == "__main__":
    # Test latency measurement system
    logging.basicConfig(level=logging.INFO)
    
    latency_system = create_latency_system("test_latency")
    
    # Test measurement
    request_id = "test_request_001"
    
    # Start measurement
    latency_system.start_measurement(request_id, {'test': True})
    
    # Simulate stages
    stages = [
        (LatencyStage.DATA_INGESTION, 0.008),
        (LatencyStage.FEATURE_PROCESSING, 0.045),
        (LatencyStage.MODEL_INFERENCE, 0.095),
        (LatencyStage.SIGNAL_GENERATION, 0.018),
        (LatencyStage.DECISION_MAKING, 0.025),
        (LatencyStage.ORDER_ROUTING, 0.042),
        (LatencyStage.EXECUTION, 0.085)
    ]
    
    for stage, delay in stages:
        latency_system.start_stage(request_id, stage)
        time.sleep(delay)
        latency_system.end_stage(request_id, stage)
    
    # End measurement
    total_duration = latency_system.end_measurement(request_id)
    
    print(f"Total duration: {total_duration:.2f}ms")
    
    # Get statistics
    stats = latency_system.get_all_stats()
    print(f"Stage statistics:")
    for stage_name, stage_stats in stats['stage_stats'].items():
        print(f"  {stage_name}: {stage_stats.mean_ms:.2f}ms avg, {stage_stats.p95_ms:.2f}ms p95")
    
    print(f"End-to-end: {stats['end_to_end_stats'].mean_ms:.2f}ms avg")
    print(f"Total measurements: {stats['metrics']['total_measurements']}")
    
    # Get breakdown
    breakdown = latency_system.get_latency_breakdown(request_id)
    if breakdown:
        print(f"\nRequest breakdown:")
        for stage_info in breakdown['stages']:
            print(f"  {stage_info['stage']}: {stage_info['duration_ms']:.2f}ms ({stage_info['percentage']:.1f}%)")
    
    print("\nLatency measurement system ready!")
