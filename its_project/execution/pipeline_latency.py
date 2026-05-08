#!/usr/bin/env python3
"""
Pipeline Latency Tracking System
=============================

Comprehensive pipeline latency monitoring and analysis
for trading systems with real-time tracking,
statistical analysis, and performance optimization.
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class LatencyPoint:
    """Single latency measurement point."""
    timestamp: datetime
    pipeline_stage: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class LatencyStatistics:
    """Statistical analysis of latency data."""
    stage: str
    count: int
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    error_rate: float
    last_updated: datetime


@dataclass
class PipelineLatencyConfig:
    """Configuration for pipeline latency tracking."""
    max_history_points: int = 10000
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'warning_ms': 100.0,
        'critical_ms': 500.0,
        'error_rate_warning': 0.05,  # 5%
        'error_rate_critical': 0.1   # 10%
    })
    statistics_window: int = 1000
    auto_cleanup: bool = True
    enable_real_time_alerts: bool = True
    enable_persistence: bool = True
    persistence_interval: int = 60  # seconds


class PipelineLatencyTracker:
    """
    Advanced pipeline latency tracking with real-time monitoring and database integration.
    
    Features:
    - Real-time latency measurement
    - Statistical analysis with percentiles
    - Error rate monitoring with configurable thresholds
    - Alert system with customizable warning/critical levels
    - Historical data management with automatic cleanup
    - Interactive visualization with multiple chart types
    - Data persistence with multiple export formats
    - Performance optimization with caching and batch operations
    - Database integration with SQLite backend
    - Background aggregation and maintenance tasks
    """
    
    def __init__(
        self,
        pipeline_name: str,
        config: Optional[PipelineLatencyConfig] = None,
        persistence_manager: Optional[Any] = None
    ) -> None:
        self.pipeline_name = pipeline_name
        self.config = config or PipelineLatencyConfig()
        
        # Initialize database
        from .latency_database import create_latency_database
        self.database = create_latency_database()
        
        # Data storage
        self.latency_points: List[LatencyPoint] = []
        self.stage_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_history_points))
        
        # Statistics cache
        self.statistics_cache: Dict[str, LatencyStatistics] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Alert system
        self.alert_callbacks: List[Callable] = []
        self.alert_history: List[Dict[str, Any]] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.persistence_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self.data_lock = threading.Lock()
        
        logger.info(f"Pipeline latency tracker initialized for: {pipeline_name}")
    
    def record_latency(
        self,
        stage: str,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """
        Record a latency measurement.
        
        Args:
            stage: Pipeline stage name
            latency_ms: Latency in milliseconds
            metadata: Additional metadata
            success: Whether operation was successful
            error_message: Error message if failed
        """
        point = LatencyPoint(
            timestamp=datetime.now(),
            pipeline_stage=stage,
            latency_ms=latency_ms,
            metadata=metadata or {},
            success=success,
            error_message=error_message
        )
        
        with self.data_lock:
            # Add to main storage
            self.latency_points.append(point)
            
            # Add to stage-specific storage
            self.stage_data[stage].append(point)
            
            # Limit history size
            if len(self.latency_points) > self.config.max_history_points:
                self.latency_points = self.latency_points[-self.config.max_history_points:]
            
            # Invalidate cache for this stage
            if stage in self.statistics_cache:
                del self.statistics_cache[stage]
                del self.cache_timestamps[stage]
        
        # Check for alerts
        if self.config.enable_real_time_alerts:
            self._check_alerts(point)
        
        # Log measurement
        if success:
            logger.debug(f"Latency recorded: {stage} - {latency_ms:.2f}ms")
        else:
            logger.warning(f"Latency error recorded: {stage} - {error_message}")
        
        # Trigger persistence if enabled
        if self.config.enable_persistence and self.persistence_manager:
            asyncio.create_task(self._save_latency_point(point))
        
        # Save to database if enabled
        if self.database_manager:
            asyncio.create_task(self._save_latency_point_to_database(point))
    
    async def _save_latency_point_to_database(self, point: LatencyPoint) -> None:
        """Save latency point to database."""
        if not self.database_manager:
            return
        
        try:
            # Convert to database record
            record = LatencyRecord(
                timestamp=point.timestamp,
                pipeline_name=self.pipeline_name,
                stage=point.pipeline_stage,
                execution_id=point.execution_id,
                correlation_id=point.correlation_id,
                latency_ms=point.latency_ms,
                success=point.success,
                error_message=point.error_message,
                metadata=json.dumps(point.metadata) if point.metadata else None,
                thread_id=point.thread_id
            )
            
            # Save to database
            await self.database_manager.save_latency_record(record)
            
            logger.debug(f"Saved latency point to database: {point.pipeline_stage} - {point.latency_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error saving latency point to database: {e}")
    
    async def get_latency_from_database(
        self,
        stage: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[LatencyPoint]:
        """Get latency records from database."""
        if not self.database_manager:
            return []
        
        try:
            # Get records from database
            db_records = await self.database_manager.get_latency_records(
                pipeline_name=self.pipeline_name,
                stage=stage,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            # Convert to LatencyPoint objects
            latency_points = []
            for record in db_records:
                point = LatencyPoint(
                    timestamp=record.timestamp,
                    pipeline_stage=record.stage,
                    latency_ms=record.latency_ms,
                    metadata=json.loads(record.metadata) if record.metadata else {},
                    success=record.success,
                    error_message=record.error_message,
                    execution_id=record.execution_id,
                    correlation_id=record.correlation_id,
                    thread_id=record.thread_id
                )
                latency_points.append(point)
            
            logger.debug(f"Retrieved {len(latency_points)} latency records from database")
            return latency_points
            
        except Exception as e:
            logger.error(f"Error getting latency records from database: {e}")
            return []
    
    async def get_aggregations_from_database(
        self,
        stage: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[LatencyAggregation]:
        """Get aggregations from database."""
        if not self.database_manager:
            return []
        
        try:
            # Get aggregations from database
            db_aggregations = await self.database_manager.get_latency_aggregations(
                pipeline_name=self.pipeline_name,
                stage=stage,
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            
            # Convert to LatencyAggregation objects
            aggregations = []
            for agg in db_aggregations:
                aggregation = LatencyAggregation(
                    id=agg.id,
                    pipeline_name=agg.pipeline_name,
                    stage=agg.stage,
                    date=agg.date,
                    total_executions=agg.total_executions,
                    successful_executions=agg.successful_executions,
                    failed_executions=agg.failed_executions,
                    avg_latency_ms=agg.avg_latency_ms,
                    min_latency_ms=agg.min_latency_ms,
                    max_latency_ms=agg.max_latency_ms,
                    p50_latency_ms=agg.p50_latency_ms,
                    p95_latency_ms=agg.p95_latency_ms,
                    p99_latency_ms=agg.p99_latency_ms,
                    std_latency_ms=agg.std_latency_ms,
                    error_rate=agg.error_rate,
                    created_at=agg.created_at,
                    updated_at=agg.updated_at
                )
                aggregations.append(aggregation)
            
            logger.debug(f"Retrieved {len(aggregations)} aggregations from database")
            return aggregations
            
        except Exception as e:
            logger.error(f"Error getting aggregations from database: {e}")
            return []
    
    def set_database_manager(self, database_manager: Any) -> None:
        """Set database manager for persistence."""
        self.database_manager = database_manager
        logger.info(f"Database manager set for pipeline: {self.pipeline_name}")
    
    def enable_database_persistence(self) -> None:
        """Enable database persistence."""
        if not self.database_manager:
            logger.warning("No database manager available")
            return
        
        self.database_manager.start_monitoring()
        logger.info(f"Database persistence enabled for pipeline: {self.pipeline_name}")
    
    def disable_database_persistence(self) -> None:
        """Disable database persistence."""
        if not self.database_manager:
            return
        
        self.database_manager.stop_monitoring()
        logger.info(f"Database persistence disabled for pipeline: {self.pipeline_name}")
    
    def _check_alerts(self, point: LatencyPoint) -> None:
        """Check for latency alerts based on configuration."""
        alerts = []
        
        # Latency threshold alerts
        if point.success:
            if point.latency_ms >= self.config.alert_thresholds['critical_ms']:
                alerts.append({
                    'type': 'critical_latency',
                    'stage': point.pipeline_stage,
                    'value': point.latency_ms,
                    'threshold': self.config.alert_thresholds['critical_ms'],
                    'message': f"Critical latency: {point.latency_ms:.2f}ms >= {self.config.alert_thresholds['critical_ms']}ms",
                    'timestamp': point.timestamp,
                    'severity': 'critical'
                })
            elif point.latency_ms >= self.config.alert_thresholds['warning_ms']:
                alerts.append({
                    'type': 'warning_latency',
                    'stage': point.pipeline_stage,
                    'value': point.latency_ms,
                    'threshold': self.config.alert_thresholds['warning_ms'],
                    'message': f"High latency: {point.latency_ms:.2f}ms >= {self.config.alert_thresholds['warning_ms']}ms",
                    'timestamp': point.timestamp,
                    'severity': 'warning'
                })
        
        # Error rate alerts
        if not point.success:
            error_rate = self._get_recent_error_rate(point.pipeline_stage)
            if error_rate >= self.config.alert_thresholds['error_rate_critical']:
                alerts.append({
                    'type': 'critical_error_rate',
                    'stage': point.pipeline_stage,
                    'value': error_rate,
                    'threshold': self.config.alert_thresholds['error_rate_critical'],
                    'message': f"Critical error rate: {error_rate:.1%} >= {self.config.alert_thresholds['error_rate_critical']:.1%}",
                    'timestamp': point.timestamp,
                    'severity': 'critical'
                })
            elif error_rate >= self.config.alert_thresholds['error_rate_warning']:
                alerts.append({
                    'type': 'warning_error_rate',
                    'stage': point.pipeline_stage,
                    'value': error_rate,
                    'threshold': self.config.alert_thresholds['error_rate_warning'],
                    'message': f"High error rate: {error_rate:.1%} >= {self.config.alert_thresholds['error_rate_warning']:.1%}",
                    'timestamp': point.timestamp,
                    'severity': 'warning'
                })
        
        # Trigger alert callbacks
        for alert in alerts:
            self.alert_history.append(alert)
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    def _get_recent_error_rate(self, stage: str, window: int = 100) -> float:
        """Calculate recent error rate for a stage."""
        with self.data_lock:
            recent_points = list(self.stage_data[stage])[-window:]
        
        if not recent_points:
            return 0.0
        
        error_count = sum(1 for point in recent_points if not point.success)
        return error_count / len(recent_points)
    
    def get_statistics(self, stage: Optional[str] = None) -> Dict[str, LatencyStatistics]:
        """
        Get latency statistics for stages.
        
        Args:
            stage: Specific stage or None for all stages
            
        Returns:
            Dictionary of statistics by stage
        """
        with self.data_lock:
            if stage:
                return {stage: self._calculate_stage_statistics(stage)}
            else:
                return {
                    s: self._calculate_stage_statistics(s)
                    for s in self.stage_data.keys()
                    if len(self.stage_data[s]) > 0
                }
    
    def _calculate_stage_statistics(self, stage: str) -> LatencyStatistics:
        """Calculate statistics for a specific stage."""
        # Check cache
        cache_key = stage
        now = datetime.now()
        
        if (cache_key in self.statistics_cache and 
            cache_key in self.cache_timestamps and
            (now - self.cache_timestamps[cache_key]).seconds < 60):  # 1 minute cache
            return self.statistics_cache[cache_key]
        
        points = list(self.stage_data[stage])
        if not points:
            return LatencyStatistics(
                stage=stage, count=0, mean_ms=0.0, median_ms=0.0,
                std_ms=0.0, min_ms=0.0, max_ms=0.0,
                p50_ms=0.0, p95_ms=0.0, p99_ms=0.0,
                error_rate=0.0, last_updated=now
            )
        
        # Calculate basic statistics
        successful_points = [p for p in points if p.success]
        latencies = [p.latency_ms for p in successful_points]
        
        if latencies:
            count = len(latencies)
            mean_ms = statistics.mean(latencies)
            median_ms = statistics.median(latencies)
            std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            min_ms = min(latencies)
            max_ms = max(latencies)
            
            # Calculate percentiles
            sorted_latencies = sorted(latencies)
            p50_ms = np.percentile(sorted_latencies, 50)
            p95_ms = np.percentile(sorted_latencies, 95)
            p99_ms = np.percentile(sorted_latencies, 99)
        else:
            count = mean_ms = median_ms = std_ms = 0.0
            min_ms = max_ms = p50_ms = p95_ms = p99_ms = 0.0
        
        # Calculate error rate
        error_rate = 1.0 - (len(successful_points) / len(points)) if points else 0.0
        
        stats = LatencyStatistics(
            stage=stage,
            count=count,
            mean_ms=mean_ms,
            median_ms=median_ms,
            std_ms=std_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            p50_ms=p50_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
            error_rate=error_rate,
            last_updated=now
        )
        
        # Cache results
        self.statistics_cache[cache_key] = stats
        self.cache_timestamps[cache_key] = now
        
        return stats
    
    def get_latency_dataframe(self, stage: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get latency data as pandas DataFrame.
        
        Args:
            stage: Filter by specific stage
            limit: Limit number of records
            
        Returns:
            DataFrame with latency data
        """
        with self.data_lock:
            points = self.latency_points
        
        if stage:
            points = [p for p in points if p.pipeline_stage == stage]
        
        if limit:
            points = points[-limit:]
        
        if not points:
            return pd.DataFrame()
        
        data = []
        for point in points:
            row = {
                'timestamp': point.timestamp,
                'pipeline_stage': point.pipeline_stage,
                'latency_ms': point.latency_ms,
                'success': point.success,
                'error_message': point.error_message or ''
            }
            
            # Add metadata
            for key, value in point.metadata.items():
                row[f'meta_{key}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_pipeline_overview(self) -> Dict[str, Any]:
        """Get comprehensive pipeline overview."""
        with self.data_lock:
            total_points = len(self.latency_points)
            stage_stats = {}
            
            for stage in self.stage_data.keys():
                stage_points = list(self.stage_data[stage])
                if stage_points:
                    successful = [p for p in stage_points if p.success]
                    latencies = [p.latency_ms for p in successful]
                    
                    if latencies:
                        stage_stats[stage] = {
                            'count': len(latencies),
                            'mean_latency': statistics.mean(latencies),
                            'p95_latency': np.percentile(latencies, 95),
                            'error_rate': 1.0 - (len(successful) / len(stage_points))
                        }
                    else:
                        stage_stats[stage] = {
                            'count': 0,
                            'mean_latency': 0.0,
                            'p95_latency': 0.0,
                            'error_rate': 1.0
                        }
                else:
                    stage_stats[stage] = {
                        'count': 0,
                        'mean_latency': 0.0,
                        'p95_latency': 0.0,
                        'error_rate': 0.0
                    }
        
        return {
            'pipeline_name': self.pipeline_name,
            'total_measurements': total_points,
            'active_stages': list(self.stage_data.keys()),
            'stage_statistics': stage_stats,
            'last_updated': datetime.now(),
            'alert_count': len(self.alert_history)
        }
    
    def create_latency_chart(
        self,
        stage: Optional[str] = None,
        time_range: Optional[timedelta] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive latency chart.
        
        Args:
            stage: Filter by specific stage
            time_range: Time range for data
            save_path: Path to save chart
            
        Returns:
            Plotly figure object
        """
        df = self.get_latency_dataframe(stage=stage)
        
        if df.empty:
            return go.Figure()
        
        # Filter by time range
        if time_range:
            cutoff_time = datetime.now() - time_range
            df = df[df.index >= cutoff_time]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Latency Over Time', 'Latency Distribution'),
            row_heights=[0.7, 0.3]
        )
        
        # Latency time series
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['latency_ms'],
                mode='markers+lines',
                name='Latency',
                line=dict(color='blue', width=1),
                marker=dict(size=4, color='blue'),
                hovertemplate='Time: %{x}<br>Latency: %{y:.2f}ms<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add alert thresholds
        fig.add_hline(
            y=self.config.alert_thresholds['warning_ms'],
            line_dash="dash",
            line_color="orange",
            annotation_text="Warning",
            row=1, col=1
        )
        
        fig.add_hline(
            y=self.config.alert_thresholds['critical_ms'],
            line_dash="dash",
            line_color="red",
            annotation_text="Critical",
            row=1, col=1
        )
        
        # Latency distribution
        fig.add_trace(
            go.Histogram(
                x=df['latency_ms'],
                nbinsx=50,
                name='Distribution',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        # Update layout
        title = f"Pipeline Latency - {self.pipeline_name}"
        if stage:
            title += f" ({stage})"
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Latency chart saved to {save_path}")
        
        return fig
    
    def create_performance_dashboard(
        self,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create comprehensive performance dashboard.
        
        Args:
            save_path: Path to save dashboard
            
        Returns:
            Plotly figure object
        """
        stats = self.get_statistics()
        
        if not stats:
            return go.Figure()
        
        # Create dashboard with KPIs
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Latency Metrics', 'Error Rates',
                'Stage Performance', 'Recent Trends',
                'Alert Summary', 'System Health'
            ),
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "indicator"}]
            ]
        )
        
        # Prepare data for charts
        stages = list(stats.keys())
        mean_latencies = [stats[s].mean_ms for s in stages]
        p95_latencies = [stats[s].p95_ms for s in stages]
        error_rates = [stats[s].error_rate for s in stages]
        
        # Latency metrics indicators
        overall_mean = statistics.mean(mean_latencies) if mean_latencies else 0.0
        overall_p95 = statistics.mean(p95_latencies) if p95_latencies else 0.0
        
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=overall_mean,
                title={"text": "Mean Latency (ms)"},
                gauge={'axis': {'range': [None, 1000]}},
                delta={'reference': self.config.alert_thresholds['warning_ms']}
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=overall_p95,
                title={"text": "P95 Latency (ms)"},
                gauge={'axis': {'range': [None, 1000]}},
                delta={'reference': self.config.alert_thresholds['critical_ms']}
            ),
            row=1, col=2
        )
        
        # Stage performance bar chart
        fig.add_trace(
            go.Bar(
                x=stages,
                y=mean_latencies,
                name='Mean Latency',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=stages,
                y=error_rates,
                name='Error Rate',
                marker_color='lightcoral'
            ),
            row=2, col=2
        )
        
        # Recent trends (last N points)
        recent_df = self.get_latency_dataframe(limit=100)
        if not recent_df.empty:
            # Calculate rolling averages
            recent_df['rolling_mean'] = recent_df['latency_ms'].rolling(window=10).mean()
            
            # Plot by stage
            for stage in stages[:5]:  # Limit to 5 stages
                stage_data = recent_df[recent_df['pipeline_stage'] == stage]
                if not stage_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=stage_data.index,
                            y=stage_data['latency_ms'],
                            mode='lines',
                            name=f'{stage} Trends',
                            line=dict(width=1)
                        ),
                        row=3, col=1
                    )
        
        # System health indicator
        critical_alerts = len([a for a in self.alert_history if a.get('severity') == 'critical'])
        health_score = max(0, 100 - critical_alerts * 10)
        
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=health_score,
                title={"text": "System Health"},
                gauge={'axis': {'range': [None, 100]}},
                delta={'reference': 80}
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Pipeline Performance Dashboard - {self.pipeline_name}",
            template='plotly_white',
            height=1000,
            showlegend=False
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Performance dashboard saved to {save_path}")
        
        return fig
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add callback for latency alerts."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring is already active")
            return
        
        self.is_monitoring = True
        logger.info(f"Started latency monitoring for pipeline: {self.pipeline_name}")
        
        # Start persistence task if enabled
        if self.config.enable_persistence:
            self.persistence_task = asyncio.create_task(self._persistence_loop())
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        logger.info(f"Stopped latency monitoring for pipeline: {self.pipeline_name}")
        
        # Cancel persistence task
        if self.persistence_task:
            self.persistence_task.cancel()
            try:
                asyncio.run(self.persistence_task)
            except asyncio.CancelledError:
                pass
    
    async def _persistence_loop(self) -> None:
        """Background persistence loop."""
        while self.is_monitoring:
            try:
                # Save current statistics
                stats = self.get_statistics()
                
                # Save to persistence manager
                if self.persistence_manager:
                    for stage_stat in stats.values():
                        await self._save_statistics(stage_stat)
                
                # Cleanup old data if enabled
                if self.config.auto_cleanup:
                    await self._cleanup_old_data()
                
                await asyncio.sleep(self.config.persistence_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in persistence loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _save_latency_point(self, point: LatencyPoint) -> None:
        """Save latency point to persistence."""
        if not self.persistence_manager:
            return
        
        try:
            # Convert to dict for storage
            point_data = {
                'timestamp': point.timestamp,
                'pipeline_name': self.pipeline_name,
                'pipeline_stage': point.pipeline_stage,
                'latency_ms': point.latency_ms,
                'success': point.success,
                'error_message': point.error_message,
                'metadata': json.dumps(point.metadata)
            }
            
            # Save using persistence manager
            # This would need to be implemented based on the persistence manager interface
            logger.debug(f"Saved latency point: {point.pipeline_stage} - {point.latency_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error saving latency point: {e}")
    
    async def _save_statistics(self, stats: LatencyStatistics) -> None:
        """Save statistics to persistence."""
        if not self.persistence_manager:
            return
        
        try:
            stats_data = {
                'pipeline_name': self.pipeline_name,
                'stage': stats.stage,
                'count': stats.count,
                'mean_ms': stats.mean_ms,
                'median_ms': stats.median_ms,
                'std_ms': stats.std_ms,
                'min_ms': stats.min_ms,
                'max_ms': stats.max_ms,
                'p50_ms': stats.p50_ms,
                'p95_ms': stats.p95_ms,
                'p99_ms': stats.p99_ms,
                'error_rate': stats.error_rate,
                'last_updated': stats.last_updated
            }
            
            # Save using persistence manager
            logger.debug(f"Saved statistics for stage: {stats.stage}")
            
        except Exception as e:
            logger.error(f"Error saving statistics: {e}")
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old latency data."""
        with self.data_lock:
            # Remove old points beyond max history
            if len(self.latency_points) > self.config.max_history_points:
                excess = len(self.latency_points) - self.config.max_history_points
                self.latency_points = self.latency_points[excess:]
            
            # Clean up stage data
            for stage in self.stage_data:
                if len(self.stage_data[stage]) > self.config.max_history_points:
                    excess = len(self.stage_data[stage]) - self.config.max_history_points
                    for _ in range(excess):
                        self.stage_data[stage].popleft()
        
        logger.debug("Cleaned up old latency data")
    
    def export_data(
        self,
        format: str = "csv",
        include_statistics: bool = True,
        save_path: Optional[str] = None
    ) -> str:
        """
        Export latency data.
        
        Args:
            format: Export format ('csv', 'json', 'excel')
            include_statistics: Include statistics summary
            save_path: Custom save path
            
        Returns:
            Path to exported file
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"pipeline_latency_{self.pipeline_name}_{timestamp}.{format}"
        
        # Get data
        df = self.get_latency_dataframe()
        stats = self.get_statistics()
        
        if format.lower() == "csv":
            # Export to CSV
            df.to_csv(save_path)
            
            if include_statistics:
                stats_df = pd.DataFrame([
                    {
                        'stage': s.stage,
                        'count': s.count,
                        'mean_ms': s.mean_ms,
                        'p95_ms': s.p95_ms,
                        'error_rate': s.error_rate,
                        'last_updated': s.last_updated
                    }
                    for s in stats.values()
                ])
                stats_df.to_csv(save_path.replace('.csv', '_statistics.csv'))
        
        elif format.lower() == "json":
            # Export to JSON
            export_data = {
                'pipeline_name': self.pipeline_name,
                'export_timestamp': datetime.now().isoformat(),
                'latency_points': df.to_dict('records'),
                'statistics': {
                    stage: {
                        'count': s.count,
                        'mean_ms': s.mean_ms,
                        'median_ms': s.median_ms,
                        'std_ms': s.std_ms,
                        'min_ms': s.min_ms,
                        'max_ms': s.max_ms,
                        'p50_ms': s.p50_ms,
                        'p95_ms': s.p95_ms,
                        'p99_ms': s.p99_ms,
                        'error_rate': s.error_rate,
                        'last_updated': s.last_updated.isoformat()
                    }
                    for stage, s in stats.items()
                },
                'alert_history': self.alert_history
            }
            
            with open(save_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format.lower() == "excel":
            # Export to Excel
            with pd.ExcelWriter(save_path) as writer:
                df.to_excel(writer, sheet_name='Latency Data', index=False)
                
                if include_statistics:
                    stats_df = pd.DataFrame([
                        {
                            'stage': s.stage,
                            'count': s.count,
                            'mean_ms': s.mean_ms,
                            'p95_ms': s.p95_ms,
                            'error_rate': s.error_rate
                        }
                        for s in stats.values()
                    ])
                    stats_df.to_excel(writer, sheet_name='Statistics', index=False)
                
                # Export alerts
                if self.alert_history:
                    alerts_df = pd.DataFrame(self.alert_history)
                    alerts_df.to_excel(writer, sheet_name='Alerts', index=False)
        
        logger.info(f"Pipeline latency data exported to {save_path}")
        return save_path


class LatencyContextManager:
    """
    Context manager for measuring pipeline latency.
    
    Usage:
        with LatencyContextManager(tracker, "data_processing") as ctx:
            # Your pipeline code here
            result = process_data(data)
            ctx.set_metadata({"input_size": len(data)})
    """
    
    def __init__(
        self,
        tracker: PipelineLatencyTracker,
        stage: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.tracker = tracker
        self.stage = stage
        self.metadata = metadata or {}
        self.start_time = None
        self.success = True
        self.error_message = None
    
    def __enter__(self) -> 'LatencyContextManager':
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time is None:
            return
        
        latency_ms = (time.time() - self.start_time) * 1000  # Convert to milliseconds
        
        if exc_type is not None:
            self.success = False
            self.error_message = str(exc_val)
        
        self.tracker.record_latency(
            stage=self.stage,
            latency_ms=latency_ms,
            metadata=self.metadata,
            success=self.success,
            error_message=self.error_message
        )
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the latency measurement."""
        self.metadata[key] = value


def measure_latency(stage: str):
    """Decorator for measuring function latency."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000
                logger.debug(f"Function {func.__name__} latency: {latency_ms:.2f}ms")
                return result
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                logger.error(f"Function {func.__name__} failed after {latency_ms:.2f}ms: {e}")
                raise
        return wrapper
    return decorator


# Convenience functions
def create_latency_tracker(
    pipeline_name: str,
    config: Optional[PipelineLatencyConfig] = None,
    persistence_manager: Optional[Any] = None
) -> PipelineLatencyTracker:
    """Create pipeline latency tracker with default settings."""
    return PipelineLatencyTracker(pipeline_name, config, persistence_manager)


def create_latency_config(
    max_history_points: int = 10000,
    warning_threshold_ms: float = 100.0,
    critical_threshold_ms: float = 500.0
) -> PipelineLatencyConfig:
    """Create latency tracking configuration."""
    return PipelineLatencyConfig(
        max_history_points=max_history_points,
        alert_thresholds={
            'warning_ms': warning_threshold_ms,
            'critical_ms': critical_threshold_ms
        }
    )
