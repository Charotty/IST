#!/usr/bin/env python3
"""
Pipeline Logging System
====================

Comprehensive logging system for data → execution pipeline
with structured logging, stage tracking, and performance monitoring.
"""

from __future__ import annotations

import logging
import time
import json
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import asyncio
import gzip
import hashlib

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels for pipeline logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class PipelineStage(Enum):
    """Pipeline stages for logging."""
    DATA_INGESTION = "data_ingestion"
    DATA_VALIDATION = "data_validation"
    DATA_TRANSFORMATION = "data_transformation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_INFERENCE = "model_inference"
    DECISION_MAKING = "decision_making"
    ORDER_GENERATION = "order_generation"
    ORDER_EXECUTION = "order_execution"
    EXECUTION_CONFIRMATION = "execution_confirmation"
    ERROR_HANDLING = "error_handling"
    CLEANUP = "cleanup"


@dataclass
class PipelineLogEntry:
    """Single pipeline log entry."""
    timestamp: datetime
    stage: PipelineStage
    level: LogLevel
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    execution_id: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    thread_id: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""
    stage: PipelineStage
    total_executions: int
    successful_executions: int
    failed_executions: int
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    p95_duration_ms: float
    error_rate: float
    last_execution: Optional[datetime] = None
    last_error: Optional[str] = None


@dataclass
class PipelineLoggerConfig:
    """Configuration for pipeline logger."""
    log_level: LogLevel = LogLevel.INFO
    enable_file_logging: bool = True
    enable_database_logging: bool = True
    log_directory: str = "logs"
    log_file_prefix: str = "pipeline"
    max_file_size_mb: int = 100
    enable_compression: bool = True
    enable_structured_logging: bool = True
    enable_performance_tracking: bool = True
    metrics_window_size: int = 1000
    enable_correlation_tracking: bool = True
    retention_days: int = 30


class PipelineLogger:
    """
    Advanced pipeline logging system with structured logging,
    performance tracking, and comprehensive monitoring.
    
    Features:
    - Multi-level logging with structured format
    - Stage-based execution tracking
    - Performance metrics collection
    - Correlation and traceability
    - File and database logging
    - Real-time monitoring
    - Log aggregation and analysis
    """
    
    def __init__(
        self,
        pipeline_name: str,
        config: Optional[PipelineLoggerConfig] = None
    ) -> None:
        self.pipeline_name = pipeline_name
        self.config = config or PipelineLoggerConfig()
        
        # Data storage
        self.log_entries: List[PipelineLogEntry] = []
        self.stage_metrics: Dict[PipelineStage, PipelineMetrics] = {}
        self.correlation_map: Dict[str, str] = {}
        
        # Performance tracking
        self.stage_timers: Dict[str, float] = {}
        self.error_counts: Dict[str, int] = {}
        self.success_counts: Dict[str, int] = {}
        
        # Thread safety
        self.data_lock = threading.Lock()
        
        # Logging setup
        self._setup_logging()
        
        # Background tasks
        self.monitoring_active = False
        self.aggregation_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"Pipeline logger initialized for: {pipeline_name}")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        # Create log directory
        log_dir = Path(self.config.log_directory)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file handler
        if self.config.enable_file_logging:
            self._setup_file_handler(log_dir)
        
        # Setup database logging (would need database connection)
        if self.config.enable_database_logging:
            self._setup_database_logging()
        
        # Configure root logger
        logging.getLogger().setLevel(getattr(logging, self.config.log_level.value))
    
    def _setup_file_handler(self, log_dir: Path) -> None:
        """Setup file logging with rotation and compression."""
        import logging.handlers
        
        # Create file handler
        log_file = log_dir / f"{self.config.log_file_prefix}_{self.pipeline_name}.log"
        
        handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file),
            maxBytes=self.config.max_file_size_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=5,
            encoding='utf-8'
        )
        
        # Setup formatter
        if self.config.enable_structured_logging:
            formatter = self._create_structured_formatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        handler.setFormatter(formatter)
        
        # Add handler to logger
        pipeline_logger = logging.getLogger(f"pipeline.{self.pipeline_name}")
        pipeline_logger.addHandler(handler)
        pipeline_logger.setLevel(getattr(logging, self.config.log_level.value))
    
    def _create_structured_formatter(self) -> logging.Formatter:
        """Create structured JSON formatter for logging."""
        class StructuredFormatter(logging.Formatter):
            def format(self, record):
                # Create structured log entry
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'pipeline': self.pipeline_name,
                    'level': record.levelname,
                    'stage': getattr(record, 'stage', 'unknown'),
                    'message': record.getMessage(),
                    'thread': record.threadName,
                    'process': record.process,
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add extra fields if available
                if hasattr(record, 'data'):
                    log_entry['data'] = record.data
                if hasattr(record, 'execution_id'):
                    log_entry['execution_id'] = record.execution_id
                if hasattr(record, 'duration_ms'):
                    log_entry['duration_ms'] = record.duration_ms
                if hasattr(record, 'correlation_id'):
                    log_entry['correlation_id'] = record.correlation_id
                
                return json.dumps(log_entry, default=str)
        
        return StructuredFormatter()
    
    def _setup_database_logging(self) -> None:
        """Setup database logging (placeholder for implementation)."""
        # This would integrate with the existing database system
        logger.debug("Database logging setup (implementation needed)")
    
    def start_stage(
        self,
        stage: PipelineStage,
        execution_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start timing a pipeline stage.
        
        Args:
            stage: Pipeline stage being executed
            execution_id: Unique execution identifier
            correlation_id: Correlation ID for traceability
            metadata: Additional metadata
            
        Returns:
            Stage execution ID for correlation
        """
        stage_id = f"{stage.value}_{int(time.time() * 1000)}"
        
        with self.data_lock:
            self.stage_timers[stage_id] = time.time()
            
            if correlation_id:
                self.correlation_map[stage_id] = correlation_id
            
            # Log stage start
            self._log(
                level=LogLevel.INFO,
                stage=stage,
                message=f"Stage started: {stage.value}",
                execution_id=execution_id,
                correlation_id=correlation_id,
                metadata=metadata or {}
            )
        
        return stage_id
    
    def end_stage(
        self,
        stage: PipelineStage,
        stage_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        End timing a pipeline stage.
        
        Args:
            stage: Pipeline stage being completed
            stage_id: Stage execution ID from start_stage
            success: Whether stage completed successfully
            error_message: Error message if failed
            metadata: Additional metadata
        """
        with self.data_lock:
            if stage_id in self.stage_timers:
                duration_ms = (time.time() - self.stage_timers[stage_id]) * 1000
                del self.stage_timers[stage_id]
                
                # Update metrics
                self._update_metrics(stage, success, duration_ms)
                
                # Log stage completion
                level = LogLevel.INFO if success else LogLevel.ERROR
                message = f"Stage completed: {stage.value}" if success else f"Stage failed: {stage.value}"
                
                self._log(
                    level=level,
                    stage=stage,
                    message=message,
                    duration_ms=duration_ms,
                    error_message=error_message,
                    metadata=metadata or {}
                )
            else:
                logger.warning(f"Stage ID {stage_id} not found in timers")
    
    def log_event(
        self,
        stage: PipelineStage,
        level: LogLevel,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a pipeline event.
        
        Args:
            stage: Pipeline stage
            level: Log level
            message: Log message
            data: Additional data
            execution_id: Execution identifier
            correlation_id: Correlation ID
            metadata: Additional metadata
        """
        entry = PipelineLogEntry(
            timestamp=datetime.now(),
            stage=stage,
            level=level,
            message=message,
            data=data or {},
            execution_id=execution_id,
            metadata=metadata or {},
            correlation_id=correlation_id
        )
        
        with self.data_lock:
            self.log_entries.append(entry)
            
            # Update error counts
            if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                self.error_counts[stage.value] = self.error_counts.get(stage.value, 0) + 1
            else:
                self.success_counts[stage.value] = self.success_counts.get(stage.value, 0) + 1
            
            # Limit log entries
            if len(self.log_entries) > 10000:  # Configurable limit
                self.log_entries = self.log_entries[-5000:]
        
        # Log using standard logging
        logger = logging.getLogger(f"pipeline.{self.pipeline_name}")
        log_level = getattr(logging, level.value)
        
        # Add extra fields for structured logging
        extra = {
            'stage': stage.value,
            'data': data,
            'execution_id': execution_id,
            'correlation_id': correlation_id
        }
        
        if metadata:
            extra['metadata'] = metadata
        
        logger.log(log_level, message, extra=extra)
    
    def _update_metrics(self, stage: PipelineStage, success: bool, duration_ms: float) -> None:
        """Update performance metrics for a stage."""
        if stage not in self.stage_metrics:
            self.stage_metrics[stage] = PipelineMetrics(
                stage=stage,
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                avg_duration_ms=0.0,
                min_duration_ms=float('inf'),
                max_duration_ms=0.0,
                p95_duration_ms=0.0,
                error_rate=0.0
            )
        
        metrics = self.stage_metrics[stage]
        metrics.total_executions += 1
        
        if success:
            metrics.successful_executions += 1
        else:
            metrics.failed_executions += 1
        
        # Update duration statistics
        if metrics.min_duration_ms == float('inf'):
            metrics.min_duration_ms = duration_ms
        else:
            metrics.min_duration_ms = min(metrics.min_duration_ms, duration_ms)
        
        metrics.max_duration_ms = max(metrics.max_duration_ms, duration_ms)
        
        # Calculate average and p95
        all_durations = []
        if hasattr(self, '_get_all_durations'):
            all_durations = self._get_all_durations()
            all_durations.append(duration_ms)
        
        if all_durations:
            metrics.avg_duration_ms = sum(all_durations) / len(all_durations)
            metrics.p95_duration_ms = np.percentile(all_durations, 95)
        
        # Calculate error rate
        metrics.error_rate = metrics.failed_executions / metrics.total_executions if metrics.total_executions > 0 else 0.0
        
        metrics.last_execution = datetime.now()
        metrics.last_error = None if success else "Stage failed"
    
    def _get_all_durations(self) -> List[float]:
        """Get all duration measurements for metrics calculation."""
        # This would collect from log entries or separate storage
        # For now, return empty list
        return []
    
    def get_stage_metrics(self, stage: Optional[PipelineStage] = None) -> Dict[PipelineStage, PipelineMetrics]:
        """Get metrics for specific stage or all stages."""
        if stage:
            return {stage: self.stage_metrics.get(stage, PipelineMetrics(
                stage=stage, total_executions=0, successful_executions=0,
                failed_executions=0, avg_duration_ms=0.0, min_duration_ms=float('inf'),
                max_duration_ms=0.0, p95_duration_ms=0.0, error_rate=0.0,
                last_execution=None, last_error=None
            ))}
        return self.stage_metrics
    
    def get_pipeline_overview(self) -> Dict[str, Any]:
        """Get comprehensive pipeline overview."""
        total_executions = sum(m.total_executions for m in self.stage_metrics.values())
        total_successful = sum(m.successful_executions for m in self.stage_metrics.values())
        total_failed = sum(m.failed_executions for m in self.stage_metrics.values())
        
        return {
            'pipeline_name': self.pipeline_name,
            'total_stages': len(self.stage_metrics),
            'total_executions': total_executions,
            'successful_executions': total_successful,
            'failed_executions': total_failed,
            'overall_success_rate': total_successful / total_executions if total_executions > 0 else 0.0,
            'stage_metrics': {
                stage.value: asdict(metrics) for stage, metrics in self.stage_metrics.items()
            },
            'error_counts': self.error_counts,
            'success_counts': self.success_counts,
            'log_entries_count': len(self.log_entries),
            'last_updated': datetime.now()
        }
    
    def get_logs_dataframe(
        self,
        stage: Optional[PipelineStage] = None,
        level: Optional[LogLevel] = None,
        limit: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get logs as pandas DataFrame."""
        with self.data_lock:
            logs = self.log_entries
        
        # Apply filters
        if stage:
            logs = [log for log in logs if log.stage == stage]
        
        if level:
            logs = [log for log in logs if log.level == level]
        
        if start_time:
            logs = [log for log in logs if log.timestamp >= start_time]
        
        if end_time:
            logs = [log for log in logs if log.timestamp <= end_time]
        
        if limit:
            logs = logs[-limit:]
        
        if not logs:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for log in logs:
            row = {
                'timestamp': log.timestamp,
                'stage': log.stage.value,
                'level': log.level.value,
                'message': log.message,
                'execution_id': log.execution_id,
                'correlation_id': log.correlation_id,
                'duration_ms': log.duration_ms
            }
            
            # Add data fields
            for key, value in log.data.items():
                row[f'data_{key}'] = value
            
            # Add metadata fields
            for key, value in log.metadata.items():
                row[f'meta_{key}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def export_logs(
        self,
        format: str = "csv",
        stage: Optional[PipelineStage] = None,
        level: Optional[LogLevel] = None,
        include_metrics: bool = True,
        save_path: Optional[str] = None
    ) -> str:
        """Export logs to various formats."""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"pipeline_logs_{self.pipeline_name}_{timestamp}.{format}"
        
        # Get logs
        df = self.get_logs_dataframe(stage=stage, level=level)
        
        if format.lower() == "csv":
            df.to_csv(save_path)
            
            if include_metrics:
                metrics_df = self._get_metrics_dataframe()
                metrics_df.to_csv(save_path.replace('.csv', '_metrics.csv'))
        
        elif format.lower() == "json":
            export_data = {
                'pipeline_name': self.pipeline_name,
                'export_timestamp': datetime.now().isoformat(),
                'logs': df.to_dict('records'),
                'stage_metrics': {
                    stage.value: asdict(metrics) for stage, metrics in self.stage_metrics.items()
                },
                'pipeline_overview': self.get_pipeline_overview()
            }
            
            with open(save_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format.lower() == "excel":
            with pd.ExcelWriter(save_path) as writer:
                df.to_excel(writer, sheet_name='Logs', index=False)
                
                if include_metrics:
                    metrics_df = self._get_metrics_dataframe()
                    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        
        logger.info(f"Logs exported to {save_path}")
        return save_path
    
    def _get_metrics_dataframe(self) -> pd.DataFrame:
        """Get metrics as DataFrame."""
        metrics_data = []
        
        for stage, metrics in self.stage_metrics.items():
            row = {
                'stage': stage.value,
                'total_executions': metrics.total_executions,
                'successful_executions': metrics.successful_executions,
                'failed_executions': metrics.failed_executions,
                'avg_duration_ms': metrics.avg_duration_ms,
                'min_duration_ms': metrics.min_duration_ms,
                'max_duration_ms': metrics.max_duration_ms,
                'p95_duration_ms': metrics.p95_duration_ms,
                'error_rate': metrics.error_rate,
                'last_execution': metrics.last_execution,
                'last_error': metrics.last_error
            }
            metrics_data.append(row)
        
        return pd.DataFrame(metrics_data)
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        logger.info(f"Started monitoring for pipeline: {self.pipeline_name}")
        
        # Start background tasks
        self.aggregation_task = asyncio.create_task(self._aggregation_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        logger.info(f"Stopped monitoring for pipeline: {self.pipeline_name}")
        
        # Cancel background tasks
        if self.aggregation_task:
            self.aggregation_task.cancel()
            try:
                asyncio.run(self.aggregation_task)
            except asyncio.CancelledError:
                pass
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                asyncio.run(self.cleanup_task)
            except asyncio.CancelledError:
                pass
    
    async def _aggregation_loop(self) -> None:
        """Background loop for log aggregation."""
        while self.monitoring_active:
            try:
                # Perform aggregation tasks
                await self._aggregate_logs()
                await self._update_performance_stats()
                
                # Sleep for configurable interval
                await asyncio.sleep(60)  # 1 minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _aggregate_logs(self) -> None:
        """Aggregate logs for performance analysis."""
        # This would implement various aggregation tasks
        # - Error pattern detection
        # - Performance trend analysis
        # - Bottleneck identification
        logger.debug("Performing log aggregation")
    
    async def _update_performance_stats(self) -> None:
        """Update performance statistics."""
        # This would update performance statistics
        # - Calculate moving averages
        # - Detect performance degradation
        # - Generate alerts
        logger.debug("Updating performance statistics")
    
    async def _cleanup_loop(self) -> None:
        """Background loop for cleanup tasks."""
        while self.monitoring_active:
            try:
                # Perform cleanup
                await self._cleanup_old_logs()
                await self._cleanup_old_metrics()
                
                # Sleep for longer interval
                await asyncio.sleep(3600)  # 1 hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _cleanup_old_logs(self) -> None:
        """Clean up old log entries."""
        with self.data_lock:
            if len(self.log_entries) > 5000:  # Configurable limit
                # Keep recent entries
                cutoff_time = datetime.now() - timedelta(days=self.config.retention_days)
                self.log_entries = [
                    log for log in self.log_entries 
                    if log.timestamp > cutoff_time
                ]
                
                # Clean up old files
                await self._cleanup_old_log_files()
        
        logger.debug("Cleaned up old logs")
    
    async def _cleanup_old_log_files(self) -> None:
        """Clean up old log files."""
        log_dir = Path(self.config.log_directory)
        if not log_dir.exists():
            return
        
        # Find old log files
        cutoff_time = datetime.now() - timedelta(days=self.config.retention_days)
        
        for log_file in log_dir.glob(f"{self.config.log_file_prefix}_{self.pipeline_name}*.log*"):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_time:
                    log_file.unlink()
                    logger.debug(f"Removed old log file: {log_file}")
            except Exception as e:
                logger.error(f"Error removing old log file {log_file}: {e}")
    
    async def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics data."""
        # This would implement cleanup of old metrics
        # - Remove old error counts
        # - Archive old performance data
        logger.debug("Cleaned up old metrics")


class PipelineCorrelationTracker:
    """
    Track correlations between pipeline stages and executions.
    
    Features:
    - Execution flow tracking
    - Stage dependency analysis
    - Performance correlation
    - Bottleneck identification
    """
    
    def __init__(self, pipeline_name: str) -> None:
        self.pipeline_name = pipeline_name
        self.execution_flows: Dict[str, List[str]] = {}
        self.stage_dependencies: Dict[str, List[str]] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"Correlation tracker initialized for: {pipeline_name}")
    
    def track_execution_flow(self, execution_id: str, stages: List[str]) -> None:
        """Track execution flow through stages."""
        self.execution_flows[execution_id] = stages
        
        # Update dependencies
        for i in range(len(stages) - 1):
            current_stage = stages[i]
            next_stage = stages[i + 1]
            
            if current_stage not in self.stage_dependencies:
                self.stage_dependencies[current_stage] = []
            
            if next_stage not in self.stage_dependencies[current_stage]:
                self.stage_dependencies[current_stage].append(next_stage)
        
        logger.debug(f"Tracked execution flow: {execution_id} -> {stages}")
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze pipeline bottlenecks."""
        bottleneck_analysis = {}
        
        for stage, dependencies in self.stage_dependencies.items():
            # Calculate bottleneck metrics
            avg_wait_time = self._calculate_stage_wait_time(stage)
            error_rate = self._calculate_stage_error_rate(stage)
            
            bottleneck_score = avg_wait_time * error_rate
            
            bottleneck_analysis[stage] = {
                'avg_wait_time': avg_wait_time,
                'error_rate': error_rate,
                'bottleneck_score': bottleneck_score,
                'dependencies': dependencies,
                'recommendation': self._generate_bottleneck_recommendation(bottleneck_score)
            }
        
        return bottleneck_analysis
    
    def _calculate_stage_wait_time(self, stage: str) -> float:
        """Calculate average wait time for a stage."""
        # This would analyze wait times between stages
        # For now, return placeholder
        return 0.0
    
    def _calculate_stage_error_rate(self, stage: str) -> float:
        """Calculate error rate for a stage."""
        # This would calculate actual error rates
        # For now, return placeholder
        return 0.0
    
    def _generate_bottleneck_recommendation(self, score: float) -> str:
        """Generate bottleneck recommendation based on score."""
        if score > 100:
            return "Critical bottleneck detected - immediate optimization required"
        elif score > 50:
            return "Significant bottleneck - optimization recommended"
        elif score > 20:
            return "Moderate bottleneck - consider optimization"
        else:
            return "No significant bottleneck"


# Convenience functions
def create_pipeline_logger(
    pipeline_name: str,
    config: Optional[PipelineLoggerConfig] = None
) -> PipelineLogger:
    """Create pipeline logger with default settings."""
    return PipelineLogger(pipeline_name, config)


def create_pipeline_config(
    log_level: LogLevel = LogLevel.INFO,
    enable_file_logging: bool = True,
    enable_database_logging: bool = True,
    retention_days: int = 30
) -> PipelineLoggerConfig:
    """Create pipeline logger configuration."""
    return PipelineLoggerConfig(
        log_level=log_level,
        enable_file_logging=enable_file_logging,
        enable_database_logging=enable_database_logging,
        retention_days=retention_days
    )


# Context manager for pipeline logging
class PipelineLoggingContext:
    """Context manager for automatic pipeline stage logging."""
    
    def __init__(
        self,
        logger: PipelineLogger,
        stage: PipelineStage,
        execution_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.logger = logger
        self.stage = stage
        self.execution_id = execution_id
        self.metadata = metadata or {}
        self.stage_id = None
    
    def __enter__(self) -> 'PipelineLoggingContext':
        self.stage_id = self.logger.start_stage(
            self.stage,
            self.execution_id,
            metadata=self.metadata
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.stage_id is not None:
            success = exc_type is None
            error_message = str(exc_val) if exc_val else None
            
            self.logger.end_stage(
                self.stage,
                self.stage_id,
                success=success,
                error_message=error_message,
                metadata=self.metadata
            )


# Decorator for automatic pipeline logging
def log_pipeline_stage(stage: PipelineStage):
    """Decorator for automatic pipeline stage logging."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract logger from first argument if available
            logger = None
            execution_id = None
            
            if args and hasattr(args[0], 'pipeline_logger'):
                logger = args[0].pipeline_logger
                execution_id = kwargs.get('execution_id')
            
            # Create context if logger available
            if logger:
                with PipelineLoggingContext(logger, stage, execution_id) as ctx:
                    # Add metadata from kwargs
                    for key, value in kwargs.items():
                        if key.startswith('meta_'):
                            ctx.set_metadata({key[5:]: value})
                    
                    # Call function
                    result = func(*args, **kwargs)
                    
                    # Set result in context
                    ctx.set_metadata({'result': str(result)})
                    
                    return result
            else:
                # Fallback to manual logging
                return func(*args, **kwargs)
        
        return wrapper
    return decorator
