#!/usr/bin/env python3
"""
Structured Logging System
=========================

Production-ready structured logging and error tracking:
- Structured JSON logging
- Error aggregation and tracking
- Log rotation and archiving
- Performance monitoring
- Alert integration
"""

from __future__ import annotations

import os
import json
import logging
import time
import traceback
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import sqlite3
from collections import defaultdict, deque
import gzip
import hashlib

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels with numeric values."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    FATAL = 50


class ErrorCategory(Enum):
    """Error categories for tracking."""
    SYSTEM = "system"
    NETWORK = "network"
    DATA = "data"
    MODEL = "model"
    EXECUTION = "execution"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    MEMORY = "memory"
    PERMISSION = "permission"
    UNKNOWN = "unknown"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: int
    level: LogLevel
    message: str
    logger_name: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    
    # Optional fields
    error_category: Optional[ErrorCategory] = None
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Context data
    context: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # System info
    hostname: Optional[str] = None
    environment: Optional[str] = None
    version: Optional[str] = None


@dataclass
class ErrorStats:
    """Error statistics."""
    error_type: str
    count: int
    first_seen: int
    last_seen: int
    affected_users: int
    affected_sessions: int
    stack_trace_hash: str
    sample_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogConfig:
    """Logging configuration."""
    log_level: LogLevel = LogLevel.INFO
    log_directory: str = "logs"
    max_file_size_mb: int = 100
    max_files: int = 10
    compress_old_logs: bool = True
    enable_error_tracking: bool = True
    enable_metrics: bool = True
    enable_performance_tracking: bool = True
    
    # Output formats
    console_format: str = "json"  # "json" or "text"
    file_format: str = "json"
    
    # Filtering
    exclude_modules: List[str] = field(default_factory=list)
    include_only_modules: List[str] = field(default_factory=list)
    
    # Performance
    buffer_size: int = 1000
    flush_interval_seconds: int = 5
    async_logging: bool = True


class StructuredLogger:
    """
    Production-ready structured logging system.
    
    Features:
    - JSON structured logging
    - Error tracking and aggregation
    - Log rotation and compression
    - Performance monitoring
    - Context propagation
    """
    
    def __init__(self, name: str, config: LogConfig) -> None:
        self.name = name
        self.config = config
        
        # Directory setup
        self.log_dir = Path(config.log_directory)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Database for error tracking
        self.db_path = self.log_dir / "logging.db"
        self._init_database()
        
        # Log files
        self.current_log_file = self.log_dir / f"{name}.log"
        self.error_log_file = self.log_dir / f"{name}_errors.log"
        
        # Buffer for async logging
        self.log_buffer = deque(maxlen=config.buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Background flush thread
        self.flush_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Error tracking
        self.error_stats: Dict[str, ErrorStats] = {}
        self.error_patterns: Dict[str, List[str]] = defaultdict(list)
        
        # Context management
        self.context: Dict[str, Any] = {}
        self.context_lock = threading.Lock()
        
        # Performance tracking
        self.log_counts: Dict[LogLevel, int] = defaultdict(int)
        self.performance_metrics: Dict[str, float] = {}
        
        # Callbacks
        self.error_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # System info
        self.hostname = os.environ.get('HOSTNAME', 'unknown')
        self.environment = os.environ.get('ENVIRONMENT', 'development')
        self.version = os.environ.get('APP_VERSION', '1.0.0')
    
    def _init_database(self) -> None:
        """Initialize SQLite database for error tracking."""
        with sqlite3.connect(self.db_path) as conn:
            # Error tracking table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS error_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT,
                    error_code TEXT,
                    stack_trace_hash TEXT,
                    count INTEGER,
                    first_seen INTEGER,
                    last_seen INTEGER,
                    affected_users INTEGER,
                    affected_sessions INTEGER,
                    sample_context TEXT,
                    created_at INTEGER DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            # Log entries table (for search)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS log_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    level INTEGER,
                    message TEXT,
                    logger_name TEXT,
                    module TEXT,
                    error_category TEXT,
                    error_code TEXT,
                    context TEXT,
                    tags TEXT,
                    created_at INTEGER DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_error_type ON error_tracking(error_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_log_timestamp ON log_entries(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_log_level ON log_entries(level)")
    
    def trace(self, message: str, **kwargs) -> None:
        """Log trace message."""
        self._log(LogLevel.TRACE, message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warn(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(LogLevel.WARN, message, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """Log error message."""
        if error:
            kwargs['stack_trace'] = traceback.format_exc()
            kwargs['error_category'] = self._categorize_error(error)
        
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def fatal(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """Log fatal message."""
        if error:
            kwargs['stack_trace'] = traceback.format_exc()
            kwargs['error_category'] = self._categorize_error(error)
        
        self._log(LogLevel.FATAL, message, **kwargs)
    
    def _log(self, level: LogLevel, message: str, **kwargs) -> None:
        """Internal logging method."""
        # Check log level
        if level.value < self.config.log_level.value:
            return
        
        # Get caller info
        frame = traceback.extract_stack()[-3]  # Skip internal frames
        module = frame.filename.split('/')[-1] if frame.filename else 'unknown'
        function = frame.name
        line_number = frame.lineno
        
        # Create log entry
        entry = LogEntry(
            timestamp=int(time.time() * 1000),
            level=level,
            message=message,
            logger_name=self.name,
            module=module,
            function=function,
            line_number=line_number,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            hostname=self.hostname,
            environment=self.environment,
            version=self.version,
            **kwargs
        )
        
        # Add global context
        with self.context_lock:
            if self.context:
                entry.context.update(self.context)
        
        # Track errors
        if level in [LogLevel.ERROR, LogLevel.FATAL]:
            self._track_error(entry)
        
        # Update metrics
        self.log_counts[level] += 1
        
        # Buffer for async writing
        if self.config.async_logging:
            with self.buffer_lock:
                self.log_buffer.append(entry)
        else:
            self._write_log_entry(entry)
        
        # Console output
        if self.config.console_format == "json":
            print(json.dumps(asdict(entry), default=str))
        else:
            print(f"[{level.name}] {message}")
    
    def _write_log_entry(self, entry: LogEntry) -> None:
        """Write log entry to file."""
        try:
            # Check file rotation
            if self.current_log_file.exists():
                size_mb = self.current_log_file.stat().st_size / (1024 * 1024)
                if size_mb > self.config.max_file_size_mb:
                    self._rotate_log_file()
            
            # Write to file
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                if self.config.file_format == "json":
                    f.write(json.dumps(asdict(entry), default=str) + '\n')
                else:
                    f.write(f"[{entry.level.name}] {entry.message}\n")
            
            # Separate error file
            if entry.level in [LogLevel.ERROR, LogLevel.FATAL]:
                with open(self.error_log_file, 'a', encoding='utf-8') as f:
                    if self.config.file_format == "json":
                        f.write(json.dumps(asdict(entry), default=str) + '\n')
                    else:
                        f.write(f"[{entry.level.name}] {entry.message}\n")
            
            # Save to database for search
            self._save_to_database(entry)
            
        except Exception as e:
            # Fallback to stderr
            print(f"LOGGING ERROR: {e}", file=sys.stderr)
    
    def _save_to_database(self, entry: LogEntry) -> None:
        """Save log entry to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO log_entries VALUES (
                        NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    entry.timestamp, entry.level.value, entry.message,
                    entry.logger_name, entry.module,
                    entry.error_category.value if entry.error_category else None,
                    entry.error_code,
                    json.dumps(entry.context),
                    json.dumps(entry.tags),
                    int(time.time())
                ))
        except Exception as e:
            # Don't log errors to avoid infinite recursion
            pass
    
    def _track_error(self, entry: LogEntry) -> None:
        """Track error statistics."""
        if not self.config.enable_error_tracking:
            return
        
        # Generate error key
        error_key = entry.error_code or entry.message[:100]
        stack_trace_hash = self._hash_stack_trace(entry.stack_trace) if entry.stack_trace else ""
        
        # Update or create error stats
        if error_key in self.error_stats:
            stats = self.error_stats[error_key]
            stats.count += 1
            stats.last_seen = entry.timestamp
            stats.affected_users += 1 if entry.user_id and entry.user_id not in entry.context.get('seen_users', []) else 0
            stats.affected_sessions += 1 if entry.session_id and entry.session_id not in entry.context.get('seen_sessions', []) else 0
        else:
            stats = ErrorStats(
                error_type=error_key,
                count=1,
                first_seen=entry.timestamp,
                last_seen=entry.timestamp,
                affected_users=1 if entry.user_id else 0,
                affected_sessions=1 if entry.session_id else 0,
                stack_trace_hash=stack_trace_hash,
                sample_context=entry.context.copy()
            )
            self.error_stats[error_key] = stats
        
        # Save to database
        self._save_error_stats(stats)
        
        # Trigger callbacks
        for callback in self.error_callbacks:
            try:
                callback(entry, stats)
            except Exception:
                pass
        
        # Check for alert conditions
        self._check_error_alerts(error_key, stats)
    
    def _save_error_stats(self, stats: ErrorStats) -> None:
        """Save error statistics to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO error_tracking VALUES (
                        NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    stats.error_type, stats.error_code, stats.stack_trace_hash,
                    stats.count, stats.first_seen, stats.last_seen,
                    stats.affected_users, stats.affected_sessions,
                    json.dumps(stats.sample_context), int(time.time())
                ))
        except Exception:
            pass
    
    def _check_error_alerts(self, error_key: str, stats: ErrorStats) -> None:
        """Check for error alert conditions."""
        # Alert on high error count
        if stats.count > 10:
            alert_data = {
                'type': 'high_error_count',
                'error_type': error_key,
                'count': stats.count,
                'severity': 'high' if stats.count > 50 else 'medium'
            }
            
            for callback in self.alert_callbacks:
                try:
                    callback(alert_data)
                except Exception:
                    pass
        
        # Alert on new error pattern
        if stats.count == 1:
            alert_data = {
                'type': 'new_error_pattern',
                'error_type': error_key,
                'first_seen': stats.first_seen,
                'severity': 'medium'
            }
            
            for callback in self.alert_callbacks:
                try:
                    callback(alert_data)
                except Exception:
                    pass
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error type."""
        error_type = type(error).__name__.lower()
        
        if any(keyword in error_type for keyword in ['connection', 'network', 'timeout', 'socket']):
            return ErrorCategory.NETWORK
        elif any(keyword in error_type for keyword in ['memory', 'memoryerror', 'outofmemory']):
            return ErrorCategory.MEMORY
        elif any(keyword in error_type for keyword in ['permission', 'access', 'auth']):
            return ErrorCategory.PERMISSION
        elif any(keyword in error_type for keyword in ['value', 'type', 'attribute']):
            return ErrorCategory.VALIDATION
        elif any(keyword in error_type for keyword in ['timeout']):
            return ErrorCategory.TIMEOUT
        elif 'data' in error_type:
            return ErrorCategory.DATA
        elif 'model' in error_type:
            return ErrorCategory.MODEL
        elif 'execution' in error_type:
            return ErrorCategory.EXECUTION
        else:
            return ErrorCategory.UNKNOWN
    
    def _hash_stack_trace(self, stack_trace: str) -> str:
        """Generate hash of stack trace for grouping."""
        if not stack_trace:
            return ""
        
        # Normalize stack trace (remove line numbers, memory addresses)
        normalized = stack_trace
        lines = normalized.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove memory addresses and line numbers
            line = line.replace('0x', '0x')  # Keep hex indicator but normalize
            line = line.split(',')[0] if ',' in line else line
            cleaned_lines.append(line.strip())
        
        normalized = '\n'.join(cleaned_lines)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _rotate_log_file(self) -> None:
        """Rotate log files."""
        if not self.current_log_file.exists():
            return
        
        # Move current file to timestamped name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_file = self.log_dir / f"{self.name}_{timestamp}.log"
        
        try:
            self.current_log_file.rename(rotated_file)
            
            # Compress old file if enabled
            if self.config.compress_old_logs:
                self._compress_log_file(rotated_file)
            
            # Clean up old files
            self._cleanup_old_files()
            
        except Exception as e:
            print(f"Error rotating log file: {e}")
    
    def _compress_log_file(self, file_path: Path) -> None:
        """Compress log file."""
        try:
            with open(file_path, 'rb') as f_in:
                with gzip.open(f"{file_path}.gz", 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove original file
            file_path.unlink()
            
        except Exception as e:
            print(f"Error compressing log file: {e}")
    
    def _cleanup_old_files(self) -> None:
        """Clean up old log files."""
        try:
            # Get all log files
            log_files = list(self.log_dir.glob(f"{self.name}_*.log*"))
            log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep only the most recent files
            if len(log_files) > self.config.max_files:
                for old_file in log_files[self.config.max_files:]:
                    old_file.unlink()
            
        except Exception as e:
            print(f"Error cleaning up old files: {e}")
    
    def _flush_buffer(self) -> None:
        """Flush log buffer to file."""
        if not self.log_buffer:
            return
        
        entries_to_write = []
        with self.buffer_lock:
            while self.log_buffer:
                entries_to_write.append(self.log_buffer.popleft())
        
        for entry in entries_to_write:
            self._write_log_entry(entry)
    
    def _background_flush(self) -> None:
        """Background thread for flushing logs."""
        while self.running:
            try:
                self._flush_buffer()
                time.sleep(self.config.flush_interval_seconds)
            except Exception as e:
                print(f"Error in background flush: {e}")
    
    def start_background_flushing(self) -> None:
        """Start background flushing thread."""
        if self.flush_thread and self.flush_thread.is_alive():
            return
        
        self.running = True
        self.flush_thread = threading.Thread(target=self._background_flush, daemon=True)
        self.flush_thread.start()
    
    def stop_background_flushing(self) -> None:
        """Stop background flushing thread."""
        self.running = False
        
        if self.flush_thread:
            self.flush_thread.join(timeout=5)
        
        # Flush remaining buffer
        self._flush_buffer()
    
    def set_context(self, **context) -> None:
        """Set global context for all log entries."""
        with self.context_lock:
            self.context.update(context)
    
    def clear_context(self) -> None:
        """Clear global context."""
        with self.context_lock:
            self.context.clear()
    
    def with_context(self, **context):
        """Context manager for temporary context."""
        class ContextManager:
            def __init__(self, logger, context):
                self.logger = logger
                self.context = context
                self.old_context = {}
            
            def __enter__(self):
                with self.logger.context_lock:
                    self.old_context = self.logger.context.copy()
                    self.logger.context.update(self.context)
                return self.logger
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                with self.logger.context_lock:
                    self.logger.context = self.old_context
        
        return ContextManager(self, context)
    
    def add_error_callback(self, callback: Callable) -> None:
        """Add error callback."""
        self.error_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert callback."""
        self.alert_callbacks.append(callback)
    
    def get_error_stats(self, limit: int = 100) -> List[ErrorStats]:
        """Get error statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM error_tracking 
                    ORDER BY count DESC, last_seen DESC 
                    LIMIT ?
                """, (limit,))
                
                stats = []
                for row in cursor.fetchall():
                    stats.append(ErrorStats(
                        error_type=row[1],
                        error_code=row[2],
                        stack_trace_hash=row[3],
                        count=row[4],
                        first_seen=row[5],
                        last_seen=row[6],
                        affected_users=row[7],
                        affected_sessions=row[8],
                        sample_context=json.loads(row[9]) if row[9] else {}
                    ))
                
                return stats
                
        except Exception as e:
            print(f"Error getting error stats: {e}")
            return []
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            'log_counts': dict(self.log_counts),
            'total_logs': sum(self.log_counts.values()),
            'error_count': self.log_counts.get(LogLevel.ERROR, 0) + self.log_counts.get(LogLevel.FATAL, 0),
            'buffer_size': len(self.log_buffer),
            'error_types_tracked': len(self.error_stats),
            'hostname': self.hostname,
            'environment': self.environment,
            'version': self.version
        }
    
    def search_logs(
        self,
        level: Optional[LogLevel] = None,
        module: Optional[str] = None,
        message_contains: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Search logs in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM log_entries WHERE 1=1"
                params = []
                
                if level:
                    query += " AND level = ?"
                    params.append(level.value)
                
                if module:
                    query += " AND module LIKE ?"
                    params.append(f"%{module}%")
                
                if message_contains:
                    query += " AND message LIKE ?"
                    params.append(f"%{message_contains}%")
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [
                    {
                        'id': row[0],
                        'timestamp': row[1],
                        'level': LogLevel(row[2]).name,
                        'message': row[3],
                        'logger_name': row[4],
                        'module': row[5],
                        'error_category': row[6],
                        'error_code': row[7],
                        'context': json.loads(row[8]) if row[8] else {},
                        'tags': json.loads(row[9]) if row[9] else []
                    }
                    for row in rows
                ]
                
        except Exception as e:
            print(f"Error searching logs: {e}")
            return []


# Convenience functions
def create_structured_logger(name: str, config: Optional[LogConfig] = None) -> StructuredLogger:
    """Create structured logger with default configuration."""
    if config is None:
        config = LogConfig()
    
    logger = StructuredLogger(name, config)
    logger.start_background_flushing()
    return logger


def create_production_logger(name: str) -> StructuredLogger:
    """Create production-ready logger."""
    config = LogConfig(
        log_level=LogLevel.INFO,
        log_directory="logs",
        max_file_size_mb=50,
        max_files=20,
        compress_old_logs=True,
        enable_error_tracking=True,
        enable_metrics=True,
        console_format="json",
        file_format="json",
        buffer_size=2000,
        flush_interval_seconds=3,
        async_logging=True
    )
    
    return create_structured_logger(name, config)


if __name__ == "__main__":
    # Test structured logging
    config = LogConfig(
        log_level=LogLevel.DEBUG,
        log_directory="test_logs",
        enable_error_tracking=True
    )
    
    logger = create_structured_logger("test_logger", config)
    
    # Test logging
    logger.info("Test message", user_id="user123", request_id="req456")
    logger.warn("Warning message", module="test_module")
    
    try:
        raise ValueError("Test error for tracking")
    except Exception as e:
        logger.error("Caught test error", error=e, session_id="sess789")
    
    # Test context
    with logger.with_context(operation="test_operation", component="test_component"):
        logger.info("Message with context")
    
    # Get statistics
    stats = logger.get_log_statistics()
    print(f"Log statistics: {stats}")
    
    # Get error stats
    error_stats = logger.get_error_stats()
    print(f"Error stats: {error_stats}")
    
    # Cleanup
    logger.stop_background_flushing()
    
    print("Structured logging system ready!")
