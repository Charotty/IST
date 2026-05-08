#!/usr/bin/env python3
"""
Advanced Reconnection Manager for Data Sources
==========================================

Comprehensive reconnection system with exponential backoff,
circuit breaker pattern, and health monitoring.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class ReconnectionConfig:
    """Reconnection configuration."""
    # Basic settings
    max_reconnect_attempts: int = 10
    initial_delay: float = 1.0  # seconds
    max_delay: float = 300.0  # 5 minutes
    backoff_multiplier: float = 2.0
    jitter: bool = True  # Add randomness to delay
    
    # Circuit breaker settings
    enable_circuit_breaker: bool = True
    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds to wait before trying again
    
    # Health check settings
    enable_health_checks: bool = True
    health_check_interval: float = 30.0  # seconds
    health_check_timeout: float = 10.0  # seconds
    
    # Connection timeout settings
    connection_timeout: float = 30.0  # seconds
    read_timeout: float = 60.0  # seconds
    
    # Monitoring settings
    enable_monitoring: bool = True
    connection_log_size: int = 1000
    alert_callbacks: List[Callable] = field(default_factory=list)


@dataclass
class ConnectionEvent:
    """Connection event record."""
    timestamp: datetime
    state: ConnectionState
    source: str
    error: Optional[str] = None
    attempt: int = 0
    duration_ms: Optional[int] = None


@dataclass
class ConnectionStats:
    """Connection statistics."""
    total_connections: int = 0
    total_disconnections: int = 0
    total_reconnections: int = 0
    total_failures: int = 0
    average_connection_time: float = 0.0
    longest_connection_time: float = 0.0
    current_uptime: float = 0.0
    last_connection_time: Optional[datetime] = None
    last_disconnection_time: Optional[datetime] = None


class CircuitBreaker:
    """Circuit breaker implementation for connection management."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
    
    def call_success(self) -> None:
        """Record successful call."""
        self.failure_count = 0
        self.state = "closed"
    
    def call_failure(self) -> None:
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout:
                self.state = "half_open"
                logger.info("Circuit breaker moved to half-open state")
                return True
            return False
        else:  # half_open
            return True


class ReconnectionManager:
    """
    Advanced reconnection manager for data sources.
    
    Features:
    - Exponential backoff with jitter
    - Circuit breaker pattern
    - Health monitoring
    - Connection statistics
    - Alert system
    - Automatic recovery
    """
    
    def __init__(self, config: ReconnectionConfig) -> None:
        self.config = config
        
        # Connection management
        self.connection_states: Dict[str, ConnectionState] = {}
        self.connection_callbacks: Dict[str, Callable] = {}
        self.reconnection_tasks: Dict[str, asyncio.Task] = {}
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Event tracking
        self.connection_events: List[ConnectionEvent] = []
        self.connection_stats: Dict[str, ConnectionStats] = {}
        
        # Health monitoring
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self.last_health_check: Dict[str, datetime] = {}
        
        # State
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        logger.info("Reconnection manager initialized")
    
    def register_source(
        self,
        source_name: str,
        connect_callback: Callable,
        disconnect_callback: Optional[Callable] = None,
        health_check_callback: Optional[Callable] = None
    ) -> None:
        """Register a data source for reconnection management."""
        self.connection_callbacks[source_name] = connect_callback
        self.connection_states[source_name] = ConnectionState.DISCONNECTED
        self.connection_stats[source_name] = ConnectionStats()
        
        # Initialize circuit breaker
        if self.config.enable_circuit_breaker:
            self.circuit_breakers[source_name] = CircuitBreaker(
                failure_threshold=self.config.failure_threshold,
                recovery_timeout=self.config.recovery_timeout
            )
        
        logger.info(f"Registered data source: {source_name}")
    
    async def start(self) -> None:
        """Start reconnection manager."""
        if self._running:
            logger.warning("Reconnection manager already running")
            return
        
        self._running = True
        
        # Start monitoring task
        if self.config.enable_monitoring:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start health checks
        if self.config.enable_health_checks:
            await self._start_health_checks()
        
        logger.info("Reconnection manager started")
    
    async def stop(self) -> None:
        """Stop reconnection manager."""
        self._running = False
        
        # Cancel monitoring task
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
        
        # Cancel reconnection tasks
        for task in self.reconnection_tasks.values():
            task.cancel()
        
        # Cancel health check tasks
        for task in self.health_check_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(
            *self.reconnection_tasks.values(),
            *self.health_check_tasks.values(),
            return_exceptions=True
        )
        
        logger.info("Reconnection manager stopped")
    
    async def connect_source(self, source_name: str) -> bool:
        """Connect a data source."""
        if source_name not in self.connection_callbacks:
            logger.error(f"Source {source_name} not registered")
            return False
        
        # Check circuit breaker
        if self.config.enable_circuit_breaker:
            circuit_breaker = self.circuit_breakers.get(source_name)
            if circuit_breaker and not circuit_breaker.can_execute():
                self._record_event(source_name, ConnectionState.CIRCUIT_OPEN)
                logger.warning(f"Circuit breaker open for {source_name}")
                return False
        
        # Check if already connected
        if self.connection_states[source_name] == ConnectionState.CONNECTED:
            return True
        
        # Set state to connecting
        self.connection_states[source_name] = ConnectionState.CONNECTING
        self._record_event(source_name, ConnectionState.CONNECTING)
        
        try:
            # Attempt connection
            start_time = datetime.now()
            
            connect_callback = self.connection_callbacks[source_name]
            if asyncio.iscoroutinefunction(connect_callback):
                await connect_callback()
            else:
                connect_callback()
            
            # Record success
            connection_time = (datetime.now() - start_time).total_seconds()
            self.connection_states[source_name] = ConnectionState.CONNECTED
            self._record_event(source_name, ConnectionState.CONNECTED, duration_ms=int(connection_time * 1000))
            self._update_stats(source_name, connection_time, success=True)
            
            # Update circuit breaker
            if self.config.enable_circuit_breaker:
                circuit_breaker = self.circuit_breakers.get(source_name)
                if circuit_breaker:
                    circuit_breaker.call_success()
            
            # Start health check
            if self.config.enable_health_checks and source_name not in self.health_check_tasks:
                await self._start_health_check(source_name)
            
            logger.info(f"Successfully connected {source_name}")
            return True
        
        except Exception as e:
            # Record failure
            self.connection_states[source_name] = ConnectionState.FAILED
            self._record_event(source_name, ConnectionState.FAILED, error=str(e))
            self._update_stats(source_name, 0, success=False)
            
            # Update circuit breaker
            if self.config.enable_circuit_breaker:
                circuit_breaker = self.circuit_breakers.get(source_name)
                if circuit_breaker:
                    circuit_breaker.call_failure()
            
            # Start reconnection
            await self._start_reconnection(source_name)
            
            logger.error(f"Failed to connect {source_name}: {e}")
            return False
    
    async def disconnect_source(self, source_name: str) -> bool:
        """Disconnect a data source."""
        if source_name not in self.connection_states:
            return False
        
        try:
            # Cancel reconnection task if running
            if source_name in self.reconnection_tasks:
                self.reconnection_tasks[source_name].cancel()
                del self.reconnection_tasks[source_name]
            
            # Cancel health check
            if source_name in self.health_check_tasks:
                self.health_check_tasks[source_name].cancel()
                del self.health_check_tasks[source_name]
            
            # Set state
            self.connection_states[source_name] = ConnectionState.DISCONNECTED
            self._record_event(source_name, ConnectionState.DISCONNECTED)
            
            logger.info(f"Disconnected {source_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error disconnecting {source_name}: {e}")
            return False
    
    async def _start_reconnection(self, source_name: str) -> None:
        """Start reconnection process for a source."""
        if source_name in self.reconnection_tasks:
            return  # Already reconnecting
        
        task = asyncio.create_task(self._reconnection_loop(source_name))
        self.reconnection_tasks[source_name] = task
        
        logger.info(f"Started reconnection for {source_name}")
    
    async def _reconnection_loop(self, source_name: str) -> None:
        """Reconnection loop with exponential backoff."""
        attempt = 0
        delay = self.config.initial_delay
        
        while self._running and self.connection_states[source_name] != ConnectionState.CONNECTED:
            attempt += 1
            
            # Check if we should stop
            if attempt > self.config.max_reconnect_attempts:
                logger.error(f"Max reconnection attempts reached for {source_name}")
                break
            
            # Wait before attempting
            await asyncio.sleep(delay)
            
            # Check circuit breaker
            if self.config.enable_circuit_breaker:
                circuit_breaker = self.circuit_breakers.get(source_name)
                if circuit_breaker and not circuit_breaker.can_execute():
                    await asyncio.sleep(self.config.recovery_timeout)
                    continue
            
            # Attempt reconnection
            self.connection_states[source_name] = ConnectionState.RECONNECTING
            self._record_event(source_name, ConnectionState.RECONNECTING, attempt=attempt)
            
            try:
                success = await self.connect_source(source_name)
                if success:
                    break
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt} failed for {source_name}: {e}")
            
            # Calculate next delay with exponential backoff and jitter
            delay = min(delay * self.config.backoff_multiplier, self.config.max_delay)
            if self.config.jitter:
                delay *= (0.5 + random.random() * 0.5)  # 50-100% of delay
        
        # Clean up
        if source_name in self.reconnection_tasks:
            del self.reconnection_tasks[source_name]
        
        logger.info(f"Reconnection loop ended for {source_name}")
    
    async def _start_health_checks(self) -> None:
        """Start health check tasks for all sources."""
        for source_name in self.connection_callbacks:
            await self._start_health_check(source_name)
    
    async def _start_health_check(self, source_name: str) -> None:
        """Start health check for a specific source."""
        if source_name in self.health_check_tasks:
            return
        
        task = asyncio.create_task(self._health_check_loop(source_name))
        self.health_check_tasks[source_name] = task
    
    async def _health_check_loop(self, source_name: str) -> None:
        """Health check loop for a source."""
        while self._running:
            try:
                # Wait for interval
                await asyncio.sleep(self.config.health_check_interval)
                
                # Check if source is connected
                if self.connection_states[source_name] != ConnectionState.CONNECTED:
                    continue
                
                # Perform health check
                await self._perform_health_check(source_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for {source_name}: {e}")
    
    async def _perform_health_check(self, source_name: str) -> None:
        """Perform health check for a source."""
        try:
            # Check if source has health check callback
            connect_callback = self.connection_callbacks.get(source_name)
            if not connect_callback:
                return
            
            # Simple health check - try to verify connection is alive
            # This would be implemented by the specific data source
            start_time = datetime.now()
            
            # For now, we'll just check if we can call the callback
            # In real implementation, this would be a ping/health check
            if hasattr(connect_callback, '__self__') and hasattr(connect_callback.__self__, 'is_alive'):
                is_alive = await connect_callback.__self__.is_alive()
                if not is_alive:
                    raise ConnectionError("Health check failed")
            
            self.last_health_check[source_name] = datetime.now()
            
        except Exception as e:
            logger.warning(f"Health check failed for {source_name}: {e}")
            
            # Trigger reconnection if health check fails
            if self.connection_states[source_name] == ConnectionState.CONNECTED:
                await self.disconnect_source(source_name)
                await self._start_reconnection(source_name)
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        logger.info("Connection monitoring started")
        
        while self._running:
            try:
                # Update uptime statistics
                await self._update_uptime_stats()
                
                # Clean old events
                await self._cleanup_old_events()
                
                # Check for stale connections
                await self._check_stale_connections()
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)
        
        logger.info("Connection monitoring stopped")
    
    async def _update_uptime_stats(self) -> None:
        """Update uptime statistics for all sources."""
        current_time = datetime.now()
        
        for source_name, stats in self.connection_stats.items():
            if self.connection_states[source_name] == ConnectionState.CONNECTED:
                if stats.last_connection_time:
                    uptime = (current_time - stats.last_connection_time).total_seconds()
                    stats.current_uptime = uptime
    
    async def _cleanup_old_events(self) -> None:
        """Clean up old connection events."""
        if len(self.connection_events) > self.config.connection_log_size:
            excess = len(self.connection_events) - self.config.connection_log_size
            self.connection_events = self.connection_events[excess:]
    
    async def _check_stale_connections(self) -> None:
        """Check for stale connections."""
        if not self.config.enable_health_checks:
            return
        
        current_time = datetime.now()
        
        for source_name, last_check in self.last_health_check.items():
            if (current_time - last_check).total_seconds() > self.config.health_check_interval * 2:
                logger.warning(f"Stale connection detected for {source_name}")
                
                # Trigger reconnection
                if self.connection_states[source_name] == ConnectionState.CONNECTED:
                    await self.disconnect_source(source_name)
                    await self._start_reconnection(source_name)
    
    def _record_event(
        self,
        source_name: str,
        state: ConnectionState,
        error: Optional[str] = None,
        attempt: int = 0,
        duration_ms: Optional[int] = None
    ) -> None:
        """Record a connection event."""
        event = ConnectionEvent(
            timestamp=datetime.now(),
            state=state,
            source=source_name,
            error=error,
            attempt=attempt,
            duration_ms=duration_ms
        )
        
        self.connection_events.append(event)
        
        # Trigger alerts
        for callback in self.config.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def _update_stats(self, source_name: str, connection_time: float, success: bool) -> None:
        """Update connection statistics."""
        stats = self.connection_stats[source_name]
        current_time = datetime.now()
        
        if success:
            stats.total_connections += 1
            stats.last_connection_time = current_time
            
            # Update connection time stats
            if stats.total_connections == 1:
                stats.average_connection_time = connection_time
            else:
                total_time = stats.average_connection_time * (stats.total_connections - 1)
                stats.average_connection_time = (total_time + connection_time) / stats.total_connections
            
            stats.longest_connection_time = max(stats.longest_connection_time, connection_time)
        else:
            stats.total_failures += 1
            stats.last_disconnection_time = current_time
    
    def get_connection_status(self, source_name: str) -> Dict[str, Any]:
        """Get connection status for a source."""
        state = self.connection_states.get(source_name, ConnectionState.DISCONNECTED)
        stats = self.connection_stats.get(source_name, ConnectionStats())
        circuit_breaker = self.circuit_breakers.get(source_name)
        
        return {
            'source': source_name,
            'state': state.value,
            'is_connected': state == ConnectionState.CONNECTED,
            'is_reconnecting': state == ConnectionState.RECONNECTING,
            'circuit_breaker_state': circuit_breaker.state if circuit_breaker else None,
            'stats': {
                'total_connections': stats.total_connections,
                'total_failures': stats.total_failures,
                'average_connection_time': stats.average_connection_time,
                'current_uptime': stats.current_uptime,
                'last_connection_time': stats.last_connection_time.isoformat() if stats.last_connection_time else None
            },
            'last_health_check': self.last_health_check.get(source_name).isoformat() if source_name in self.last_health_check else None
        }
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get connection status for all sources."""
        return {
            source_name: self.get_connection_status(source_name)
            for source_name in self.connection_callbacks.keys()
        }
    
    def get_recent_events(self, limit: int = 100) -> List[ConnectionEvent]:
        """Get recent connection events."""
        return self.connection_events[-limit:] if self.connection_events else []
    
    def reset_statistics(self, source_name: Optional[str] = None) -> None:
        """Reset connection statistics."""
        if source_name:
            if source_name in self.connection_stats:
                self.connection_stats[source_name] = ConnectionStats()
        else:
            for name in self.connection_stats:
                self.connection_stats[name] = ConnectionStats()


# Convenience functions
def create_reconnection_manager(
    max_reconnect_attempts: int = 10,
    initial_delay: float = 1.0,
    max_delay: float = 300.0,
    enable_circuit_breaker: bool = True,
    enable_health_checks: bool = True
) -> ReconnectionManager:
    """Create reconnection manager with default configuration."""
    config = ReconnectionConfig(
        max_reconnect_attempts=max_reconnect_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        enable_circuit_breaker=enable_circuit_breaker,
        enable_health_checks=enable_health_checks
    )
    
    return ReconnectionManager(config)


# Default alert callback
async def default_connection_alert(event: ConnectionEvent) -> None:
    """Default alert callback for connection events."""
    if event.state in [ConnectionState.FAILED, ConnectionState.CIRCUIT_OPEN]:
        logger.error(f"CONNECTION ALERT: {event.source} {event.state.value} "
                     f"Attempt: {event.attempt} Error: {event.error}")
    elif event.state == ConnectionState.CONNECTED:
        logger.info(f"CONNECTION RESTORED: {event.source}")
    
    # Here you could add:
    # - Send to monitoring system
    # - Send email/SMS alert
    # - Update dashboard
    # - Trigger automated responses
