#!/usr/bin/env python3
"""
Kill Switch and Fail-Safe Mechanisms
==================================

Production-ready safety mechanisms for trading execution:
- Emergency stop functionality
- Circuit breaker patterns
- Health monitoring
- Automatic position liquidation
- System state recovery
"""

from __future__ import annotations

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
import signal
import sys

from its_project.execution.base import Order, OrderStatus, Position
from its_project.decision.decision import Decision, Action

logger = logging.getLogger(__name__)


class KillSwitchReason(Enum):
    """Reasons for triggering kill switch."""
    MANUAL = "manual"
    MAX_DRAWDOWN = "max_drawdown"
    DAILY_LOSS = "daily_loss"
    SYSTEM_ERROR = "system_error"
    CONNECTIVITY_LOSS = "connectivity_loss"
    MARKET_VOLATILITY = "market_volatility"
    RISK_BREACH = "risk_breach"
    POSITION_LIMIT = "position_limit"
    LATENCY_SPIKE = "latency_spike"
    DATA_QUALITY = "data_quality"


class SystemState(Enum):
    """System operational states."""
    NORMAL = "normal"
    WARNING = "warning"
    DEGRADED = "degraded"
    EMERGENCY_STOP = "emergency_stop"
    SHUTDOWN = "shutdown"


@dataclass
class KillSwitchConfig:
    """Kill switch configuration."""
    # Financial limits
    max_drawdown_pct: float = 0.10          # 10% max drawdown
    max_daily_loss_pct: float = 0.05          # 5% max daily loss
    max_position_value: float = 100000.0       # $100k max position
    
    # System limits
    max_latency_ms: float = 1000.0             # 1s max latency
    max_error_rate: float = 0.05                # 5% max error rate
    max_order_rejections: int = 10              # Max rejections per minute
    
    # Market conditions
    max_volatility: float = 0.10                # 10% max volatility
    min_liquidity_ratio: float = 0.01           # Min liquidity ratio
    
    # Time limits
    position_timeout_hours: float = 24.0         # 24h max position duration
    emergency_cooldown_minutes: float = 30.0    # 30min cooldown after emergency
    
    # Circuit breaker
    circuit_breaker_threshold: int = 5          # 5 failures trigger circuit breaker
    circuit_breaker_timeout_seconds: float = 300.0  # 5min timeout
    
    # Auto-recovery
    auto_recovery_enabled: bool = True
    recovery_check_interval_seconds: float = 60.0
    max_recovery_attempts: int = 3


@dataclass
class SystemHealth:
    """System health status."""
    state: SystemState
    timestamp: int
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_latency: float = 0.0
    error_rate: float = 0.0
    active_positions: int = 0
    pending_orders: int = 0
    last_trade_time: Optional[int] = None
    issues: List[str] = field(default_factory=list)


@dataclass
class KillSwitchEvent:
    """Kill switch activation event."""
    reason: KillSwitchReason
    timestamp: int
    system_state: SystemState
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[int] = None


class KillSwitch:
    """
    Advanced kill switch and fail-safe system.
    
    Features:
    - Multiple trigger conditions
    - Circuit breaker patterns
    - Health monitoring
    - Auto-recovery mechanisms
    - Emergency position liquidation
    """
    
    def __init__(self, config: KillSwitchConfig) -> None:
        self.config = config
        
        # State management
        self.current_state = SystemState.NORMAL
        self.is_emergency_stopped = False
        self.kill_switch_active = False
        self.circuit_breaker_open = False
        
        # Health monitoring
        self.health = SystemHealth(
            state=SystemState.NORMAL,
            timestamp=int(time.time() * 1000)
        )
        
        # Event tracking
        self.events: List[KillSwitchEvent] = []
        self.failure_count = 0
        self.last_failure_time = 0
        
        # Callbacks
        self.emergency_callbacks: List[Callable] = []
        self.state_change_callbacks: List[Callable] = []
        self.position_liquidation_callbacks: List[Callable] = []
        
        # Background monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.recovery_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Statistics
        self.stats = {
            'total_activations': 0,
            'manual_activations': 0,
            'auto_activations': 0,
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'positions_liquidated': 0,
            'circuit_breaker_activations': 0
        }
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self.running:
            logger.warning("Kill switch monitoring already running")
            return
        
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Kill switch monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                asyncio.run(self.monitoring_task)
            except asyncio.CancelledError:
                pass
        
        if self.recovery_task:
            self.recovery_task.cancel()
            try:
                asyncio.run(self.recovery_task)
            except asyncio.CancelledError:
                pass
        
        logger.info("Kill switch monitoring stopped")
    
    def emergency_stop(
        self,
        reason: KillSwitchReason = KillSwitchReason.MANUAL,
        description: str = "Manual emergency stop",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Trigger emergency stop.
        
        Returns:
            True if stop was successful, False if already stopped
        """
        if self.is_emergency_stopped:
            logger.warning("Emergency stop already active")
            return False
        
        self.is_emergency_stopped = True
        self.kill_switch_active = True
        self.current_state = SystemState.EMERGENCY_STOP
        
        # Create event
        event = KillSwitchEvent(
            reason=reason,
            timestamp=int(time.time() * 1000),
            system_state=self.current_state,
            description=description,
            metadata=metadata or {}
        )
        self.events.append(event)
        
        # Update statistics
        self.stats['total_activations'] += 1
        if reason == KillSwitchReason.MANUAL:
            self.stats['manual_activations'] += 1
        else:
            self.stats['auto_activations'] += 1
        
        # Log emergency
        logger.critical(f"EMERGENCY STOP ACTIVATED: {description}")
        
        # Trigger callbacks
        self._trigger_emergency_callbacks(event)
        
        # Start recovery process if enabled
        if self.config.auto_recovery_enabled:
            self._start_recovery_process()
        
        return True
    
    def reset_emergency_stop(self) -> bool:
        """
        Reset emergency stop (manual intervention).
        
        Returns:
            True if reset was successful
        """
        if not self.is_emergency_stopped:
            logger.warning("No emergency stop to reset")
            return False
        
        # Check cooldown period
        last_event = self.events[-1] if self.events else None
        if last_event:
            cooldown_elapsed = (time.time() * 1000 - last_event.timestamp) / 1000
            if cooldown_elapsed < self.config.emergency_cooldown_minutes * 60:
                logger.warning(f"Cooldown period not elapsed: {cooldown_elapsed:.1f}s remaining")
                return False
        
        # Reset state
        self.is_emergency_stopped = False
        self.kill_switch_active = False
        self.current_state = SystemState.NORMAL
        self.circuit_breaker_open = False
        self.failure_count = 0
        
        # Update last event
        if last_event:
            last_event.resolved = True
            last_event.resolution_time = int(time.time() * 1000)
        
        logger.info("Emergency stop reset - system back to normal")
        self._trigger_state_change_callbacks(SystemState.NORMAL)
        
        return True
    
    def check_health_conditions(
        self,
        portfolio_value: float,
        positions: Dict[str, Position],
        orders: Dict[str, Order],
        market_data: Dict[str, Dict[str, Any]]
    ) -> List[KillSwitchReason]:
        """
        Check all health conditions that might trigger kill switch.
        
        Returns:
            List of trigger reasons
        """
        triggers = []
        
        # 1. Check drawdown
        if hasattr(self, '_peak_portfolio_value'):
            current_drawdown = (self._peak_portfolio_value - portfolio_value) / self._peak_portfolio_value
            if current_drawdown > self.config.max_drawdown_pct:
                triggers.append(KillSwitchReason.MAX_DRAWDOWN)
        
        # 2. Check daily loss
        daily_pnl = self._calculate_daily_pnl(positions, portfolio_value)
        if daily_pnl < -self.config.max_daily_loss_pct:
            triggers.append(KillSwitchReason.DAILY_LOSS)
        
        # 3. Check position limits
        for position in positions.values():
            position_value = abs(position.size * position.current_price)
            if position_value > self.config.max_position_value:
                triggers.append(KillSwitchReason.POSITION_LIMIT)
        
        # 4. Check market volatility
        for symbol, data in market_data.items():
            volatility = data.get('volatility', 0)
            if volatility > self.config.max_volatility:
                triggers.append(KillSwitchReason.MARKET_VOLATILITY)
        
        # 5. Check system latency
        if self.health.network_latency > self.config.max_latency_ms:
            triggers.append(KillSwitchReason.LATENCY_SPIKE)
        
        # 6. Check error rate
        if self.health.error_rate > self.config.max_error_rate:
            triggers.append(KillSwitchReason.SYSTEM_ERROR)
        
        # 7. Check data quality
        data_quality_issues = self._check_data_quality(market_data)
        if data_quality_issues:
            triggers.append(KillSwitchReason.DATA_QUALITY)
        
        return triggers
    
    def liquidate_all_positions(
        self,
        positions: Dict[str, Position],
        reason: str = "Emergency liquidation"
    ) -> List[Dict[str, Any]]:
        """
        Generate liquidation orders for all positions.
        
        Returns:
            List of liquidation orders
        """
        liquidation_orders = []
        
        for symbol, position in positions.items():
            # Determine liquidation price
            if position.side == 'long':
                liquidation_price = position.current_price * 0.995  # 0.5% discount
            else:  # short
                liquidation_price = position.current_price * 1.005  # 0.5% premium
            
            liquidation_order = {
                'action': 'CLOSE',
                'symbol': symbol,
                'size': position.size,
                'price': liquidation_price,
                'order_type': 'MARKET',
                'reason': reason,
                'original_position': position,
                'emergency': True
            }
            
            liquidation_orders.append(liquidation_order)
        
        # Update statistics
        self.stats['positions_liquidated'] += len(liquidation_orders)
        
        # Trigger callbacks
        for callback in self.position_liquidation_callbacks:
            try:
                callback(liquidation_orders, reason)
            except Exception as e:
                logger.error(f"Error in liquidation callback: {e}")
        
        logger.critical(f"Emergency liquidation generated: {len(liquidation_orders)} positions")
        
        return liquidation_orders
    
    def add_emergency_callback(self, callback: Callable[[KillSwitchEvent], None]) -> None:
        """Add callback for emergency events."""
        self.emergency_callbacks.append(callback)
    
    def add_state_change_callback(self, callback: Callable[[SystemState], None]) -> None:
        """Add callback for state changes."""
        self.state_change_callbacks.append(callback)
    
    def add_position_liquidation_callback(
        self,
        callback: Callable[[List[Dict[str, Any]], str], None]
    ) -> None:
        """Add callback for position liquidation."""
        self.position_liquidation_callbacks.append(callback)
    
    async def _monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        logger.info("Health monitoring loop started")
        
        while self.running:
            try:
                # Update health metrics
                await self._update_health_metrics()
                
                # Check if we should trigger emergency stop
                if not self.is_emergency_stopped:
                    await self._check_emergency_conditions()
                
                # Check circuit breaker
                await self._check_circuit_breaker()
                
                # Sleep until next check
                await asyncio.sleep(self.config.recovery_check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error
        
        logger.info("Health monitoring loop stopped")
    
    async def _update_health_metrics(self) -> None:
        """Update system health metrics."""
        try:
            # Get system metrics (simplified)
            import psutil
            
            self.health.cpu_usage = psutil.cpu_percent()
            self.health.memory_usage = psutil.virtual_memory().percent
            self.health.disk_usage = psutil.disk_usage('/').percent
            
            # Update timestamp
            self.health.timestamp = int(time.time() * 1000)
            
            # Determine system state
            if self.health.cpu_usage > 90 or self.health.memory_usage > 90:
                self.health.state = SystemState.DEGRADED
            elif self.health.cpu_usage > 70 or self.health.memory_usage > 70:
                self.health.state = SystemState.WARNING
            else:
                self.health.state = SystemState.NORMAL
            
        except ImportError:
            # psutil not available, use dummy values
            self.health.state = SystemState.NORMAL
        except Exception as e:
            logger.error(f"Error updating health metrics: {e}")
    
    async def _check_emergency_conditions(self) -> None:
        """Check for emergency conditions."""
        # This would need current portfolio and market data
        # For now, just check basic system health
        
        if self.health.state == SystemState.DEGRADED:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.config.circuit_breaker_threshold:
                await self._trigger_circuit_breaker("System degradation")
    
    async def _check_circuit_breaker(self) -> None:
        """Check circuit breaker conditions."""
        if self.circuit_breaker_open:
            # Check if we can close circuit breaker
            time_since_failure = time.time() - self.last_failure_time
            if time_since_failure >= self.config.circuit_breaker_timeout_seconds:
                self.circuit_breaker_open = False
                self.failure_count = 0
                logger.info("Circuit breaker closed - system recovered")
                self._trigger_state_change_callbacks(SystemState.NORMAL)
    
    async def _trigger_circuit_breaker(self, reason: str) -> None:
        """Trigger circuit breaker."""
        if self.circuit_breaker_open:
            return
        
        self.circuit_breaker_open = True
        self.stats['circuit_breaker_activations'] += 1
        
        logger.warning(f"Circuit breaker opened: {reason}")
        self._trigger_state_change_callbacks(SystemState.DEGRADED)
        
        # Consider emergency stop if conditions are severe
        if self.failure_count >= self.config.circuit_breaker_threshold * 2:
            self.emergency_stop(
                reason=KillSwitchReason.SYSTEM_ERROR,
                description=f"Circuit breaker overload: {reason}"
            )
    
    def _start_recovery_process(self) -> None:
        """Start automatic recovery process."""
        if self.recovery_task and not self.recovery_task.done():
            return
        
        self.recovery_task = asyncio.create_task(self._recovery_loop())
    
    async def _recovery_loop(self) -> None:
        """Automatic recovery loop."""
        logger.info("Auto-recovery process started")
        
        recovery_attempts = 0
        
        while self.is_emergency_stopped and recovery_attempts < self.config.max_recovery_attempts:
            recovery_attempts += 1
            self.stats['recovery_attempts'] += 1
            
            # Wait for recovery interval
            await asyncio.sleep(self.config.recovery_check_interval_seconds)
            
            # Check if conditions have improved
            if await self._check_recovery_conditions():
                logger.info("Recovery conditions met - attempting reset")
                if self.reset_emergency_stop():
                    self.stats['successful_recoveries'] += 1
                    logger.info("Auto-recovery successful")
                    break
            else:
                logger.info(f"Recovery attempt {recovery_attempts} failed - conditions not met")
        
        if recovery_attempts >= self.config.max_recovery_attempts:
            logger.error("Max recovery attempts reached - manual intervention required")
        
        logger.info("Auto-recovery process ended")
    
    async def _check_recovery_conditions(self) -> bool:
        """Check if conditions are suitable for recovery."""
        # Check system health
        if self.health.state != SystemState.NORMAL:
            return False
        
        # Check cooldown period
        if self.events:
            last_event = self.events[-1]
            cooldown_elapsed = (time.time() * 1000 - last_event.timestamp) / 1000
            if cooldown_elapsed < self.config.emergency_cooldown_minutes * 60:
                return False
        
        # Additional checks would go here (market conditions, etc.)
        
        return True
    
    def _calculate_daily_pnl(self, positions: Dict[str, Position], portfolio_value: float) -> float:
        """Calculate daily PnL percentage."""
        # Simplified calculation - would need proper tracking
        total_unrealized = sum(pos.unrealized_pnl for pos in positions.values())
        return total_unrealized / portfolio_value if portfolio_value > 0 else 0
    
    def _check_data_quality(self, market_data: Dict[str, Dict[str, Any]]) -> List[str]:
        """Check data quality issues."""
        issues = []
        
        for symbol, data in market_data.items():
            # Check for stale data
            timestamp = data.get('timestamp', 0)
            age_seconds = (time.time() * 1000 - timestamp) / 1000
            if age_seconds > 60:  # 1 minute old
                issues.append(f"Stale data for {symbol}: {age_seconds:.1f}s old")
            
            # Check for missing fields
            required_fields = ['price', 'volume', 'bid', 'ask']
            for field in required_fields:
                if field not in data or data[field] is None:
                    issues.append(f"Missing {field} for {symbol}")
            
            # Check for price anomalies
            price = data.get('price', 0)
            if price <= 0:
                issues.append(f"Invalid price for {symbol}: {price}")
        
        return issues
    
    def _trigger_emergency_callbacks(self, event: KillSwitchEvent) -> None:
        """Trigger emergency event callbacks."""
        for callback in self.emergency_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in emergency callback: {e}")
    
    def _trigger_state_change_callbacks(self, new_state: SystemState) -> None:
        """Trigger state change callbacks."""
        for callback in self.state_change_callbacks:
            try:
                callback(new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum} - triggering emergency stop")
            self.emergency_stop(
                reason=KillSwitchReason.MANUAL,
                description=f"Signal {signum} received"
            )
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def update_portfolio_peak(self, portfolio_value: float) -> None:
        """Update peak portfolio value for drawdown calculation."""
        if not hasattr(self, '_peak_portfolio_value'):
            self._peak_portfolio_value = portfolio_value
        elif portfolio_value > self._peak_portfolio_value:
            self._peak_portfolio_value = portfolio_value
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive kill switch status."""
        return {
            'current_state': self.current_state.value,
            'emergency_stopped': self.is_emergency_stopped,
            'kill_switch_active': self.kill_switch_active,
            'circuit_breaker_open': self.circuit_breaker_open,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time,
            'health': {
                'state': self.health.state.value,
                'cpu_usage': self.health.cpu_usage,
                'memory_usage': self.health.memory_usage,
                'network_latency': self.health.network_latency,
                'error_rate': self.health.error_rate,
                'active_positions': self.health.active_positions,
                'pending_orders': self.health.pending_orders
            },
            'statistics': self.stats,
            'recent_events': [
                {
                    'reason': event.reason.value,
                    'timestamp': event.timestamp,
                    'description': event.description,
                    'resolved': event.resolved
                }
                for event in self.events[-10:]  # Last 10 events
            ],
            'configuration': {
                'max_drawdown_pct': self.config.max_drawdown_pct,
                'max_daily_loss_pct': self.config.max_daily_loss_pct,
                'auto_recovery_enabled': self.config.auto_recovery_enabled,
                'circuit_breaker_threshold': self.config.circuit_breaker_threshold
            }
        }


# Convenience functions
def create_conservative_kill_switch() -> KillSwitch:
    """Create conservative kill switch (lower thresholds)."""
    config = KillSwitchConfig(
        max_drawdown_pct=0.05,      # 5%
        max_daily_loss_pct=0.02,      # 2%
        max_position_value=50000.0,    # $50k
        max_latency_ms=500.0,          # 500ms
        circuit_breaker_threshold=3,     # 3 failures
        auto_recovery_enabled=False       # Manual recovery only
    )
    return KillSwitch(config)


def create_aggressive_kill_switch() -> KillSwitch:
    """Create aggressive kill switch (higher thresholds)."""
    config = KillSwitchConfig(
        max_drawdown_pct=0.20,      # 20%
        max_daily_loss_pct=0.10,      # 10%
        max_position_value=500000.0,   # $500k
        max_latency_ms=2000.0,         # 2s
        circuit_breaker_threshold=10,    # 10 failures
        auto_recovery_enabled=True
    )
    return KillSwitch(config)


if __name__ == "__main__":
    # Test kill switch
    logging.basicConfig(level=logging.INFO)
    
    config = KillSwitchConfig(
        max_drawdown_pct=0.10,
        max_daily_loss_pct=0.05,
        auto_recovery_enabled=True
    )
    
    kill_switch = KillSwitch(config)
    
    # Test emergency stop
    success = kill_switch.emergency_stop(
        reason=KillSwitchReason.MANUAL,
        description="Test emergency stop"
    )
    
    print(f"Emergency stop triggered: {success}")
    print(f"Current state: {kill_switch.current_state.value}")
    print(f"Status: {kill_switch.get_status()}")
    
    # Test reset after cooldown
    import time
    time.sleep(2)  # Wait a bit
    
    reset_success = kill_switch.reset_emergency_stop()
    print(f"Emergency stop reset: {reset_success}")
