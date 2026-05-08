#!/usr/bin/env python3
"""
Advanced Gap Detection for Data Pipeline
======================================

Comprehensive gap detection system for real-time and historical data.
Integrates with data sources and backfill systems.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

from its_project.common.types import MarketData, MarketDataType
from its_project.data_layer.integrity import DataIntegrityMonitor, GapInfo, ValidationReport

logger = logging.getLogger(__name__)


@dataclass
class GapDetectionConfig:
    """Gap detection configuration."""
    # Time thresholds (in seconds)
    ticker_gap_threshold: float = 5.0      # 5 seconds
    trade_gap_threshold: float = 1.0        # 1 second
    orderbook_gap_threshold: float = 2.0    # 2 seconds
    ohlcv_gap_threshold: float = 60.0       # 1 minute
    
    # Data validation thresholds
    max_price_jump_pct: float = 10.0         # 10% price jump
    max_volume_spike_pct: float = 1000.0     # 1000% volume spike
    min_data_points: int = 1                 # Minimum data points
    
    # Detection settings
    enable_real_time_detection: bool = True
    enable_historical_validation: bool = True
    detection_window_size: int = 1000         # Rolling window size
    alert_threshold: int = 5                  # Alert after N gaps
    
    # Integration settings
    auto_backfill: bool = True
    backfill_priority: int = 2
    alert_callback: Optional[Callable] = None


@dataclass
class GapStatistics:
    """Gap detection statistics."""
    total_gaps_detected: int = 0
    gaps_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    gaps_by_symbol: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    gaps_by_severity: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    average_gap_duration: float = 0.0
    longest_gap_duration: float = 0.0
    last_gap_time: Optional[datetime] = None
    detection_rate: float = 0.0  # Gaps per hour


class GapDetector:
    """
    Advanced gap detection system for data pipeline.
    
    Features:
    - Real-time gap detection for all data types
    - Historical data validation
    - Statistical analysis of gaps
    - Automatic backfill integration
    - Alerting and monitoring
    - Configurable thresholds per data type
    """
    
    def __init__(self, config: GapDetectionConfig) -> None:
        self.config = config
        
        # Data tracking
        self.last_timestamps: Dict[str, Dict[str, int]] = defaultdict(dict)  # symbol -> data_type -> timestamp
        self.data_buffers: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.detection_window_size))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.detection_window_size))
        
        # Gap tracking
        self.detected_gaps: List[GapInfo] = []
        self.gap_statistics = GapStatistics()
        
        # Integrity monitor
        self.integrity_monitor = DataIntegrityMonitor()
        
        # Alert system
        self.alert_callbacks: List[Callable] = []
        if config.alert_callback:
            self.alert_callbacks.append(config.alert_callback)
        
        # State
        self._running = False
        self._detection_task: Optional[asyncio.Task] = None
        
        # Backfill integration
        self.backfill_manager: Optional[Any] = None
        
        logger.info("Gap detector initialized")
    
    def add_alert_callback(self, callback: Callable[[GapInfo], None]) -> None:
        """Add alert callback for gap notifications."""
        self.alert_callbacks.append(callback)
    
    def set_backfill_manager(self, backfill_manager: Any) -> None:
        """Set backfill manager for automatic gap filling."""
        self.backfill_manager = backfill_manager
        logger.info("Backfill manager integrated")
    
    async def start(self) -> None:
        """Start gap detection."""
        if self._running:
            logger.warning("Gap detector already running")
            return
        
        self._running = True
        
        # Start background detection task
        self._detection_task = asyncio.create_task(self._detection_loop())
        
        logger.info("Gap detector started")
    
    async def stop(self) -> None:
        """Stop gap detection."""
        self._running = False
        
        if self._detection_task and not self._detection_task.done():
            self._detection_task.cancel()
            try:
                await self._detection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Gap detector stopped")
    
    async def process_market_data(self, market_data: MarketData) -> Optional[GapInfo]:
        """Process market data and detect gaps."""
        if not self._running:
            return None
        
        symbol = market_data.symbol
        data_type = market_data.type.value
        current_timestamp = market_data.timestamp_ms
        
        # Store in buffer
        self.data_buffers[symbol][data_type].append(market_data)
        
        # Limit buffer size
        if len(self.data_buffers[symbol][data_type]) > self.config.detection_window_size:
            self.data_buffers[symbol][data_type].popleft()
        
        # Detect gaps
        gap_info = await self._detect_gap(symbol, data_type, current_timestamp, market_data)
        
        if gap_info:
            await self._handle_gap_detection(gap_info)
            return gap_info
        
        # Update last timestamp
        self.last_timestamps[symbol][data_type] = current_timestamp
        
        # Update price/volume history for anomaly detection
        if data_type == "ticker" and market_data.data:
            price = market_data.data.get('last_price', 0)
            volume = market_data.data.get('volume_24h', 0)
            
            if price > 0:
                self.price_history[symbol].append(price)
            if volume > 0:
                self.volume_history[symbol].append(volume)
        
        return None
    
    async def _detect_gap(self, symbol: str, data_type: str, current_timestamp: int, 
                         market_data: MarketData) -> Optional[GapInfo]:
        """Detect gap in data stream."""
        # Check if we have previous timestamp
        if symbol not in self.last_timestamps or data_type not in self.last_timestamps[symbol]:
            return None
        
        last_timestamp = self.last_timestamps[symbol][data_type]
        
        # Get threshold based on data type
        threshold = self._get_gap_threshold(data_type)
        
        # Calculate time difference
        time_diff = (current_timestamp - last_timestamp) / 1000.0  # Convert to seconds
        
        # Check for gap
        if time_diff > threshold:
            gap_info = GapInfo(
                start_time=last_timestamp,
                end_time=current_timestamp,
                duration_ms=current_timestamp - last_timestamp,
                symbol=symbol,
                exchange=market_data.exchange,
                gap_type="missing",
                severity=self._calculate_gap_severity(time_diff, threshold)
            )
            
            return gap_info
        
        # Check for other issues
        return await self._detect_data_anomalies(symbol, data_type, market_data)
    
    async def _detect_data_anomalies(self, symbol: str, data_type: str, 
                                  market_data: MarketData) -> Optional[GapInfo]:
        """Detect data anomalies beyond time gaps."""
        if data_type != "ticker" or not market_data.data:
            return None
        
        current_price = market_data.data.get('last_price', 0)
        current_volume = market_data.data.get('volume_24h', 0)
        
        if current_price == 0:
            return None
        
        # Check for price jump anomaly
        if len(self.price_history[symbol]) > 0:
            last_price = self.price_history[symbol][-1]
            price_change_pct = abs((current_price - last_price) / last_price) * 100
            
            if price_change_pct > self.config.max_price_jump_pct:
                gap_info = GapInfo(
                    start_time=int(datetime.now().timestamp() * 1000),
                    end_time=int(datetime.now().timestamp() * 1000),
                    duration_ms=0,
                    symbol=symbol,
                    exchange=market_data.exchange,
                    gap_type="price_jump",
                    severity="high" if price_change_pct > 20 else "medium"
                )
                
                return gap_info
        
        # Check for volume spike anomaly
        if len(self.volume_history[symbol]) > 0:
            last_volume = self.volume_history[symbol][-1]
            if last_volume > 0:
                volume_change_pct = abs((current_volume - last_volume) / last_volume) * 100
                
                if volume_change_pct > self.config.max_volume_spike_pct:
                    gap_info = GapInfo(
                        start_time=int(datetime.now().timestamp() * 1000),
                        end_time=int(datetime.now().timestamp() * 1000),
                        duration_ms=0,
                        symbol=symbol,
                        exchange=market_data.exchange,
                        gap_type="volume_spike",
                        severity="medium"
                    )
                    
                    return gap_info
        
        return None
    
    def _get_gap_threshold(self, data_type: str) -> float:
        """Get gap threshold for data type."""
        thresholds = {
            "ticker": self.config.ticker_gap_threshold,
            "trade": self.config.trade_gap_threshold,
            "orderbook": self.config.orderbook_gap_threshold,
            "ohlcv": self.config.ohlcv_gap_threshold
        }
        return thresholds.get(data_type, self.config.ticker_gap_threshold)
    
    def _calculate_gap_severity(self, gap_duration: float, threshold: float) -> str:
        """Calculate gap severity based on duration."""
        ratio = gap_duration / threshold
        
        if ratio > 10:
            return "critical"
        elif ratio > 5:
            return "high"
        elif ratio > 2:
            return "medium"
        else:
            return "low"
    
    async def _handle_gap_detection(self, gap_info: GapInfo) -> None:
        """Handle detected gap."""
        # Store gap
        self.detected_gaps.append(gap_info)
        
        # Update statistics
        self._update_statistics(gap_info)
        
        # Trigger alerts
        await self._trigger_alerts(gap_info)
        
        # Auto backfill
        if self.config.auto_backfill and self.backfill_manager:
            await self._request_backfill(gap_info)
        
        logger.warning(f"Gap detected: {gap_info.symbol} {gap_info.gap_type} "
                     f"{gap_info.duration_ms}ms ({gap_info.severity})")
    
    def _update_statistics(self, gap_info: GapInfo) -> None:
        """Update gap detection statistics."""
        self.gap_statistics.total_gaps_detected += 1
        self.gap_statistics.gaps_by_type[gap_info.gap_type] += 1
        self.gap_statistics.gaps_by_symbol[gap_info.symbol] += 1
        self.gap_statistics.gaps_by_severity[gap_info.severity] += 1
        self.gap_statistics.last_gap_time = datetime.now()
        
        # Update duration statistics
        duration_seconds = gap_info.duration_ms / 1000.0
        if self.gap_statistics.total_gaps_detected == 1:
            self.gap_statistics.average_gap_duration = duration_seconds
        else:
            total_duration = self.gap_statistics.average_gap_duration * (self.gap_statistics.total_gaps_detected - 1)
            self.gap_statistics.average_gap_duration = (total_duration + duration_seconds) / self.gap_statistics.total_gaps_detected
        
        self.gap_statistics.longest_gap_duration = max(
            self.gap_statistics.longest_gap_duration, duration_seconds
        )
        
        # Calculate detection rate (gaps per hour)
        if len(self.detected_gaps) > 1:
            time_span = (datetime.now() - self.detected_gaps[0].start_time).total_seconds() / 3600.0
            if time_span > 0:
                self.gap_statistics.detection_rate = len(self.detected_gaps) / time_span
    
    async def _trigger_alerts(self, gap_info: GapInfo) -> None:
        """Trigger alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(gap_info)
                else:
                    callback(gap_info)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    async def _request_backfill(self, gap_info: GapInfo) -> None:
        """Request backfill for detected gap."""
        if not self.backfill_manager:
            return
        
        try:
            start_time = datetime.fromtimestamp(gap_info.start_time / 1000.0)
            end_time = datetime.fromtimestamp(gap_info.end_time / 1000.0)
            
            # Determine data type for backfill
            data_type = "ohlcv" if gap_info.gap_type == "missing" else gap_info.gap_type
            
            await self.backfill_manager.create_backfill_job(
                symbol=gap_info.symbol,
                exchange=gap_info.exchange,
                data_type=data_type,
                start_time=start_time,
                end_time=end_time,
                priority=self.config.backfill_priority
            )
            
            logger.info(f"Backfill requested for {gap_info.symbol} gap")
            
        except Exception as e:
            logger.error(f"Failed to request backfill: {e}")
    
    async def _detection_loop(self) -> None:
        """Background detection loop for periodic validation."""
        logger.info("Gap detection loop started")
        
        while self._running:
            try:
                # Perform periodic validation
                await self._periodic_validation()
                
                # Clean old data
                await self._cleanup_old_data()
                
                # Sleep for next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                await asyncio.sleep(10)
        
        logger.info("Gap detection loop stopped")
    
    async def _periodic_validation(self) -> None:
        """Perform periodic validation of all data streams."""
        if not self.config.enable_historical_validation:
            return
        
        current_time = datetime.now()
        current_timestamp = int(current_time.timestamp() * 1000)
        
        for symbol, data_types in self.last_timestamps.items():
            for data_type, last_timestamp in data_types.items():
                # Check for stale data
                threshold = self._get_gap_threshold(data_type)
                time_diff = (current_timestamp - last_timestamp) / 1000.0
                
                if time_diff > threshold * 2:  # Double threshold for stale data
                    gap_info = GapInfo(
                        start_time=last_timestamp,
                        end_time=current_timestamp,
                        duration_ms=current_timestamp - last_timestamp,
                        symbol=symbol,
                        exchange="unknown",  # Will be updated by data source
                        gap_type="stale_data",
                        severity="high"
                    )
                    
                    await self._handle_gap_detection(gap_info)
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old data to prevent memory leaks."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
        
        # Remove old gaps
        self.detected_gaps = [
            gap for gap in self.detected_gaps
            if gap.start_time > cutoff_timestamp
        ]
        
        # Clean old buffer data
        for symbol in self.data_buffers:
            for data_type in self.data_buffers[symbol]:
                buffer = self.data_buffers[symbol][data_type]
                while buffer and buffer[0].timestamp_ms < cutoff_timestamp:
                    buffer.popleft()
    
    def get_recent_gaps(self, limit: int = 100) -> List[GapInfo]:
        """Get recent gaps."""
        return self.detected_gaps[-limit:] if self.detected_gaps else []
    
    def get_gap_statistics(self) -> GapStatistics:
        """Get gap detection statistics."""
        return self.gap_statistics
    
    def get_symbol_status(self, symbol: str) -> Dict[str, Any]:
        """Get status for specific symbol."""
        status = {
            'symbol': symbol,
            'last_timestamps': self.last_timestamps.get(symbol, {}),
            'buffer_sizes': {
                data_type: len(buffer)
                for data_type, buffer in self.data_buffers.get(symbol, {}).items()
            },
            'recent_gaps': [
                gap for gap in self.detected_gaps[-10:]
                if gap.symbol == symbol
            ]
        }
        
        return status
    
    def reset_statistics(self) -> None:
        """Reset gap detection statistics."""
        self.gap_statistics = GapStatistics()
        self.detected_gaps.clear()
        logger.info("Gap detection statistics reset")


# Convenience functions
def create_gap_detector(
    ticker_gap_threshold: float = 5.0,
    trade_gap_threshold: float = 1.0,
    auto_backfill: bool = True,
    alert_callback: Optional[Callable] = None
) -> GapDetector:
    """Create gap detector with default configuration."""
    config = GapDetectionConfig(
        ticker_gap_threshold=ticker_gap_threshold,
        trade_gap_threshold=trade_gap_threshold,
        auto_backfill=auto_backfill,
        alert_callback=alert_callback
    )
    
    return GapDetector(config)


# Default alert callback
async def default_gap_alert(gap_info: GapInfo) -> None:
    """Default alert callback for gap detection."""
    logger.warning(f"GAP ALERT: {gap_info.symbol} {gap_info.gap_type} "
                 f"Duration: {gap_info.duration_ms}ms "
                 f"Severity: {gap_info.severity}")
    
    # Here you could add:
    # - Send to monitoring system
    # - Send email/SMS alert
    # - Update dashboard
    # - Trigger automated responses
