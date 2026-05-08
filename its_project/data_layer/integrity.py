#!/usr/bin/env python3
"""
Data Integrity and Gap Detection Module
====================================

Critical component for production trading system.
Detects gaps, validates data quality, and ensures data consistency.
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import numpy as np

from its_project.common.types import MarketData, MarketDataType

logger = logging.getLogger(__name__)


@dataclass
class GapInfo:
    """Information about detected gap."""
    start_time: int  # ms timestamp
    end_time: int    # ms timestamp
    duration_ms: int
    symbol: str
    exchange: str
    gap_type: str    # "missing", "duplicate", "out_of_order"
    severity: str    # "low", "medium", "high", "critical"


@dataclass
class ValidationReport:
    """Data validation report."""
    symbol: str
    exchange: str
    timestamp: int
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    gaps: List[GapInfo] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class DataIntegrityMonitor:
    """
    Monitors data integrity and detects gaps in real-time.
    
    Features:
    - Gap detection (missing data, duplicates, out-of-order)
    - Data quality validation (NaN, spikes, outliers)
    - Statistics tracking
    - Configurable thresholds
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        
        # Configuration
        self.max_gap_ms = config.get('max_gap_ms', 5000)  # Max allowed gap
        self.duplicate_window_ms = config.get('duplicate_window_ms', 100)
        self.out_of_order_window_ms = config.get('out_of_order_window_ms', 1000)
        self.price_spike_threshold = config.get('price_spike_threshold', 0.1)  # 10%
        self.volume_spike_threshold = config.get('volume_spike_threshold', 10.0)  # 10x
        self.history_size = config.get('history_size', 1000)
        
        # State tracking per symbol
        self.symbol_states: Dict[str, Dict] = {}
        
        # Statistics
        self.total_processed = 0
        self.total_gaps = 0
        self.total_invalid = 0
        
    def _ensure_symbol_state(self, symbol: str, exchange: str) -> Dict:
        """Ensure state exists for symbol."""
        key = f"{exchange}:{symbol}"
        if key not in self.symbol_states:
            self.symbol_states[key] = {
                'last_timestamp': 0,
                'last_price': None,
                'last_volume': None,
                'timestamps': deque(maxlen=self.history_size),
                'prices': deque(maxlen=self.history_size),
                'volumes': deque(maxlen=self.history_size),
                'gaps_detected': 0,
                'invalid_count': 0
            }
        return self.symbol_states[key]
    
    def validate_data_point(self, data: MarketData) -> ValidationReport:
        """
        Validate a single data point and detect gaps.
        
        Args:
            data: MarketData point to validate
            
        Returns:
            ValidationReport with findings
        """
        self.total_processed += 1
        
        key = f"{data.exchange}:{data.symbol}"
        state = self._ensure_symbol_state(data.symbol, data.exchange)
        
        report = ValidationReport(
            symbol=data.symbol,
            exchange=data.exchange,
            timestamp=data.timestamp_ms,
            is_valid=True,
            issues=[],
            gaps=[],
            stats={}
        )
        
        # Extract price and volume
        price = data.data.get('close') or data.data.get('last')
        volume = data.data.get('volume', 0)
        
        # 1. Gap Detection
        gaps = self._detect_gaps(data, state)
        report.gaps.extend(gaps)
        
        for gap in gaps:
            report.issues.append(f"Gap detected: {gap.gap_type} ({gap.duration_ms}ms)")
            state['gaps_detected'] += 1
            self.total_gaps += 1
        
        # 2. Data Quality Validation
        quality_issues = self._validate_data_quality(data, price, volume, state)
        report.issues.extend(quality_issues)
        
        if quality_issues:
            report.is_valid = False
            state['invalid_count'] += 1
            self.total_invalid += 1
        
        # 3. Update state
        self._update_state(state, data.timestamp_ms, price, volume)
        
        # 4. Generate stats
        report.stats = self._generate_stats(state)
        
        return report
    
    def _detect_gaps(self, data: MarketData, state: Dict) -> List[GapInfo]:
        """Detect various types of gaps."""
        gaps = []
        current_time = data.timestamp_ms
        
        # Skip first data point
        if state['last_timestamp'] == 0:
            return gaps
        
        # 1. Missing data gap
        time_diff = current_time - state['last_timestamp']
        if time_diff > self.max_gap_ms:
            gap = GapInfo(
                start_time=state['last_timestamp'],
                end_time=current_time,
                duration_ms=time_diff,
                symbol=data.symbol,
                exchange=data.exchange,
                gap_type="missing",
                severity=self._calculate_gap_severity(time_diff)
            )
            gaps.append(gap)
        
        # 2. Duplicate detection
        if time_diff <= self.duplicate_window_ms:
            gap = GapInfo(
                start_time=state['last_timestamp'],
                end_time=current_time,
                duration_ms=time_diff,
                symbol=data.symbol,
                exchange=data.exchange,
                gap_type="duplicate",
                severity="low"
            )
            gaps.append(gap)
        
        # 3. Out-of-order detection
        if time_diff < 0:
            if abs(time_diff) > self.out_of_order_window_ms:
                gap = GapInfo(
                    start_time=current_time,
                    end_time=state['last_timestamp'],
                    duration_ms=abs(time_diff),
                    symbol=data.symbol,
                    exchange=data.exchange,
                    gap_type="out_of_order",
                    severity="medium"
                )
                gaps.append(gap)
        
        return gaps
    
    def _validate_data_quality(self, data: MarketData, price: Optional[float], 
                            volume: float, state: Dict) -> List[str]:
        """Validate data quality issues."""
        issues = []
        
        # 1. NaN/None checks
        if price is None or np.isnan(price) if isinstance(price, (float, np.floating)) else False:
            issues.append("Price is None or NaN")
        
        if volume is None or np.isnan(volume) if isinstance(volume, (float, np.floating)) else False:
            issues.append("Volume is None or NaN")
        
        # 2. Price spike detection
        if price is not None and state['last_price'] is not None:
            price_change = abs(price - state['last_price']) / state['last_price']
            if price_change > self.price_spike_threshold:
                issues.append(f"Price spike detected: {price_change:.2%}")
        
        # 3. Volume spike detection
        if volume > 0 and state['last_volume'] is not None and state['last_volume'] > 0:
            volume_change = volume / state['last_volume']
            if volume_change > self.volume_spike_threshold:
                issues.append(f"Volume spike detected: {volume_change:.1f}x")
        
        # 4. Order book validation
        if 'orderbook' in data.data:
            ob_issues = self._validate_orderbook(data.data['orderbook'])
            issues.extend(ob_issues)
        
        return issues
    
    def _validate_orderbook(self, orderbook: Dict) -> List[str]:
        """Validate order book structure and data."""
        issues = []
        
        if not isinstance(orderbook, dict):
            issues.append("Order book is not a dictionary")
            return issues
        
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        # Check structure
        if not bids or not asks:
            issues.append("Empty order book (no bids or asks)")
        
        # Check bid-ask spread
        if bids and asks:
            best_bid = bids[0][0] if bids[0] else None
            best_ask = asks[0][0] if asks[0] else None
            
            if best_bid and best_ask:
                if best_bid >= best_ask:
                    issues.append(f"Invalid spread: bid {best_bid} >= ask {best_ask}")
        
        # Check price ordering
        for i, (price, _) in enumerate(bids[:-1]):
            next_price = bids[i + 1][0]
            if price <= next_price:
                issues.append(f"Bid prices not descending: {price} <= {next_price}")
                break
        
        for i, (price, _) in enumerate(asks[:-1]):
            next_price = asks[i + 1][0]
            if price >= next_price:
                issues.append(f"Ask prices not ascending: {price} >= {next_price}")
                break
        
        return issues
    
    def _calculate_gap_severity(self, gap_duration_ms: int) -> str:
        """Calculate gap severity based on duration."""
        if gap_duration_ms < 10000:  # < 10s
            return "low"
        elif gap_duration_ms < 60000:  # < 1m
            return "medium"
        elif gap_duration_ms < 300000:  # < 5m
            return "high"
        else:
            return "critical"
    
    def _update_state(self, state: Dict, timestamp: int, price: Optional[float], volume: float) -> None:
        """Update internal state."""
        state['last_timestamp'] = timestamp
        if price is not None:
            state['last_price'] = price
        state['last_volume'] = volume
        
        # Update history
        state['timestamps'].append(timestamp)
        if price is not None:
            state['prices'].append(price)
        state['volumes'].append(volume)
    
    def _generate_stats(self, state: Dict) -> Dict[str, Any]:
        """Generate statistics for the symbol."""
        return {
            'processed_count': len(state['timestamps']),
            'gaps_detected': state['gaps_detected'],
            'invalid_count': state['invalid_count'],
            'last_timestamp': state['last_timestamp'],
            'avg_time_diff': np.mean(np.diff(list(state['timestamps']))) if len(state['timestamps']) > 1 else 0,
            'price_volatility': np.std(list(state['prices'])) if len(state['prices']) > 1 else 0,
            'avg_volume': np.mean(list(state['volumes'])) if state['volumes'] else 0
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        return {
            'total_processed': self.total_processed,
            'total_gaps': self.total_gaps,
            'total_invalid': self.total_invalid,
            'symbols_tracked': len(self.symbol_states),
            'gap_rate': self.total_gaps / max(self.total_processed, 1),
            'invalid_rate': self.total_invalid / max(self.total_processed, 1),
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.total_processed = 0
        self.total_gaps = 0
        self.total_invalid = 0
        self.symbol_states.clear()
        logger.info("Data integrity monitor stats reset")


# Global instance for easy access
_integrity_monitor: Optional[DataIntegrityMonitor] = None


def get_integrity_monitor() -> DataIntegrityMonitor:
    """Get global integrity monitor instance."""
    global _integrity_monitor
    if _integrity_monitor is None:
        # Default configuration
        config = {
            'max_gap_ms': 5000,
            'duplicate_window_ms': 100,
            'out_of_order_window_ms': 1000,
            'price_spike_threshold': 0.1,
            'volume_spike_threshold': 10.0,
            'history_size': 1000
        }
        _integrity_monitor = DataIntegrityMonitor(config)
    return _integrity_monitor


def validate_market_data(data: MarketData) -> ValidationReport:
    """Convenience function to validate market data."""
    monitor = get_integrity_monitor()
    return monitor.validate_data_point(data)


if __name__ == "__main__":
    # Test the integrity monitor
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'max_gap_ms': 5000,
        'price_spike_threshold': 0.05
    }
    
    monitor = DataIntegrityMonitor(config)
    
    # Simulate some data
    base_time = int(time.time() * 1000)
    
    # Normal data
    data1 = MarketData(
        timestamp_ms=base_time,
        symbol="BTC/USDT",
        exchange="binance",
        type=MarketDataType.TICKER,
        data={'close': 42000.0, 'volume': 1.0}
    )
    
    report1 = monitor.validate_data_point(data1)
    print(f"Report 1: valid={report1.is_valid}, issues={len(report1.issues)}")
    
    # Gap data
    data2 = MarketData(
        timestamp_ms=base_time + 10000,  # 10s gap
        symbol="BTC/USDT",
        exchange="binance",
        type=MarketDataType.TICKER,
        data={'close': 42100.0, 'volume': 1.5}
    )
    
    report2 = monitor.validate_data_point(data2)
    print(f"Report 2: valid={report2.is_valid}, gaps={len(report2.gaps)}")
    for gap in report2.gaps:
        print(f"  Gap: {gap.gap_type} - {gap.duration_ms}ms ({gap.severity})")
    
    # Spike data
    data3 = MarketData(
        timestamp_ms=base_time + 11000,
        symbol="BTC/USDT",
        exchange="binance",
        type=MarketDataType.TICKER,
        data={'close': 50000.0, 'volume': 2.0}  # 19% spike
    )
    
    report3 = monitor.validate_data_point(data3)
    print(f"Report 3: valid={report3.is_valid}, issues={report3.issues}")
    
    print(f"\nSystem stats: {monitor.get_system_stats()}")
