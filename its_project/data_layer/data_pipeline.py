#!/usr/bin/env python3
"""
Integrated Data Pipeline with Gap Detection and Reconnection
========================================================

Production-ready data pipeline combining OKX data source,
gap detection, and reconnection management.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime

from its_project.common.types import MarketData, MarketDataType
from its_project.data_layer.okx_source import OKXDataSource, OKXConfig
from its_project.data_layer.gap_detector import GapDetector, GapDetectionConfig, GapInfo
from its_project.data_layer.reconnection_manager import ReconnectionManager, ReconnectionConfig, ConnectionEvent
from its_project.data_layer.backfill_manager import BackfillManager, BackfillConfig

logger = logging.getLogger(__name__)


@dataclass
class DataPipelineConfig:
    """Data pipeline configuration."""
    # OKX configuration
    okx_config: OKXConfig = field(default_factory=OKXConfig)
    
    # Gap detection configuration
    gap_detection_config: GapDetectionConfig = field(default_factory=GapDetectionConfig)
    
    # Reconnection configuration
    reconnection_config: ReconnectionConfig = field(default_factory=ReconnectionConfig)
    
    # Backfill configuration
    backfill_config: BackfillConfig = field(default_factory=BackfillConfig)
    
    # Pipeline settings
    enable_monitoring: bool = True
    monitoring_interval: float = 60.0  # seconds
    enable_alerts: bool = True
    alert_webhook: Optional[str] = None
    
    # Data processing
    enable_data_validation: bool = True
    enable_data_buffering: bool = True
    buffer_size: int = 1000
    
    # Performance settings
    max_concurrent_processing: int = 10
    processing_timeout: float = 30.0


@dataclass
class PipelineStats:
    """Pipeline statistics."""
    start_time: datetime
    total_messages_processed: int = 0
    total_gaps_detected: int = 0
    total_reconnections: int = 0
    total_backfills_triggered: int = 0
    average_processing_rate: float = 0.0  # messages per second
    uptime_percentage: float = 0.0
    last_message_time: Optional[datetime] = None
    error_count: int = 0


class DataPipeline:
    """
    Integrated data pipeline with gap detection and reconnection.
    
    Features:
    - OKX data source integration
    - Real-time gap detection
    - Automatic reconnection management
    - Backfill integration
    - Performance monitoring
    - Alert system
    - Data validation
    """
    
    def __init__(self, config: DataPipelineConfig) -> None:
        self.config = config
        
        # Initialize components
        self.okx_source = OKXDataSource(config.okx_config)
        self.gap_detector = GapDetector(config.gap_detection_config)
        self.reconnection_manager = ReconnectionManager(config.reconnection_config)
        self.backfill_manager = BackfillManager(config.backfill_config)
        
        # Data processing
        self.data_buffer: List[MarketData] = []
        self.processing_queue: asyncio.Queue[MarketData] = asyncio.Queue()
        self.output_queue: asyncio.Queue[MarketData] = asyncio.Queue()
        
        # Statistics
        self.stats = PipelineStats(start_time=datetime.now())
        
        # State
        self._running = False
        self._processing_tasks: List[asyncio.Task] = []
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Setup integration
        self._setup_integration()
        
        logger.info("Data pipeline initialized")
    
    def _setup_integration(self) -> None:
        """Setup integration between components."""
        # Set up gap detector with backfill manager
        self.gap_detector.set_backfill_manager(self.backfill_manager)
        
        # Add alert callbacks
        if self.config.enable_alerts:
            self.gap_detector.add_alert_callback(self._on_gap_detected)
            self.reconnection_manager.config.alert_callbacks.append(self._on_connection_event)
        
        # Register OKX source with reconnection manager
        self.reconnection_manager.register_source(
            "okx",
            self._connect_okx,
            health_check_callback=self._health_check_okx
        )
    
    async def start(self) -> None:
        """Start the data pipeline."""
        if self._running:
            logger.warning("Data pipeline already running")
            return
        
        self._running = True
        
        try:
            # Start components
            await self.reconnection_manager.start()
            await self.gap_detector.start()
            await self.backfill_manager.start()
            
            # Connect OKX source
            await self.reconnection_manager.connect_source("okx")
            
            # Start data processing
            await self._start_data_processing()
            
            # Start monitoring
            if self.config.enable_monitoring:
                self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Data pipeline started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start data pipeline: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the data pipeline."""
        if not self._running:
            return
        
        self._running = False
        
        try:
            # Stop monitoring
            if self._monitoring_task and not self._monitoring_task.done():
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Stop processing tasks
            for task in self._processing_tasks:
                task.cancel()
            
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)
            
            # Stop components
            await self.okx_source.disconnect()
            await self.reconnection_manager.stop()
            await self.gap_detector.stop()
            await self.backfill_manager.stop()
            
            logger.info("Data pipeline stopped")
            
        except Exception as e:
            logger.error(f"Error stopping data pipeline: {e}")
    
    async def _connect_okx(self) -> None:
        """Connect OKX data source."""
        await self.okx_source.connect()
        
        # Start data streaming
        asyncio.create_task(self._stream_okx_data())
    
    async def _stream_okx_data(self) -> None:
        """Stream data from OKX source."""
        try:
            async for market_data in self.okx_source.subscribe():
                if not self._running:
                    break
                
                # Process data
                await self._process_market_data(market_data)
                
        except Exception as e:
            logger.error(f"OKX data streaming error: {e}")
            raise
    
    async def _health_check_okx(self) -> bool:
        """Health check for OKX source."""
        try:
            return await self.okx_source.is_alive()
        except Exception as e:
            logger.error(f"OKX health check error: {e}")
            return False
    
    async def _start_data_processing(self) -> None:
        """Start data processing workers."""
        for i in range(self.config.max_concurrent_processing):
            task = asyncio.create_task(self._data_processing_worker(f"worker-{i}"))
            self._processing_tasks.append(task)
        
        logger.info(f"Started {len(self._processing_tasks)} data processing workers")
    
    async def _data_processing_worker(self, worker_name: str) -> None:
        """Data processing worker."""
        logger.debug(f"Data processing worker {worker_name} started")
        
        while self._running:
            try:
                # Get data from queue
                market_data = await asyncio.wait_for(
                    self.processing_queue.get(),
                    timeout=1.0
                )
                
                # Process data
                processed_data = await self._validate_and_enrich_data(market_data)
                
                # Add to output queue
                await self.output_queue.put(processed_data)
                
                # Update statistics
                self.stats.total_messages_processed += 1
                self.stats.last_message_time = datetime.now()
                
                # Update processing rate
                elapsed = (datetime.now() - self.stats.start_time).total_seconds()
                if elapsed > 0:
                    self.stats.average_processing_rate = self.stats.total_messages_processed / elapsed
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Data processing worker {worker_name} error: {e}")
                self.stats.error_count += 1
        
        logger.debug(f"Data processing worker {worker_name} stopped")
    
    async def _process_market_data(self, market_data: MarketData) -> None:
        """Process incoming market data."""
        try:
            # Add to buffer if enabled
            if self.config.enable_data_buffering:
                self.data_buffer.append(market_data)
                if len(self.data_buffer) > self.config.buffer_size:
                    self.data_buffer.pop(0)
            
            # Gap detection
            if self.config.gap_detection_config.enable_real_time_detection:
                gap_info = await self.gap_detector.process_market_data(market_data)
                if gap_info:
                    self.stats.total_gaps_detected += 1
            
            # Add to processing queue
            await self.processing_queue.put(market_data)
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            self.stats.error_count += 1
    
    async def _validate_and_enrich_data(self, market_data: MarketData) -> MarketData:
        """Validate and enrich market data."""
        if not self.config.enable_data_validation:
            return market_data
        
        # Add pipeline metadata
        enriched_data = MarketData(
            timestamp_ms=market_data.timestamp_ms,
            symbol=market_data.symbol,
            type=market_data.type,
            exchange=market_data.exchange,
            data={
                **market_data.data,
                '_pipeline_timestamp': int(datetime.now().timestamp() * 1000),
                '_pipeline_processed': True
            }
        )
        
        # Basic validation
        if market_data.type == MarketDataType.TICKER:
            price = market_data.data.get('last_price', 0)
            if price <= 0:
                logger.warning(f"Invalid ticker price for {market_data.symbol}: {price}")
        
        return enriched_data
    
    async def _monitoring_loop(self) -> None:
        """Pipeline monitoring loop."""
        logger.info("Pipeline monitoring started")
        
        while self._running:
            try:
                # Update statistics
                await self._update_pipeline_stats()
                
                # Check for issues
                await self._check_pipeline_health()
                
                # Sleep until next check
                await asyncio.sleep(self.config.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)
        
        logger.info("Pipeline monitoring stopped")
    
    async def _update_pipeline_stats(self) -> None:
        """Update pipeline statistics."""
        current_time = datetime.now()
        elapsed = (current_time - self.stats.start_time).total_seconds()
        
        # Calculate uptime percentage
        if self.stats.last_message_time:
            time_since_last_message = (current_time - self.stats.last_message_time).total_seconds()
            if time_since_last_message < self.config.monitoring_interval * 2:
                self.stats.uptime_percentage = 100.0
            else:
                # Reduce uptime based on staleness
                staleness_penalty = min(time_since_last_message / 3600.0, 1.0) * 100
                self.stats.uptime_percentage = max(0, 100 - staleness_penalty)
    
    async def _check_pipeline_health(self) -> None:
        """Check pipeline health and alert on issues."""
        current_time = datetime.now()
        
        # Check for stale data
        if self.stats.last_message_time:
            time_since_last = (current_time - self.stats.last_message_time).total_seconds()
            if time_since_last > self.config.monitoring_interval * 3:
                await self._send_alert(
                    "pipeline_stale",
                    f"No data received for {time_since_last:.0f} seconds",
                    severity="high"
                )
        
        # Check error rate
        if self.stats.total_messages_processed > 0:
            error_rate = self.stats.error_count / self.stats.total_messages_processed
            if error_rate > 0.05:  # 5% error rate
                await self._send_alert(
                    "high_error_rate",
                    f"Error rate: {error_rate:.2%}",
                    severity="medium"
                )
        
        # Check processing queue size
        queue_size = self.processing_queue.qsize()
        if queue_size > self.config.buffer_size * 0.8:
            await self._send_alert(
                "queue_backlog",
                f"Processing queue size: {queue_size}",
                    severity="medium"
                )
    
    async def _on_gap_detected(self, gap_info: GapInfo) -> None:
        """Handle gap detection event."""
        self.stats.total_gaps_detected += 1
        
        # Log gap
        logger.warning(f"Gap detected: {gap_info.symbol} {gap_info.gap_type} "
                     f"{gap_info.duration_ms}ms ({gap_info.severity})")
        
        # Send alert
        await self._send_alert(
            "gap_detected",
            f"Gap: {gap_info.symbol} {gap_info.gap_type} {gap_info.duration_ms}ms",
            severity=gap_info.severity
        )
        
        # Update backfill statistics
        if gap_info.gap_type == "missing":
            self.stats.total_backfills_triggered += 1
    
    async def _on_connection_event(self, event: ConnectionEvent) -> None:
        """Handle connection event."""
        if event.state.value in ["reconnecting", "failed"]:
            self.stats.total_reconnections += 1
        
        # Log connection event
        if event.state.value == "connected":
            logger.info(f"Connection restored: {event.source}")
        elif event.state.value == "failed":
            logger.error(f"Connection failed: {event.source} - {event.error}")
        
        # Send alert for critical events
        if event.state.value in ["failed", "circuit_open"]:
            await self._send_alert(
                "connection_issue",
                f"Connection issue: {event.source} {event.state.value}",
                severity="high"
            )
    
    async def _send_alert(self, alert_type: str, message: str, severity: str = "medium") -> None:
        """Send alert."""
        logger.warning(f"ALERT [{alert_type}]: {message}")
        
        # Here you could add:
        # - Send to monitoring system
        # - Send email/SMS
        # - Send to webhook
        # - Update dashboard
        
        if self.config.alert_webhook:
            # Send webhook alert (implementation would depend on webhook format)
            pass
    
    async def get_data_stream(self) -> AsyncIterator[MarketData]:
        """Get processed data stream."""
        while self._running:
            try:
                market_data = await asyncio.wait_for(
                    self.output_queue.get(),
                    timeout=1.0
                )
                yield market_data
            except asyncio.TimeoutError:
                continue
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        # Get component statuses
        okx_status = "connected" if self.okx_source._connected else "disconnected"
        gap_detector_status = "running" if self.gap_detector._running else "stopped"
        reconnection_status = "running" if self.reconnection_manager._running else "stopped"
        backfill_status = "running" if self.backfill_manager._running else "stopped"
        
        # Get detailed component information
        connection_statuses = self.reconnection_manager.get_all_status()
        gap_statistics = self.gap_detector.get_gap_statistics()
        backfill_status_info = self.backfill_manager.get_queue_status()
        
        return {
            'pipeline': {
                'running': self._running,
                'uptime_percentage': self.stats.uptime_percentage,
                'start_time': self.stats.start_time.isoformat(),
                'last_message_time': self.stats.last_message_time.isoformat() if self.stats.last_message_time else None
            },
            'components': {
                'okx_source': okx_status,
                'gap_detector': gap_detector_status,
                'reconnection_manager': reconnection_status,
                'backfill_manager': backfill_status
            },
            'statistics': {
                'total_messages_processed': self.stats.total_messages_processed,
                'total_gaps_detected': self.stats.total_gaps_detected,
                'total_reconnections': self.stats.total_reconnections,
                'total_backfills_triggered': self.stats.total_backfills_triggered,
                'average_processing_rate': self.stats.average_processing_rate,
                'error_count': self.stats.error_count,
                'queue_sizes': {
                    'processing': self.processing_queue.qsize(),
                    'output': self.output_queue.qsize()
                }
            },
            'detailed_status': {
                'connections': connection_statuses,
                'gap_statistics': {
                    'total_gaps': gap_statistics.total_gaps_detected,
                    'gaps_by_type': dict(gap_statistics.gaps_by_type),
                    'gaps_by_severity': dict(gap_statistics.gaps_by_severity),
                    'average_gap_duration': gap_statistics.average_gap_duration
                },
                'backfill_queue': backfill_status_info
            }
        }
    
    def get_recent_gaps(self, limit: int = 50) -> List[GapInfo]:
        """Get recent gaps."""
        return self.gap_detector.get_recent_gaps(limit)
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get data buffer status."""
        return {
            'buffer_size': len(self.data_buffer),
            'max_buffer_size': self.config.buffer_size,
            'buffer_utilization': len(self.data_buffer) / self.config.buffer_size if self.config.buffer_size > 0 else 0,
            'oldest_message_age': (
                (datetime.now() - self.data_buffer[0].timestamp_ms / 1000).total_seconds()
                if self.data_buffer else 0
            )
        }


# Convenience function
def create_data_pipeline(
    symbols: List[str] = None,
    enable_gap_detection: bool = True,
    enable_reconnection: bool = True,
    enable_backfill: bool = True
) -> DataPipeline:
    """Create data pipeline with default configuration."""
    # OKX configuration
    okx_config = OKXConfig(symbols=symbols or ["BTC-USDT", "ETH-USDT"])
    
    # Gap detection configuration
    gap_config = GapDetectionConfig(
        enable_real_time_detection=enable_gap_detection,
        auto_backfill=enable_backfill
    )
    
    # Reconnection configuration
    reconnection_config = ReconnectionConfig(
        enable_circuit_breaker=True,
        enable_health_checks=enable_reconnection
    )
    
    # Backfill configuration
    backfill_config = BackfillConfig(
        enable_scheduling=enable_backfill,
        max_concurrent_jobs=3
    )
    
    # Pipeline configuration
    pipeline_config = DataPipelineConfig(
        okx_config=okx_config,
        gap_detection_config=gap_config,
        reconnection_config=reconnection_config,
        backfill_config=backfill_config
    )
    
    return DataPipeline(pipeline_config)
