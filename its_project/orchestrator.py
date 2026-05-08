from __future__ import annotations

import asyncio
import logging
import time
import json
from typing import Optional, Dict, Any, Callable, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque
import signal
import uuid

from its_project.common.types import MarketData
from its_project.data_layer.base import BaseDataSource
from its_project.features.pipeline import FeaturePipeline
from its_project.models.base import BaseModel
from its_project.decision.engine import TradingDecisionEngine
from its_project.execution.base import BaseExecutor
from its_project.decision.decision import Decision, Signal
from its_project.execution.kill_switch import KillSwitch, KillSwitchReason
from its_project.monitoring.business_metrics import BusinessMetricsCollector
from its_project.execution.latency_measurement import LatencyMeasurementSystem, LatencyStage
from its_project.mlops.drift_detection import DriftDetector

logger = logging.getLogger(__name__)


class QueuePriority(Enum):
    """Queue priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class QueueType(Enum):
    """Queue types."""
    MARKET_DATA = "market_data"
    SIGNALS = "signals"
    DECISIONS = "decisions"
    ORDERS = "orders"
    METRICS = "metrics"
    ALERTS = "alerts"


@dataclass
class QueueItem:
    """Queue item with priority and metadata."""
    item_id: str
    queue_type: QueueType
    priority: QueuePriority
    data: Any
    timestamp: int
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorConfig:
    """Configuration for pipeline orchestrator."""
    data_source: BaseDataSource
    feature_pipeline: FeaturePipeline
    model: BaseModel
    decision_engine: TradingDecisionEngine
    executor: BaseExecutor
    symbols: list[str]
    
    # Queue configuration
    queue_sizes: Dict[QueueType, int] = field(default_factory=lambda: {
        QueueType.MARKET_DATA: 1000,
        QueueType.SIGNALS: 500,
        QueueType.DECISIONS: 200,
        QueueType.ORDERS: 100,
        QueueType.METRICS: 1000,
        QueueType.ALERTS: 500
    })
    
    # Performance configuration
    loop_interval_ms: int = 1000
    max_retries: int = 3
    enable_paper_trading: bool = True
    enable_kill_switch: bool = True
    enable_business_metrics: bool = True
    enable_latency_tracking: bool = True
    
    # Processing configuration
    max_concurrent_processing: int = 10
    batch_size: int = 50
    processing_timeout_seconds: int = 30


@dataclass
class PipelineMetrics:
    """Enhanced metrics for pipeline monitoring."""
    processed_ticks: int = 0
    signals_generated: int = 0
    decisions_executed: int = 0
    orders_placed: int = 0
    errors: int = 0
    
    # Queue metrics
    queue_sizes: Dict[str, int] = field(default_factory=dict)
    queue_processing_times: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    
    # Timing
    last_tick_time: Optional[datetime] = None
    last_signal_time: Optional[datetime] = None
    last_decision_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    
    # System health
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Business metrics
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0


class CentralizedQueue:
    """Centralized priority queue system."""
    
    def __init__(self, queue_type: QueueType, max_size: int = 1000) -> None:
        self.queue_type = queue_type
        self.max_size = max_size
        self._queue = deque()
        self._lock = asyncio.Lock()
        self._stats = {
            'total_processed': 0,
            'total_dropped': 0,
            'avg_wait_time_ms': 0.0,
            'current_size': 0
        }
    
    async def put(self, item: QueueItem) -> bool:
        """Add item to queue."""
        async with self._lock:
            if len(self._queue) >= self.max_size:
                self._stats['total_dropped'] += 1
                return False
            
            self._queue.append(item)
            self._stats['current_size'] = len(self._queue)
            return True
    
    async def get(self, timeout: Optional[float] = None) -> Optional[QueueItem]:
        """Get item from queue."""
        start_time = time.time()
        
        while True:
            async with self._lock:
                if self._queue:
                    item = self._queue.popleft()
                    self._stats['current_size'] = len(self._queue)
                    
                    # Update wait time
                    wait_time = (time.time() - start_time) * 1000
                    self._stats['avg_wait_time_ms'] = (
                        (self._stats['avg_wait_time_ms'] * self._stats['total_processed'] + wait_time) /
                        (self._stats['total_processed'] + 1)
                    )
                    self._stats['total_processed'] += 1
                    
                    return item
            
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            await asyncio.sleep(0.01)
    
    def size(self) -> int:
        """Get current queue size."""
        return len(self._queue)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return self._stats.copy()


class PipelineOrchestrator:
    """
    Production-ready centralized pipeline orchestrator.
    
    Features:
    - Centralized priority queues
    - Concurrent processing
    - Kill switch integration
    - Business metrics tracking
    - Latency measurement
    - Error handling and recovery
    - Performance monitoring
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        
        # Core state
        self._running = False
        self._stop_event = asyncio.Event()
        self._start_time: Optional[datetime] = None
        
        # Centralized queues
        self.queues: Dict[QueueType, CentralizedQueue] = {}
        for queue_type, size in config.queue_sizes.items():
            self.queues[queue_type] = CentralizedQueue(queue_type, size)
        
        # Metrics
        self._metrics = PipelineMetrics()
        
        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "on_tick": [],
            "on_signal": [],
            "on_decision": [],
            "on_error": [],
            "on_order": [],
            "on_alert": []
        }
        
        # Production components
        self.kill_switch: Optional[KillSwitch] = None
        self.business_metrics: Optional[BusinessMetricsCollector] = None
        self.latency_system: Optional[LatencyMeasurementSystem] = None
        self.drift_detector: Optional[DriftDetector] = None
        
        # Processing workers
        self._workers: List[asyncio.Task] = []
        self._processing_semaphore = asyncio.Semaphore(config.max_concurrent_processing)
        
        # Performance tracking
        self._processing_times: deque = deque(maxlen=1000)
        self._last_throughput_update = time.time()
        self._last_processed_count = 0

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for pipeline events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        else:
            raise ValueError(f"Unknown event: {event}")

    async def _emit(self, event: str, *args, **kwargs) -> None:
        """Emit event to registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Initializing production pipeline orchestrator")
        
        try:
            # Initialize core components
            await self.config.data_source.connect()
            logger.info("Data source connected")
            
            # Initialize production components
            if self.config.enable_kill_switch:
                self.kill_switch = KillSwitch()
                self.kill_switch.add_callback(self._handle_kill_switch)
                logger.info("Kill switch initialized")
            
            if self.config.enable_business_metrics:
                self.business_metrics = BusinessMetricsCollector()
                logger.info("Business metrics initialized")
            
            if self.config.enable_latency_tracking:
                self.latency_system = LatencyMeasurementSystem()
                self.latency_system.add_alert_callback(self._handle_latency_alert)
                logger.info("Latency tracking initialized")
            
            # Initialize drift detector
            self.drift_detector = DriftDetector()
            self.drift_detector.add_alert_callback(self._handle_drift_alert)
            logger.info("Drift detection initialized")
            
            # Start processing workers
            await self._start_workers()
            
            logger.info("Pipeline components initialized successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    async def _start_workers(self) -> None:
        """Start processing workers."""
        # Market data worker
        self._workers.append(asyncio.create_task(self._market_data_worker()))
        
        # Signal processing worker
        self._workers.append(asyncio.create_task(self._signal_processing_worker()))
        
        # Decision processing worker
        self._workers.append(asyncio.create_task(self._decision_processing_worker()))
        
        # Order execution worker
        self._workers.append(asyncio.create_task(self._order_execution_worker()))
        
        # Metrics collection worker
        self._workers.append(asyncio.create_task(self._metrics_collection_worker()))
        
        logger.info(f"Started {len(self._workers)} processing workers")

    async def _market_data_worker(self) -> None:
        """Worker for processing market data."""
        logger.info("Market data worker started")
        
        while self._running and not self._stop_event.is_set():
            try:
                # Fetch latest data for all symbols
                for symbol in self.config.symbols:
                    if self._stop_event.is_set():
                        break
                    
                    market_data = await self.config.data_source.fetch(symbol)
                    if market_data:
                        # Add to market data queue
                        item = QueueItem(
                            item_id=str(uuid.uuid4()),
                            queue_type=QueueType.MARKET_DATA,
                            priority=QueuePriority.NORMAL,
                            data=market_data,
                            timestamp=int(time.time() * 1000)
                        )
                        
                        await self.queues[QueueType.MARKET_DATA].put(item)
                
                await asyncio.sleep(self.config.loop_interval_ms / 1000.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in market data worker: {e}")
                self._metrics.errors += 1
                await asyncio.sleep(1)
        
        logger.info("Market data worker stopped")

    async def _signal_processing_worker(self) -> None:
        """Worker for processing signals."""
        logger.info("Signal processing worker started")
        
        while self._running and not self._stop_event.is_set():
            try:
                # Get market data from queue
                item = await self.queues[QueueType.MARKET_DATA].get(timeout=1.0)
                if not item:
                    continue
                
                async with self._processing_semaphore:
                    await self._process_market_data(item)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in signal processing worker: {e}")
                self._metrics.errors += 1
                await asyncio.sleep(0.1)
        
        logger.info("Signal processing worker stopped")

    async def _decision_processing_worker(self) -> None:
        """Worker for processing decisions."""
        logger.info("Decision processing worker started")
        
        while self._running and not self._stop_event.is_set():
            try:
                # Get signal from queue
                item = await self.queues[QueueType.SIGNALS].get(timeout=1.0)
                if not item:
                    continue
                
                async with self._processing_semaphore:
                    await self._process_signal(item)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in decision processing worker: {e}")
                self._metrics.errors += 1
                await asyncio.sleep(0.1)
        
        logger.info("Decision processing worker stopped")

    async def _order_execution_worker(self) -> None:
        """Worker for order execution."""
        logger.info("Order execution worker started")
        
        while self._running and not self._stop_event.is_set():
            try:
                # Get decision from queue
                item = await self.queues[QueueType.DECISIONS].get(timeout=1.0)
                if not item:
                    continue
                
                async with self._processing_semaphore:
                    await self._execute_decision(item)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in order execution worker: {e}")
                self._metrics.errors += 1
                await asyncio.sleep(0.1)
        
        logger.info("Order execution worker stopped")

    async def _metrics_collection_worker(self) -> None:
        """Worker for metrics collection."""
        logger.info("Metrics collection worker started")
        
        while self._running and not self._stop_event.is_set():
            try:
                # Update metrics
                await self._update_metrics()
                
                # Process metrics queue
                while True:
                    item = await self.queues[QueueType.METRICS].get(timeout=0.1)
                    if not item:
                        break
                    
                    await self._process_metrics_item(item)
                
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection worker: {e}")
                await asyncio.sleep(1)
        
        logger.info("Metrics collection worker stopped")

    async def _process_market_data(self, item: QueueItem) -> None:
        """Process market data item."""
        start_time = time.time()
        request_id = item.item_id
        
        try:
            # Start latency measurement
            if self.latency_system:
                self.latency_system.start_measurement(request_id, {'queue_type': 'market_data'})
                self.latency_system.start_stage(request_id, LatencyStage.DATA_INGESTION)
            
            market_data = item.data
            self._metrics.processed_ticks += 1
            self._metrics.last_tick_time = datetime.now()
            
            await self._emit("on_tick", market_data)
            
            if self.latency_system:
                self.latency_system.end_stage(request_id, LatencyStage.DATA_INGESTION)
                self.latency_system.start_stage(request_id, LatencyStage.FEATURE_PROCESSING)
            
            # Convert to DataFrame for feature pipeline
            import pandas as pd
            df = pd.DataFrame([{
                "timestamp": market_data.timestamp_ms,
                "symbol": market_data.symbol,
                **market_data.data
            }])
            
            # Feature extraction
            features = self.config.feature_pipeline.transform(df)
            
            if self.latency_system:
                self.latency_system.end_stage(request_id, LatencyStage.FEATURE_PROCESSING)
                self.latency_system.start_stage(request_id, LatencyStage.MODEL_INFERENCE)
            
            # Model prediction
            prediction = self.config.model.predict(features)
            proba = self.config.model.predict_proba(features) if hasattr(self.config.model, "predict_proba") else None
            
            if self.latency_system:
                self.latency_system.end_stage(request_id, LatencyStage.MODEL_INFERENCE)
                self.latency_system.start_stage(request_id, LatencyStage.SIGNAL_GENERATION)
            
            # Signal generation
            signal = Signal(
                symbol=market_data.symbol,
                action=prediction[0] if prediction is not None else "HOLD",
                confidence=float(proba[0][prediction[0]]) if proba is not None else 0.0,
                timestamp=market_data.timestamp_ms,
                metadata={"prediction": prediction, "proba": proba}
            )
            
            self._metrics.signals_generated += 1
            self._metrics.last_signal_time = datetime.now()
            
            if self.latency_system:
                self.latency_system.end_stage(request_id, LatencyStage.SIGNAL_GENERATION)
            
            await self._emit("on_signal", signal)
            
            # Add signal to queue
            signal_item = QueueItem(
                item_id=str(uuid.uuid4()),
                queue_type=QueueType.SIGNALS,
                priority=QueuePriority.NORMAL,
                data=signal,
                timestamp=int(time.time() * 1000),
                metadata={'original_request_id': request_id}
            )
            
            await self.queues[QueueType.SIGNALS].put(signal_item)
            
            # Update processing time
            processing_time = (time.time() - start_time) * 1000
            self._processing_times.append(processing_time)
            
        except Exception as e:
            self._metrics.errors += 1
            logger.error(f"Error processing market data: {e}")
            await self._emit("on_error", e)

    async def _process_signal(self, item: QueueItem) -> None:
        """Process signal item."""
        start_time = time.time()
        request_id = item.item_id
        
        try:
            signal = item.data
            
            if self.latency_system:
                self.latency_system.start_stage(request_id, LatencyStage.DECISION_MAKING)
            
            # Decision making
            market_state = {"price": 50000.0}  # Placeholder
            balance = await self.config.executor.fetch_balance()
            account_balance = balance.get("USDT", 0.0)
            
            decision = self.config.decision_engine.process_signal(
                signal, market_state, account_balance
            )
            
            if self.latency_system:
                self.latency_system.end_stage(request_id, LatencyStage.DECISION_MAKING)
            
            if decision:
                self._metrics.decisions_executed += 1
                self._metrics.last_decision_time = datetime.now()
                await self._emit("on_decision", decision)
                
                # Add decision to queue
                decision_item = QueueItem(
                    item_id=str(uuid.uuid4()),
                    queue_type=QueueType.DECISIONS,
                    priority=QueuePriority.HIGH,
                    data=decision,
                    timestamp=int(time.time() * 1000),
                    metadata={'original_request_id': request_id}
                )
                
                await self.queues[QueueType.DECISIONS].put(decision_item)
            
            # Update processing time
            processing_time = (time.time() - start_time) * 1000
            self._processing_times.append(processing_time)
            
        except Exception as e:
            self._metrics.errors += 1
            logger.error(f"Error processing signal: {e}")
            await self._emit("on_error", e)

    async def _execute_decision(self, item: QueueItem) -> None:
        """Execute decision item."""
        start_time = time.time()
        request_id = item.item_id
        
        try:
            decision = item.data
            
            if self.latency_system:
                self.latency_system.start_stage(request_id, LatencyStage.ORDER_ROUTING)
            
            # Create order
            from its_project.execution.base import OrderType
            
            order = await self.config.executor.create_order(
                symbol=decision.symbol,
                order_type=OrderType.MARKET,
                side=decision.action.lower(),
                amount=decision.size,
            )
            
            if self.latency_system:
                self.latency_system.end_stage(request_id, LatencyStage.ORDER_ROUTING)
                self.latency_system.start_stage(request_id, LatencyStage.EXECUTION)
            
            self._metrics.orders_placed += 1
            await self._emit("on_order", order)
            
            # Record trade in business metrics
            if self.business_metrics:
                self.business_metrics.record_trade({
                    'order_id': order.id,
                    'symbol': decision.symbol,
                    'side': decision.action,
                    'quantity': decision.size,
                    'price': order.price,
                    'timestamp': int(time.time() * 1000)
                })
            
            if self.latency_system:
                self.latency_system.end_stage(request_id, LatencyStage.EXECUTION)
                self.latency_system.end_measurement(request_id)
            
            logger.info(f"Order executed: {order.id} {decision.action} {decision.size} {decision.symbol}")
            
            # Update processing time
            processing_time = (time.time() - start_time) * 1000
            self._processing_times.append(processing_time)
            
        except Exception as e:
            self._metrics.errors += 1
            logger.error(f"Error executing decision: {e}")
            await self._emit("on_error", e)

    async def _process_metrics_item(self, item: QueueItem) -> None:
        """Process metrics item."""
        try:
            # Add to business metrics
            if self.business_metrics:
                self.business_metrics.update_positions(item.data)
            
        except Exception as e:
            logger.error(f"Error processing metrics item: {e}")

    async def _update_metrics(self) -> None:
        """Update pipeline metrics."""
        try:
            # Update queue sizes
            for queue_type, queue in self.queues.items():
                self._metrics.queue_sizes[queue_type.value] = queue.size()
            
            # Update processing time metrics
            if self._processing_times:
                times = list(self._processing_times)
                self._metrics.avg_processing_time_ms = sum(times) / len(times)
                self._metrics.max_processing_time_ms = max(times)
            
            # Update throughput
            current_time = time.time()
            time_diff = current_time - self._last_throughput_update
            if time_diff > 0:
                processed_diff = self._metrics.processed_ticks - self._last_processed_count
                self._metrics.throughput_per_second = processed_diff / time_diff
                self._last_throughput_update = current_time
                self._last_processed_count = self._metrics.processed_ticks
            
            # Update uptime
            if self._start_time:
                self._metrics.uptime_seconds = (datetime.now() - self._start_time).total_seconds()
            
            # Update system health
            try:
                import psutil
                process = psutil.Process()
                self._metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                self._metrics.cpu_usage_percent = process.cpu_percent()
            except:
                pass
            
            # Update business metrics
            if self.business_metrics:
                business_stats = self.business_metrics.get_current_stats()
                self._metrics.total_pnl = business_stats.get('total_pnl', 0.0)
                self._metrics.win_rate = business_stats.get('win_rate', 0.0)
                self._metrics.sharpe_ratio = business_stats.get('sharpe_ratio', 0.0)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    async def _handle_kill_switch(self, event_data: Dict[str, Any]) -> None:
        """Handle kill switch event."""
        logger.warning(f"Kill switch triggered: {event_data}")
        
        # Add alert to queue
        alert_item = QueueItem(
            item_id=str(uuid.uuid4()),
            queue_type=QueueType.ALERTS,
            priority=QueuePriority.CRITICAL,
            data=event_data,
            timestamp=int(time.time() * 1000)
        )
        
        await self.queues[QueueType.ALERTS].put(alert_item)
        
        # Stop processing
        await self.stop()

    async def _handle_latency_alert(self, alert_data: Dict[str, Any]) -> None:
        """Handle latency alert."""
        logger.warning(f"Latency alert: {alert_data}")
        
        # Add alert to queue
        alert_item = QueueItem(
            item_id=str(uuid.uuid4()),
            queue_type=QueueType.ALERTS,
            priority=QueuePriority.HIGH,
            data=alert_data,
            timestamp=int(time.time() * 1000)
        )
        
        await self.queues[QueueType.ALERTS].put(alert_item)

    async def _handle_drift_alert(self, alert: Any) -> None:
        """Handle drift alert."""
        logger.warning(f"Drift alert: {alert}")
        
        # Add alert to queue
        alert_item = QueueItem(
            item_id=str(uuid.uuid4()),
            queue_type=QueueType.ALERTS,
            priority=QueuePriority.HIGH,
            data={'drift_alert': str(alert)},
            timestamp=int(time.time() * 1000)
        )
        
        await self.queues[QueueType.ALERTS].put(alert_item)

    async def shutdown(self) -> None:
        """Shutdown all pipeline components gracefully."""
        logger.info("Shutting down pipeline orchestrator")
        self._running = False
        self._stop_event.set()
        
        try:
            # Stop workers
            for worker in self._workers:
                worker.cancel()
            
            # Wait for workers to finish
            await asyncio.gather(*self._workers, return_exceptions=True)
            
            # Shutdown components
            await self.config.data_source.disconnect()
            
            if self.kill_switch:
                self.kill_switch.shutdown()
            
            if self.latency_system:
                self.latency_system.stop_monitoring()
            
            logger.info("Pipeline components shutdown successfully")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    async def start(self) -> None:
        """Start the production pipeline orchestrator."""
        if self._running:
            logger.warning("Orchestrator already running")
            return
        
        logger.info("Starting production pipeline orchestrator")
        self._running = True
        self._start_time = datetime.now()
        self._stop_event.clear()
        
        await self.initialize()
        
        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
        
        # Start latency monitoring
        if self.latency_system:
            self.latency_system.start_monitoring()
        
        logger.info("Production pipeline orchestrator started successfully")

    async def stop(self) -> None:
        """Stop the pipeline orchestrator."""
        logger.info("Stopping pipeline orchestrator")
        await self.shutdown()

    def get_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        return self._metrics

    def is_running(self) -> bool:
        """Check if orchestrator is running."""
        return self._running

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "running": self._running,
            "uptime_seconds": self._metrics.uptime_seconds,
            "processed_ticks": self._metrics.processed_ticks,
            "signals_generated": self._metrics.signals_generated,
            "decisions_executed": self._metrics.decisions_executed,
            "orders_placed": self._metrics.orders_placed,
            "errors": self._metrics.errors,
            "avg_processing_time_ms": self._metrics.avg_processing_time_ms,
            "max_processing_time_ms": self._metrics.max_processing_time_ms,
            "throughput_per_second": self._metrics.throughput_per_second,
            "queue_sizes": self._metrics.queue_sizes,
            "memory_usage_mb": self._metrics.memory_usage_mb,
            "cpu_usage_percent": self._metrics.cpu_usage_percent,
            "total_pnl": self._metrics.total_pnl,
            "win_rate": self._metrics.win_rate,
            "sharpe_ratio": self._metrics.sharpe_ratio,
            "last_tick_time": self._metrics.last_tick_time.isoformat() if self._metrics.last_tick_time else None,
            "last_signal_time": self._metrics.last_signal_time.isoformat() if self._metrics.last_signal_time else None,
            "last_decision_time": self._metrics.last_decision_time.isoformat() if self._metrics.last_decision_time else None,
            "symbols": self.config.symbols,
            "workers_active": len(self._workers),
            "kill_switch_enabled": self.config.enable_kill_switch,
            "business_metrics_enabled": self.config.enable_business_metrics,
            "latency_tracking_enabled": self.config.enable_latency_tracking,
        }

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        stats = {}
        for queue_type, queue in self.queues.items():
            stats[queue_type.value] = queue.get_stats()
        return stats

    async def emergency_stop(self, reason: str = "Manual emergency stop") -> None:
        """Emergency stop the orchestrator."""
        logger.critical(f"Emergency stop triggered: {reason}")
        
        # Trigger kill switch if available
        if self.kill_switch:
            self.kill_switch.emergency_stop(KillSwitchReason.MANUAL, reason)
        
        # Stop all processing
        await self.stop()

    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        health = {
            "overall": "healthy",
            "components": {},
            "alerts": []
        }
        
        # Check queue health
        for queue_type, queue in self.queues.items():
            queue_size = queue.size()
            max_size = queue.max_size
            utilization = queue_size / max_size
            
            if utilization > 0.9:
                health["components"][queue_type.value] = "critical"
                health["overall"] = "critical"
                health["alerts"].append(f"{queue_type.value} queue at {utilization:.1%} capacity")
            elif utilization > 0.7:
                health["components"][queue_type.value] = "warning"
                if health["overall"] == "healthy":
                    health["overall"] = "warning"
            else:
                health["components"][queue_type.value] = "healthy"
        
        # Check processing latency
        if self._metrics.avg_processing_time_ms > 1000:  # 1 second
            health["components"]["processing_latency"] = "warning"
            if health["overall"] == "healthy":
                health["overall"] = "warning"
            health["alerts"].append(f"High processing latency: {self._metrics.avg_processing_time_ms:.2f}ms")
        
        # Check error rate
        if self._metrics.processed_ticks > 0:
            error_rate = self._metrics.errors / self._metrics.processed_ticks
            if error_rate > 0.1:  # 10% error rate
                health["components"]["error_rate"] = "critical"
                health["overall"] = "critical"
                health["alerts"].append(f"High error rate: {error_rate:.2%}")
            elif error_rate > 0.05:  # 5% error rate
                health["components"]["error_rate"] = "warning"
                if health["overall"] == "healthy":
                    health["overall"] = "warning"
        
        return health
