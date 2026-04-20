from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable

from its_project.core.queue_manager import QueueManager, QueueConfig

logger = logging.getLogger(__name__)


class PipelineController:
    """Advanced pipeline controller with queue management and fault tolerance."""
    
    def __init__(
        self,
        queue_configs: Dict[str, QueueConfig],
        max_retries: int = 3,
        retry_delay: timedelta = timedelta(seconds=5),
        health_check_interval: timedelta = timedelta(seconds=30),
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: timedelta = timedelta(minutes=1)
    ) -> None:
        self.queue_configs = queue_configs
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.health_check_interval = health_check_interval
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        
        # Queue manager
        self.queue_manager = QueueManager(queue_configs)
        
        # Pipeline tasks
        self.pipeline_tasks: Dict[str, asyncio.Task] = {}
        self.task_handlers: Dict[str, Callable] = {}
        
        # Fault tolerance
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.task_health: Dict[str, TaskHealth] = {}
        
        # Monitoring
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.pipeline_stats = {
            "total_processed": 0,
            "total_errors": 0,
            "total_retries": 0,
            "queue_overflows": 0,
            "circuit_breaker_trips": 0,
            "uptime": datetime.now()
        }
    
    async def start(self) -> None:
        """Start the pipeline controller."""
        if self._running:
            return
        
        # Initialize queue manager
        await self.queue_manager.start()
        
        # Initialize circuit breakers
        for queue_name in self.queue_configs.keys():
            self.circuit_breakers[queue_name] = CircuitBreaker(
                threshold=self.circuit_breaker_threshold,
                timeout=self.circuit_breaker_timeout
            )
            
            self.task_health[queue_name] = TaskHealth(queue_name)
        
        # Start health monitoring
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_monitoring_loop())
        self._metrics_task = asyncio.create_task(self._metrics_collection_loop())
        
        logger.info("Pipeline controller started")
    
    async def stop(self) -> None:
        """Stop the pipeline controller."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel monitoring tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._metrics_task:
            self._metrics_task.cancel()
        
        # Stop pipeline tasks
        for task_name, task in self.pipeline_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop queue manager
        await self.queue_manager.stop()
        
        logger.info("Pipeline controller stopped")
    
    def register_task_handler(self, queue_name: str, handler: Callable) -> None:
        """Register task handler for a queue."""
        self.task_handlers[queue_name] = handler
        
        # Start processing task if not already running
        if queue_name not in self.pipeline_tasks:
            task = asyncio.create_task(self._process_queue(queue_name))
            self.pipeline_tasks[queue_name] = task
    
    async def _process_queue(self, queue_name: str) -> None:
        """Process items from a specific queue with fault tolerance."""
        circuit_breaker = self.circuit_breakers[queue_name]
        task_health = self.task_health[queue_name]
        handler = self.task_handlers.get(queue_name)
        
        if not handler:
            logger.error(f"No handler registered for queue {queue_name}")
            return
        
        while self._running:
            try:
                # Check circuit breaker
                if circuit_breaker.is_open():
                    await asyncio.sleep(circuit_breaker.remaining_timeout())
                    continue
                
                # Get item from queue with timeout
                try:
                    item = await asyncio.wait_for(
                        self.queue_manager.get(queue_name),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process item with retry logic
                await self._process_item_with_retry(
                    queue_name, item, handler, circuit_breaker, task_health
                )
                
                # Update statistics
                self.pipeline_stats["total_processed"] += 1
                task_health.record_success()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing queue {queue_name}: {e}")
                
                # Update health and statistics
                task_health.record_error()
                self.pipeline_stats["total_errors"] += 1
                
                # Check circuit breaker
                if task_health.error_rate > 0.5:  # 50% error rate
                    circuit_breaker.trip()
                    self.pipeline_stats["circuit_breaker_trips"] += 1
                
                # Wait before retrying
                await asyncio.sleep(self.retry_delay.total_seconds())
    
    async def _process_item_with_retry(
        self,
        queue_name: str,
        item: Any,
        handler: Callable,
        circuit_breaker: CircuitBreaker,
        task_health: TaskHealth
    ) -> None:
        """Process item with retry mechanism."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Call handler
                if asyncio.iscoroutinefunction(handler):
                    await handler(item)
                else:
                    handler(item)
                
                # Success - break retry loop
                return
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Handler error for {queue_name} (attempt {attempt + 1}): {e}")
                
                # Record retry
                self.pipeline_stats["total_retries"] += 1
                task_health.record_retry()
                
                # Wait before retry
                if attempt < self.max_retries:
                    await asyncio.sleep(
                        self.retry_delay.total_seconds() * (2 ** attempt)
                    )
        
        # All retries failed
        logger.error(f"All retries failed for {queue_name}: {last_exception}")
        raise last_exception
    
    async def _health_monitoring_loop(self) -> None:
        """Monitor health of all pipeline components."""
        while self._running:
            try:
                # Check queue health
                queue_health = await self.queue_manager.get_health_status()
                
                # Check task health
                for queue_name, health in self.task_health.items():
                    health_stats = health.get_stats()
                    
                    # Log warnings for unhealthy components
                    if health_stats["error_rate"] > 0.3:
                        logger.warning(f"High error rate for {queue_name}: {health_stats['error_rate']:.2%}")
                    
                    if health_stats["avg_processing_time"] > 5.0:
                        logger.warning(f"Slow processing for {queue_name}: {health_stats['avg_processing_time']:.2f}s")
                
                # Check circuit breakers
                for queue_name, breaker in self.circuit_breakers.items():
                    if breaker.is_open():
                        logger.warning(f"Circuit breaker open for {queue_name}")
                
                await asyncio.sleep(self.health_check_interval.total_seconds())
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _metrics_collection_loop(self) -> None:
        """Collect and log pipeline metrics."""
        while self._running:
            try:
                # Get queue metrics
                queue_metrics = await self.queue_manager.get_metrics()
                
                # Calculate uptime
                uptime = datetime.now() - self.pipeline_stats["uptime"]
                
                # Log comprehensive metrics
                logger.info(
                    f"Pipeline Metrics - "
                    f"Processed: {self.pipeline_stats['total_processed']}, "
                    f"Errors: {self.pipeline_stats['total_errors']}, "
                    f"Retries: {self.pipeline_stats['total_retries']}, "
                    f"Queue Overflows: {self.pipeline_stats['queue_overflows']}, "
                    f"Circuit Breaker Trips: {self.pipeline_stats['circuit_breaker_trips']}, "
                    f"Uptime: {uptime}"
                )
                
                # Log queue-specific metrics
                for queue_name, metrics in queue_metrics.items():
                    logger.info(f"Queue {queue_name}: {metrics}")
                
                await asyncio.sleep(60)  # Log every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def submit_item(self, queue_name: str, item: Any, priority: bool = False) -> bool:
        """Submit item to pipeline with backpressure handling."""
        try:
            # Check circuit breaker
            if self.circuit_breakers[queue_name].is_open():
                logger.warning(f"Circuit breaker open, rejecting item for {queue_name}")
                return False
            
            # Submit to queue
            success = await self.queue_manager.put(queue_name, item, priority)
            
            if not success:
                self.pipeline_stats["queue_overflows"] += 1
                logger.warning(f"Queue overflow for {queue_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error submitting item to {queue_name}: {e}")
            return False
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        status = {
            "running": self._running,
            "statistics": self.pipeline_stats.copy(),
            "queues": {},
            "tasks": {},
            "circuit_breakers": {}
        }
        
        # Queue status
        for queue_name in self.queue_configs.keys():
            status["queues"][queue_name] = {
                "size": self.queue_manager.get_queue_size(queue_name),
                "max_size": self.queue_configs[queue_name].max_size,
                "overflow_count": self.queue_manager.get_overflow_count(queue_name)
            }
        
        # Task health
        for queue_name, health in self.task_health.items():
            status["tasks"][queue_name] = health.get_stats()
        
        # Circuit breaker status
        for queue_name, breaker in self.circuit_breakers.items():
            status["circuit_breakers"][queue_name] = {
                "is_open": breaker.is_open(),
                "failure_count": breaker.failure_count,
                "remaining_timeout": breaker.remaining_timeout()
            }
        
        return status
    
    async def reset_circuit_breaker(self, queue_name: str) -> bool:
        """Manually reset circuit breaker for a queue."""
        if queue_name in self.circuit_breakers:
            self.circuit_breakers[queue_name].reset()
            logger.info(f"Circuit breaker reset for {queue_name}")
            return True
        return False
    
    async def scale_queue(self, queue_name: str, new_max_size: int) -> bool:
        """Dynamically resize queue."""
        return await self.queue_manager.resize_queue(queue_name, new_max_size)


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, threshold: int = 5, timeout: timedelta = timedelta(minutes=1)):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.state == "closed":
            return False
        
        if self.state == "open":
            if self.last_failure_time and \
               datetime.now() - self.last_failure_time > self.timeout:
                self.state = "half_open"
                return False
            return True
        
        return False
    
    def record_success(self) -> None:
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self) -> None:
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.threshold:
            self.state = "open"
    
    def trip(self) -> None:
        """Manually trip circuit breaker."""
        self.state = "open"
        self.last_failure_time = datetime.now()
    
    def reset(self) -> None:
        """Reset circuit breaker."""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None
    
    def remaining_timeout(self) -> timedelta:
        """Get remaining timeout duration."""
        if self.state != "open" or not self.last_failure_time:
            return timedelta(0)
        
        elapsed = datetime.now() - self.last_failure_time
        remaining = self.timeout - elapsed
        
        return max(timedelta(0), remaining)


class TaskHealth:
    """Monitor task health and performance."""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.success_count = 0
        self.error_count = 0
        self.retry_count = 0
        self.processing_times: List[float] = []
        self.last_update = datetime.now()
    
    def record_success(self, processing_time: Optional[float] = None) -> None:
        """Record successful operation."""
        self.success_count += 1
        if processing_time is not None:
            self.processing_times.append(processing_time)
        self.last_update = datetime.now()
    
    def record_error(self) -> None:
        """Record failed operation."""
        self.error_count += 1
        self.last_update = datetime.now()
    
    def record_retry(self) -> None:
        """Record retry operation."""
        self.retry_count += 1
        self.last_update = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get health statistics."""
        total_operations = self.success_count + self.error_count
        
        return {
            "success_count": self.success_count,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "total_operations": total_operations,
            "error_rate": self.error_count / total_operations if total_operations > 0 else 0,
            "retry_rate": self.retry_count / total_operations if total_operations > 0 else 0,
            "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
            "last_update": self.last_update
        }
