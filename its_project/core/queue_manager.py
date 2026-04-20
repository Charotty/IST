from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QueueConfig:
    """Configuration for a managed queue."""
    max_size: int = 1000
    overflow_strategy: str = "drop_oldest"  # "drop_oldest", "drop_newest", "block"
    priority_levels: int = 2  # Normal and high priority
    batch_size: int = 10
    batch_timeout: float = 1.0  # seconds
    health_check_interval: float = 30.0  # seconds


class ManagedQueue:
    """Enhanced queue with overflow handling and monitoring."""
    
    def __init__(self, name: str, config: QueueConfig) -> None:
        self.name = name
        self.config = config
        
        # Queue storage with priority
        self.normal_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_size)
        self.high_priority_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_size // 4)
        
        # Overflow handling
        self.overflow_count = 0
        self.dropped_items = []
        
        # Monitoring
        self.total_put_count = 0
        self.total_get_count = 0
        self.last_activity = datetime.now()
        
        # Batch processing
        self.batch_buffer: List[Any] = []
        self.batch_event = asyncio.Event()
    
    async def put(self, item: Any, priority: bool = False) -> bool:
        """
        Put item in queue with overflow handling.
        
        Args:
            item: Item to put
            priority: Whether to use high priority queue
            
        Returns:
            True if item was added, False if dropped
        """
        self.total_put_count += 1
        self.last_activity = datetime.now()
        
        queue = self.high_priority_queue if priority else self.normal_queue
        
        try:
            # Try to put item
            queue.put_nowait(item)
            return True
            
        except asyncio.QueueFull:
            # Handle overflow based on strategy
            return await self._handle_overflow(item, queue, priority)
    
    async def get(self, timeout: Optional[float] = None) -> Any:
        """
        Get item from queue with priority handling.
        
        Args:
            timeout: Get timeout
            
        Returns:
            Item or None if timeout
        """
        try:
            # Check high priority queue first
            if not self.high_priority_queue.empty():
                item = self.high_priority_queue.get_nowait()
            elif not self.normal_queue.empty():
                item = self.normal_queue.get_nowait()
            else:
                # Wait for any item
                if timeout:
                    item = await asyncio.wait_for(
                        self._get_from_any_queue(),
                        timeout=timeout
                    )
                else:
                    item = await self._get_from_any_queue()
            
            self.total_get_count += 1
            self.last_activity = datetime.now()
            return item
            
        except asyncio.TimeoutError:
            return None
    
    async def _get_from_any_queue(self) -> Any:
        """Get from any queue with priority."""
        # Create tasks for both queues
        high_priority_task = asyncio.create_task(self.high_priority_queue.get())
        normal_priority_task = asyncio.create_task(self.normal_queue.get())
        
        try:
            # Wait for first completed task
            done, pending = await asyncio.wait(
                [high_priority_task, normal_priority_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Get result from completed task
            return list(done)[0].result()
            
        except Exception as e:
            # Clean up on error
            high_priority_task.cancel()
            normal_priority_task.cancel()
            raise e
    
    async def _handle_overflow(self, item: Any, queue: asyncio.Queue, priority: bool) -> bool:
        """Handle queue overflow based on strategy."""
        self.overflow_count += 1
        
        if self.config.overflow_strategy == "drop_oldest":
            # Drop oldest item
            try:
                dropped_item = queue.get_nowait()
                self.dropped_items.append(dropped_item)
                queue.put_nowait(item)
                logger.warning(f"Dropped oldest item from queue {self.name}")
                return True
            except asyncio.QueueEmpty:
                # Shouldn't happen if queue is full
                pass
        
        elif self.config.overflow_strategy == "drop_newest":
            # Drop the new item
            self.dropped_items.append(item)
            logger.warning(f"Dropped new item from queue {self.name}")
            return False
        
        elif self.config.overflow_strategy == "block":
            # Block until space is available
            try:
                await queue.put(item)
                return True
            except asyncio.CancelledError:
                return False
        
        else:
            logger.error(f"Unknown overflow strategy: {self.config.overflow_strategy}")
            return False
    
    def get_size(self) -> int:
        """Get total queue size."""
        return self.normal_queue.qsize() + self.high_priority_queue.qsize()
    
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self.get_size() >= self.config.max_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "name": self.name,
            "total_size": self.get_size(),
            "normal_size": self.normal_queue.qsize(),
            "high_priority_size": self.high_priority_queue.qsize(),
            "max_size": self.config.max_size,
            "overflow_count": self.overflow_count,
            "dropped_items_count": len(self.dropped_items),
            "total_put_count": self.total_put_count,
            "total_get_count": self.total_get_count,
            "last_activity": self.last_activity,
            "utilization": self.get_size() / self.config.max_size
        }
    
    async def clear(self) -> int:
        """Clear queue and return number of cleared items."""
        cleared = 0
        
        while not self.normal_queue.empty():
            self.normal_queue.get_nowait()
            cleared += 1
        
        while not self.high_priority_queue.empty():
            self.high_priority_queue.get_nowait()
            cleared += 1
        
        self.dropped_items.clear()
        logger.info(f"Cleared {cleared} items from queue {self.name}")
        
        return cleared


class QueueManager:
    """Manages multiple queues with backpressure and monitoring."""
    
    def __init__(self, queue_configs: Dict[str, QueueConfig]) -> None:
        self.queue_configs = queue_configs
        self.queues: Dict[str, ManagedQueue] = {}
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Initialize queues
        for name, config in queue_configs.items():
            self.queues[name] = ManagedQueue(name, config)
    
    async def start(self) -> None:
        """Start queue manager."""
        if self._running:
            return
        
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Queue manager started")
    
    async def stop(self) -> None:
        """Stop queue manager."""
        if not self._running:
            return
        
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Queue manager stopped")
    
    async def put(self, queue_name: str, item: Any, priority: bool = False) -> bool:
        """Put item in specified queue."""
        if queue_name not in self.queues:
            logger.error(f"Queue {queue_name} not found")
            return False
        
        return await self.queues[queue_name].put(item, priority)
    
    async def get(self, queue_name: str, timeout: Optional[float] = None) -> Any:
        """Get item from specified queue."""
        if queue_name not in self.queues:
            logger.error(f"Queue {queue_name} not found")
            return None
        
        return await self.queues[queue_name].get(timeout)
    
    def get_queue_size(self, queue_name: str) -> int:
        """Get size of specified queue."""
        if queue_name not in self.queues:
            return 0
        
        return self.queues[queue_name].get_size()
    
    def get_overflow_count(self, queue_name: str) -> int:
        """Get overflow count for specified queue."""
        if queue_name not in self.queues:
            return 0
        
        return self.queues[queue_name].overflow_count
    
    async def resize_queue(self, queue_name: str, new_max_size: int) -> bool:
        """Resize queue dynamically."""
        if queue_name not in self.queues:
            return False
        
        queue = self.queues[queue_name]
        old_max_size = queue.config.max_size
        
        # Update configuration
        queue.config.max_size = new_max_size
        
        # Note: asyncio.Queue doesn't support resizing
        # This is a configuration update that affects future operations
        logger.info(f"Resized queue {queue_name} from {old_max_size} to {new_max_size}")
        
        return True
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all queues."""
        health_status = {}
        
        for name, queue in self.queues.items():
            stats = queue.get_stats()
            
            # Determine health status
            health = "healthy"
            if stats["utilization"] > 0.9:
                health = "warning"
            if stats["utilization"] >= 1.0:
                health = "critical"
            
            # Check for staleness
            time_since_activity = datetime.now() - stats["last_activity"]
            if time_since_activity > timedelta(minutes=5):
                health = "stale"
            
            health_status[name] = {
                "health": health,
                "stats": stats,
                "time_since_activity": time_since_activity
            }
        
        return health_status
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        metrics = {}
        total_stats = {
            "total_items": 0,
            "total_overflow": 0,
            "total_put": 0,
            "total_get": 0,
            "avg_utilization": 0
        }
        
        queue_count = len(self.queues)
        
        for name, queue in self.queues.items():
            stats = queue.get_stats()
            metrics[name] = stats
            
            # Aggregate totals
            total_stats["total_items"] += stats["total_size"]
            total_stats["total_overflow"] += stats["overflow_count"]
            total_stats["total_put"] += stats["total_put_count"]
            total_stats["total_get"] += stats["total_get_count"]
            total_stats["avg_utilization"] += stats["utilization"]
        
        # Calculate averages
        if queue_count > 0:
            total_stats["avg_utilization"] /= queue_count
        
        metrics["_total"] = total_stats
        
        return metrics
    
    async def _health_check_loop(self) -> None:
        """Periodic health check for all queues."""
        while self._running:
            try:
                health_status = await self.get_health_status()
                
                for queue_name, status in health_status.items():
                    health = status["health"]
                    stats = status["stats"]
                    
                    # Log warnings for unhealthy queues
                    if health == "critical":
                        logger.error(f"Queue {queue_name} is CRITICAL: {stats['utilization']:.1%} full")
                    elif health == "warning":
                        logger.warning(f"Queue {queue_name} is WARNING: {stats['utilization']:.1%} full")
                    elif health == "stale":
                        logger.warning(f"Queue {queue_name} is STALE: no activity for {status['time_since_activity']}")
                    
                    # Check for high overflow rates
                    if stats["overflow_count"] > 0:
                        overflow_rate = stats["overflow_count"] / max(stats["total_put_count"], 1)
                        if overflow_rate > 0.1:  # 10% overflow rate
                            logger.warning(f"High overflow rate for {queue_name}: {overflow_rate:.1%}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)
    
    async def clear_queue(self, queue_name: str) -> int:
        """Clear specified queue."""
        if queue_name not in self.queues:
            return 0
        
        return await self.queues[queue_name].clear()
    
    async def clear_all_queues(self) -> int:
        """Clear all queues."""
        total_cleared = 0
        
        for queue_name in self.queues.keys():
            cleared = await self.clear_queue(queue_name)
            total_cleared += cleared
        
        logger.info(f"Cleared {total_cleared} items from all queues")
        return total_cleared
    
    def get_queue_config(self, queue_name: str) -> Optional[QueueConfig]:
        """Get configuration for specified queue."""
        return self.queue_configs.get(queue_name)
    
    def update_queue_config(self, queue_name: str, config: QueueConfig) -> bool:
        """Update configuration for specified queue."""
        if queue_name not in self.queues:
            return False
        
        self.queue_configs[queue_name] = config
        self.queues[queue_name].config = config
        
        logger.info(f"Updated configuration for queue {queue_name}")
        return True
