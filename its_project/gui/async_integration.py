from __future__ import annotations

import asyncio
import logging
import threading
import queue
import weakref
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread, QEventLoop
from PyQt6.QtWidgets import QApplication

logger = logging.getLogger(__name__)


@dataclass
class AsyncTask:
    """Async task wrapper for PyQt integration."""
    id: str
    coro: Coroutine
    future: Future
    created_at: datetime
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None


class AsyncEventLoopManager(QObject):
    """Manages asyncio event loop integration with PyQt."""
    
    # Signals for async operation results
    task_completed = pyqtSignal(str, object)  # task_id, result
    task_failed = pyqtSignal(str, str)  # task_id, error
    task_cancelled = pyqtSignal(str)  # task_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Event loop management
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Task management
        self.tasks: Dict[str, AsyncTask] = {}
        self.task_counter = 0
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="AsyncWorker")
        
        # Queue for cross-thread communication
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Setup result processing timer
        self.result_timer = QTimer()
        self.result_timer.timeout.connect(self._process_results)
        self.result_timer.start(16)  # ~60 FPS
        
        logger.info("AsyncEventLoopManager initialized")
    
    def start_event_loop(self) -> bool:
        """Start the asyncio event loop in a separate thread."""
        if self.is_running:
            logger.warning("Event loop already running")
            return True
        
        try:
            # Create and start event loop thread
            self.loop_thread = threading.Thread(
                target=self._run_event_loop,
                name="AsyncEventLoop",
                daemon=True
            )
            self.loop_thread.start()
            
            # Wait for loop to start
            for _ in range(50):  # 5 seconds timeout
                if self.is_running:
                    break
                threading.Event().wait(0.1)
            
            if self.is_running:
                logger.info("Async event loop started successfully")
                return True
            else:
                logger.error("Failed to start event loop")
                return False
                
        except Exception as e:
            logger.error(f"Error starting event loop: {e}")
            return False
    
    def stop_event_loop(self):
        """Stop the asyncio event loop."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all running tasks
        if self.loop:
            for task_id, async_task in list(self.tasks.items()):
                if not async_task.future.done():
                    async_task.future.cancel()
                    self.task_cancelled.emit(task_id)
            
            # Stop the loop
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        # Wait for thread to finish
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=2.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Clear tasks
        self.tasks.clear()
        
        logger.info("Async event loop stopped")
    
    def _run_event_loop(self):
        """Run the asyncio event loop in the background thread."""
        try:
            # Create new event loop
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            self.is_running = True
            logger.info("Event loop thread started")
            
            # Run the loop
            self.loop.run_forever()
            
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            self.is_running = False
            if self.loop:
                self.loop.close()
                self.loop = None
    
    def submit_task(
        self,
        coro: Coroutine,
        timeout: Optional[float] = None,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None
    ) -> str:
        """Submit a coroutine to be executed in the event loop."""
        if not self.is_running:
            raise RuntimeError("Event loop is not running")
        
        # Generate task ID
        task_id = f"task_{self.task_counter}"
        self.task_counter += 1
        
        # Create future for the task
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        
        # Create async task wrapper
        async_task = AsyncTask(
            id=task_id,
            coro=coro,
            future=future,
            created_at=datetime.now(),
            timeout=timeout,
            callback=callback,
            error_callback=error_callback
        )
        
        # Store task
        self.tasks[task_id] = async_task
        
        # Add timeout if specified
        if timeout:
            asyncio.run_coroutine_threadsafe(
                self._add_task_timeout(task_id, timeout),
                self.loop
            )
        
        # Schedule result checking
        asyncio.run_coroutine_threadsafe(
            self._monitor_task(task_id),
            self.loop
        )
        
        logger.debug(f"Submitted async task {task_id}")
        return task_id
    
    async def _add_task_timeout(self, task_id: str, timeout: float):
        """Add timeout to a task."""
        try:
            async_task = self.tasks.get(task_id)
            if async_task and not async_task.future.done():
                await asyncio.wait_for(async_task.future, timeout=timeout)
        except asyncio.TimeoutError:
            if task_id in self.tasks:
                async_task = self.tasks[task_id]
                async_task.future.cancel()
                self.result_queue.put(("timeout", task_id, None))
        except Exception as e:
            if task_id in self.tasks:
                self.result_queue.put(("error", task_id, str(e)))
    
    async def _monitor_task(self, task_id: str):
        """Monitor task completion and send results."""
        try:
            async_task = self.tasks.get(task_id)
            if not async_task:
                return
            
            # Wait for task completion
            try:
                result = await asyncio.wrap_future(async_task.future)
                self.result_queue.put(("success", task_id, result))
            except asyncio.CancelledError:
                self.result_queue.put(("cancelled", task_id, None))
            except Exception as e:
                self.result_queue.put(("error", task_id, str(e)))
                
        except Exception as e:
            logger.error(f"Error monitoring task {task_id}: {e}")
            self.result_queue.put(("error", task_id, str(e)))
    
    def _process_results(self):
        """Process completed task results from the queue."""
        try:
            while not self.result_queue.empty():
                try:
                    status, task_id, result = self.result_queue.get_nowait()
                    
                    if task_id in self.tasks:
                        async_task = self.tasks[task_id]
                        
                        if status == "success":
                            self.task_completed.emit(task_id, result)
                            if async_task.callback:
                                async_task.callback(result)
                                
                        elif status == "error":
                            self.task_failed.emit(task_id, str(result))
                            if async_task.error_callback:
                                async_task.error_callback(str(result))
                                
                        elif status == "cancelled" or status == "timeout":
                            self.task_cancelled.emit(task_id)
                        
                        # Remove completed task
                        del self.tasks[task_id]
                        
                except queue.Empty:
                    break
                    
        except Exception as e:
            logger.error(f"Error processing results: {e}")
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id not in self.tasks:
            return False
        
        async_task = self.tasks[task_id]
        if not async_task.future.done():
            async_task.future.cancel()
            return True
        return False
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get the status of a task."""
        if task_id not in self.tasks:
            return None
        
        async_task = self.tasks[task_id]
        future = async_task.future
        
        if future.done():
            if future.cancelled():
                return "cancelled"
            elif future.exception():
                return "error"
            else:
                return "completed"
        else:
            return "running"
    
    def get_active_tasks(self) -> List[str]:
        """Get list of active task IDs."""
        return [task_id for task_id, task in self.tasks.items() 
                if not task.future.done()]
    
    def get_task_count(self) -> int:
        """Get total number of tasks."""
        return len(self.tasks)


class AsyncSignalBridge(QObject):
    """Bridge for async operations with PyQt signals."""
    
    # Generic signals for different data types
    signal_result = pyqtSignal(str, object)  # signal_name, result
    signal_error = pyqtSignal(str, str)  # signal_name, error
    signal_progress = pyqtSignal(str, float)  # signal_name, progress
    
    def __init__(self, event_manager: AsyncEventLoopManager, parent=None):
        super().__init__(parent)
        self.event_manager = event_manager
        
        # Connect to event manager signals
        self.event_manager.task_completed.connect(self._on_task_completed)
        self.event_manager.task_failed.connect(self._on_task_failed)
        self.event_manager.task_cancelled.connect(self._on_task_cancelled)
        
        # Signal registry
        self.signal_registry: Dict[str, str] = {}  # signal_name -> task_id
        
        logger.info("AsyncSignalBridge initialized")
    
    def emit_async(self, signal_name: str, coro: Coroutine, **kwargs) -> str:
        """Emit an async operation result through a signal."""
        task_id = self.event_manager.submit_task(
            coro=coro,
            callback=lambda result: self._handle_signal_result(signal_name, result),
            error_callback=lambda error: self._handle_signal_error(signal_name, error),
            **kwargs
        )
        
        self.signal_registry[signal_name] = task_id
        return task_id
    
    def _handle_signal_result(self, signal_name: str, result: Any):
        """Handle successful async result."""
        self.signal_result.emit(signal_name, result)
    
    def _handle_signal_error(self, signal_name: str, error: str):
        """Handle async error."""
        self.signal_error.emit(signal_name, error)
    
    def _on_task_completed(self, task_id: str, result: Any):
        """Handle task completion from event manager."""
        # Find signal name for this task
        signal_name = next((name for name, tid in self.signal_registry.items() 
                          if tid == task_id), None)
        if signal_name:
            del self.signal_registry[signal_name]
    
    def _on_task_failed(self, task_id: str, error: str):
        """Handle task failure from event manager."""
        signal_name = next((name for name, tid in self.signal_registry.items() 
                          if tid == task_id), None)
        if signal_name:
            self.signal_error.emit(signal_name, error)
            del self.signal_registry[signal_name]
    
    def _on_task_cancelled(self, task_id: str):
        """Handle task cancellation from event manager."""
        signal_name = next((name for name, tid in self.signal_registry.items() 
                          if tid == task_id), None)
        if signal_name:
            self.signal_error.emit(signal_name, "Task cancelled")
            del self.signal_registry[signal_name]


class AsyncDataStream(QObject):
    """Async data streaming utilities for PyQt."""
    
    # Data stream signals
    data_received = pyqtSignal(str, object)  # stream_id, data
    stream_error = pyqtSignal(str, str)  # stream_id, error
    stream_completed = pyqtSignal(str)  # stream_id
    stream_started = pyqtSignal(str)  # stream_id
    
    def __init__(self, event_manager: AsyncEventLoopManager, parent=None):
        super().__init__(parent)
        self.event_manager = event_manager
        self.streams: Dict[str, asyncio.Queue] = {}
        self.stream_tasks: Dict[str, str] = {}  # stream_id -> task_id
        
        logger.info("AsyncDataStream initialized")
    
    async def create_stream(self, stream_id: str, max_size: int = 1000) -> asyncio.Queue:
        """Create a new async data stream."""
        if stream_id in self.streams:
            raise ValueError(f"Stream {stream_id} already exists")
        
        queue = asyncio.Queue(maxsize=max_size)
        self.streams[stream_id] = queue
        
        # Emit stream started signal
        QApplication.instance().postEvent(
            self,
            QEvent(QEvent.Type(QEvent.registerEventType())),
        )
        self.stream_started.emit(stream_id)
        
        return queue
    
    async def send_data(self, stream_id: str, data: Any):
        """Send data to a stream."""
        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        queue = self.streams[stream_id]
        await queue.put(data)
        
        # Emit data received signal
        self.data_received.emit(stream_id, data)
    
    async def receive_data(self, stream_id: str, timeout: Optional[float] = None) -> Any:
        """Receive data from a stream."""
        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        queue = self.streams[stream_id]
        
        try:
            if timeout:
                data = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                data = await queue.get()
            return data
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timeout receiving data from stream {stream_id}")
    
    def start_stream_processor(
        self,
        stream_id: str,
        processor_func: Callable[[Any], Any],
        error_handler: Optional[Callable] = None
    ) -> str:
        """Start processing data from a stream."""
        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        async def process_stream():
            try:
                queue = self.streams[stream_id]
                while True:
                    data = await queue.get()
                    if data is None:  # Sentinel value to stop
                        break
                    
                    try:
                        result = processor_func(data)
                        if result is not None:
                            self.data_received.emit(stream_id, result)
                    except Exception as e:
                        if error_handler:
                            error_handler(e)
                        else:
                            self.stream_error.emit(stream_id, str(e))
                            
            except Exception as e:
                self.stream_error.emit(stream_id, str(e))
            finally:
                self.stream_completed.emit(stream_id)
        
        # Submit processing task
        task_id = self.event_manager.submit_task(process_stream())
        self.stream_tasks[stream_id] = task_id
        
        return task_id
    
    def stop_stream(self, stream_id: str):
        """Stop a data stream."""
        if stream_id in self.stream_tasks:
            task_id = self.stream_tasks[stream_id]
            self.event_manager.cancel_task(task_id)
            del self.stream_tasks[stream_id]
        
        if stream_id in self.streams:
            # Send sentinel value to stop processing
            queue = self.streams[stream_id]
            try:
                asyncio.run_coroutine_threadsafe(queue.put(None), self.event_manager.loop)
            except:
                pass
            del self.streams[stream_id]
    
    def get_stream_stats(self, stream_id: str) -> Dict[str, Any]:
        """Get statistics for a stream."""
        if stream_id not in self.streams:
            return {}
        
        queue = self.streams[stream_id]
        return {
            "queue_size": queue.qsize(),
            "max_size": queue.maxsize,
            "is_processing": stream_id in self.stream_tasks
        }


class AsyncBatchProcessor(QObject):
    """Async batch processing utilities for PyQt."""
    
    # Batch processing signals
    batch_completed = pyqtSignal(str, list)  # batch_id, results
    batch_progress = pyqtSignal(str, int, int)  # batch_id, completed, total
    batch_error = pyqtSignal(str, str)  # batch_id, error
    
    def __init__(self, event_manager: AsyncEventLoopManager, parent=None):
        super().__init__(parent)
        self.event_manager = event_manager
        self.batches: Dict[str, Dict] = {}
        
        logger.info("AsyncBatchProcessor initialized")
    
    def process_batch(
        self,
        batch_id: str,
        items: List[Any],
        processor_func: Callable[[Any], Any],
        batch_size: int = 10,
        max_concurrent: int = 4
    ) -> str:
        """Process a batch of items asynchronously."""
        if batch_id in self.batches:
            raise ValueError(f"Batch {batch_id} already exists")
        
        # Store batch info
        self.batches[batch_id] = {
            "items": items,
            "total": len(items),
            "completed": 0,
            "results": [],
            "batch_size": batch_size,
            "max_concurrent": max_concurrent
        }
        
        # Submit batch processing task
        task_id = self.event_manager.submit_task(
            self._process_batch_async(batch_id, items, processor_func, batch_size, max_concurrent)
        )
        
        return task_id
    
    async def _process_batch_async(
        self,
        batch_id: str,
        items: List[Any],
        processor_func: Callable[[Any], Any],
        batch_size: int,
        max_concurrent: int
    ):
        """Process batch items asynchronously."""
        try:
            results = []
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_item(item):
                async with semaphore:
                    try:
                        if asyncio.iscoroutinefunction(processor_func):
                            result = await processor_func(item)
                        else:
                            # Run blocking function in thread pool
                            result = await self.event_manager.loop.run_in_executor(
                                self.event_manager.executor,
                                processor_func,
                                item
                            )
                        return result
                    except Exception as e:
                        logger.error(f"Error processing item: {e}")
                        return None
            
            # Process items in batches
            for i in range(0, len(items), batch_size):
                batch_items = items[i:i + batch_size]
                
                # Process batch concurrently
                batch_results = await asyncio.gather(
                    *[process_item(item) for item in batch_items],
                    return_exceptions=True
                )
                
                # Filter out exceptions and None results
                valid_results = [
                    result for result in batch_results
                    if not isinstance(result, Exception) and result is not None
                ]
                
                results.extend(valid_results)
                
                # Update progress
                completed = min(i + batch_size, len(items))
                self.batch_progress.emit(batch_id, completed, len(items))
                
                # Small delay to prevent blocking
                await asyncio.sleep(0.001)
            
            # Emit completion
            self.batch_completed.emit(batch_id, results)
            
        except Exception as e:
            self.batch_error.emit(batch_id, str(e))
        finally:
            # Clean up batch info
            if batch_id in self.batches:
                del self.batches[batch_id]
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a batch."""
        return self.batches.get(batch_id)


class AsyncCache(QObject):
    """Async cache with PyQt integration."""
    
    # Cache signals
    cache_hit = pyqtSignal(str, object)  # key, value
    cache_miss = pyqtSignal(str)  # key
    cache_updated = pyqtSignal(str, object)  # key, value
    
    def __init__(self, event_manager: AsyncEventLoopManager, ttl: float = 300.0, parent=None):
        super().__init__(parent)
        self.event_manager = event_manager
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        # Start cleanup task
        self.cleanup_task_id = self.event_manager.submit_task(self._cleanup_loop())
        
        logger.info("AsyncCache initialized")
    
    async def get(self, key: str, loader_func: Optional[Callable] = None) -> Any:
        """Get value from cache, loading if necessary."""
        now = datetime.now()
        
        # Check cache
        if key in self.cache:
            entry = self.cache[key]
            if (now - entry["created"]).total_seconds() < self.ttl:
                self.cache_hit.emit(key, entry["value"])
                return entry["value"]
            else:
                # Expired, remove
                del self.cache[key]
        
        # Cache miss
        self.cache_miss.emit(key)
        
        # Load value if loader provided
        if loader_func:
            if asyncio.iscoroutinefunction(loader_func):
                value = await loader_func(key)
            else:
                value = await self.event_manager.loop.run_in_executor(
                    self.event_manager.executor,
                    loader_func,
                    key
                )
            
            await self.set(key, value)
            return value
        
        return None
    
    async def set(self, key: str, value: Any):
        """Set value in cache."""
        self.cache[key] = {
            "value": value,
            "created": datetime.now()
        }
        self.cache_updated.emit(key, value)
    
    async def invalidate(self, key: str):
        """Invalidate cache entry."""
        if key in self.cache:
            del self.cache[key]
    
    async def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
    
    async def _cleanup_loop(self):
        """Background task to clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                now = datetime.now()
                expired_keys = [
                    key for key, entry in self.cache.items()
                    if (now - entry["created"]).total_seconds() >= self.ttl
                ]
                
                for key in expired_keys:
                    del self.cache[key]
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")


# Global instance management
_async_manager: Optional[AsyncEventLoopManager] = None
_signal_bridge: Optional[AsyncSignalBridge] = None
_data_stream: Optional[AsyncDataStream] = None
_batch_processor: Optional[AsyncBatchProcessor] = None
_cache: Optional[AsyncCache] = None


def get_async_manager() -> AsyncEventLoopManager:
    """Get global async event manager."""
    global _async_manager
    if _async_manager is None:
        _async_manager = AsyncEventLoopManager()
        _async_manager.start_event_loop()
    return _async_manager


def get_signal_bridge() -> AsyncSignalBridge:
    """Get global async signal bridge."""
    global _signal_bridge
    if _signal_bridge is None:
        _signal_bridge = AsyncSignalBridge(get_async_manager())
    return _signal_bridge


def get_data_stream() -> AsyncDataStream:
    """Get global async data stream."""
    global _data_stream
    if _data_stream is None:
        _data_stream = AsyncDataStream(get_async_manager())
    return _data_stream


def get_batch_processor() -> AsyncBatchProcessor:
    """Get global async batch processor."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = AsyncBatchProcessor(get_async_manager())
    return _batch_processor


def get_cache(ttl: float = 300.0) -> AsyncCache:
    """Get global async cache."""
    global _cache
    if _cache is None:
        _cache = AsyncCache(get_async_manager(), ttl)
    return _cache


def shutdown_async_system():
    """Shutdown the async system."""
    global _async_manager, _signal_bridge, _data_stream, _batch_processor, _cache
    
    if _async_manager:
        _async_manager.stop_event_loop()
        _async_manager = None
    
    _signal_bridge = None
    _data_stream = None
    _batch_processor = None
    _cache = None
    
    logger.info("Async system shutdown")


# Decorators for easy async integration
def async_slot(*args, **kwargs):
    """Decorator for async PyQt slots."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_async_manager()
            task_id = manager.submit_task(func(*args, **kwargs))
            return task_id
        return wrapper
    return decorator


def async_cached(ttl: float = 300.0):
    """Decorator for async cached functions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = get_cache(ttl)
            key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            result = await cache.get(key)
            if result is None:
                result = await func(*args, **kwargs)
                await cache.set(key, result)
            
            return result
        return wrapper
    return decorator
