from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from its_project.common.types import Order, OrderStatus, OrderType

logger = logging.getLogger(__name__)


class OrderManager:
    """Advanced order management with execution control and monitoring."""
    
    def __init__(
        self,
        max_concurrent_orders: int = 50,
        order_timeout: timedelta = timedelta(minutes=5),
        retry_attempts: int = 3,
        retry_delay: timedelta = timedelta(seconds=1)
    ) -> None:
        self.max_concurrent_orders = max_concurrent_orders
        self.order_timeout = order_timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.order_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        
        # Execution control
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_orders)
        self._running = False
        self._execution_task: Optional[asyncio.Task] = None
        
        # Monitoring
        self.order_stats = {
            "total_created": 0,
            "total_filled": 0,
            "total_cancelled": 0,
            "total_rejected": 0,
            "avg_execution_time": 0.0,
            "success_rate": 0.0
        }
        
        # Order callbacks
        self._order_callbacks: Dict[str, List[callable]] = {}
    
    def generate_order_id(self) -> str:
        """Generate unique order ID."""
        return str(uuid.uuid4())
    
    async def submit_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
        callback: Optional[callable] = None
    ) -> str:
        """
        Submit order for execution with queue management.
        
        Args:
            symbol: Trading symbol
            order_type: Order type
            side: Order side
            amount: Order amount
            price: Order price
            params: Additional parameters
            callback: Callback function for order updates
            
        Returns:
            Order ID
        """
        # Create order
        order = Order(
            id=self.generate_order_id(),
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount,
            price=price,
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
            params=params or {}
        )
        
        # Register callback
        if callback:
            self._order_callbacks[order.id] = [callback]
        
        # Add to queue
        try:
            await self.order_queue.put(order)
            logger.info(f"Order {order.id} queued for execution")
        except asyncio.QueueFull:
            raise RuntimeError("Order queue is full")
        
        return order.id
    
    async def start(self) -> None:
        """Start order manager execution."""
        if self._running:
            return
        
        self._running = True
        self._execution_task = asyncio.create_task(self._execution_loop())
        logger.info("Order manager started")
    
    async def stop(self) -> None:
        """Stop order manager execution."""
        if not self._running:
            return
        
        self._running = False
        
        if self._execution_task:
            self._execution_task.cancel()
            try:
                await self._execution_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Order manager stopped")
    
    async def _execution_loop(self) -> None:
        """Main execution loop for processing orders."""
        while self._running:
            try:
                # Get order from queue
                order = await asyncio.wait_for(
                    self.order_queue.get(),
                    timeout=1.0
                )
                
                # Execute order with semaphore control
                async with self.execution_semaphore:
                    await self._execute_order_with_retry(order)
                
            except asyncio.TimeoutError:
                # No orders in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                continue
    
    async def _execute_order_with_retry(self, order: Order) -> None:
        """Execute order with retry mechanism."""
        attempt = 0
        
        while attempt < self.retry_attempts:
            try:
                # Add to active orders
                self.active_orders[order.id] = order
                
                # Update statistics
                self.order_stats["total_created"] += 1
                
                # Execute order (this would be implemented by the specific exchange)
                success = await self._execute_order(order)
                
                if success:
                    # Order executed successfully
                    self.order_stats["total_filled"] += 1
                    await self._notify_callbacks(order)
                    break
                else:
                    # Execution failed, check if we should retry
                    attempt += 1
                    if attempt < self.retry_attempts:
                        await asyncio.sleep(self.retry_delay.total_seconds() * attempt)
                    else:
                        # Max retries reached, reject order
                        order.status = OrderStatus.REJECTED
                        order.reason = "Max retry attempts reached"
                        self.order_stats["total_rejected"] += 1
                        await self._notify_callbacks(order)
                        
            except Exception as e:
                logger.error(f"Order execution error for {order.id}: {e}")
                attempt += 1
                
                if attempt >= self.retry_attempts:
                    order.status = OrderStatus.REJECTED
                    order.reason = str(e)
                    self.order_stats["total_rejected"] += 1
                    await self._notify_callbacks(order)
        
        # Remove from active orders
        self.active_orders.pop(order.id, None)
        
        # Add to history
        self.order_history.append(order)
        
        # Update statistics
        self._update_statistics()
    
    async def _execute_order(self, order: Order) -> bool:
        """
        Execute order (to be implemented by specific exchange).
        This is a placeholder implementation.
        """
        # Simulate execution time
        await asyncio.sleep(0.1)
        
        # Simulate execution success/failure
        success_rate = 0.95  # 95% success rate
        
        if np.random.random() < success_rate:
            # Simulate partial fill
            fill_amount = order.amount * np.random.uniform(0.5, 1.0)
            order.filled_amount = fill_amount
            order.average_price = order.price or 100.0  # Placeholder
            
            if abs(fill_amount - order.amount) < 1e-8:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
            
            order.updated_at = datetime.now()
            return True
        else:
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        # Check if order is active
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                return False
            
            # Cancel order
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            
            # Remove from active orders
            self.active_orders.pop(order_id, None)
            
            # Update statistics
            self.order_stats["total_cancelled"] += 1
            self._update_statistics()
            
            # Notify callbacks
            await self._notify_callbacks(order)
            
            logger.info(f"Order {order_id} cancelled")
            return True
        
        return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all orders, optionally filtered by symbol."""
        orders_to_cancel = []
        
        for order_id, order in list(self.active_orders.items()):
            if symbol is None or order.symbol == symbol:
                orders_to_cancel.append(order_id)
        
        cancelled_count = 0
        for order_id in orders_to_cancel:
            if await self.cancel_order(order_id):
                cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} orders")
        return cancelled_count
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        # Check active orders first
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        
        # Check history
        for order in reversed(self.order_history):  # Check recent orders first
            if order.id == order_id:
                return order
        
        return None
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get active orders, optionally filtered by symbol."""
        orders = list(self.active_orders.values())
        
        if symbol:
            orders = [order for order in orders if order.symbol == symbol]
        
        return orders
    
    def get_order_history(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        limit: Optional[int] = None
    ) -> List[Order]:
        """Get order history with filters."""
        orders = self.order_history.copy()
        
        # Apply filters
        if symbol:
            orders = [order for order in orders if order.symbol == symbol]
        
        if status:
            orders = [order for order in orders if order.status == status]
        
        # Sort by creation time (newest first)
        orders.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply limit
        if limit:
            orders = orders[:limit]
        
        return orders
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get order execution statistics."""
        return self.order_stats.copy()
    
    def _update_statistics(self) -> None:
        """Update execution statistics."""
        total_orders = self.order_stats["total_created"]
        
        if total_orders > 0:
            self.order_stats["success_rate"] = (
                self.order_stats["total_filled"] / total_orders
            )
        
        # Calculate average execution time
        execution_times = []
        for order in self.order_history:
            if order.updated_at and order.created_at:
                exec_time = (order.updated_at - order.created_at).total_seconds()
                execution_times.append(exec_time)
        
        if execution_times:
            self.order_stats["avg_execution_time"] = np.mean(execution_times)
    
    async def _notify_callbacks(self, order: Order) -> None:
        """Notify registered callbacks of order updates."""
        callbacks = self._order_callbacks.get(order.id, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order)
                else:
                    callback(order)
            except Exception as e:
                logger.error(f"Callback error for order {order.id}: {e}")
        
        # Remove callbacks for completed orders
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            self._order_callbacks.pop(order.id, None)
    
    def add_order_callback(self, order_id: str, callback: callable) -> None:
        """Add callback for order updates."""
        if order_id not in self._order_callbacks:
            self._order_callbacks[order_id] = []
        self._order_callbacks[order_id].append(callback)
    
    def remove_order_callback(self, order_id: str, callback: callable) -> None:
        """Remove callback for order updates."""
        if order_id in self._order_callbacks:
            try:
                self._order_callbacks[order_id].remove(callback)
                if not self._order_callbacks[order_id]:
                    del self._order_callbacks[order_id]
            except ValueError:
                pass
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status information."""
        return {
            "queue_size": self.order_queue.qsize(),
            "max_queue_size": self.order_queue.maxsize,
            "active_orders": len(self.active_orders),
            "max_concurrent_orders": self.max_concurrent_orders,
            "semaphore_available": self.execution_semaphore._value,
            "is_running": self._running
        }
    
    async def cleanup_stale_orders(self, max_age: timedelta = timedelta(hours=1)) -> int:
        """Clean up stale orders."""
        now = datetime.now()
        stale_orders = []
        
        for order_id, order in list(self.active_orders.items()):
            age = now - order.created_at
            if age > max_age and order.status in [OrderStatus.PENDING, OrderStatus.OPEN]:
                stale_orders.append(order_id)
        
        for order_id in stale_orders:
            order = self.active_orders[order_id]
            order.status = OrderStatus.CANCELLED
            order.reason = "Order timeout"
            self.order_stats["total_cancelled"] += 1
            
            self.active_orders.pop(order_id, None)
            self.order_history.append(order)
            
            await self._notify_callbacks(order)
        
        if stale_orders:
            logger.info(f"Cleaned up {len(stale_orders)} stale orders")
        
        return len(stale_orders)
