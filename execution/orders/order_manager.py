"""
Order manager for execution layer.

Manages order lifecycle, tracking, and reconciliation.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..brokers.base_broker import (
    BaseBroker, Order, OrderType, OrderSide, OrderStatus
)


@dataclass
class OrderRequest:
    """Order request data structure."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderManager:
    """
    Manages order lifecycle and tracking.
    
    Handles order submission, status updates, and reconciliation.
    Maintains order history and provides query capabilities.
    """
    
    def __init__(self, broker: BaseBroker):
        """
        Initialize order manager.
        
        Args:
            broker: Broker instance for order execution
        """
        self.broker = broker
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.client_order_counter = 0
    
    def generate_client_order_id(self) -> str:
        """
        Generate unique client order ID.
        
        Returns:
            Client order ID
        """
        self.client_order_counter += 1
        timestamp = int(time.time())
        return f"client_{self.client_order_counter}_{timestamp}"
    
    def submit_order(self, request: OrderRequest) -> Order:
        """
        Submit order to broker.
        
        Args:
            request: Order request
            
        Returns:
            Order with broker response
        """
        # Generate client order ID if not provided
        if request.client_order_id is None:
            request.client_order_id = self.generate_client_order_id()
        
        # Create order object
        order = Order(
            order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            price=request.price,
            stop_price=request.stop_price,
            status=OrderStatus.PENDING,
            timestamp=time.time()
        )
        
        # Submit to broker
        try:
            result_order = self.broker.place_order(order)
            
            # Store order
            self.orders[result_order.order_id] = result_order
            self.order_history.append(result_order)
            
            return result_order
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            self.orders[order.order_id] = order
            self.order_history.append(order)
            return order
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        order = self.orders.get(order_id)
        if not order:
            return False
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return False
        
        success = self.broker.cancel_order(order_id)
        
        if success:
            order.status = OrderStatus.CANCELLED
        
        return success
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order if found, None otherwise
        """
        # Update status from broker
        if order_id in self.orders:
            current_status = self.broker.get_order_status(order_id)
            self.orders[order_id].status = current_status
        
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """
        Get all orders for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of orders
        """
        return [order for order in self.orders.values() if order.symbol == symbol]
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """
        Get all orders with specific status.
        
        Args:
            status: Order status
            
        Returns:
            List of orders
        """
        return [order for order in self.orders.values() if order.status == status]
    
    def get_open_orders(self) -> List[Order]:
        """
        Get all open orders.
        
        Returns:
            List of open orders
        """
        return self.get_orders_by_status(OrderStatus.OPEN)
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            Number of orders cancelled
        """
        open_orders = self.get_open_orders()
        
        if symbol:
            open_orders = [order for order in open_orders if order.symbol == symbol]
        
        cancelled_count = 0
        for order in open_orders:
            if self.cancel_order(order.order_id):
                cancelled_count += 1
        
        return cancelled_count
    
    def update_order_status(self, order_id: str) -> OrderStatus:
        """
        Update order status from broker.
        
        Args:
            order_id: Order ID
            
        Returns:
            Updated order status
        """
        order = self.orders.get(order_id)
        if not order:
            return OrderStatus.REJECTED
        
        new_status = self.broker.get_order_status(order_id)
        order.status = new_status
        
        return new_status
    
    def get_order_history(self, limit: Optional[int] = None) -> List[Order]:
        """
        Get order history.
        
        Args:
            limit: Optional limit on number of orders
            
        Returns:
            List of orders
        """
        if limit:
            return self.order_history[-limit:]
        return self.order_history
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """
        Get order statistics.
        
        Returns:
            Dictionary with order statistics
        """
        total_orders = len(self.order_history)
        
        status_counts = {}
        for order in self.order_history:
            status = order.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        filled_orders = self.get_orders_by_status(OrderStatus.FILLED)
        total_fees = sum(order.fees for order in filled_orders)
        
        return {
            'total_orders': total_orders,
            'status_counts': status_counts,
            'filled_orders': len(filled_orders),
            'total_fees': total_fees,
            'fill_rate': len(filled_orders) / total_orders if total_orders > 0 else 0
        }
    
    def reconcile_orders(self) -> List[str]:
        """
        Reconcile orders with broker.
        
        Updates status of all tracked orders from broker.
        
        Returns:
            List of order IDs with status changes
        """
        changed_orders = []
        
        for order_id in self.orders:
            old_status = self.orders[order_id].status
            new_status = self.update_order_status(order_id)
            
            if old_status != new_status:
                changed_orders.append(order_id)
        
        return changed_orders
