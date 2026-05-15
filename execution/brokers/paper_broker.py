"""
Paper broker for simulation trading.

Simulates trading execution without real money, using live quotes for realistic simulation.
Matches Backtester assumptions (0.06% commission, 0.02% slippage).
"""

import time
import uuid
from typing import Dict, Any, Optional
import pandas as pd

from .base_broker import (
    BaseBroker, Order, OrderType, OrderSide, OrderStatus,
    Position, AccountInfo
)


class PaperBroker(BaseBroker):
    """
    Paper trading broker for simulation.
    
    Simulates order execution with realistic slippage and commission.
    Maintains virtual portfolio and positions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize paper broker.
        
        Args:
            config: Broker configuration with initial_balance
        """
        super().__init__(config)
        self.initial_balance = config.get('initial_balance', 10000.0)
        self.balance = self.initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
        
        # Price feed (simulated market data)
        self.current_prices: Dict[str, float] = {}
    
    def connect(self) -> bool:
        """
        Connect to paper trading environment.
        
        Returns:
            True (always successful for paper trading)
        """
        return True
    
    def disconnect(self) -> bool:
        """
        Disconnect from paper trading environment.
        
        Returns:
            True (always successful for paper trading)
        """
        return True
    
    def update_price(self, symbol: str, price: float):
        """
        Update current price for a symbol.
        
        Args:
            symbol: Trading symbol
            price: Current market price
        """
        self.current_prices[symbol] = price
        self._update_positions_pnl()
    
    def place_order(self, order: Order) -> Order:
        """
        Place an order (simulated execution).
        
        For market orders, executes immediately with slippage and commission.
        For limit orders, checks if price is favorable and executes if so.
        
        Args:
            order: Order to place
            
        Returns:
            Order with updated status
        """
        if not self.validate_order(order):
            order.status = OrderStatus.REJECTED
            order.error_message = "Invalid order"
            return order
        
        # Generate order ID if not provided
        if order.order_id is None:
            order.order_id = f"paper_{self.order_counter}_{int(time.time())}"
            self.order_counter += 1
        
        order.timestamp = time.time()
        
        # Get current price
        current_price = self.current_prices.get(order.symbol)
        if current_price is None:
            order.status = OrderStatus.REJECTED
            order.error_message = "No price data available"
            return order
        
        # Execute based on order type
        if order.order_type == OrderType.MARKET:
            return self._execute_market_order(order, current_price)
        elif order.order_type == OrderType.LIMIT:
            return self._execute_limit_order(order, current_price)
        else:
            order.status = OrderStatus.REJECTED
            order.error_message = f"Order type {order.order_type} not implemented yet"
            return order
    
    def _execute_market_order(self, order: Order, current_price: float) -> Order:
        """
        Execute market order with slippage and commission.
        
        Args:
            order: Order to execute
            current_price: Current market price
            
        Returns:
            Executed order
        """
        # Apply slippage
        execution_price = self.calculate_slippage(current_price, order.side)
        
        # Calculate commission
        commission = self.calculate_commission(order.quantity, execution_price)
        
        # Check if sufficient balance
        if order.side == OrderSide.BUY:
            required = order.quantity * execution_price + commission
            if required > self.balance:
                order.status = OrderStatus.REJECTED
                order.error_message = "Insufficient balance"
                return order
        else:
            # For sell, check if we have position
            position = self.positions.get(order.symbol)
            if position is None or position.quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                order.error_message = "Insufficient position"
                return order
        
        # Update balance and positions
        if order.side == OrderSide.BUY:
            self.balance -= required
            self._update_position(order.symbol, order.quantity, execution_price, order.side)
        else:
            proceeds = order.quantity * execution_price - commission
            self.balance += proceeds
            self._update_position(order.symbol, order.quantity, execution_price, order.side)
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.fees = commission
        
        # Store order
        self.orders[order.order_id] = order
        
        return order
    
    def _execute_limit_order(self, order: Order, current_price: float) -> Order:
        """
        Execute limit order if price is favorable.
        
        Args:
            order: Order to execute
            current_price: Current market price
            
        Returns:
            Order with updated status
        """
        if order.price is None:
            order.status = OrderStatus.REJECTED
            order.error_message = "Limit price not specified"
            return order
        
        # Check if limit order can be filled
        can_fill = False
        if order.side == OrderSide.BUY and current_price <= order.price:
            can_fill = True
        elif order.side == OrderSide.SELL and current_price >= order.price:
            can_fill = True
        
        if can_fill:
            # Execute as market order at limit price
            execution_price = order.price
            commission = self.calculate_commission(order.quantity, execution_price)
            
            # Check balance/position
            if order.side == OrderSide.BUY:
                required = order.quantity * execution_price + commission
                if required > self.balance:
                    order.status = OrderStatus.REJECTED
                    order.error_message = "Insufficient balance"
                    return order
            else:
                position = self.positions.get(order.symbol)
                if position is None or position.quantity < order.quantity:
                    order.status = OrderStatus.REJECTED
                    order.error_message = "Insufficient position"
                    return order
            
            # Update balance and positions
            if order.side == OrderSide.BUY:
                self.balance -= required
                self._update_position(order.symbol, order.quantity, execution_price, order.side)
            else:
                proceeds = order.quantity * execution_price - commission
                self.balance += proceeds
                self._update_position(order.symbol, order.quantity, execution_price, order.side)
            
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = execution_price
            order.fees = commission
        else:
            # Order remains open
            order.status = OrderStatus.OPEN
        
        self.orders[order.order_id] = order
        return order
    
    def _update_position(self, symbol: str, quantity: float, price: float, side: OrderSide):
        """
        Update position after trade execution.
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Trade price
            side: Order side
        """
        current_position = self.positions.get(symbol)
        
        if current_position is None:
            if side == OrderSide.BUY:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    current_price=price,
                    unrealized_pnl=0.0
                )
        else:
            if side == OrderSide.BUY:
                # Add to position
                total_cost = current_position.quantity * current_position.entry_price + quantity * price
                total_quantity = current_position.quantity + quantity
                current_position.entry_price = total_cost / total_quantity
                current_position.quantity = total_quantity
                current_position.current_price = price
            else:
                # Reduce position
                current_position.quantity -= quantity
                if current_position.quantity <= 0:
                    # Position closed
                    realized_pnl = (price - current_position.entry_price) * quantity
                    current_position.realized_pnl += realized_pnl
                    del self.positions[symbol]
                else:
                    current_position.current_price = price
    
    def _update_positions_pnl(self):
        """Update unrealized PnL for all positions."""
        for symbol, position in self.positions.items():
            current_price = self.current_prices.get(symbol, position.current_price)
            position.current_price = current_price
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancelled, False otherwise
        """
        order = self.orders.get(order_id)
        if order and order.status == OrderStatus.OPEN:
            order.status = OrderStatus.CANCELLED
            return True
        return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status
        """
        order = self.orders.get(order_id)
        return order.status if order else OrderStatus.REJECTED
    
    def get_account_info(self) -> AccountInfo:
        """
        Get account information.
        
        Returns:
            AccountInfo with balance and positions
        """
        self._update_positions_pnl()
        
        return AccountInfo(
            balance=self.balance,
            available_balance=self.balance,
            positions=self.positions.copy()
        )
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position if exists, None otherwise
        """
        self._update_positions_pnl()
        return self.positions.get(symbol)
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price
        """
        return self.current_prices.get(symbol, 0.0)
    
    def reset(self):
        """Reset broker to initial state."""
        self.balance = self.initial_balance
        self.positions.clear()
        self.orders.clear()
        self.current_prices.clear()
        self.order_counter = 0
