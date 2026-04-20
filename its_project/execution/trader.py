from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from its_project.common.types import Order, OrderStatus, OrderType
from its_project.execution.order_manager import OrderManager

logger = logging.getLogger(__name__)


class PaperTrader:
    """Enhanced paper trading with realistic execution and partial fills."""
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission_rate: float = 0.001,
        slippage_model: str = "volume_impact",
        latency_simulation: bool = True,
        partial_fill_probability: float = 0.1,
        market_impact_model: bool = True
    ) -> None:
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        self.latency_simulation = latency_simulation
        self.partial_fill_probability = partial_fill_probability
        self.market_impact_model = market_impact_model
        
        # Account state
        self.balance = initial_balance
        self.positions: Dict[str, float] = {}  # symbol -> position size
        self.equity = initial_balance
        self.unrealized_pnl = 0.0
        
        # Order management
        self.order_manager = OrderManager()
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        
        # Market data simulation
        self.market_data: Dict[str, Dict[str, Any]] = {}
        self.order_book_depth: Dict[str, Dict[str, Any]] = {}
        
        # Execution statistics
        self.execution_stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "partial_fills": 0,
            "rejected_orders": 0,
            "total_commission": 0.0,
            "total_slippage": 0.0,
            "avg_fill_time": 0.0,
            "fill_rate": 0.0
        }
        
        # Latency tracking
        self.order_latencies: List[float] = []
    
    def update_market_data(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """Update market data for realistic execution simulation."""
        self.market_data[symbol] = market_data
        
        # Update order book if available
        if "orderbook" in market_data:
            self.order_book_depth[symbol] = market_data["orderbook"]
        
        # Update unrealized P&L
        self._update_unrealized_pnl()
    
    async def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Order:
        """
        Create order with realistic execution simulation.
        
        Args:
            symbol: Trading symbol
            order_type: Order type (market, limit)
            side: Order side (buy, sell)
            amount: Order amount
            price: Order price (for limit orders)
            params: Additional parameters
            
        Returns:
            Created order
        """
        # Validate order
        if not self._validate_order(symbol, side, amount, price):
            raise ValueError("Order validation failed")
        
        # Create order
        order = Order(
            id=self.order_manager.generate_order_id(),
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount,
            price=price,
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
            params=params or {}
        )
        
        # Store order
        self.active_orders[order.id] = order
        self.order_history.append(order)
        
        # Simulate order execution
        asyncio.create_task(self._execute_order(order))
        
        # Update statistics
        self.execution_stats["total_orders"] += 1
        
        logger.info(f"Created order {order.id}: {side} {amount} {symbol} @ {price}")
        
        return order
    
    async def _execute_order(self, order: Order) -> None:
        """Execute order with realistic simulation."""
        try:
            # Simulate latency
            if self.latency_simulation:
                latency = self._simulate_latency()
                await asyncio.sleep(latency)
                self.order_latencies.append(latency)
            
            # Update order status
            order.status = OrderStatus.OPEN
            order.updated_at = datetime.now()
            
            # Get market price
            market_price = self._get_market_price(order.symbol)
            if market_price is None:
                order.status = OrderStatus.REJECTED
                order.reason = "No market data available"
                self.execution_stats["rejected_orders"] += 1
                return
            
            # Execute based on order type
            if order.type == OrderType.MARKET:
                await self._execute_market_order(order, market_price)
            elif order.type == OrderType.LIMIT:
                await self._execute_limit_order(order, market_price)
            
        except Exception as e:
            logger.error(f"Order execution failed for {order.id}: {e}")
            order.status = OrderStatus.REJECTED
            order.reason = str(e)
            self.execution_stats["rejected_orders"] += 1
    
    async def _execute_market_order(self, order: Order, market_price: float) -> None:
        """Execute market order with partial fills."""
        # Calculate slippage
        slippage = self._calculate_slippage(order)
        execution_price = market_price + (slippage if order.side == "buy" else -slippage)
        
        # Determine fill amount (partial fills possible)
        fill_amount = self._calculate_fill_amount(order)
        
        if fill_amount <= 0:
            order.status = OrderStatus.REJECTED
            order.reason = "Insufficient liquidity"
            self.execution_stats["rejected_orders"] += 1
            return
        
        # Execute the fill
        await self._process_fill(order, fill_amount, execution_price)
        
        # Check if order is fully filled
        if abs(order.filled_amount - order.amount) < 1e-8:
            order.status = OrderStatus.FILLED
            self.execution_stats["filled_orders"] += 1
        else:
            # Partial fill - continue execution
            order.status = OrderStatus.PARTIALLY_FILLED
            self.execution_stats["partial_fills"] += 1
            
            # Continue with remaining amount
            remaining_amount = order.amount - order.filled_amount
            order.amount = remaining_amount
            
            # Simulate additional fills
            await asyncio.sleep(0.1)  # Small delay for partial fills
            await self._execute_market_order(order, market_price)
    
    async def _execute_limit_order(self, order: Order, market_price: float) -> None:
        """Execute limit order with price checking."""
        if order.price is None:
            order.status = OrderStatus.REJECTED
            order.reason = "Limit order requires price"
            self.execution_stats["rejected_orders"] += 1
            return
        
        # Check if limit price is crossed
        price_crossed = False
        if order.side == "buy" and market_price <= order.price:
            price_crossed = True
        elif order.side == "sell" and market_price >= order.price:
            price_crossed = True
        
        if not price_crossed:
            # Order stays open
            # In a real system, we'd monitor market data
            # For simulation, we'll check again after delay
            await asyncio.sleep(1.0)
            await self._execute_limit_order(order, market_price)
            return
        
        # Execute at limit price (or better)
        execution_price = order.price
        
        # Calculate slippage (usually less for limit orders)
        slippage = self._calculate_slippage(order) * 0.5  # Reduced slippage for limit orders
        if order.side == "buy":
            execution_price = min(execution_price, market_price + slippage)
        else:
            execution_price = max(execution_price, market_price - slippage)
        
        # Execute fill
        fill_amount = self._calculate_fill_amount(order)
        if fill_amount > 0:
            await self._process_fill(order, fill_amount, execution_price)
            order.status = OrderStatus.FILLED
            self.execution_stats["filled_orders"] += 1
        else:
            order.status = OrderStatus.REJECTED
            order.reason = "Insufficient liquidity"
            self.execution_stats["rejected_orders"] += 1
    
    async def _process_fill(self, order: Order, fill_amount: float, fill_price: float) -> None:
        """Process order fill and update positions."""
        # Calculate commission
        commission = fill_amount * fill_price * self.commission_rate
        
        # Calculate slippage cost
        market_price = self._get_market_price(order.symbol)
        slippage_cost = abs(fill_price - market_price) * fill_amount
        
        # Update order
        order.filled_amount += fill_amount
        order.average_price = (
            (order.average_price * (order.filled_amount - fill_amount) + fill_price * fill_amount) 
            / order.filled_amount
        ) if order.filled_amount > 0 else fill_price
        order.updated_at = datetime.now()
        
        # Update position
        current_position = self.positions.get(order.symbol, 0.0)
        if order.side == "buy":
            new_position = current_position + fill_amount
            cost = fill_amount * fill_price + commission
            self.balance -= cost
        else:
            new_position = current_position - fill_amount
            proceeds = fill_amount * fill_price - commission
            self.balance += proceeds
        
        self.positions[order.symbol] = new_position
        
        # Update statistics
        self.execution_stats["total_commission"] += commission
        self.execution_stats["total_slippage"] += slippage_cost
        
        # Update equity
        self._update_equity()
        
        logger.info(f"Processed fill: {fill_amount} {order.symbol} @ {fill_price}")
    
    def _validate_order(self, symbol: str, side: str, amount: float, price: Optional[float]) -> bool:
        """Validate order parameters."""
        # Check amount
        if amount <= 0:
            return False
        
        # Check balance for buy orders
        if side == "buy":
            required_balance = amount * (price or self._get_market_price(symbol) or 0)
            if self.balance < required_balance * 1.1:  # 10% buffer for commission/slippage
                return False
        
        # Check position for sell orders
        if side == "sell":
            current_position = self.positions.get(symbol, 0.0)
            if current_position < amount:
                return False
        
        return True
    
    def _get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol."""
        if symbol not in self.market_data:
            return None
        
        market_data = self.market_data[symbol]
        
        # Try different price sources
        if "price" in market_data:
            return float(market_data["price"])
        elif "last_price" in market_data:
            return float(market_data["last_price"])
        elif "close" in market_data:
            return float(market_data["close"])
        
        # Extract from order book
        if symbol in self.order_book_depth:
            order_book = self.order_book_depth[symbol]
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            
            if bids and asks:
                mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2
                return mid_price
        
        return None
    
    def _calculate_slippage(self, order: Order) -> float:
        """Calculate realistic slippage based on market conditions."""
        market_price = self._get_market_price(order.symbol)
        if market_price is None:
            return 0.0
        
        # Base slippage rate
        base_slippage_rate = 0.0001  # 0.01%
        
        # Size impact
        size_factor = (order.amount / 100.0) * 0.00001  # Scales with order size
        
        # Market impact (if enabled)
        market_impact = 0.0
        if self.market_impact_model and order.symbol in self.order_book_depth:
            order_book = self.order_book_depth[order.symbol]
            market_impact = self._calculate_market_impact(order, order_book)
        
        # Total slippage
        total_slippage_rate = base_slippage_rate + size_factor + market_impact
        
        return market_price * total_slippage_rate
    
    def _calculate_market_impact(self, order: Order, order_book: Dict[str, Any]) -> float:
        """Calculate market impact based on order book depth."""
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        
        if not bids or not asks:
            return 0.0
        
        # Calculate available volume at market
        market_price = self._get_market_price(order.symbol)
        if market_price is None:
            return 0.0
        
        available_volume = 0.0
        if order.side == "buy":
            # Sum volume from asks up to market price
            for ask_price, ask_volume in asks:
                if float(ask_price) <= market_price * 1.01:  # 1% tolerance
                    available_volume += float(ask_volume)
        else:
            # Sum volume from bids down to market price
            for bid_price, bid_volume in bids:
                if float(bid_price) >= market_price * 0.99:  # 1% tolerance
                    available_volume += float(bid_volume)
        
        # Calculate impact
        if available_volume > 0:
            volume_ratio = order.amount / available_volume
            impact_rate = 0.0001 * (volume_ratio ** 0.5)  # Square root impact function
            return impact_rate
        
        return 0.0005  # Default impact if no depth available
    
    def _calculate_fill_amount(self, order: Order) -> float:
        """Calculate fill amount (partial fills possible)."""
        # Check for partial fill
        if np.random.random() < self.partial_fill_probability:
            # Partial fill - fill 50-90% of order
            fill_ratio = np.random.uniform(0.5, 0.9)
            return order.amount * fill_ratio
        
        # Full fill
        return order.amount
    
    def _simulate_latency(self) -> float:
        """Simulate realistic order execution latency."""
        # Base latency in seconds
        base_latency = 0.01  # 10ms
        
        # Add random component
        random_latency = np.random.exponential(0.02)  # Exponential distribution
        
        # Add market impact latency
        market_latency = 0.005 if self.market_impact_model else 0.0
        
        total_latency = base_latency + random_latency + market_latency
        
        # Cap at reasonable maximum
        return min(total_latency, 0.5)  # Max 500ms
    
    def _update_unrealized_pnl(self) -> None:
        """Update unrealized P&L for all positions."""
        total_unrealized = 0.0
        
        for symbol, position_size in self.positions.items():
            if position_size == 0:
                continue
            
            market_price = self._get_market_price(symbol)
            if market_price is None:
                continue
            
            # This is simplified - in practice, we'd track entry prices
            # For now, assume break-even
            unrealized = 0.0
            total_unrealized += unrealized
        
        self.unrealized_pnl = total_unrealized
    
    def _update_equity(self) -> None:
        """Update total equity."""
        self.equity = self.balance + self.unrealized_pnl
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        if order_id not in self.active_orders:
            return False
        
        order = self.active_orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return False
        
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        
        # Remove from active orders
        del self.active_orders[order_id]
        
        logger.info(f"Cancelled order {order_id}")
        return True
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get comprehensive account summary."""
        # Update statistics
        if self.execution_stats["total_orders"] > 0:
            self.execution_stats["fill_rate"] = (
                self.execution_stats["filled_orders"] / self.execution_stats["total_orders"]
            )
        
        if self.order_latencies:
            self.execution_stats["avg_fill_time"] = np.mean(self.order_latencies)
        
        return {
            "balance": self.balance,
            "equity": self.equity,
            "unrealized_pnl": self.unrealized_pnl,
            "total_return": (self.equity - self.initial_balance) / self.initial_balance,
            "positions": self.positions.copy(),
            "active_orders": len(self.active_orders),
            "execution_stats": self.execution_stats.copy(),
            "order_history_count": len(self.order_history)
        }
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of a specific order."""
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        
        # Check in history
        for order in self.order_history:
            if order.id == order_id:
                return order
        
        return None
    
    def reset(self) -> None:
        """Reset trader to initial state."""
        self.balance = self.initial_balance
        self.positions.clear()
        self.equity = self.initial_balance
        self.unrealized_pnl = 0.0
        
        self.active_orders.clear()
        self.order_history.clear()
        
        self.execution_stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "partial_fills": 0,
            "rejected_orders": 0,
            "total_commission": 0.0,
            "total_slippage": 0.0,
            "avg_fill_time": 0.0,
            "fill_rate": 0.0
        }
        
        self.order_latencies.clear()
        
        logger.info("Paper trader reset to initial state")
