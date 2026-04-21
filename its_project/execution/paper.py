from __future__ import annotations

import asyncio
import time
from typing import Dict, Any, List

try:
    from its_project.execution.base import BaseExecutor, Order, OrderType, OrderStatus, Position
except ImportError:
    from .base import BaseExecutor, Order, OrderType, OrderStatus, Position


class PaperTradingExecutor(BaseExecutor):
    """
    Paper trading simulation without real money.
    
    Used for:
    - Real-time strategy testing
    - Exchange integration verification
    - Execution logic debugging
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        # Virtual balance
        self.balances = {
            "USDT": config.get("initial_balance", 10000.0),
            "BTC": 0.0,
            "ETH": 0.0,
        }
        
        # Order counter
        self.order_counter = 0
        
        # Virtual positions
        self.positions: List[Position] = []
        
        # Simulation latency
        self.latency = config.get("latency_ms", 50) / 1000  # seconds
        
        # Current prices (simplified - should come from market data)
        self.current_prices = {
            "BTC/USDT": 42000.0,
            "ETH/USDT": 2500.0,
        }

    async def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Order:
        """Create virtual order."""
        
        # Simulate network latency
        await asyncio.sleep(self.latency)
        
        # Generate ID
        self.order_counter += 1
        order_id = f"paper_{self.order_counter}"
        
        # Get current price
        current_price = await self._fetch_current_price(symbol)
        
        # Determine execution price
        if order_type == OrderType.MARKET:
            execution_price = current_price
        else:
            execution_price = price or current_price
        
        # Create order
        order = Order(
            id=order_id,
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount,
            price=execution_price,
            status=OrderStatus.PENDING,
            filled=0.0,
            remaining=amount,
            timestamp=int(time.time() * 1000),
            info={},
        )
        
        # Immediately fill market orders
        if order_type == OrderType.MARKET:
            await self._fill_order(order, execution_price)
        
        self.active_orders[order_id] = order
        
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel virtual order."""
        await asyncio.sleep(self.latency)
        
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            if order.status in {OrderStatus.PENDING, OrderStatus.OPEN}:
                order.status = OrderStatus.CANCELLED
                return True
        return False

    async def fetch_order_status(self, order_id: str, symbol: str) -> Order:
        """Get order status."""
        await asyncio.sleep(self.latency)
        
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        
        # Return dummy order if not found
        return Order(
            id=order_id,
            symbol=symbol,
            type=OrderType.MARKET,
            side="buy",
            amount=0.0,
            price=0.0,
            status=OrderStatus.REJECTED,
            filled=0.0,
            remaining=0.0,
            timestamp=int(time.time() * 1000),
            info={"error": "Order not found"},
        )

    async def fetch_balance(self) -> Dict[str, float]:
        """Get virtual balance."""
        await asyncio.sleep(self.latency)
        return self.balances.copy()

    async def fetch_positions(self) -> List[Position]:
        """Get virtual positions."""
        await asyncio.sleep(self.latency)
        return self.positions.copy()

    def is_live(self) -> bool:
        """This is paper trading."""
        return False

    async def _fetch_current_price(self, symbol: str) -> float:
        """Get current price (simplified)."""
        # In real implementation, this would come from market data
        return self.current_prices.get(symbol, 0.0)

    async def _fill_order(self, order: Order, price: float) -> None:
        """Fill order."""
        base, quote = order.symbol.split("/")
        
        if order.side == "buy":
            # Buy base currency with quote
            cost = order.amount * price
            
            if self.balances.get(quote, 0) < cost:
                order.status = OrderStatus.REJECTED
                return
            
            self.balances[quote] -= cost
            self.balances[base] = self.balances.get(base, 0) + order.amount
            
        else:  # sell
            # Sell base currency for quote
            if self.balances.get(base, 0) < order.amount:
                order.status = OrderStatus.REJECTED
                return
            
            self.balances[base] -= order.amount
            self.balances[quote] = self.balances.get(quote, 0) + (order.amount * price)
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled = order.amount
        order.remaining = 0.0
        
        # Update position
        await self._update_position(order, price)

    async def _update_position(self, order: Order, price: float) -> None:
        """Update virtual position."""
        # Simplified position tracking
        # In real implementation, this would be more sophisticated
        pass
