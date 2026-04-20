from __future__ import annotations

from typing import Dict, Any, List

import ccxt.async_support as ccxt

from its_project.execution.base import BaseExecutor, Order, OrderType, OrderStatus, Position


class LiveExecutor(BaseExecutor):
    """
    Live trading executor using ccxt exchange integration.
    
    WARNING: Use only after extensive paper trading validation!
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        # Initialize exchange
        self.exchange_id = config.get("exchange", "binance")
        self.api_key = config.get("api_key")
        self.api_secret = config.get("api_secret")
        self.sandbox = config.get("sandbox", True)  # Use testnet by default
        
        # Initialize ccxt exchange
        self.exchange = getattr(ccxt, self.exchange_id)({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "sandbox": self.sandbox,
            "enableRateLimit": True,
        })
        
        # Safety settings
        self.max_order_size = config.get("max_order_size", 1.0)
        self.daily_loss_limit = config.get("daily_loss_limit", 100.0)

    async def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Order:
        """Create real order on exchange."""
        
        # Safety checks
        if amount > self.max_order_size:
            raise ValueError(f"Order size {amount} exceeds maximum {self.max_order_size}")
        
        # Convert order type
        ccxt_type = self._convert_order_type(order_type)
        
        try:
            # Create order on exchange
            response = await self.exchange.create_order(
                symbol=symbol,
                type=ccxt_type,
                side=side,
                amount=amount,
                price=price,
                params=params or {},
            )
            
            # Convert to our Order format
            order = self._convert_ccxt_order(response)
            self.active_orders[order.id] = order
            
            return order
            
        except Exception as e:
            # Log error and re-raise
            raise RuntimeError(f"Failed to create order: {e}")

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on exchange."""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            
            if order_id in self.active_orders:
                self.active_orders[order_id].status = OrderStatus.CANCELLED
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to cancel order {order_id}: {e}")

    async def fetch_order_status(self, order_id: str, symbol: str) -> Order:
        """Get order status from exchange."""
        try:
            response = await self.exchange.fetch_order(order_id, symbol)
            order = self._convert_ccxt_order(response)
            
            # Update local cache
            self.active_orders[order_id] = order
            
            return order
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch order status {order_id}: {e}")

    async def fetch_balance(self) -> Dict[str, float]:
        """Get balance from exchange."""
        try:
            response = await self.exchange.fetch_balance()
            
            # Extract free balances
            balances = {}
            for currency, balance in response["free"].items():
                if balance > 0:
                    balances[currency] = float(balance)
            
            return balances
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch balance: {e}")

    async def fetch_positions(self) -> List[Position]:
        """Get positions from exchange."""
        try:
            response = await self.exchange.fetch_positions()
            
            positions = []
            for pos in response:
                if float(pos["contracts"]) > 0:  # Only open positions
                    position = Position(
                        symbol=pos["symbol"],
                        side="long" if pos["side"] == "long" else "short",
                        size=float(pos["contracts"]),
                        entry_price=float(pos["entryPrice"]),
                        current_price=float(pos["markPrice"]),
                        unrealized_pnl=float(pos["unrealizedPnl"]),
                        timestamp=int(pos["timestamp"]) if pos.get("timestamp") else 0,
                    )
                    positions.append(position)
            
            return positions
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch positions: {e}")

    def is_live(self) -> bool:
        """This is live trading."""
        return True

    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert our OrderType to ccxt type."""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_LOSS: "stop",
            OrderType.TAKE_PROFIT: "take_profit",
        }
        return mapping.get(order_type, "market")

    def _convert_ccxt_order(self, ccxt_order: Dict[str, Any]) -> Order:
        """Convert ccxt order to our Order format."""
        return Order(
            id=ccxt_order["id"],
            symbol=ccxt_order["symbol"],
            type=OrderType(ccxt_order["type"]),
            side=ccxt_order["side"],
            amount=float(ccxt_order["amount"]),
            price=float(ccxt_order["price"]) if ccxt_order["price"] else None,
            status=OrderStatus(ccxt_order["status"]),
            filled=float(ccxt_order["filled"]),
            remaining=float(ccxt_order["remaining"]),
            timestamp=int(ccxt_order["timestamp"]),
            info=ccxt_order.get("info", {}),
        )
