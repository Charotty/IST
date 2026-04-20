from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(Enum):
    """Order statuses."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order on exchange."""
    id: str
    symbol: str
    type: OrderType
    side: str  # 'buy' or 'sell'
    amount: float
    price: Optional[float]  # None for market orders
    status: OrderStatus
    filled: float  # How much filled
    remaining: float  # How much remaining
    timestamp: int
    info: Dict[str, Any]  # Additional info from exchange


@dataclass
class Position:
    """Open position."""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: int


class BaseExecutor(ABC):
    """Base class for order execution."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.active_orders: Dict[str, Order] = {}

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Order:
        """
        Create order.

        Args:
            symbol: trading pair (BTC/USDT)
            order_type: order type
            side: buy/sell
            amount: quantity
            price: price (for limit orders)
            params: additional parameters

        Returns:
            Order object
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order."""
        pass

    @abstractmethod
    async def fetch_order_status(self, order_id: str, symbol: str) -> Order:
        """Get order status."""
        pass

    @abstractmethod
    async def fetch_balance(self) -> Dict[str, float]:
        """Get balance."""
        pass

    @abstractmethod
    async def fetch_positions(self) -> List[Position]:
        """Get open positions."""
        pass

    def is_live(self) -> bool:
        """Is this live trading or paper?"""
        return False
