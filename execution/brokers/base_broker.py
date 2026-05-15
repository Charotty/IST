"""
Base broker interface for execution layer.

Defines the abstract interface that all broker implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class OrderType(Enum):
    """Order types supported by brokers."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order data structure."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    timestamp: Optional[float] = None
    fees: float = 0.0
    error_message: Optional[str] = None


@dataclass
class Position:
    """Position data structure."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0


@dataclass
class AccountInfo:
    """Account information."""
    balance: float
    available_balance: float
    positions: Dict[str, Position]


class BaseBroker(ABC):
    """
    Abstract base class for broker implementations.
    
    All broker implementations (PaperBroker, OKXBroker) must inherit from this class
    and implement the abstract methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize broker with configuration.
        
        Args:
            config: Broker configuration dictionary
        """
        self.config = config
        self.sandbox = config.get('sandbox', True)
        self.commission_rate = config.get('commission_rate', 0.0006)  # 0.06% default
        self.slippage_rate = config.get('slippage_rate', 0.0002)  # 0.02% default
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the exchange.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the exchange.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """
        Place an order on the exchange.
        
        Args:
            order: Order to place
            
        Returns:
            Order with updated status and order_id
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get current status of an order.
        
        Args:
            order_id: ID of order
            
        Returns:
            Current order status
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """
        Get current account information.
        
        Returns:
            AccountInfo with balance and positions
        """
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get current position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position if exists, None otherwise
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price
        """
        pass
    
    def calculate_commission(self, quantity: float, price: float) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            quantity: Trade quantity
            price: Trade price
            
        Returns:
            Commission amount
        """
        return quantity * price * self.commission_rate
    
    def calculate_slippage(self, price: float, side: OrderSide) -> float:
        """
        Calculate slippage for a trade.
        
        Args:
            price: Original price
            side: Order side
            
        Returns:
            Price with slippage applied
        """
        if side == OrderSide.BUY:
            return price * (1 + self.slippage_rate)
        else:
            return price * (1 - self.slippage_rate)
    
    def validate_order(self, order: Order) -> bool:
        """
        Validate order before submission.
        
        Args:
            order: Order to validate
            
        Returns:
            True if valid, False otherwise
        """
        if order.quantity <= 0:
            return False
        
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price is None:
            return False
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            return False
        
        return True
