"""
Unit tests for BaseExecutor and related classes.
"""
import pytest
from its_project.execution.base import OrderType, OrderStatus, Order, Position, BaseExecutor


@pytest.mark.unit
@pytest.mark.execution_layer
class TestOrderEnums:
    """Test OrderType and OrderStatus enums."""
    
    def test_order_type_values(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP_LOSS.value == "stop_loss"
        assert OrderType.TAKE_PROFIT.value == "take_profit"
    
    def test_order_status_values(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.OPEN.value == "open"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"


@pytest.mark.unit
@pytest.mark.execution_layer
class TestOrder:
    """Test Order dataclass."""
    
    def test_order_creation(self):
        """Test creating an Order."""
        order = Order(
            id="order123",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=0.1,
            price=None,
            status=OrderStatus.PENDING,
            filled=0.0,
            remaining=0.1,
            timestamp=1704067200000,
            info={}
        )
        
        assert order.id == "order123"
        assert order.symbol == "BTC/USDT"
        assert order.type == OrderType.MARKET
        assert order.side == "buy"
        assert order.amount == 0.1
        assert order.price is None
        assert order.status == OrderStatus.PENDING
        assert order.filled == 0.0
        assert order.remaining == 0.1
    
    def test_order_with_price(self):
        """Test creating a limit order with price."""
        order = Order(
            id="order456",
            symbol="BTC/USDT",
            type=OrderType.LIMIT,
            side="sell",
            amount=0.5,
            price=43000.0,
            status=OrderStatus.OPEN,
            filled=0.2,
            remaining=0.3,
            timestamp=1704067200000,
            info={}
        )
        
        assert order.price == 43000.0
        assert order.filled == 0.2
        assert order.remaining == 0.3


@pytest.mark.unit
@pytest.mark.execution_layer
class TestPosition:
    """Test Position dataclass."""
    
    def test_position_creation(self):
        """Test creating a Position."""
        position = Position(
            symbol="BTC/USDT",
            side="long",
            size=0.5,
            entry_price=42000.0,
            current_price=42500.0,
            unrealized_pnl=250.0,
            timestamp=1704067200000
        )
        
        assert position.symbol == "BTC/USDT"
        assert position.side == "long"
        assert position.size == 0.5
        assert position.entry_price == 42000.0
        assert position.current_price == 42500.0
        assert position.unrealized_pnl == 250.0
    
    def test_position_short(self):
        """Test creating a short position."""
        position = Position(
            symbol="BTC/USDT",
            side="short",
            size=0.3,
            entry_price=43000.0,
            current_price=42500.0,
            unrealized_pnl=150.0,
            timestamp=1704067200000
        )
        
        assert position.side == "short"
        assert position.unrealized_pnl == 150.0  # Profit from short


@pytest.mark.unit
@pytest.mark.execution_layer
class TestBaseExecutor:
    """Test BaseExecutor abstract class."""
    
    def test_initialization(self):
        """Test BaseExecutor initialization."""
        
        class ConcreteExecutor(BaseExecutor):
            async def create_order(self, symbol, order_type, side, amount, price=None, params=None):
                return Order(
                    id="test",
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=price,
                    status=OrderStatus.PENDING,
                    filled=0.0,
                    remaining=amount,
                    timestamp=1704067200000,
                    info={}
                )
            
            async def cancel_order(self, order_id, symbol):
                return True
            
            async def fetch_order_status(self, order_id, symbol):
                return Order(
                    id=order_id,
                    symbol=symbol,
                    type=OrderType.MARKET,
                    side="buy",
                    amount=0.1,
                    price=None,
                    status=OrderStatus.FILLED,
                    filled=0.1,
                    remaining=0.0,
                    timestamp=1704067200000,
                    info={}
                )
            
            async def fetch_balance(self):
                return {"USDT": 1000.0, "BTC": 0.5}
            
            async def fetch_positions(self):
                return []
        
        config = {"test_param": "value"}
        executor = ConcreteExecutor(config)
        
        assert executor.config == config
        assert executor.active_orders == {}
    
    def test_is_live(self):
        """Test is_live method."""
        
        class ConcreteExecutor(BaseExecutor):
            async def create_order(self, symbol, order_type, side, amount, price=None, params=None):
                pass
            
            async def cancel_order(self, order_id, symbol):
                pass
            
            async def fetch_order_status(self, order_id, symbol):
                pass
            
            async def fetch_balance(self):
                pass
            
            async def fetch_positions(self):
                pass
        
        executor = ConcreteExecutor({})
        
        # Base executor returns False by default
        assert executor.is_live() is False
