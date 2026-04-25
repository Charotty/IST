"""
Unit tests for OrderManager.
"""
import pytest
import asyncio
from its_project.execution.manager import OrderManager
from its_project.execution.base import BaseExecutor, Order, OrderType, OrderStatus


# Mock executor for testing
class MockExecutor(BaseExecutor):
    """Mock executor for testing."""
    
    def __init__(self):
        self.orders = {}
        self.balance = {"USDT": 10000.0}
        self.positions = []
        self.order_counter = 0
    
    async def create_order(self, symbol, order_type, side, amount, price=None):
        self.order_counter += 1
        order = Order(
            id=f"order_{self.order_counter}",
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount,
            price=price,
            status=OrderStatus.OPEN,
            filled=0.0,
            remaining=amount,
            timestamp=1704067200000,
            info={}
        )
        self.orders[order.id] = order
        return order
    
    async def fetch_order_status(self, order_id, symbol):
        return self.orders.get(order_id, Order(
            id=order_id,
            symbol=symbol,
            type=OrderType.MARKET,
            side="buy",
            amount=0,
            price=0,
            status=OrderStatus.CANCELLED,
            filled=0.0,
            remaining=0.0,
            timestamp=1704067200000,
            info={}
        ))
    
    async def cancel_order(self, order_id, symbol):
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
    
    async def fetch_balance(self):
        return self.balance
    
    async def fetch_positions(self):
        return self.positions


@pytest.mark.unit
@pytest.mark.execution_layer
class TestOrderManager:
    """Test OrderManager functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test OrderManager initialization."""
        executor = MockExecutor()
        config = {"monitor_interval": 0.5}
        
        manager = OrderManager(executor, config)
        
        assert manager.executor is executor
        assert manager.config == config
        assert manager.active_orders == {}
        assert manager.sl_tp_orders == {}
        assert manager.monitor_task is None
    
    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping order monitoring."""
        executor = MockExecutor()
        config = {"monitor_interval": 0.1}
        manager = OrderManager(executor, config)
        
        await manager.start()
        assert manager.monitor_task is not None
        
        await asyncio.sleep(0.2)  # Let it run briefly
        await manager.stop()
        assert manager.stop_event.is_set()
    
    @pytest.mark.asyncio
    async def test_execute_decision_market_order(self):
        """Test executing a market order."""
        executor = MockExecutor()
        config = {}
        manager = OrderManager(executor, config)
        
        order = await manager.execute_decision(
            symbol="BTC/USDT",
            side="buy",
            amount=0.5,
            order_type="market"
        )
        
        assert order.symbol == "BTC/USDT"
        assert order.side == "buy"
        assert order.amount == 0.5
        assert order.type == OrderType.MARKET
        assert order.id in manager.active_orders
    
    @pytest.mark.asyncio
    async def test_execute_decision_limit_order(self):
        """Test executing a limit order."""
        executor = MockExecutor()
        config = {}
        manager = OrderManager(executor, config)
        
        order = await manager.execute_decision(
            symbol="BTC/USDT",
            side="buy",
            amount=0.5,
            order_type="limit",
            price=42000.0
        )
        
        assert order.type == OrderType.LIMIT
        assert order.price == 42000.0
    
    @pytest.mark.asyncio
    async def test_execute_decision_with_stop_loss(self):
        """Test executing order with stop loss."""
        executor = MockExecutor()
        config = {}
        manager = OrderManager(executor, config)
        
        order = await manager.execute_decision(
            symbol="BTC/USDT",
            side="buy",
            amount=0.5,
            order_type="market",
            stop_loss=41000.0
        )
        
        assert order.id in manager.active_orders
        assert order.id in manager.sl_tp_orders
        assert len(manager.sl_tp_orders[order.id]) == 1
    
    @pytest.mark.asyncio
    async def test_execute_decision_with_take_profit(self):
        """Test executing order with take profit."""
        executor = MockExecutor()
        config = {}
        manager = OrderManager(executor, config)
        
        order = await manager.execute_decision(
            symbol="BTC/USDT",
            side="buy",
            amount=0.5,
            order_type="market",
            take_profit=43000.0
        )
        
        assert order.id in manager.active_orders
        assert order.id in manager.sl_tp_orders
        assert len(manager.sl_tp_orders[order.id]) == 1
    
    @pytest.mark.asyncio
    async def test_execute_decision_with_sl_and_tp(self):
        """Test executing order with both stop loss and take profit."""
        executor = MockExecutor()
        config = {}
        manager = OrderManager(executor, config)
        
        order = await manager.execute_decision(
            symbol="BTC/USDT",
            side="buy",
            amount=0.5,
            order_type="market",
            stop_loss=41000.0,
            take_profit=43000.0
        )
        
        assert order.id in manager.active_orders
        assert order.id in manager.sl_tp_orders
        assert len(manager.sl_tp_orders[order.id]) == 2
    
    @pytest.mark.asyncio
    async def test_create_stop_loss_order_side(self):
        """Test stop loss order has opposite side."""
        executor = MockExecutor()
        config = {}
        manager = OrderManager(executor, config)
        
        # Buy order should have sell stop loss
        sl_order = await manager._create_stop_loss_order(
            symbol="BTC/USDT",
            side="buy",
            amount=0.5,
            stop_loss=41000.0
        )
        
        assert sl_order.side == "sell"
        assert sl_order.type == OrderType.STOP_LOSS
        
        # Sell order should have buy stop loss
        sl_order = await manager._create_stop_loss_order(
            symbol="BTC/USDT",
            side="sell",
            amount=0.5,
            stop_loss=43000.0
        )
        
        assert sl_order.side == "buy"
    
    @pytest.mark.asyncio
    async def test_create_take_profit_order_side(self):
        """Test take profit order has opposite side."""
        executor = MockExecutor()
        config = {}
        manager = OrderManager(executor, config)
        
        # Buy order should have sell take profit
        tp_order = await manager._create_take_profit_order(
            symbol="BTC/USDT",
            side="buy",
            amount=0.5,
            take_profit=43000.0
        )
        
        assert tp_order.side == "sell"
        assert tp_order.type == OrderType.TAKE_PROFIT
    
    @pytest.mark.asyncio
    async def test_get_active_orders(self):
        """Test getting active orders."""
        executor = MockExecutor()
        config = {}
        manager = OrderManager(executor, config)
        
        order1 = await manager.execute_decision(
            symbol="BTC/USDT",
            side="buy",
            amount=0.5,
            order_type="market"
        )
        
        order2 = await manager.execute_decision(
            symbol="ETH/USDT",
            side="buy",
            amount=1.0,
            order_type="market"
        )
        
        active_orders = await manager.get_active_orders()
        
        assert len(active_orders) == 2
        assert order1 in active_orders
        assert order2 in active_orders
    
    @pytest.mark.asyncio
    async def test_get_balance(self):
        """Test getting balance."""
        executor = MockExecutor()
        config = {}
        manager = OrderManager(executor, config)
        
        balance = await manager.get_balance()
        
        assert balance == {"USDT": 10000.0}
    
    @pytest.mark.asyncio
    async def test_get_positions(self):
        """Test getting positions."""
        executor = MockExecutor()
        config = {}
        manager = OrderManager(executor, config)
        
        positions = await manager.get_positions()
        
        assert positions == []
    
    @pytest.mark.asyncio
    async def test_handle_closed_order(self):
        """Test handling closed order."""
        executor = MockExecutor()
        config = {}
        manager = OrderManager(executor, config)
        
        order = await manager.execute_decision(
            symbol="BTC/USDT",
            side="buy",
            amount=0.5,
            order_type="market",
            stop_loss=41000.0
        )
        
        # Simulate order being cancelled
        order.status = OrderStatus.CANCELLED
        await manager._handle_closed_order(order)
        
        assert order.id not in manager.active_orders
