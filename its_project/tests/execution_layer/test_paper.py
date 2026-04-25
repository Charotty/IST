"""
Unit tests for PaperTradingExecutor.
"""
import pytest
import asyncio
from its_project.execution.paper import PaperTradingExecutor
from its_project.execution.base import OrderType, OrderStatus


@pytest.mark.unit
@pytest.mark.execution_layer
class TestPaperTradingExecutor:
    """Test PaperTradingExecutor functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test PaperTradingExecutor initialization."""
        config = {
            'initial_balance': 50000.0,
            'latency_ms': 100
        }
        
        executor = PaperTradingExecutor(config)
        
        assert executor.config == config
        assert executor.balances['USDT'] == 50000.0
        assert executor.balances['BTC'] == 0.0
        assert executor.balances['ETH'] == 0.0
        assert executor.order_counter == 0
        assert executor.latency == 0.1
        assert executor.is_live() is False
    
    @pytest.mark.asyncio
    async def test_default_config(self):
        """Test default configuration values."""
        executor = PaperTradingExecutor({})
        
        assert executor.balances['USDT'] == 10000.0
        assert executor.latency == 0.05  # 50ms / 1000
    
    @pytest.mark.asyncio
    async def test_create_market_order_buy(self):
        """Test creating a market buy order."""
        executor = PaperTradingExecutor({'initial_balance': 10000.0})
        
        order = await executor.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.MARKET,
            side="buy",
            amount=0.1
        )
        
        assert order.symbol == "BTC/USDT"
        assert order.type == OrderType.MARKET
        assert order.side == "buy"
        assert order.amount == 0.1
        assert order.status == OrderStatus.FILLED  # Market orders are filled immediately
        assert order.filled == 0.1
        assert order.remaining == 0.0
        assert order.id.startswith("paper_")
    
    @pytest.mark.asyncio
    async def test_create_market_order_sell(self):
        """Test creating a market sell order."""
        executor = PaperTradingExecutor({'initial_balance': 10000.0})
        executor.balances['BTC'] = 0.5  # Give some BTC to sell
        
        order = await executor.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.MARKET,
            side="sell",
            amount=0.1
        )
        
        assert order.side == "sell"
        assert order.status == OrderStatus.FILLED
    
    @pytest.mark.asyncio
    async def test_create_limit_order(self):
        """Test creating a limit order."""
        executor = PaperTradingExecutor({})
        
        order = await executor.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.LIMIT,
            side="buy",
            amount=0.1,
            price=41000.0
        )
        
        assert order.type == OrderType.LIMIT
        assert order.price == 41000.0
        assert order.status == OrderStatus.PENDING  # Limit orders are not filled immediately
        assert order.filled == 0.0
        assert order.remaining == 0.1
    
    @pytest.mark.asyncio
    async def test_buy_order_updates_balance(self):
        """Test that buy order updates balance correctly."""
        executor = PaperTradingExecutor({'initial_balance': 10000.0})
        
        initial_usdt = executor.balances['USDT']
        initial_btc = executor.balances['BTC']
        
        await executor.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.MARKET,
            side="buy",
            amount=0.1
        )
        
        # USDT should decrease
        assert executor.balances['USDT'] < initial_usdt
        # BTC should increase
        assert executor.balances['BTC'] > initial_btc
        assert executor.balances['BTC'] == 0.1
    
    @pytest.mark.asyncio
    async def test_sell_order_updates_balance(self):
        """Test that sell order updates balance correctly."""
        executor = PaperTradingExecutor({'initial_balance': 10000.0})
        executor.balances['BTC'] = 0.5
        
        initial_usdt = executor.balances['USDT']
        initial_btc = executor.balances['BTC']
        
        await executor.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.MARKET,
            side="sell",
            amount=0.1
        )
        
        # USDT should increase
        assert executor.balances['USDT'] > initial_usdt
        # BTC should decrease
        assert executor.balances['BTC'] < initial_btc
        assert executor.balances['BTC'] == 0.4
    
    @pytest.mark.asyncio
    async def test_insufficient_balance_buy(self):
        """Test buy order with insufficient balance."""
        executor = PaperTradingExecutor({'initial_balance': 100.0})
        
        order = await executor.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.MARKET,
            side="buy",
            amount=1.0  # Would cost ~42000 USDT
        )
        
        assert order.status == OrderStatus.REJECTED
        assert executor.balances['USDT'] == 100.0  # Balance unchanged
    
    @pytest.mark.asyncio
    async def test_insufficient_balance_sell(self):
        """Test sell order with insufficient balance."""
        executor = PaperTradingExecutor({})
        executor.balances['BTC'] = 0.01
        
        order = await executor.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.MARKET,
            side="sell",
            amount=0.1  # More than we have
        )
        
        assert order.status == OrderStatus.REJECTED
        assert executor.balances['BTC'] == 0.01  # Balance unchanged
    
    @pytest.mark.asyncio
    async def test_cancel_order(self):
        """Test cancelling an order."""
        executor = PaperTradingExecutor({})
        
        # Create a limit order (won't be filled)
        order = await executor.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.LIMIT,
            side="buy",
            amount=0.1,
            price=41000.0
        )
        
        result = await executor.cancel_order(order.id, "BTC/USDT")
        
        assert result is True
        assert executor.active_orders[order.id].status == OrderStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self):
        """Test cancelling a non-existent order."""
        executor = PaperTradingExecutor({})
        
        result = await executor.cancel_order("nonexistent", "BTC/USDT")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_cancel_filled_order(self):
        """Test cancelling an already filled order."""
        executor = PaperTradingExecutor({})
        
        # Create a market order (will be filled)
        order = await executor.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.MARKET,
            side="buy",
            amount=0.1
        )
        
        result = await executor.cancel_order(order.id, "BTC/USDT")
        
        assert result is False  # Cannot cancel filled order
    
    @pytest.mark.asyncio
    async def test_fetch_order_status(self):
        """Test fetching order status."""
        executor = PaperTradingExecutor({})
        
        order = await executor.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.LIMIT,
            side="buy",
            amount=0.1,
            price=41000.0
        )
        
        fetched = await executor.fetch_order_status(order.id, "BTC/USDT")
        
        assert fetched.id == order.id
        assert fetched.status == OrderStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_fetch_order_status_not_found(self):
        """Test fetching status of non-existent order."""
        executor = PaperTradingExecutor({})
        
        fetched = await executor.fetch_order_status("nonexistent", "BTC/USDT")
        
        assert fetched.status == OrderStatus.REJECTED
        assert "error" in fetched.info
    
    @pytest.mark.asyncio
    async def test_fetch_balance(self):
        """Test fetching balance."""
        executor = PaperTradingExecutor({'initial_balance': 15000.0})
        
        balance = await executor.fetch_balance()
        
        assert balance['USDT'] == 15000.0
        assert balance['BTC'] == 0.0
        assert balance['ETH'] == 0.0
    
    @pytest.mark.asyncio
    async def test_fetch_positions(self):
        """Test fetching positions."""
        executor = PaperTradingExecutor({})
        
        positions = await executor.fetch_positions()
        
        assert isinstance(positions, list)
        assert len(positions) == 0  # No positions initially
    
    @pytest.mark.asyncio
    async def test_is_live(self):
        """Test is_live method."""
        executor = PaperTradingExecutor({})
        
        assert executor.is_live() is False
    
    @pytest.mark.asyncio
    async def test_order_counter_increments(self):
        """Test that order counter increments."""
        executor = PaperTradingExecutor({})
        
        order1 = await executor.create_order("BTC/USDT", OrderType.MARKET, "buy", 0.1)
        order2 = await executor.create_order("BTC/USDT", OrderType.MARKET, "buy", 0.1)
        
        assert order1.id == "paper_1"
        assert order2.id == "paper_2"
