"""
Tests for paper trading engine.
"""

import pytest
from backend.paper_trading import PaperPosition, PaperPortfolio, PaperTradingEngine


class TestPaperPosition:
    """Test PaperPosition class."""
    
    def test_create_position(self):
        """Test creating a position."""
        position = PaperPosition("BTC/USDT", "buy", 0.1, 50000.0)
        assert position.symbol == "BTC/USDT"
        assert position.side == "buy"
        assert position.qty == 0.1
        assert position.entry_price == 50000.0
        assert position.pnl is None
    
    def test_close_position_buy(self):
        """Test closing a buy position."""
        position = PaperPosition("BTC/USDT", "buy", 0.1, 50000.0)
        position.close(51000.0, 5.0)
        
        assert position.exit_price == 51000.0
        assert position.fee == 5.0
        assert position.pnl is not None
        assert position.pnl > 0  # Profit
    
    def test_close_position_sell(self):
        """Test closing a sell position."""
        position = PaperPosition("BTC/USDT", "sell", 0.1, 50000.0)
        position.close(49000.0, 5.0)
        
        assert position.exit_price == 49000.0
        assert position.fee == 5.0
        assert position.pnl is not None
        assert position.pnl > 0  # Profit from short
    
    def test_unrealized_pnl(self):
        """Test unrealized PnL calculation."""
        position = PaperPosition("BTC/USDT", "buy", 0.1, 50000.0)
        
        # Profit scenario
        unrealized = position.unrealized_pnl(51000.0)
        assert unrealized > 0
        
        # Loss scenario
        unrealized = position.unrealized_pnl(49000.0)
        assert unrealized < 0
    
    def test_position_value(self):
        """Test position value calculation."""
        position = PaperPosition("BTC/USDT", "buy", 0.1, 50000.0)
        value = position.value()
        assert value == 5000.0  # 0.1 * 50000


class TestPaperPortfolio:
    """Test PaperPortfolio class."""
    
    def test_create_portfolio(self):
        """Test creating a portfolio."""
        portfolio = PaperPortfolio(initial_balance=10000.0)
        assert portfolio.initial_balance == 10000.0
        assert portfolio.balance == 10000.0
        assert len(portfolio.positions) == 0
        assert len(portfolio.closed_trades) == 0
    
    def test_execute_buy_order(self):
        """Test executing a buy order."""
        portfolio = PaperPortfolio(initial_balance=10000.0)
        event = portfolio.execute_order("BTC/USDT", "buy", 0.1, 50000.0)
        
        assert event is not None
        assert event.symbol == "BTC/USDT"
        assert event.side == "BUY"
        assert event.status == "OPEN"
        assert "BTC/USDT" in portfolio.positions
        assert portfolio.balance < 10000.0  # Balance decreased
    
    def test_execute_sell_order(self):
        """Test executing a sell order (closing position)."""
        portfolio = PaperPortfolio(initial_balance=10000.0)
        
        # Open position first
        portfolio.execute_order("BTC/USDT", "buy", 0.1, 50000.0)
        
        # Close position
        event = portfolio.execute_order("BTC/USDT", "sell", 0.1, 51000.0)
        
        assert event is not None
        assert event.status == "CLOSED"
        assert event.pnl is not None
        assert "BTC/USDT" not in portfolio.positions
        assert len(portfolio.closed_trades) == 1
    
    def test_insufficient_balance(self):
        """Test order with insufficient balance."""
        portfolio = PaperPortfolio(initial_balance=100.0)
        event = portfolio.execute_order("BTC/USDT", "buy", 1.0, 50000.0)
        
        assert event is None  # Order rejected
    
    def test_close_nonexistent_position(self):
        """Test closing a position that doesn't exist."""
        portfolio = PaperPortfolio(initial_balance=10000.0)
        event = portfolio.execute_order("BTC/USDT", "sell", 0.1, 50000.0)
        
        assert event is None  # Order rejected
    
    def test_get_metrics(self):
        """Test getting portfolio metrics."""
        portfolio = PaperPortfolio(initial_balance=10000.0)
        
        # Execute some trades
        portfolio.execute_order("BTC/USDT", "buy", 0.1, 50000.0)
        portfolio.execute_order("BTC/USDT", "sell", 0.1, 51000.0)
        
        metrics = portfolio.get_metrics()
        
        assert metrics.trades_count == 1
        assert metrics.win_rate == 100.0  # One winning trade
        assert metrics.pnl > 0
    
    def test_reset_portfolio(self):
        """Test resetting portfolio."""
        portfolio = PaperPortfolio(initial_balance=10000.0)
        portfolio.execute_order("BTC/USDT", "buy", 0.1, 50000.0)
        
        portfolio.reset()
        
        assert portfolio.balance == 10000.0
        assert len(portfolio.positions) == 0
        assert len(portfolio.closed_trades) == 0
        assert portfolio.trades_count == 0


class TestPaperTradingEngine:
    """Test PaperTradingEngine class."""
    
    def test_create_engine(self):
        """Test creating a trading engine."""
        engine = PaperTradingEngine()
        assert engine.portfolio is not None
        assert engine.running is False
    
    def test_execute_signal_buy(self):
        """Test executing a BUY signal."""
        engine = PaperTradingEngine()
        event = engine.execute_signal("BTC/USDT", "BUY", 0.8, 50000.0)
        
        assert event is not None
        assert event.side == "BUY"
        assert event.status == "OPEN"
    
    def test_execute_signal_hold(self):
        """Test executing a HOLD signal."""
        engine = PaperTradingEngine()
        event = engine.execute_signal("BTC/USDT", "HOLD", 0.5, 50000.0)
        
        assert event is None  # No trade for HOLD
    
    def test_execute_signal_invalid_action(self):
        """Test executing an invalid signal action."""
        engine = PaperTradingEngine()
        event = engine.execute_signal("BTC/USDT", "INVALID", 0.5, 50000.0)
        
        assert event is None
    
    def test_get_metrics(self):
        """Test getting engine metrics."""
        engine = PaperTradingEngine()
        metrics = engine.get_metrics()
        
        assert metrics.trades_count == 0
        assert metrics.win_rate == 0.0
    
    def test_reset_engine(self):
        """Test resetting the engine."""
        engine = PaperTradingEngine()
        engine.execute_signal("BTC/USDT", "BUY", 0.8, 50000.0)
        
        engine.reset()
        
        assert engine.portfolio.balance == 10000.0
        assert len(engine.portfolio.positions) == 0
    
    def test_trade_callback(self):
        """Test trade callback registration."""
        engine = PaperTradingEngine()
        callback_called = []
        
        def callback(event):
            callback_called.append(event)
        
        engine.on_trade(callback)
        engine.execute_signal("BTC/USDT", "BUY", 0.8, 50000.0)
        
        assert len(callback_called) == 1
