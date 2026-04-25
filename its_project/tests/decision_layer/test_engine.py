"""
Unit tests for TradingDecisionEngine.
"""
import pytest
from its_project.decision.engine import TradingDecisionEngine
from its_project.decision.decision import Signal, Action
from its_project.decision.risk import Position


@pytest.mark.unit
@pytest.mark.decision_layer
class TestTradingDecisionEngine:
    """Test TradingDecisionEngine functionality."""
    
    def test_initialization(self):
        """Test TradingDecisionEngine initialization."""
        config = {
            'decision': {'confidence_threshold': 0.7},
            'risk': {'max_position_size': 0.1, 'max_portfolio_risk': 1.0},
            'sizing': {'risk_per_trade': 0.02},
            'portfolio': {'max_positions': 5}
        }
        
        engine = TradingDecisionEngine(config)
        
        assert engine.decision_maker is not None
        assert engine.risk_manager is not None
        assert engine.position_sizer is not None
        assert engine.portfolio is not None
    
    def test_process_signal_buy(self):
        """Test processing a buy signal."""
        config = {
            'decision': {'confidence_threshold': 0.5},
            'risk': {'max_position_size': 0.5, 'max_portfolio_risk': 1.0},
            'sizing': {'risk_per_trade': 0.02},
            'portfolio': {'max_positions': 5}
        }
        
        engine = TradingDecisionEngine(config)
        
        signal = Signal(
            symbol="BTC/USDT",
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        account_balance = 10000.0
        
        decision = engine.process_signal(signal, market_state, account_balance)
        
        assert decision is not None
        assert decision.action == Action.BUY
        assert decision.symbol == "BTC/USDT"
    
    def test_process_signal_hold(self):
        """Test processing a hold signal returns None."""
        config = {
            'decision': {'confidence_threshold': 0.5},
            'risk': {'max_position_size': 0.5, 'max_portfolio_risk': 1.0},
            'sizing': {'risk_per_trade': 0.02},
            'portfolio': {'max_positions': 5}
        }
        
        engine = TradingDecisionEngine(config)
        
        signal = Signal(
            symbol="BTC/USDT",
            action=Action.HOLD,
            confidence=0.8,
            timestamp=1704067200000,
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        account_balance = 10000.0
        
        decision = engine.process_signal(signal, market_state, account_balance)
        
        assert decision is None
    
    def test_process_signal_low_confidence(self):
        """Test processing signal with low confidence returns None."""
        config = {
            'decision': {'confidence_threshold': 0.7},
            'risk': {'max_position_size': 0.5, 'max_portfolio_risk': 1.0},
            'sizing': {'risk_per_trade': 0.02},
            'portfolio': {'max_positions': 5}
        }
        
        engine = TradingDecisionEngine(config)
        
        signal = Signal(
            symbol="BTC/USDT",
            action=Action.BUY,
            confidence=0.5,  # Below threshold
            timestamp=1704067200000,
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        account_balance = 10000.0
        
        decision = engine.process_signal(signal, market_state, account_balance)
        
        assert decision is None
    
    def test_process_signal_position_exists(self):
        """Test processing signal when position already exists."""
        config = {
            'decision': {'confidence_threshold': 0.5},
            'risk': {'max_position_size': 0.5, 'max_portfolio_risk': 1.0},
            'sizing': {'risk_per_trade': 0.02},
            'portfolio': {'max_positions': 5}
        }
        
        engine = TradingDecisionEngine(config)
        
        # Add existing position
        position = Position(
            symbol="BTC/USDT",
            side="long",
            size=0.5,
            entry_price=42000.0,
            current_price=42500.0,
            unrealized_pnl=250.0,
            timestamp=1704067200000
        )
        engine.portfolio.add_position(position)
        
        signal = Signal(
            symbol="BTC/USDT",
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        account_balance = 10000.0
        
        decision = engine.process_signal(signal, market_state, account_balance)
        
        assert decision is None  # Cannot open position for existing symbol
    
    def test_process_signal_max_positions_reached(self):
        """Test processing signal when max positions reached."""
        config = {
            'decision': {'confidence_threshold': 0.5},
            'risk': {'max_position_size': 0.5, 'max_portfolio_risk': 1.0},
            'sizing': {'risk_per_trade': 0.02},
            'portfolio': {'max_positions': 1}
        }
        
        engine = TradingDecisionEngine(config)
        
        # Add existing position
        position = Position(
            symbol="ETH/USDT",
            side="long",
            size=0.5,
            entry_price=2500.0,
            current_price=2550.0,
            unrealized_pnl=50.0,
            timestamp=1704067200000
        )
        engine.portfolio.add_position(position)
        
        signal = Signal(
            symbol="BTC/USDT",
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        account_balance = 10000.0
        
        decision = engine.process_signal(signal, market_state, account_balance)
        
        assert decision is None  # Max positions reached
    
    def test_process_signal_risk_rejection(self):
        """Test processing signal rejected by risk manager."""
        config = {
            'decision': {'confidence_threshold': 0.5},
            'risk': {'max_position_size': 0.001, 'max_portfolio_risk': 1.0},  # Very small max
            'sizing': {'risk_per_trade': 0.5},  # Large size
            'portfolio': {'max_positions': 5}
        }
        
        engine = TradingDecisionEngine(config)
        
        signal = Signal(
            symbol="BTC/USDT",
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        account_balance = 10000.0
        
        decision = engine.process_signal(signal, market_state, account_balance)
        
        # Decision should be rejected due to size exceeding max_position_size
        assert decision is None
    
    def test_process_signal_sell(self):
        """Test processing a sell signal."""
        config = {
            'decision': {'confidence_threshold': 0.5},
            'risk': {'max_position_size': 0.5, 'max_portfolio_risk': 1.0},
            'sizing': {'risk_per_trade': 0.02},
            'portfolio': {'max_positions': 5}
        }
        
        engine = TradingDecisionEngine(config)
        
        signal = Signal(
            symbol="BTC/USDT",
            action=Action.SELL,
            confidence=0.8,
            timestamp=1704067200000,
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        account_balance = 10000.0
        
        decision = engine.process_signal(signal, market_state, account_balance)
        
        assert decision is not None
        assert decision.action == Action.SELL
