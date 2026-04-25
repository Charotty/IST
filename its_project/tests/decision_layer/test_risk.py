"""
Unit tests for RiskManager and Position.
"""
import pytest
from its_project.decision.risk import Position, RiskManager
from its_project.decision.decision import Decision, Action


@pytest.mark.unit
@pytest.mark.decision_layer
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
        assert position.unrealized_pnl == 150.0


@pytest.mark.unit
@pytest.mark.decision_layer
class TestRiskManager:
    """Test RiskManager functionality."""
    
    def test_initialization(self):
        """Test RiskManager initialization."""
        config = {
            'max_position_size': 0.2,
            'max_portfolio_risk': 0.1,
            'max_drawdown': 0.2,
            'max_correlation': 0.8
        }
        
        manager = RiskManager(config)
        
        assert manager.config == config
        assert manager.max_position_size == 0.2
        assert manager.max_portfolio_risk == 0.1
        assert manager.max_drawdown == 0.2
        assert manager.max_correlation == 0.8
        assert manager.current_drawdown == 0.0
        assert manager.peak_equity == 0.0
    
    def test_default_config(self):
        """Test default configuration values."""
        manager = RiskManager({})
        
        assert manager.max_position_size == 0.1
        assert manager.max_portfolio_risk == 0.05
        assert manager.max_drawdown == 0.15
        assert manager.max_correlation == 0.7
    
    def test_check_decision_size_too_large(self):
        """Test check_decision with size too large."""
        manager = RiskManager({'max_position_size': 0.1})
        
        decision = Decision(
            action=Action.BUY,
            symbol="BTC/USDT",
            size=0.15,
            price=None,
            stop_loss=41000.0,
            take_profit=43000.0,
            timestamp=1704067200000,
            reason="Test"
        )
        
        allowed, reason = manager.check_decision(decision, [], 10000)
        
        assert allowed is False
        assert "Size 0.15 > max 0.1" in reason
    
    def test_check_decision_size_ok(self):
        """Test check_decision with acceptable size."""
        manager = RiskManager({'max_position_size': 0.1, 'max_portfolio_risk': 1.0})
        
        decision = Decision(
            action=Action.BUY,
            symbol="BTC/USDT",
            size=0.05,
            price=None,
            stop_loss=41000.0,
            take_profit=43000.0,
            timestamp=1704067200000,
            reason="Test"
        )
        
        allowed, reason = manager.check_decision(decision, [], 10000)
        
        assert allowed is True
        assert reason == "OK"
    
    def test_check_decision_portfolio_risk_too_high(self):
        """Test check_decision with portfolio risk too high."""
        manager = RiskManager({'max_portfolio_risk': 0.05})
        
        existing_position = Position(
            symbol="ETH/USDT",
            side="long",
            size=1.0,
            entry_price=2000.0,
            current_price=2000.0,
            unrealized_pnl=0.0,
            timestamp=1704067200000
        )
        
        decision = Decision(
            action=Action.BUY,
            symbol="BTC/USDT",
            size=0.1,
            price=42000.0,
            stop_loss=40000.0,
            take_profit=44000.0,
            timestamp=1704067200000,
            reason="Test"
        )
        
        allowed, reason = manager.check_decision(decision, [existing_position], 1000)
        
        assert allowed is False
        assert "Portfolio risk" in reason
    
    def test_check_decision_drawdown_too_high(self):
        """Test check_decision with drawdown too high."""
        manager = RiskManager({'max_drawdown': 0.1, 'max_portfolio_risk': 1.0})
        manager.current_drawdown = 0.15
        
        decision = Decision(
            action=Action.BUY,
            symbol="BTC/USDT",
            size=0.05,
            price=None,
            stop_loss=41000.0,
            take_profit=43000.0,
            timestamp=1704067200000,
            reason="Test"
        )
        
        allowed, reason = manager.check_decision(decision, [], 10000)
        
        assert allowed is False
        assert "Drawdown 15.00% > max 10.00%" in reason
    
    def test_check_decision_high_correlation(self):
        """Test check_decision with high correlation (same symbol)."""
        manager = RiskManager({'max_correlation': 0.7, 'max_portfolio_risk': 1.0})
        
        existing_position = Position(
            symbol="BTC/USDT",
            side="long",
            size=0.1,
            entry_price=42000.0,
            current_price=42500.0,
            unrealized_pnl=50.0,
            timestamp=1704067200000
        )
        
        decision = Decision(
            action=Action.BUY,
            symbol="BTC/USDT",
            size=0.05,
            price=None,
            stop_loss=41000.0,
            take_profit=43000.0,
            timestamp=1704067200000,
            reason="Test"
        )
        
        allowed, reason = manager.check_decision(decision, [existing_position], 10000)
        
        assert allowed is False
        assert "High correlation" in reason
    
    def test_check_decision_different_symbol(self):
        """Test check_decision with different symbol (no correlation)."""
        manager = RiskManager({'max_correlation': 0.7, 'max_portfolio_risk': 1.0})
        
        existing_position = Position(
            symbol="ETH/USDT",
            side="long",
            size=0.1,
            entry_price=2000.0,
            current_price=2050.0,
            unrealized_pnl=5.0,
            timestamp=1704067200000
        )
        
        decision = Decision(
            action=Action.BUY,
            symbol="BTC/USDT",
            size=0.05,
            price=None,
            stop_loss=41000.0,
            take_profit=43000.0,
            timestamp=1704067200000,
            reason="Test"
        )
        
        allowed, reason = manager.check_decision(decision, [existing_position], 10000)
        
        assert allowed is True
    
    def test_calculate_portfolio_risk(self):
        """Test portfolio risk calculation."""
        manager = RiskManager({})
        
        positions = [
            Position(
                symbol="BTC/USDT",
                side="long",
                size=0.1,
                entry_price=42000.0,
                current_price=42000.0,
                unrealized_pnl=0.0,
                timestamp=1704067200000
            )
        ]
        
        decision = Decision(
            action=Action.BUY,
            symbol="ETH/USDT",
            size=0.5,
            price=2000.0,
            stop_loss=1900.0,
            take_profit=2100.0,
            timestamp=1704067200000,
            reason="Test"
        )
        
        risk = manager._calculate_portfolio_risk(positions, decision, 10000)
        
        assert risk > 0
        assert risk < 1  # Should be a percentage
    
    def test_update_drawdown_new_peak(self):
        """Test update_drawdown with new equity peak."""
        manager = RiskManager({})
        manager.peak_equity = 10000.0
        
        manager.update_drawdown(11000.0)
        
        assert manager.peak_equity == 11000.0
        assert manager.current_drawdown == 0.0
    
    def test_update_drawdown_decline(self):
        """Test update_drawdown with equity decline."""
        manager = RiskManager({})
        manager.peak_equity = 10000.0
        
        manager.update_drawdown(9000.0)
        
        assert manager.peak_equity == 10000.0
        assert manager.current_drawdown == 0.1  # 10% drawdown
    
    def test_update_drawdown_initial(self):
        """Test update_drawdown with initial equity."""
        manager = RiskManager({})
        
        manager.update_drawdown(10000.0)
        
        assert manager.peak_equity == 10000.0
        assert manager.current_drawdown == 0.0
