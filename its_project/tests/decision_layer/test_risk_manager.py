"""
Unit tests for RiskManager.
"""
import pytest
from its_project.decision.risk_manager import RiskManager
from its_project.decision.decision import Decision, Action
from its_project.decision.portfolio import Position


@pytest.mark.unit
@pytest.mark.decision_layer
class TestRiskManager:
    """Test RiskManager class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "max_position_size": 0.1,
            "max_position_value": 10000.0,
            "max_leverage": 1.0,
            "max_portfolio_risk": 0.05,
            "max_total_exposure": 0.5,
            "max_correlation": 0.7,
            "max_drawdown": 0.15,
            "daily_loss_limit": 0.02,
            "consecutive_loss_limit": 3,
            "default_stop_loss_pct": 0.02,
            "default_take_profit_pct": 0.04,
            "trailing_stop_pct": 0.01,
            "atr_multiplier": 2.0,
            "volatility_window": 20,
            "volatility_target": 0.15,
            "position_risk_pct": 0.02
        }
    
    @pytest.fixture
    def risk_manager(self, config):
        """Create RiskManager instance."""
        return RiskManager(config)
    
    @pytest.fixture
    def sample_decision(self):
        """Create sample decision."""
        return Decision(
            action=Action.BUY,
            symbol="BTCUSDT",
            size=0.05,
            price=50000.0,
            stop_loss=None,
            take_profit=None,
            timestamp=1234567890000,
            reason="Test signal"
        )
    
    @pytest.fixture
    def sample_position(self):
        """Create sample position."""
        return Position(
            symbol="ETHUSDT",
            side="long",
            size=0.03,
            entry_price=3000.0,
            current_price=3100.0,
            unrealized_pnl=30.0,
            timestamp=1234567890000
        )
    
    def test_initialization(self, risk_manager, config):
        """Test RiskManager initialization."""
        assert risk_manager.max_position_size == 0.1
        assert risk_manager.max_position_value == 10000.0
        assert risk_manager.max_leverage == 1.0
        assert risk_manager.max_portfolio_risk == 0.05
        assert risk_manager.max_total_exposure == 0.5
        assert risk_manager.max_correlation == 0.7
        assert risk_manager.max_drawdown == 0.15
        assert risk_manager.daily_loss_limit == 0.02
        assert risk_manager.consecutive_loss_limit == 3
        assert risk_manager.default_stop_loss_pct == 0.02
        assert risk_manager.default_take_profit_pct == 0.04
        assert risk_manager.atr_multiplier == 2.0
        assert risk_manager.current_drawdown == 0.0
        assert risk_manager.peak_equity == 0.0
        assert risk_manager.daily_pnl == 0.0
        assert risk_manager.consecutive_losses == 0
    
    def test_initialization_defaults(self):
        """Test initialization with default values."""
        risk_manager = RiskManager({})
        assert risk_manager.max_position_size == 0.1
        assert risk_manager.max_position_value == 10000.0
        assert risk_manager.max_leverage == 1.0
        assert risk_manager.max_portfolio_risk == 0.05
        assert risk_manager.max_total_exposure == 0.5
        assert risk_manager.max_correlation == 0.7
        assert risk_manager.max_drawdown == 0.15
        assert risk_manager.daily_loss_limit == 0.02
        assert risk_manager.consecutive_loss_limit == 3
    
    def test_check_decision_approved(self, risk_manager, sample_decision):
        """Test decision approval when all checks pass."""
        allowed, reason = risk_manager.check_decision(
            decision=sample_decision,
            positions=[],
            account_balance=10000.0
        )
        
        assert allowed is True
        assert reason == "Decision approved"
    
    def test_check_decision_max_drawdown_exceeded(self, risk_manager, sample_decision):
        """Test decision rejection when max drawdown exceeded."""
        risk_manager.peak_equity = 10000.0
        risk_manager.current_drawdown = 0.20  # 20% > 15% limit
        
        allowed, reason = risk_manager.check_decision(
            decision=sample_decision,
            positions=[],
            account_balance=8000.0
        )
        
        assert allowed is False
        assert "Max drawdown exceeded" in reason
    
    def test_check_decision_daily_loss_limit_exceeded(self, risk_manager, sample_decision):
        """Test decision rejection when daily loss limit exceeded."""
        risk_manager.peak_equity = 10000.0
        risk_manager.daily_pnl = -300.0  # 3% > 2% limit
        
        allowed, reason = risk_manager.check_decision(
            decision=sample_decision,
            positions=[],
            account_balance=9700.0
        )
        
        assert allowed is False
        assert "Daily loss limit exceeded" in reason
    
    def test_check_decision_consecutive_losses_exceeded(self, risk_manager, sample_decision):
        """Test decision rejection when consecutive losses exceeded."""
        risk_manager.consecutive_losses = 3
        
        allowed, reason = risk_manager.check_decision(
            decision=sample_decision,
            positions=[],
            account_balance=10000.0
        )
        
        assert allowed is False
        assert "Too many consecutive losses" in reason
    
    def test_check_decision_position_size_exceeded(self, risk_manager):
        """Test decision rejection when position size exceeded."""
        decision = Decision(
            action=Action.BUY,
            symbol="BTCUSDT",
            size=0.15,  # 15% > 10% limit
            price=50000.0,
            stop_loss=None,
            take_profit=None,
            timestamp=1234567890000,
            reason="Test signal"
        )
        
        allowed, reason = risk_manager.check_decision(
            decision=decision,
            positions=[],
            account_balance=10000.0
        )
        
        assert allowed is False
        assert "Position size exceeds limits" in reason
    
    def test_check_decision_position_value_exceeded(self, risk_manager):
        """Test decision rejection when position value exceeded."""
        decision = Decision(
            action=Action.BUY,
            symbol="BTCUSDT",
            size=0.05,
            price=250000.0,  # $12,500 > $10,000 limit
            stop_loss=None,
            take_profit=None,
            timestamp=1234567890000,
            reason="Test signal"
        )
        
        allowed, reason = risk_manager.check_decision(
            decision=decision,
            positions=[],
            account_balance=10000.0
        )
        
        assert allowed is False
        assert "Position size exceeds limits" in reason
    
    def test_check_decision_portfolio_risk_exceeded(self, risk_manager, sample_decision, sample_position):
        """Test decision rejection when portfolio risk exceeded."""
        positions = [sample_position] * 20  # Very high exposure (0.6 > 0.5 limit)
        
        allowed, reason = risk_manager.check_decision(
            decision=sample_decision,
            positions=positions,
            account_balance=10000.0
        )
        
        assert allowed is False
        assert "Portfolio risk exceeds limits" in reason
    
    def test_check_decision_correlation_too_high(self, risk_manager, sample_decision):
        """Test decision rejection when correlation too high."""
        position = Position(
            symbol="BTCUSDT",  # Same symbol as decision
            side="long",
            size=0.03,
            entry_price=50000.0,
            current_price=51000.0,
            unrealized_pnl=30.0,
            timestamp=1234567890000
        )
        
        allowed, reason = risk_manager.check_decision(
            decision=sample_decision,
            positions=[position],
            account_balance=10000.0
        )
        
        assert allowed is False
        assert "Position correlation too high" in reason
    
    def test_check_decision_unfavorable_market_conditions(self, risk_manager, sample_decision):
        """Test decision rejection when market conditions unfavorable."""
        market_data = {
            "spread": 0.002,  # 0.2% > 0.1% limit
            "volume": 500000  # $500K < $1M limit
        }
        
        allowed, reason = risk_manager.check_decision(
            decision=sample_decision,
            positions=[],
            account_balance=10000.0,
            market_data=market_data
        )
        
        assert allowed is False
        assert "Unfavorable market conditions" in reason
    
    def test_check_position_size_within_limits(self, risk_manager, sample_decision):
        """Test position size check within limits."""
        result = risk_manager._check_position_size(sample_decision, 10000.0)
        assert result is True
    
    def test_check_position_size_percentage_exceeded(self, risk_manager):
        """Test position size check when percentage exceeded."""
        decision = Decision(
            action=Action.BUY,
            symbol="BTCUSDT",
            size=0.15,
            price=50000.0,
            stop_loss=None,
            take_profit=None,
            timestamp=1234567890000,
            reason="Test signal"
        )
        
        result = risk_manager._check_position_size(decision, 10000.0)
        assert result is False
    
    def test_check_position_size_value_exceeded(self, risk_manager):
        """Test position size check when value exceeded."""
        decision = Decision(
            action=Action.BUY,
            symbol="BTCUSDT",
            size=0.05,
            price=250000.0,
            stop_loss=None,
            take_profit=None,
            timestamp=1234567890000,
            reason="Test signal"
        )
        
        result = risk_manager._check_position_size(decision, 10000.0)
        assert result is False
    
    def test_check_position_size_leverage_exceeded(self, risk_manager):
        """Test position size check when leverage exceeded."""
        decision = Decision(
            action=Action.SELL,
            symbol="BTCUSDT",
            size=1.5,  # 150% > 100% leverage limit
            price=50000.0,
            stop_loss=None,
            take_profit=None,
            timestamp=1234567890000,
            reason="Test signal"
        )
        
        result = risk_manager._check_position_size(decision, 10000.0)
        assert result is False
    
    def test_check_portfolio_risk_within_limits(self, risk_manager, sample_decision):
        """Test portfolio risk check within limits."""
        result = risk_manager._check_portfolio_risk(sample_decision, [], 10000.0)
        assert result is True
    
    def test_check_portfolio_risk_exposure_exceeded(self, risk_manager, sample_decision, sample_position):
        """Test portfolio risk check when exposure exceeded."""
        positions = [sample_position] * 20  # Very high exposure
        
        result = risk_manager._check_portfolio_risk(sample_decision, positions, 10000.0)
        assert result is False
    
    def test_check_portfolio_risk_ratio_exceeded(self, risk_manager, sample_decision, sample_position):
        """Test portfolio risk check when ratio exceeded."""
        positions = [sample_position] * 20  # High ratio (0.6 / 100 = 0.6% > 5% limit)
        
        result = risk_manager._check_portfolio_risk(sample_decision, positions, 100.0)
        assert result is False
    
    def test_check_correlation_no_conflict(self, risk_manager, sample_decision, sample_position):
        """Test correlation check with no conflict."""
        result = risk_manager._check_correlation(sample_decision, [sample_position])
        assert result is True
    
    def test_check_correlation_same_asset(self, risk_manager, sample_decision):
        """Test correlation check with same asset."""
        position = Position(
            symbol="BTCUSDT",
            side="long",
            size=0.03,
            entry_price=50000.0,
            current_price=51000.0,
            unrealized_pnl=30.0,
            timestamp=1234567890000
        )
        
        result = risk_manager._check_correlation(sample_decision, [position])
        assert result is False
    
    def test_check_market_conditions_favorable(self, risk_manager, sample_decision):
        """Test market conditions check with favorable conditions."""
        market_data = {
            "volatility": 0.02,
            "spread": 0.0005,
            "volume": 2000000
        }
        
        result = risk_manager._check_market_conditions(sample_decision, market_data)
        assert result is True
    
    def test_check_market_conditions_high_volatility(self, risk_manager, sample_decision):
        """Test market conditions check with high volatility."""
        market_data = {
            "volatility": 0.06  # 6% > 5% threshold
        }
        
        result = risk_manager._check_market_conditions(sample_decision, market_data)
        assert result is True  # Warning only, still passes
    
    def test_check_market_conditions_high_spread(self, risk_manager, sample_decision):
        """Test market conditions check with high spread."""
        market_data = {
            "spread": 0.002  # 0.2% > 0.1% limit
        }
        
        result = risk_manager._check_market_conditions(sample_decision, market_data)
        assert result is False
    
    def test_check_market_conditions_low_volume(self, risk_manager, sample_decision):
        """Test market conditions check with low volume."""
        market_data = {
            "volume": 500000  # $500K < $1M limit
        }
        
        result = risk_manager._check_market_conditions(sample_decision, market_data)
        assert result is False
    
    def test_calculate_position_size_base(self, risk_manager):
        """Test position size calculation with base parameters."""
        size = risk_manager.calculate_position_size(
            signal_strength=0.8,
            account_balance=10000.0
        )
        
        assert size == pytest.approx(0.08)  # 0.1 * 0.8
    
    def test_calculate_position_size_with_volatility(self, risk_manager):
        """Test position size calculation with volatility adjustment."""
        size = risk_manager.calculate_position_size(
            signal_strength=0.8,
            volatility=0.3,  # High volatility
            account_balance=10000.0
        )
        
        assert size < 0.08  # Should be reduced due to high volatility
    
    def test_calculate_position_size_with_atr(self, risk_manager):
        """Test position size calculation with ATR."""
        size = risk_manager.calculate_position_size(
            signal_strength=0.8,
            atr=1000.0,
            account_balance=10000.0
        )
        
        assert size >= 0.0
        assert size <= 0.1
    
    def test_calculate_position_size_max_limit(self, risk_manager):
        """Test position size calculation respects max limit."""
        size = risk_manager.calculate_position_size(
            signal_strength=1.5,  # Very high signal
            account_balance=10000.0
        )
        
        assert size == 0.1  # Capped at max_position_size
    
    def test_calculate_position_size_zero_signal(self, risk_manager):
        """Test position size calculation with zero signal."""
        size = risk_manager.calculate_position_size(
            signal_strength=0.0,
            account_balance=10000.0
        )
        
        assert size == 0.0
    
    def test_set_stop_loss_take_profit_buy(self, risk_manager, sample_decision):
        """Test stop loss and take profit for buy decision."""
        stop_loss, take_profit = risk_manager.set_stop_loss_take_profit(
            decision=sample_decision,
            entry_price=50000.0
        )
        
        assert stop_loss < 50000.0
        assert take_profit > 50000.0
        assert stop_loss == 49000.0  # 50000 * (1 - 0.02)
        assert take_profit == 52000.0  # 50000 * (1 + 0.04)
    
    def test_set_stop_loss_take_profit_sell(self, risk_manager):
        """Test stop loss and take profit for sell decision."""
        decision = Decision(
            action=Action.SELL,
            symbol="BTCUSDT",
            size=0.05,
            price=50000.0,
            stop_loss=None,
            take_profit=None,
            timestamp=1234567890000,
            reason="Test signal"
        )
        
        stop_loss, take_profit = risk_manager.set_stop_loss_take_profit(
            decision=decision,
            entry_price=50000.0
        )
        
        assert stop_loss > 50000.0
        assert take_profit < 50000.0
    
    def test_set_stop_loss_take_profit_with_atr(self, risk_manager, sample_decision):
        """Test stop loss and take profit with ATR."""
        stop_loss, take_profit = risk_manager.set_stop_loss_take_profit(
            decision=sample_decision,
            entry_price=50000.0,
            atr=500.0
        )
        
        assert stop_loss < 50000.0
        assert take_profit > 50000.0
        # ATR-based: stop = 50000 - (500 * 2) = 49000
        assert stop_loss == 49000.0
    
    def test_set_stop_loss_take_profit_with_volatility(self, risk_manager, sample_decision):
        """Test stop loss and take profit with volatility adjustment."""
        stop_loss, take_profit = risk_manager.set_stop_loss_take_profit(
            decision=sample_decision,
            entry_price=50000.0,
            volatility=0.04
        )
        
        assert stop_loss < 50000.0
        assert take_profit > 50000.0
    
    def test_update_daily_pnl_profit(self, risk_manager):
        """Test updating daily P&L with profit."""
        risk_manager.update_daily_pnl(100.0)
        
        assert risk_manager.daily_pnl == 100.0
        assert risk_manager.consecutive_losses == 0
    
    def test_update_daily_pnl_loss(self, risk_manager):
        """Test updating daily P&L with loss."""
        risk_manager.update_daily_pnl(-50.0)
        
        assert risk_manager.daily_pnl == -50.0
        assert risk_manager.consecutive_losses == 1
    
    def test_update_daily_pnl_consecutive_losses(self, risk_manager):
        """Test consecutive loss counter."""
        risk_manager.update_daily_pnl(-50.0)
        risk_manager.update_daily_pnl(-30.0)
        
        assert risk_manager.consecutive_losses == 2
    
    def test_update_daily_pnl_reset_on_profit(self, risk_manager):
        """Test consecutive losses reset on profit."""
        risk_manager.update_daily_pnl(-50.0)
        risk_manager.update_daily_pnl(100.0)
        
        assert risk_manager.consecutive_losses == 0
    
    def test_reset_daily_pnl(self, risk_manager):
        """Test resetting daily P&L."""
        risk_manager.daily_pnl = 100.0
        risk_manager.consecutive_losses = 2
        
        risk_manager.reset_daily_pnl()
        
        assert risk_manager.daily_pnl == 0.0
        assert risk_manager.consecutive_losses == 2  # Not reset
    
    def test_get_risk_metrics(self, risk_manager):
        """Test getting risk metrics."""
        risk_manager.current_drawdown = 0.05
        risk_manager.peak_equity = 10000.0
        risk_manager.daily_pnl = 100.0
        risk_manager.consecutive_losses = 1
        
        metrics = risk_manager.get_risk_metrics()
        
        assert metrics["current_drawdown"] == 0.05
        assert metrics["peak_equity"] == 10000.0
        assert metrics["daily_pnl"] == 100.0
        assert metrics["consecutive_losses"] == 1
        assert metrics["max_position_size"] == 0.1
        assert metrics["max_portfolio_risk"] == 0.05
        assert metrics["max_drawdown_limit"] == 0.15
        assert metrics["risk_utilization"] == 0.05 / 0.15
    
    def test_should_reduce_risk_drawdown(self, risk_manager):
        """Test risk reduction due to high drawdown."""
        risk_manager.current_drawdown = 0.13  # 13% > 80% of 15% (0.12)
        
        result = risk_manager.should_reduce_risk()
        assert result is True
    
    def test_should_reduce_risk_consecutive_losses(self, risk_manager):
        """Test risk reduction due to consecutive losses."""
        risk_manager.consecutive_losses = 2  # 2 >= 3 - 1
        
        result = risk_manager.should_reduce_risk()
        assert result is True
    
    def test_should_reduce_risk_daily_losses(self, risk_manager):
        """Test risk reduction due to daily losses."""
        risk_manager.peak_equity = 10000.0
        risk_manager.daily_pnl = -170.0  # 1.7% > 80% of 2% (0.016)
        
        result = risk_manager.should_reduce_risk()
        assert result is True
    
    def test_should_reduce_risk_no_reduction(self, risk_manager):
        """Test no risk reduction needed."""
        risk_manager.current_drawdown = 0.05
        risk_manager.consecutive_losses = 1
        risk_manager.daily_pnl = -10.0
        risk_manager.peak_equity = 10000.0
        
        result = risk_manager.should_reduce_risk()
        assert result is False
    
    def test_get_risk_reduction_factor_drawdown(self, risk_manager):
        """Test risk reduction factor for high drawdown."""
        risk_manager.current_drawdown = 0.12  # 12% > 10%
        
        factor = risk_manager.get_risk_reduction_factor()
        assert factor == 0.5
    
    def test_get_risk_reduction_factor_consecutive_losses(self, risk_manager):
        """Test risk reduction factor for consecutive losses."""
        risk_manager.consecutive_losses = 2
        
        factor = risk_manager.get_risk_reduction_factor()
        assert factor == 0.7
    
    def test_get_risk_reduction_factor_daily_losses(self, risk_manager):
        """Test risk reduction factor for daily losses."""
        risk_manager.peak_equity = 10000.0
        risk_manager.daily_pnl = -110.0  # 1.1% > 50% of 2% (0.01)
        
        factor = risk_manager.get_risk_reduction_factor()
        assert factor == 0.6
    
    def test_get_risk_reduction_factor_no_reduction(self, risk_manager):
        """Test risk reduction factor when no reduction needed."""
        factor = risk_manager.get_risk_reduction_factor()
        assert factor == 1.0
