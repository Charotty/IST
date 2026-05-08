"""
Tests for Fixed Percentage Position Sizing (1-2%).

Tests the implementation of fixed percentage position sizing
within risk management framework.
"""

import pytest
import numpy as np

from its_project.decision.sizing import (
    PositionSizer, 
    SizingConfig, 
    SizingMethod,
    SizingResult
)
from its_project.decision.decision import Decision, Action


class TestFixedPercentageSizing:
    """Test fixed percentage position sizing functionality."""
    
    @pytest.fixture
    def fixed_percentage_config(self):
        """Create configuration for fixed percentage sizing."""
        return SizingConfig(
            method=SizingMethod.FIXED_PERCENTAGE,
            fixed_percentage_min=0.01,    # 1%
            fixed_percentage_max=0.02,    # 2%
            fixed_percentage_default=0.015,  # 1.5%
            max_risk_per_trade=0.01,
            max_portfolio_risk=0.05
        )
    
    @pytest.fixture
    def sample_decision(self):
        """Create sample trading decision."""
        return Decision(
            action=Action.BUY,
            symbol="BTCUSDT",
            size=0.0,
            price=50000.0,
            stop_loss=49000.0,
            take_profit=51000.0,
            timestamp=1234567890,
            reason="Test decision"
        )
    
    def test_fixed_percentage_sizing_min(self, fixed_percentage_config, sample_decision):
        """Test fixed percentage sizing at minimum (1%)."""
        sizer = PositionSizer(fixed_percentage_config)
        
        portfolio_value = 100000.0  # $100k portfolio
        current_positions = {}
        market_data = {"price": 50000.0}
        
        result = sizer.calculate_position_size(
            sample_decision, portfolio_value, current_positions, market_data
        )
        
        # Expected: 1% of portfolio = $1000
        expected_size = 1000.0
        expected_units = expected_size / 50000.0  # 0.02 BTC
        
        assert result.method_used == SizingMethod.FIXED_PERCENTAGE.value
        assert result.size_fraction == 0.01
        assert abs(result.size - expected_units) < 0.0001
        assert result.confidence_adjusted is False
        assert result.metadata['fixed_percentage'] == 0.01
        assert result.metadata['portfolio_value'] == portfolio_value
    
    def test_fixed_percentage_sizing_max(self, fixed_percentage_config, sample_decision):
        """Test fixed percentage sizing at maximum (2%)."""
        # Override config for max percentage
        config_max = SizingConfig(
            method=SizingMethod.FIXED_PERCENTAGE,
            fixed_percentage_min=0.01,
            fixed_percentage_max=0.02,
            fixed_percentage_default=0.02,  # Use max
            max_risk_per_trade=0.01
        )
        
        sizer = PositionSizer(config_max)
        
        portfolio_value = 100000.0
        current_positions = {}
        market_data = {"price": 50000.0}
        
        result = sizer.calculate_position_size(
            sample_decision, portfolio_value, current_positions, market_data
        )
        
        # Expected: 2% of portfolio = $2000
        expected_size = 2000.0
        expected_units = expected_size / 50000.0  # 0.04 BTC
        
        assert result.method_used == SizingMethod.FIXED_PERCENTAGE.value
        assert result.size_fraction == 0.02
        assert abs(result.size - expected_units) < 0.0001
        assert result.metadata['fixed_percentage'] == 0.02
    
    def test_fixed_percentage_sizing_default(self, fixed_percentage_config, sample_decision):
        """Test fixed percentage sizing at default (1.5%)."""
        sizer = PositionSizer(fixed_percentage_config)
        
        portfolio_value = 100000.0
        current_positions = {}
        market_data = {"price": 50000.0}
        
        result = sizer.calculate_position_size(
            sample_decision, portfolio_value, current_positions, market_data
        )
        
        # Expected: 1.5% of portfolio = $1500
        expected_size = 1500.0
        expected_units = expected_size / 50000.0  # 0.03 BTC
        
        assert result.method_used == SizingMethod.FIXED_PERCENTAGE.value
        assert result.size_fraction == 0.015
        assert abs(result.size - expected_units) < 0.0001
        assert result.metadata['fixed_percentage'] == 0.015
    
    def test_fixed_percentage_clamping(self, fixed_percentage_config, sample_decision):
        """Test percentage clamping within min-max range."""
        # Test below minimum
        config_below_min = SizingConfig(
            method=SizingMethod.FIXED_PERCENTAGE,
            fixed_percentage_min=0.01,
            fixed_percentage_max=0.02,
            fixed_percentage_default=0.005,  # Below min
        )
        
        sizer = PositionSizer(config_below_min)
        
        portfolio_value = 100000.0
        current_positions = {}
        market_data = {"price": 50000.0}
        
        result = sizer.calculate_position_size(
            sample_decision, portfolio_value, current_positions, market_data
        )
        
        # Should clamp to minimum (1%)
        assert result.size_fraction == 0.01  # Clamped to min
        assert result.metadata['fixed_percentage'] == 0.01
        
        # Test above maximum
        config_above_max = SizingConfig(
            method=SizingMethod.FIXED_PERCENTAGE,
            fixed_percentage_min=0.01,
            fixed_percentage_max=0.02,
            fixed_percentage_default=0.025,  # Above max
        )
        
        sizer = PositionSizer(config_above_max)
        
        result = sizer.calculate_position_size(
            sample_decision, portfolio_value, current_positions, market_data
        )
        
        # Should clamp to maximum (2%)
        assert result.size_fraction == 0.02  # Clamped to max
        assert result.metadata['fixed_percentage'] == 0.02
    
    def test_existing_position_adjustment(self, fixed_percentage_config, sample_decision):
        """Test position size adjustment for existing positions."""
        sizer = PositionSizer(fixed_percentage_config)
        
        portfolio_value = 100000.0
        # Simulate existing position
        current_positions = {"BTCUSDT": 0.01}  # 0.01 BTC existing
        market_data = {"price": 50000.0}
        
        result = sizer.calculate_position_size(
            sample_decision, portfolio_value, current_positions, market_data
        )
        
        # Should reduce size by 50% for existing position
        expected_fraction = fixed_percentage_config.fixed_percentage_default * 0.5  # 1.5% * 0.5 = 0.75%
        expected_size = portfolio_value * expected_fraction
        expected_units = expected_size / 50000.0
        
        assert result.size_fraction == expected_fraction
        assert abs(result.size - expected_units) < 0.0001
    
    def test_risk_calculation(self, fixed_percentage_config, sample_decision):
        """Test risk amount calculation."""
        sizer = PositionSizer(fixed_percentage_config)
        
        portfolio_value = 100000.0
        current_positions = {}
        market_data = {"price": 50000.0}
        
        result = sizer.calculate_position_size(
            sample_decision, portfolio_value, current_positions, market_data
        )
        
        # Risk amount should be size_value * max_risk_per_trade
        expected_size_value = portfolio_value * fixed_percentage_config.fixed_percentage_default
        expected_risk = expected_size_value * fixed_percentage_config.max_risk_per_trade
        
        assert abs(result.risk_amount - expected_risk) < 0.01
        assert result.risk_amount > 0
    
    def test_sell_decision_sizing(self, fixed_percentage_config):
        """Test sizing for SELL decisions."""
        sell_decision = Decision(
            action=Action.SELL,
            symbol="BTCUSDT",
            size=0.0,
            price=50000.0,
            stop_loss=51000.0,
            take_profit=49000.0,
            timestamp=1234567890,
            reason="Test SELL decision"
        )
        
        sizer = PositionSizer(fixed_percentage_config)
        
        portfolio_value = 100000.0
        current_positions = {}
        market_data = {"price": 50000.0}
        
        result = sizer.calculate_position_size(
            sell_decision, portfolio_value, current_positions, market_data
        )
        
        # Should work the same as BUY
        expected_size = portfolio_value * fixed_percentage_config.fixed_percentage_default
        expected_units = expected_size / 50000.0
        
        assert result.method_used == SizingMethod.FIXED_PERCENTAGE.value
        assert abs(result.size - expected_units) < 0.0001
        assert result.size_fraction == fixed_percentage_config.fixed_percentage_default
    
    def test_hold_decision_sizing(self, fixed_percentage_config):
        """Test sizing for HOLD decisions."""
        hold_decision = Decision(
            action=Action.HOLD,
            symbol="BTCUSDT",
            size=0.0,
            price=50000.0,
            stop_loss=None,
            take_profit=None,
            timestamp=1234567890,
            reason="Test HOLD decision"
        )
        
        sizer = PositionSizer(fixed_percentage_config)
        
        portfolio_value = 100000.0
        current_positions = {}
        market_data = {"price": 50000.0}
        
        result = sizer.calculate_position_size(
            hold_decision, portfolio_value, current_positions, market_data
        )
        
        # HOLD should return zero size
        assert result.size == 0.0
        assert result.size_fraction == 0.0
        assert result.risk_amount == 0.0
    
    def test_metadata_completeness(self, fixed_percentage_config, sample_decision):
        """Test metadata completeness for fixed percentage sizing."""
        sizer = PositionSizer(fixed_percentage_config)
        
        portfolio_value = 100000.0
        current_positions = {}
        market_data = {"price": 50000.0}
        
        result = sizer.calculate_position_size(
            sample_decision, portfolio_value, current_positions, market_data
        )
        
        # Check all required metadata fields
        required_metadata = [
            'size_value',
            'fixed_percentage',
            'portfolio_value',
            'percentage_range'
        ]
        
        for field in required_metadata:
            assert field in result.metadata, f"Missing metadata field: {field}"
        
        # Check percentage range format
        percentage_range = result.metadata['percentage_range']
        assert "1%" in percentage_range
        assert "2%" in percentage_range
    
    def test_statistics_tracking(self, fixed_percentage_config, sample_decision):
        """Test statistics tracking for sizing decisions."""
        sizer = PositionSizer(fixed_percentage_config)
        
        portfolio_value = 100000.0
        current_positions = {}
        market_data = {"price": 50000.0}
        
        # Make multiple decisions
        for i in range(5):
            sizer.calculate_position_size(
                sample_decision, portfolio_value, current_positions, market_data
            )
        
        stats = sizer.get_statistics()
        
        # Check statistics
        assert stats['total_sizing_decisions'] == 5
        assert SizingMethod.FIXED_PERCENTAGE.value in stats['method_usage']
        assert stats['method_usage'][SizingMethod.FIXED_PERCENTAGE.value] == 5


class TestFixedPercentageIntegration:
    """Test integration of fixed percentage sizing with other components."""
    
    def test_integration_with_enhanced_decision(self):
        """Test integration with EnhancedDecisionMaker."""
        # This would require importing EnhancedDecisionMaker
        # For now, test the sizing logic independently
        
        config = SizingConfig(
            method=SizingMethod.FIXED_PERCENTAGE,
            fixed_percentage_min=0.01,
            fixed_percentage_max=0.02,
            fixed_percentage_default=0.015
        )
        
        sizer = PositionSizer(config)
        
        # Test decision with confidence
        decision = Decision(
            action=Action.BUY,
            symbol="BTCUSDT",
            size=0.0,
            price=50000.0,
            stop_loss=49000.0,
            take_profit=51000.0,
            timestamp=1234567890,
            reason="Test decision with high confidence"
        )
        
        portfolio_value = 100000.0
        current_positions = {}
        market_data = {"price": 50000.0}
        
        result = sizer.calculate_position_size(
            decision, portfolio_value, current_positions, market_data
        )
        
        # Verify sizing is consistent and reasonable
        assert result.size > 0
        assert result.size_fraction == 0.015
        assert result.method_used == SizingMethod.FIXED_PERCENTAGE.value
        assert result.risk_amount > 0
    
    def test_portfolio_risk_limits(self):
        """Test portfolio risk limits with fixed percentage."""
        config = SizingConfig(
            method=SizingMethod.FIXED_PERCENTAGE,
            fixed_percentage_min=0.01,
            fixed_percentage_max=0.02,
            fixed_percentage_default=0.015,
            max_portfolio_risk=0.05  # 5% max portfolio risk
        )
        
        sizer = PositionSizer(config)
        
        portfolio_value = 100000.0
        current_positions = {}
        market_data = {"price": 50000.0}
        
        decision = Decision(
            action=Action.BUY,
            symbol="BTCUSDT",
            size=0.0,
            price=50000.0,
            stop_loss=49000.0,
            take_profit=51000.0,
            timestamp=1234567890,
            reason="Test decision"
        )
        
        result = sizer.calculate_position_size(
            decision, portfolio_value, current_positions, market_data
        )
        
        # Risk amount should be within portfolio limits
        portfolio_risk_pct = (result.risk_amount / portfolio_value) * 100
        assert portfolio_risk_pct <= config.max_portfolio_risk * 100


if __name__ == "__main__":
    pytest.main([__file__])
