"""
Tests for Enhanced Decision Maker with cooldown and volatility filtering.

Tests the additional filtering mechanisms:
- Cooldown: Prevents rapid successive signals
- Volatility filter: Blocks signals during high volatility periods
"""

import pytest
import numpy as np
from unittest.mock import Mock

from its_project.decision.enhanced_decision import (
    EnhancedDecisionMaker, 
    ConfidenceBasedDecisionMaker
)
from its_project.decision.decision import Signal, Action


class TestCooldownFiltering:
    """Test cooldown functionality."""
    
    @pytest.fixture
    def cooldown_config(self):
        """Create configuration with cooldown enabled."""
        return {
            "confidence_threshold": 0.7,
            "min_price_move": 0.0005,
            "use_price_sign_logic": True,
            "cooldown_enabled": True,
            "cooldown_period": 300,  # 5 minutes
            "position_size": 0.01,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04
        }
    
    def test_cooldown_initialization(self, cooldown_config):
        """Test cooldown initialization."""
        decision_maker = EnhancedDecisionMaker(cooldown_config)
        
        assert decision_maker.cooldown_enabled is True
        assert decision_maker.cooldown_period == 300
        assert len(decision_maker.last_signal_time) == 0
    
    def test_cooldown_first_signal(self, cooldown_config):
        """Test first signal (no cooldown)."""
        decision_maker = EnhancedDecisionMaker(cooldown_config)
        
        # Create first signal
        signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1000000,
            symbol="BTCUSDT",
            metadata={"delta_p_hat": 0.01}
        )
        
        decision = decision_maker.decide(signal, {"price": 50000.0})
        
        # Should allow the signal (no previous signal)
        assert decision is not None
        assert decision.action == Action.BUY
        assert "Cooldown" not in decision.reason
    
    def test_cooldown_second_signal_too_soon(self, cooldown_config):
        """Test second signal too soon (should be blocked)."""
        decision_maker = EnhancedDecisionMaker(cooldown_config)
        
        # First signal
        first_signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1000000,
            symbol="BTCUSDT",
            metadata={"delta_p_hat": 0.01}
        )
        
        decision_maker.decide(first_signal, {"price": 50000.0})
        
        # Second signal too soon (within cooldown)
        second_signal = Signal(
            action=Action.SELL,
            confidence=0.9,
            timestamp=1000200,  # 200 seconds later (within 300s cooldown)
            symbol="BTCUSDT",
            metadata={"delta_p_hat": -0.015}
        )
        
        decision = decision_maker.decide(second_signal, {"price": 50000.0})
        
        # Should be blocked due to cooldown
        assert decision is not None
        assert decision.action == Action.HOLD
        assert "Cooldown active" in decision.reason
        assert decision.size == 0.0
    
    def test_cooldown_after_period(self, cooldown_config):
        """Test signal after cooldown period (should be allowed)."""
        decision_maker = EnhancedDecisionMaker(cooldown_config)
        
        # First signal
        first_signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1000000,
            symbol="BTCUSDT",
            metadata={"delta_p_hat": 0.01}
        )
        
        decision_maker.decide(first_signal, {"price": 50000.0})
        
        # Second signal after cooldown period
        second_signal = Signal(
            action=Action.SELL,
            confidence=0.9,
            timestamp=1000400,  # 400 seconds later (beyond 300s cooldown)
            symbol="BTCUSDT",
            metadata={"delta_p_hat": -0.015}
        )
        
        decision = decision_maker.decide(second_signal, {"price": 50000.0})
        
        # Should be allowed (cooldown expired)
        assert decision is not None
        assert decision.action == Action.SELL
        assert "Cooldown" not in decision.reason
    
    def test_cooldown_per_symbol(self, cooldown_config):
        """Test cooldown tracking per symbol."""
        decision_maker = EnhancedDecisionMaker(cooldown_config)
        
        # First signal for BTC
        btc_signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1000000,
            symbol="BTCUSDT",
            metadata={"delta_p_hat": 0.01}
        )
        
        decision_maker.decide(btc_signal, {"price": 50000.0})
        
        # Second signal for BTC (should be blocked)
        btc_signal2 = Signal(
            action=Action.SELL,
            confidence=0.9,
            timestamp=1000100,
            symbol="BTCUSDT",
            metadata={"delta_p_hat": -0.015}
        )
        
        decision2 = decision_maker.decide(btc_signal2, {"price": 50000.0})
        
        # Should be blocked
        assert decision2.action == Action.HOLD
        assert "Cooldown active" in decision2.reason
        
        # Signal for ETH (different symbol, should be allowed)
        eth_signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1000200,
            symbol="ETHUSDT",
            metadata={"delta_p_hat": 0.01}
        )
        
        decision3 = decision_maker.decide(eth_signal, {"price": 50000.0})
        
        # Should be allowed (different symbol)
        assert decision3.action == Action.BUY
        assert "Cooldown" not in decision3.reason


class TestVolatilityFiltering:
    """Test volatility filtering functionality."""
    
    @pytest.fixture
    def volatility_config(self):
        """Create configuration with volatility filter enabled."""
        return {
            "confidence_threshold": 0.7,
            "min_price_move": 0.0005,
            "use_price_sign_logic": True,
            "volatility_filter_enabled": True,
            "volatility_window": 20,
            "max_volatility_threshold": 0.05,  # 5%
            "position_size": 0.01,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04
        }
    
    def test_volatility_initialization(self, volatility_config):
        """Test volatility filter initialization."""
        decision_maker = EnhancedDecisionMaker(volatility_config)
        
        assert decision_maker.volatility_filter_enabled is True
        assert decision_maker.volatility_window == 20
        assert decision_maker.max_volatility_threshold == 0.05
        assert len(decision_maker.volatility_history) == 0
    
    def test_volatility_normal_market(self, volatility_config):
        """Test signal in normal volatility (should be allowed)."""
        decision_maker = EnhancedDecisionMaker(volatility_config)
        
        # Setup market state with normal volatility
        market_state = {
            "price": 50000.0,
            "price_history": [50000, 50100, 49900, 50200] * 10  # 2% volatility
        }
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1000000,
            symbol="BTCUSDT",
            metadata={"delta_p_hat": 0.01}
        )
        
        decision = decision_maker.decide(signal, market_state)
        
        # Should be allowed (volatility = 0.02 < 0.05)
        assert decision is not None
        assert decision.action == Action.BUY
        assert "Volatility" not in decision.reason
    
    def test_volatility_high_market(self, volatility_config):
        """Test signal in high volatility (should be blocked)."""
        decision_maker = EnhancedDecisionMaker(volatility_config)
        
        # Setup market state with high volatility
        market_state = {
            "price": 50000.0,
            "price_history": [50000, 51000, 49000, 53000] * 10  # 6% volatility
        }
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1000000,
            symbol="BTCUSDT",
            metadata={"delta_p_hat": 0.01}
        )
        
        decision = decision_maker.decide(signal, market_state)
        
        # Should be blocked (volatility = 0.06 > 0.05)
        assert decision is not None
        assert decision.action == Action.HOLD
        assert "Volatility too high" in decision.reason
        assert "0.0600" in decision.reason  # Current volatility
    
    def test_volatility_history_tracking(self, volatility_config):
        """Test volatility history tracking."""
        decision_maker = EnhancedDecisionMaker(volatility_config)
        
        # Simulate multiple volatility readings
        volatilities = [0.02, 0.03, 0.04, 0.06, 0.03]
        
        for vol in volatilities:
            market_state = {
                "price": 50000.0,
                "price_history": [50000] * 20  # Dummy history
            }
            
            signal = Signal(
                action=Action.BUY,
                confidence=0.8,
                timestamp=1000000,
                symbol="BTCUSDT",
                metadata={"delta_p_hat": 0.01}
            )
            
            decision = decision_maker.decide(signal, market_state)
            # Update volatility history manually for testing
            decision_maker.volatility_history.append(vol)
        
        # Check that history is maintained correctly
        assert len(decision_maker.volatility_history) == len(volatilities)
        assert max(decision_maker.volatility_history) == 0.06
    
    def test_volatility_insufficient_history(self, volatility_config):
        """Test volatility filter with insufficient history."""
        decision_maker = EnhancedDecisionMaker(volatility_config)
        
        # Market state with very short history
        market_state = {
            "price": 50000.0,
            "price_history": [50000, 50100]  # Only 2 points
        }
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1000000,
            symbol="BTCUSDT",
            metadata={"delta_p_hat": 0.01}
        )
        
        decision = decision_maker.decide(signal, market_state)
        
        # Should be allowed (insufficient history, no filter)
        assert decision is not None
        assert decision.action == Action.BUY
        assert "Volatility" not in decision.reason


class TestCombinedFilters:
    """Test cooldown and volatility filters working together."""
    
    @pytest.fixture
    def combined_config(self):
        """Create configuration with both filters enabled."""
        return {
            "confidence_threshold": 0.7,
            "min_price_move": 0.0005,
            "use_price_sign_logic": True,
            "cooldown_enabled": True,
            "cooldown_period": 300,
            "volatility_filter_enabled": True,
            "volatility_window": 20,
            "max_volatility_threshold": 0.05,
            "position_size": 0.01,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04
        }
    
    def test_combined_filters_first_signal(self, combined_config):
        """Test first signal with both filters (should be allowed)."""
        decision_maker = EnhancedDecisionMaker(combined_config)
        
        # Normal market state
        market_state = {
            "price": 50000.0,
            "price_history": [50000, 50100, 49900, 50200] * 5  # 2% volatility
        }
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1000000,
            symbol="BTCUSDT",
            metadata={"delta_p_hat": 0.01}
        )
        
        decision = decision_maker.decide(signal, market_state)
        
        # Should be allowed
        assert decision is not None
        assert decision.action == Action.BUY
        assert "Cooldown" not in decision.reason
        assert "Volatility" not in decision.reason
    
    def test_combined_filters_cooldown_block(self, combined_config):
        """Test signal blocked by cooldown."""
        decision_maker = EnhancedDecisionMaker(combined_config)
        
        # First signal
        first_signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1000000,
            symbol="BTCUSDT",
            metadata={"delta_p_hat": 0.01}
        )
        
        decision_maker.decide(first_signal, {"price": 50000.0})
        
        # Second signal during cooldown
        second_signal = Signal(
            action=Action.SELL,
            confidence=0.9,
            timestamp=1000100,  # Within cooldown
            symbol="BTCUSDT",
            metadata={"delta_p_hat": -0.015}
        )
        
        decision2 = decision_maker.decide(second_signal, {"price": 50000.0})
        
        # Should be blocked by cooldown (even if volatility is normal)
        assert decision2.action == Action.HOLD
        assert "Cooldown active" in decision2.reason
    
    def test_combined_filters_volatility_block(self, combined_config):
        """Test signal blocked by high volatility."""
        decision_maker = EnhancedDecisionMaker(combined_config)
        
        # High volatility market state
        market_state = {
            "price": 50000.0,
            "price_history": [50000, 51000, 49000, 53000] * 10  # 6% volatility
        }
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1000400,  # After cooldown period
            symbol="BTCUSDT",
            metadata={"delta_p_hat": 0.01}
        )
        
        decision = decision_maker.decide(signal, market_state)
        
        # Should be blocked by volatility
        assert decision.action == Action.HOLD
        assert "Volatility too high" in decision.reason
        assert "0.0600" in decision.reason


class TestFilterConfiguration:
    """Test filter configuration and updates."""
    
    def test_config_update_add_filters(self, combined_config):
        """Test adding filters to existing configuration."""
        decision_maker = EnhancedDecisionMaker({
            "confidence_threshold": 0.7,
            "min_price_move": 0.0005,
            "use_price_sign_logic": True
        })
        
        # Initially no filters
        assert decision_maker.cooldown_enabled is False
        assert decision_maker.volatility_filter_enabled is False
        
        # Update with filters
        new_config = {
            "cooldown_enabled": True,
            "cooldown_period": 600,
            "volatility_filter_enabled": True,
            "volatility_window": 30,
            "max_volatility_threshold": 0.08
        }
        
        decision_maker.update_config(new_config)
        
        # Verify filters are enabled
        assert decision_maker.cooldown_enabled is True
        assert decision_maker.cooldown_period == 600
        assert decision_maker.volatility_filter_enabled is True
        assert decision_maker.volatility_window == 30
        assert decision_maker.max_volatility_threshold == 0.08
    
    def test_config_update_disable_filters(self, combined_config):
        """Test disabling filters."""
        decision_maker = EnhancedDecisionMaker(combined_config)
        
        # Disable filters
        new_config = {
            "cooldown_enabled": False,
            "volatility_filter_enabled": False
        }
        
        decision_maker.update_config(new_config)
        
        # Verify filters are disabled
        assert decision_maker.cooldown_enabled is False
        assert decision_maker.volatility_filter_enabled is False
    
    def test_filter_statistics(self, combined_config):
        """Test filter statistics reporting."""
        decision_maker = EnhancedDecisionMaker(combined_config)
        
        # Setup some history
        decision_maker.volatility_history = [0.02, 0.03, 0.04]
        decision_maker.last_signal_time = {"BTCUSDT": 1000000 - 1000}  # In cooldown
        
        stats = decision_maker.get_decision_stats()
        
        # Check that filter parameters are included
        assert "cooldown_enabled" in stats
        assert "volatility_filter_enabled" in stats
        assert "cooldown_period" in stats
        assert "max_volatility_threshold" in stats
        assert "volatility_window" in stats


if __name__ == "__main__":
    pytest.main([__file__])
