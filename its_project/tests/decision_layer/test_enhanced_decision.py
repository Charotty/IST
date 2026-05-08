"""
Tests for Enhanced Decision Maker with ΔP_hat sign logic.

Tests the implementation of:
- action = sign(ΔP_hat)
- Confidence thresholding: p_hat < threshold → HOLD  
- Minimum movement filtering: |ΔP_hat| < min_move → HOLD
"""

import pytest
import numpy as np

from its_project.decision.enhanced_decision import (
    EnhancedDecisionMaker, 
    ConfidenceBasedDecisionMaker
)
from its_project.decision.decision import Signal, Action


class TestEnhancedDecisionMaker:
    """Test enhanced decision maker functionality."""
    
    @pytest.fixture
    def enhanced_config(self):
        """Create enhanced decision maker configuration."""
        return {
            "confidence_threshold": 0.7,
            "min_price_move": 0.0005,
            "use_price_sign_logic": True,
            "fallback_to_hold": True,
            "position_size": 0.01,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04
        }
    
    def test_initialization(self, enhanced_config):
        """Test enhanced decision maker initialization."""
        decision_maker = EnhancedDecisionMaker(enhanced_config)
        
        assert decision_maker.confidence_threshold == 0.7
        assert decision_maker.min_price_move == 0.0005
        assert decision_maker.use_price_sign_logic is True
        assert decision_maker.fallback_to_hold is True
    
    def test_delta_p_hat_sign_logic(self, enhanced_config):
        """Test ΔP_hat sign to action conversion."""
        decision_maker = EnhancedDecisionMaker(enhanced_config)
        
        # Test positive ΔP_hat
        action = decision_maker._delta_p_to_action(0.01)
        assert action == Action.BUY
        
        # Test negative ΔP_hat
        action = decision_maker._delta_p_to_action(-0.01)
        assert action == Action.SELL
        
        # Test zero ΔP_hat
        action = decision_maker._delta_p_to_action(0.0)
        assert action == Action.HOLD
    
    def test_confidence_filtering(self, enhanced_config):
        """Test confidence thresholding."""
        decision_maker = EnhancedDecisionMaker(enhanced_config)
        
        # Create signal with low confidence
        low_conf_signal = Signal(
            action=Action.BUY,
            confidence=0.5,  # Below threshold
            timestamp=1234567890,
            symbol="BTCUSDT",
            metadata={"delta_p_hat": 0.01}
        )
        
        decision = decision_maker.decide(low_conf_signal, {})
        
        # Should return HOLD due to low confidence
        assert decision.action == Action.HOLD
        assert "Low confidence" in decision.reason
        assert decision.size == 0.0
    
    def test_minimum_movement_filtering(self, enhanced_config):
        """Test minimum movement filtering."""
        decision_maker = EnhancedDecisionMaker(enhanced_config)
        
        # Create signal with small ΔP_hat
        small_move_signal = Signal(
            action=Action.BUY,
            confidence=0.8,  # High confidence
            timestamp=1234567890,
            symbol="BTCUSDT",
            metadata={"delta_p_hat": 0.0003}  # Below min_move
        )
        
        decision = decision_maker.decide(small_move_signal, {})
        
        # Should return HOLD due to insufficient movement
        assert decision.action == Action.HOLD
        assert "Insufficient price move" in decision.reason
        assert decision.size == 0.0
    
    def test_position_size_calculation(self, enhanced_config):
        """Test enhanced position size calculation."""
        decision_maker = EnhancedDecisionMaker(enhanced_config)
        
        # Test with different ΔP_hat and confidence levels
        test_cases = [
            {"delta": 0.01, "conf": 0.8, "expected_factor": 1.6},  # Large move, high conf
            {"delta": 0.005, "conf": 0.6, "expected_factor": 1.0},  # Medium move, medium conf
            {"delta": 0.002, "conf": 0.9, "expected_factor": 1.8},  # Small move, very high conf
        ]
        
        for case in test_cases:
            signal = Signal(
                action=Action.BUY,
                confidence=case["conf"],
                timestamp=1234567890,
                symbol="BTCUSDT",
                metadata={"delta_p_hat": case["delta"]}
            )
            
            position_size = decision_maker._calculate_position_size(signal, {}, case["delta"], case["conf"])
            
            # Check calculation: base_size * conf * magnitude_factor
            expected_size = 0.01 * case["conf"] * case["expected_factor"]
            assert abs(position_size - expected_size) < 0.0001
    
    def test_risk_level_calculation(self, enhanced_config):
        """Test dynamic risk level calculation."""
        decision_maker = EnhancedDecisionMaker(enhanced_config)
        
        # Test with different ΔP_hat magnitudes
        test_cases = [
            {"delta": 0.01, "action": Action.BUY, "expected_adjustment": 1.0},
            {"delta": 0.002, "action": Action.BUY, "expected_adjustment": 0.5},
            {"delta": -0.01, "action": Action.SELL, "expected_adjustment": 1.0},
        ]
        
        market_state = {"price": 50000.0}
        
        for case in test_cases:
            signal = Signal(
                action=case["action"],
                confidence=0.8,
                timestamp=1234567890,
                symbol="BTCUSDT",
                metadata={"delta_p_hat": case["delta"]}
            )
            
            stop_loss, take_profit = decision_maker._calculate_risk_levels(
                signal, market_state, case["delta"], case["action"]
            )
            
            # Verify risk adjustment was applied
            assert stop_loss is not None
            assert take_profit is not None
            
            if case["action"] == Action.BUY:
                assert stop_loss < 50000.0  # Stop loss below current price
                assert take_profit > 50000.0  # Take profit above current price
            elif case["action"] == Action.SELL:
                assert stop_loss > 50000.0  # Stop loss above current price
                assert take_profit < 50000.0  # Take profit below current price
    
    def test_end_to_end_decision_flow(self, enhanced_config):
        """Test complete decision making flow."""
        decision_maker = EnhancedDecisionMaker(enhanced_config)
        
        # Test case 1: High confidence, sufficient movement
        signal1 = Signal(
            action=Action.BUY,
            confidence=0.9,
            timestamp=1234567890,
            symbol="BTCUSDT",
            metadata={"delta_p_hat": 0.015}
        )
        
        decision1 = decision_maker.decide(signal1, {"price": 50000.0})
        
        assert decision1.action == Action.BUY
        assert decision1.size > 0
        assert "ΔP_hat sign logic" in decision1.reason
        assert decision1.stop_loss is not None
        assert decision1.take_profit is not None
        
        # Test case 2: Low confidence
        signal2 = Signal(
            action=Action.SELL,
            confidence=0.4,
            timestamp=1234567890,
            symbol="BTCUSDT",
            metadata={"delta_p_hat": -0.02}
        )
        
        decision2 = decision_maker.decide(signal2, {"price": 50000.0})
        
        assert decision2.action == Action.HOLD
        assert "Low confidence" in decision2.reason
        assert decision2.size == 0.0
        
        # Test case 3: Insufficient movement
        signal3 = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1234567890,
            symbol="BTCUSDT",
            metadata={"delta_p_hat": 0.0003}
        )
        
        decision3 = decision_maker.decide(signal3, {"price": 50000.0})
        
        assert decision3.action == Action.HOLD
        assert "Insufficient price move" in decision3.reason
        assert decision3.size == 0.0
    
    def test_config_update(self, enhanced_config):
        """Test configuration update functionality."""
        decision_maker = EnhancedDecisionMaker(enhanced_config)
        
        # Update configuration
        new_config = {
            "confidence_threshold": 0.8,
            "min_price_move": 0.001,
            "use_price_sign_logic": False
        }
        
        decision_maker.update_config(new_config)
        
        assert decision_maker.confidence_threshold == 0.8
        assert decision_maker.min_price_move == 0.001
        assert decision_maker.use_price_sign_logic is False
    
    def test_decision_stats(self, enhanced_config):
        """Test decision statistics functionality."""
        decision_maker = EnhancedDecisionMaker(enhanced_config)
        
        stats = decision_maker.get_decision_stats()
        
        required_keys = [
            "confidence_threshold", "min_price_move", "use_price_sign_logic",
            "fallback_to_hold", "config"
        ]
        
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"
        
        assert stats["confidence_threshold"] == 0.7
        assert stats["min_price_move"] == 0.0005
        assert stats["use_price_sign_logic"] is True


class TestConfidenceBasedDecisionMaker:
    """Test confidence-based decision maker."""
    
    @pytest.fixture
    def confidence_config(self):
        """Create confidence-based decision maker configuration."""
        return {
            "confidence_threshold": 0.6,
            "strict_confidence": True,
            "confidence_decay": 0.1,
            "min_confidence_for_trade": 0.8,
            "confidence_history_length": 100,
            "position_size": 0.01
        }
    
    def test_adjusted_confidence_calculation(self, confidence_config):
        """Test confidence adjustment based on history."""
        decision_maker = ConfidenceBasedDecisionMaker(confidence_config)
        
        # Simulate confidence history
        decision_maker.confidence_history = [0.7, 0.8, 0.9, 0.85, 0.75, 0.6, 0.5]
        
        # Test current confidence higher than recent average
        adjusted = decision_maker._calculate_adjusted_confidence(0.9)
        # Should not apply decay (confidence is improving)
        assert adjusted == 0.9
        
        # Test current confidence lower than recent average
        adjusted = decision_maker._calculate_adjusted_confidence(0.6)
        # Should apply decay
        expected = 0.6 * (1 - 0.1)  # 0.54
        assert abs(adjusted - expected) < 0.01
    
    def test_confidence_based_sizing(self, confidence_config):
        """Test confidence-based position sizing."""
        decision_maker = ConfidenceBasedDecisionMaker(confidence_config)
        
        # Test different confidence levels
        test_cases = [
            {"conf": 0.5, "expected_multiplier": 0.5 ** 1.5},  # 0.353
            {"conf": 0.8, "expected_multiplier": 0.8 ** 1.5},  # 0.715
            {"conf": 1.0, "expected_multiplier": 1.0 ** 1.5},  # 1.0
        ]
        
        for case in test_cases:
            signal = Signal(
                action=Action.BUY,
                confidence=case["conf"],
                timestamp=1234567890,
                symbol="BTCUSDT"
            )
            
            position_size = decision_maker._calculate_confidence_based_size(case["conf"], signal)
            
            assert abs(position_size - (0.01 * case["expected_multiplier"])) < 0.0001
    
    def test_confidence_statistics(self, confidence_config):
        """Test confidence statistics calculation."""
        decision_maker = ConfidenceBasedDecisionMaker(confidence_config)
        
        # Setup confidence history
        decision_maker.confidence_history = [0.6, 0.7, 0.8, 0.9, 0.85, 0.75, 0.65, 0.8, 0.7]
        
        stats = decision_maker.get_confidence_stats()
        
        assert stats["current_confidence"] == 0.7
        assert abs(stats["avg_confidence"] - 0.75) < 0.01
        assert stats["min_confidence"] == 0.6
        assert stats["max_confidence"] == 0.9
        assert stats["history_length"] == 10
        assert stats["strict_mode"] is True
        assert stats["decay_factor"] == 0.1


class TestIntegrationWithWeightedEnsemble:
    """Test integration with WeightedEnsemble."""
    
    def test_signal_metadata_compatibility(self):
        """Test that enhanced decision maker works with WeightedEnsemble signals."""
        # This test would require importing WeightedEnsemble
        # For now, test the expected metadata structure
        
        enhanced_config = {
            "confidence_threshold": 0.7,
            "min_price_move": 0.0005,
            "use_price_sign_logic": True
        }
        
        decision_maker = EnhancedDecisionMaker(enhanced_config)
        
        # Test signal with ΔP_hat metadata
        signal_with_delta = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1234567890,
            symbol="BTCUSDT",
            metadata={
                "delta_p_hat": 0.015,  # From WeightedEnsemble
                "p_hat": 0.8,           # From WeightedEnsemble
                "classification": 2,        # From WeightedEnsemble
                "threshold": 0.7,          # From WeightedEnsemble
                "action_logic": "sign(delta_p_hat)"
            }
        )
        
        decision = decision_maker.decide(signal_with_delta, {"price": 50000.0})
        
        # Should use ΔP_hat sign logic
        assert decision.action == Action.BUY
        assert "ΔP_hat sign logic" in decision.reason
        assert decision.size > 0


if __name__ == "__main__":
    pytest.main([__file__])
