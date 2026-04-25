"""
Unit tests for SimpleDecisionMaker.
"""
import pytest
from its_project.decision.decision import Action, Signal, Decision
from its_project.decision.simple import SimpleDecisionMaker


@pytest.mark.unit
@pytest.mark.decision_layer
class TestSimpleDecisionMaker:
    """Test SimpleDecisionMaker functionality."""
    
    def test_initialization(self):
        """Test SimpleDecisionMaker initialization."""
        config = {
            'confidence_threshold': 0.8,
            'position_size': 0.02,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.05
        }
        
        maker = SimpleDecisionMaker(config)
        
        assert maker.config == config
        assert maker.confidence_threshold == 0.8
    
    def test_decide_buy_with_high_confidence(self):
        """Test buy decision with high confidence."""
        config = {'confidence_threshold': 0.7, 'position_size': 0.01}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.85,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        
        decision = maker.decide(signal, market_state)
        
        assert decision is not None
        assert decision.action == Action.BUY
        assert decision.symbol == 'BTC/USDT'
        assert decision.size == 0.01 * 0.85  # base_size * confidence
        assert decision.stop_loss == 42000.0 * (1 - 0.02)  # default stop_loss_pct
        assert decision.take_profit == 42000.0 * (1 + 0.04)  # default take_profit_pct
    
    def test_decide_sell_with_high_confidence(self):
        """Test sell decision with high confidence."""
        config = {'confidence_threshold': 0.7, 'position_size': 0.01}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.SELL,
            confidence=0.9,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        
        decision = maker.decide(signal, market_state)
        
        assert decision is not None
        assert decision.action == Action.SELL
        assert decision.stop_loss == 42000.0 * (1 + 0.02)  # stop loss above for sell
        assert decision.take_profit == 42000.0 * (1 - 0.04)  # take profit below for sell
    
    def test_decide_hold_with_low_confidence(self):
        """Test that decision returns None when confidence is below threshold."""
        config = {'confidence_threshold': 0.8}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.6,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        
        decision = maker.decide(signal, market_state)
        
        # When confidence is below threshold, validate_signal returns False, so decide returns None
        assert decision is None
    
    def test_decide_with_no_price(self):
        """Test decision when no price in market state."""
        config = {'confidence_threshold': 0.7, 'position_size': 0.01}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.85,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {'price': 0}
        
        decision = maker.decide(signal, market_state)
        
        assert decision is not None
        assert decision.stop_loss is None
        assert decision.take_profit is None
    
    def test_validate_signal_invalid_confidence(self):
        """Test signal validation with invalid confidence."""
        config = {'confidence_threshold': 0.7}
        maker = SimpleDecisionMaker(config)
        
        # Confidence > 1
        signal = Signal(
            action=Action.BUY,
            confidence=1.5,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        assert maker.validate_signal(signal) is False
        
        # Confidence < 0
        signal.confidence = -0.1
        assert maker.validate_signal(signal) is False
    
    def test_validate_signal_below_threshold(self):
        """Test signal validation when confidence below threshold."""
        config = {'confidence_threshold': 0.8}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.75,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        assert maker.validate_signal(signal) is False
    
    def test_validate_signal_valid(self):
        """Test signal validation with valid signal."""
        config = {'confidence_threshold': 0.7}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        assert maker.validate_signal(signal) is True
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        config = {'position_size': 0.05}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {}
        
        size = maker._calculate_position_size(signal, market_state)
        
        assert size == 0.05 * 0.8  # base_size * confidence
    
    def test_calculate_risk_levels_buy(self):
        """Test risk level calculation for buy."""
        config = {'stop_loss_pct': 0.02, 'take_profit_pct': 0.04}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        
        stop_loss, take_profit = maker._calculate_risk_levels(signal, market_state)
        
        assert stop_loss == 42000.0 * 0.98  # 1 - 0.02
        assert take_profit == 42000.0 * 1.04  # 1 + 0.04
    
    def test_calculate_risk_levels_sell(self):
        """Test risk level calculation for sell."""
        config = {'stop_loss_pct': 0.02, 'take_profit_pct': 0.04}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.SELL,
            confidence=0.8,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        
        stop_loss, take_profit = maker._calculate_risk_levels(signal, market_state)
        
        assert stop_loss == 42000.0 * 1.02  # 1 + 0.02
        assert take_profit == 42000.0 * 0.96  # 1 - 0.04
    
    def test_calculate_risk_levels_hold(self):
        """Test risk level calculation for hold action."""
        config = {'stop_loss_pct': 0.02, 'take_profit_pct': 0.04}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.HOLD,
            confidence=0.8,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        
        stop_loss, take_profit = maker._calculate_risk_levels(signal, market_state)
        
        assert stop_loss is None
        assert take_profit is None
    
    def test_decide_hold_action(self):
        """Test decision with HOLD action."""
        config = {'confidence_threshold': 0.7, 'position_size': 0.01}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.HOLD,
            confidence=0.85,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        
        decision = maker.decide(signal, market_state)
        
        assert decision is not None
        assert decision.action == Action.HOLD
        assert decision.size == 0.01 * 0.85
        assert decision.stop_loss is None
        assert decision.take_profit is None
    
    def test_decide_with_custom_risk_params(self):
        """Test decision with custom stop loss and take profit percentages."""
        config = {
            'confidence_threshold': 0.7,
            'position_size': 0.01,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10
        }
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.85,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        
        decision = maker.decide(signal, market_state)
        
        assert decision is not None
        assert decision.stop_loss == 42000.0 * 0.95  # 1 - 0.05
        assert decision.take_profit == 42000.0 * 1.10  # 1 + 0.10
    
    def test_calculate_position_size_default_config(self):
        """Test position size calculation with default config."""
        config = {}  # No position_size in config
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {}
        
        size = maker._calculate_position_size(signal, market_state)
        
        assert size == 0.01 * 0.8  # default position_size is 0.01
    
    def test_calculate_risk_levels_default_config(self):
        """Test risk level calculation with default config."""
        config = {}  # No risk params in config
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        
        stop_loss, take_profit = maker._calculate_risk_levels(signal, market_state)
        
        assert stop_loss == 42000.0 * 0.98  # default stop_loss_pct is 0.02
        assert take_profit == 42000.0 * 1.04  # default take_profit_pct is 0.04
    
    def test_calculate_risk_levels_missing_price_key(self):
        """Test risk level calculation when price key is missing."""
        config = {'stop_loss_pct': 0.02, 'take_profit_pct': 0.04}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {}  # No price key
        
        stop_loss, take_profit = maker._calculate_risk_levels(signal, market_state)
        
        assert stop_loss is None
        assert take_profit is None
    
    def test_decide_with_empty_market_state(self):
        """Test decision with empty market state."""
        config = {'confidence_threshold': 0.7, 'position_size': 0.01}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.85,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {}
        
        decision = maker.decide(signal, market_state)
        
        assert decision is not None
        assert decision.stop_loss is None
        assert decision.take_profit is None
    
    def test_decide_with_zero_confidence(self):
        """Test decision with zero confidence."""
        config = {'confidence_threshold': 0.7}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.0,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        
        decision = maker.decide(signal, market_state)
        
        # Zero confidence should fail validation
        assert decision is None
    
    def test_decide_with_edge_confidence(self):
        """Test decision with confidence exactly at threshold."""
        config = {'confidence_threshold': 0.7, 'position_size': 0.01}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.7,  # Exactly at threshold
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        
        decision = maker.decide(signal, market_state)
        
        # At threshold, should pass validation but might return HOLD due to low confidence
        # Actually, looking at the code, if confidence < threshold, it returns HOLD decision
        # So at exactly threshold, it should return a BUY decision
        assert decision is not None
        assert decision.action == Action.BUY
    
    def test_decide_reason_high_confidence(self):
        """Test that decision reason includes confidence info when above threshold."""
        config = {'confidence_threshold': 0.7, 'position_size': 0.01}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.85,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        
        decision = maker.decide(signal, market_state)
        
        assert decision is not None
        assert "Confidence" in decision.reason
        assert "0.85" in decision.reason
    
    def test_calculate_position_size_with_zero_confidence(self):
        """Test position size calculation with zero confidence."""
        config = {'position_size': 0.05}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.0,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {}
        
        size = maker._calculate_position_size(signal, market_state)
        
        assert size == 0.0
    
    def test_calculate_position_size_with_max_confidence(self):
        """Test position size calculation with max confidence."""
        config = {'position_size': 0.05}
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.BUY,
            confidence=1.0,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {}
        
        size = maker._calculate_position_size(signal, market_state)
        
        assert size == 0.05
    
    def test_decide_sell_with_custom_risk_params(self):
        """Test sell decision with custom risk parameters."""
        config = {
            'confidence_threshold': 0.7,
            'position_size': 0.01,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.06
        }
        maker = SimpleDecisionMaker(config)
        
        signal = Signal(
            action=Action.SELL,
            confidence=0.9,
            timestamp=1704067200000,
            symbol='BTC/USDT',
            metadata={}
        )
        
        market_state = {'price': 42000.0}
        
        decision = maker.decide(signal, market_state)
        
        assert decision is not None
        assert decision.action == Action.SELL
        assert decision.stop_loss == 42000.0 * 1.03  # 1 + 0.03
        assert decision.take_profit == 42000.0 * 0.94  # 1 - 0.06
