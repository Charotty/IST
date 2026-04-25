"""
Unit tests for PositionSizer.
"""
import pytest
from its_project.decision.sizer import PositionSizer
from its_project.decision.decision import Signal, Action


@pytest.mark.unit
@pytest.mark.decision_layer
class TestPositionSizer:
    """Test PositionSizer functionality."""
    
    def test_initialization(self):
        """Test PositionSizer initialization."""
        config = {'method': 'fixed', 'size': 0.05}
        sizer = PositionSizer(config)
        
        assert sizer.config == config
        assert sizer.method == 'fixed'
    
    def test_default_config(self):
        """Test default configuration values."""
        sizer = PositionSizer({})
        
        assert sizer.method == 'fixed'
    
    def test_calculate_size_fixed(self):
        """Test fixed size calculation."""
        config = {'method': 'fixed', 'size': 0.1}
        sizer = PositionSizer(config)
        
        signal = Signal(
            symbol="BTC/USDT",
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            metadata={}
        )
        
        size = sizer.calculate_size(signal, 10000.0)
        
        assert size == 0.1
    
    def test_calculate_size_fixed_fraction(self):
        """Test fixed fraction calculation."""
        config = {'method': 'fixed_fraction', 'fraction': 0.05}
        sizer = PositionSizer(config)
        
        signal = Signal(
            symbol="BTC/USDT",
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            metadata={}
        )
        
        size = sizer.calculate_size(signal, 10000.0)
        
        assert size == 500.0  # 10000 * 0.05
    
    def test_calculate_size_kelly(self):
        """Test Kelly Criterion calculation."""
        config = {'method': 'kelly', 'win_loss_ratio': 2.0, 'kelly_fraction': 0.25}
        sizer = PositionSizer(config)
        
        signal = Signal(
            symbol="BTC/USDT",
            action=Action.BUY,
            confidence=0.6,
            timestamp=1704067200000,
            metadata={}
        )
        
        size = sizer.calculate_size(signal, 10000.0)
        
        # Kelly: (0.6 * 2 - 0.4) / 2 = 0.4
        # Conservative: 0.4 * 0.25 = 0.1
        # Size: 10000 * 0.1 = 1000
        assert size == pytest.approx(1000.0)
    
    def test_calculate_size_kelly_negative(self):
        """Test Kelly Criterion with negative result (should return 0)."""
        config = {'method': 'kelly', 'win_loss_ratio': 2.0, 'kelly_fraction': 0.25}
        sizer = PositionSizer(config)
        
        signal = Signal(
            symbol="BTC/USDT",
            action=Action.BUY,
            confidence=0.3,  # Low confidence
            timestamp=1704067200000,
            metadata={}
        )
        
        size = sizer.calculate_size(signal, 10000.0)
        
        # Kelly: (0.3 * 2 - 0.7) / 2 = -0.05 -> max(0, -0.05) = 0
        assert size == 0.0
    
    def test_calculate_size_risk_based(self):
        """Test risk-based sizing."""
        config = {'method': 'risk_based', 'stop_loss_pct': 0.02}
        sizer = PositionSizer(config)
        
        signal = Signal(
            symbol="BTC/USDT",
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            metadata={'price': 42000.0}
        )
        
        size = sizer.calculate_size(signal, 10000.0, risk_per_trade=0.02)
        
        # Risk amount: 10000 * 0.02 = 200
        # Stop distance: 42000 * 0.02 = 840
        # Size: 200 / 840 ≈ 0.238
        assert size > 0
        assert size < 1
    
    def test_calculate_size_unknown_method(self):
        """Test with unknown method."""
        config = {'method': 'unknown'}
        sizer = PositionSizer(config)
        
        signal = Signal(
            symbol="BTC/USDT",
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            metadata={}
        )
        
        with pytest.raises(ValueError, match="Unknown sizing method"):
            sizer.calculate_size(signal, 10000.0)
    
    def test_fixed_size_default(self):
        """Test fixed size with default value."""
        config = {'method': 'fixed'}
        sizer = PositionSizer(config)
        
        signal = Signal(
            symbol="BTC/USDT",
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            metadata={}
        )
        
        size = sizer.calculate_size(signal, 10000.0)
        
        assert size == 0.01  # Default size
    
    def test_fixed_fraction_default(self):
        """Test fixed fraction with default value."""
        config = {'method': 'fixed_fraction'}
        sizer = PositionSizer(config)
        
        signal = Signal(
            symbol="BTC/USDT",
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            metadata={}
        )
        
        size = sizer.calculate_size(signal, 10000.0)
        
        assert size == 200.0  # 10000 * 0.02 (default fraction)
    
    def test_risk_based_default_stop_loss(self):
        """Test risk-based sizing with default stop loss."""
        config = {'method': 'risk_based'}
        sizer = PositionSizer(config)
        
        signal = Signal(
            symbol="BTC/USDT",
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            metadata={'price': 42000.0}
        )
        
        size = sizer.calculate_size(signal, 10000.0, risk_per_trade=0.02)
        
        # Should use default stop_loss_pct of 0.02
        assert size > 0
    
    def test_risk_based_no_price_metadata(self):
        """Test risk-based sizing without price in metadata."""
        config = {'method': 'risk_based', 'stop_loss_pct': 0.02}
        sizer = PositionSizer(config)
        
        signal = Signal(
            symbol="BTC/USDT",
            action=Action.BUY,
            confidence=0.8,
            timestamp=1704067200000,
            metadata={}  # No price
        )
        
        size = sizer.calculate_size(signal, 10000.0, risk_per_trade=0.02)
        
        # Should use default price of 100
        # Risk amount: 10000 * 0.02 = 200
        # Stop distance: 100 * 0.02 = 2
        # Size: 200 / 2 = 100
        assert size == 100.0
