"""
Unit tests for MicrostructureFeatures.
"""
import pytest
import numpy as np
import pandas as pd
from its_project.features.microstructure import (
    MicrostructureFeatures,
    _roll_effective_spread,
    _vpin,
    _realized_volatility,
    _amihud_illiquidity,
    _kyle_lambda
)


@pytest.mark.unit
@pytest.mark.features_layer
class TestMicrostructureHelpers:
    """Test microstructure helper functions."""
    
    def test_roll_effective_spread(self):
        """Test Roll effective spread calculation."""
        prices = np.array([100, 101, 102, 101, 100])
        result = _roll_effective_spread(prices)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_roll_effective_spread_insufficient_data(self):
        """Test Roll with insufficient data."""
        prices = np.array([100])
        result = _roll_effective_spread(prices)
        
        assert np.isnan(result)
    
    def test_vpin(self):
        """Test VPIN calculation."""
        buy_volume = np.array([100, 150, 200])
        sell_volume = np.array([50, 100, 150])
        result = _vpin(buy_volume, sell_volume)
        
        assert isinstance(result, (float, np.ndarray))
        if isinstance(result, np.ndarray):
            assert not np.any(np.isnan(result))
            assert np.all((result >= 0) & (result <= 1))
        else:
            assert not np.isnan(result)
            assert 0 <= result <= 1
    
    def test_vpin_zero_volume(self):
        """Test VPIN with zero total volume."""
        buy_volume = np.array([0, 0])
        sell_volume = np.array([0, 0])
        result = _vpin(buy_volume, sell_volume)
        
        if isinstance(result, np.ndarray):
            assert np.all(np.isnan(result))
        else:
            assert np.isnan(result)
    
    def test_realized_volatility(self):
        """Test realized volatility calculation."""
        returns = np.array([0.01, -0.02, 0.03, -0.01])
        result = _realized_volatility(returns)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert result >= 0
    
    def test_realized_volatility_empty(self):
        """Test realized volatility with empty data."""
        returns = np.array([])
        result = _realized_volatility(returns)
        
        assert np.isnan(result)
    
    def test_amihud_illiquidity(self):
        """Test Amihud illiquidity calculation."""
        returns = np.array([0.01, -0.02, 0.03])
        volumes = np.array([100, 150, 200])
        result = _amihud_illiquidity(returns, volumes)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert result >= 0
    
    def test_amihud_illiquidity_empty(self):
        """Test Amihud with empty data."""
        returns = np.array([])
        volumes = np.array([])
        result = _amihud_illiquidity(returns, volumes)
        
        assert np.isnan(result)
    
    def test_amihud_illiquidity_zero_volume(self):
        """Test Amihud with zero volumes."""
        returns = np.array([0.01, -0.02])
        volumes = np.array([0, 0])
        result = _amihud_illiquidity(returns, volumes)
        
        assert np.isnan(result)
    
    def test_kyle_lambda(self):
        """Test Kyle lambda calculation."""
        order_flow = np.array([10, -5, 15, -10])
        returns = np.array([0.01, -0.02, 0.03, -0.01])
        result = _kyle_lambda(order_flow, returns)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_kyle_lambda_insufficient_data(self):
        """Test Kyle lambda with insufficient data."""
        order_flow = np.array([10])
        returns = np.array([0.01])
        result = _kyle_lambda(order_flow, returns)
        
        assert np.isnan(result)
    
    def test_kyle_lambda_zero_variance(self):
        """Test Kyle lambda with zero variance."""
        order_flow = np.array([10, 10, 10])
        returns = np.array([0.01, 0.02, 0.03])
        result = _kyle_lambda(order_flow, returns)
        
        assert np.isnan(result)


@pytest.mark.unit
@pytest.mark.features_layer
class TestMicrostructureFeatures:
    """Test MicrostructureFeatures class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'open': np.random.normal(42000, 100, n),
            'close': np.random.normal(42000, 100, n),
            'high': np.random.normal(42100, 100, n),
            'low': np.random.normal(41900, 100, n),
            'volume': np.random.normal(1000, 100, n)
        })
    
    def test_initialization(self):
        """Test MicrostructureFeatures initialization."""
        config = {'window': 20}
        features = MicrostructureFeatures(config)
        
        assert features.window == 20
    
    def test_initialization_default(self):
        """Test initialization with default config."""
        config = {}
        features = MicrostructureFeatures(config)
        
        assert features.window == 20
    
    def test_calculate(self, sample_data):
        """Test feature calculation."""
        config = {'window': 20}
        features = MicrostructureFeatures(config)
        
        result = features.calculate(sample_data)
        
        assert result.shape[0] == len(sample_data)
        assert result.shape[1] == 5  # roll, realized_vol, amihud, vpin, kyle_lambda
        assert not np.any(np.isnan(result))
    
    def test_calculate_small_window(self, sample_data):
        """Test calculation with small window."""
        config = {'window': 5}
        features = MicrostructureFeatures(config)
        
        result = features.calculate(sample_data)
        
        assert result.shape[0] == len(sample_data)
        assert result.shape[1] == 5
        assert not np.any(np.isnan(result))
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        config = {}
        features = MicrostructureFeatures(config)
        
        names = features.get_feature_names()
        
        # Initially empty before calculate
        assert names == []
        
        # After calculate, should have names
        data = pd.DataFrame({
            'open': [42000, 42100, 42200, 42300, 42400, 42500],
            'close': [42050, 42150, 42250, 42350, 42450, 42550],
            'high': [42100, 42200, 42300, 42400, 42500, 42600],
            'low': [42000, 42100, 42200, 42300, 42400, 42500],
            'volume': [100, 150, 200, 250, 300, 350]
        })
        features.calculate(data)
        
        names = features.get_feature_names()
        
        assert 'roll' in names
        assert 'realized_vol' in names
        assert 'amihud' in names
        assert 'vpin' in names
        assert 'kyle_lambda' in names
        assert len(names) == 5
    
    def test_calculate_insufficient_data(self):
        """Test calculation with insufficient data (less than window)."""
        config = {'window': 20}
        features = MicrostructureFeatures(config)
        
        # Create data with only 10 rows
        data = pd.DataFrame({
            'open': np.random.normal(42000, 100, 10),
            'close': np.random.normal(42000, 100, 10),
            'high': np.random.normal(42100, 100, 10),
            'low': np.random.normal(41900, 100, 10),
            'volume': np.random.normal(1000, 100, 10)
        })
        
        result = features.calculate(data)
        
        # Should still return array of correct shape, filled with zeros
        assert result.shape[0] == len(data)
        assert result.shape[1] == 5
        assert np.all(result == 0.0)
    
    def test_calculate_zero_volume(self):
        """Test calculation with zero volumes."""
        config = {'window': 5}
        features = MicrostructureFeatures(config)
        
        data = pd.DataFrame({
            'open': [42000, 42100, 42200, 42300, 42400, 42500],
            'close': [42050, 42150, 42250, 42350, 42450, 42550],
            'high': [42100, 42200, 42300, 42400, 42500, 42600],
            'low': [42000, 42100, 42200, 42300, 42400, 42500],
            'volume': [0, 0, 0, 0, 0, 0]
        })
        
        result = features.calculate(data)
        
        assert result.shape[0] == len(data)
        assert result.shape[1] == 5
        # First window rows should be 0, rest may have NaNs due to zero volume
        # This is expected behavior for microstructure measures with no volume
