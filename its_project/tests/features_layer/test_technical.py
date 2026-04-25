"""
Unit tests for TechnicalFeatures.
"""
import pytest
import numpy as np
import pandas as pd
from its_project.features.technical import TechnicalFeatures


@pytest.mark.unit
@pytest.mark.features_layer
class TestTechnicalFeatures:
    """Test TechnicalFeatures functionality."""
    
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
        """Test TechnicalFeatures initialization."""
        config = {
            'indicators': ['rsi', 'macd'],
            'rsi_period': 14,
            'macd_fast': 12
        }
        features = TechnicalFeatures(config)
        
        assert features.indicators == ['rsi', 'macd']
        assert features.rsi_period == 14
        assert features.macd_fast == 12
    
    def test_calculate_rsi(self, sample_data):
        """Test RSI calculation."""
        config = {'indicators': ['rsi'], 'rsi_period': 14}
        features = TechnicalFeatures(config)
        
        result = features.calculate(sample_data)
        
        assert result.shape[0] == len(sample_data)
        assert result.shape[1] == 1  # Only RSI
        assert not np.any(np.isnan(result))
    
    def test_calculate_macd(self, sample_data):
        """Test MACD calculation."""
        config = {'indicators': ['macd']}
        features = TechnicalFeatures(config)
        
        result = features.calculate(sample_data)
        
        assert result.shape[0] == len(sample_data)
        assert result.shape[1] == 3  # macd, signal, histogram
        assert not np.any(np.isnan(result))
    
    def test_calculate_bollinger_bands(self, sample_data):
        """Test Bollinger Bands calculation."""
        config = {'indicators': ['bbands']}
        features = TechnicalFeatures(config)
        
        result = features.calculate(sample_data)
        
        assert result.shape[0] == len(sample_data)
        assert result.shape[1] == 4  # upper, middle, lower, width
        assert not np.any(np.isnan(result))
    
    def test_calculate_atr(self, sample_data):
        """Test ATR calculation."""
        config = {'indicators': ['atr']}
        features = TechnicalFeatures(config)
        
        result = features.calculate(sample_data)
        
        assert result.shape[0] == len(sample_data)
        assert result.shape[1] == 1  # ATR
        assert not np.any(np.isnan(result))
    
    def test_calculate_stochastic(self, sample_data):
        """Test Stochastic calculation."""
        config = {'indicators': ['stoch']}
        features = TechnicalFeatures(config)
        
        result = features.calculate(sample_data)
        
        assert result.shape[0] == len(sample_data)
        assert result.shape[1] == 2  # %K, %D
        assert not np.any(np.isnan(result))
    
    def test_calculate_all_indicators(self, sample_data):
        """Test calculating all indicators."""
        config = {'indicators': ['rsi', 'macd', 'bbands', 'atr', 'stoch']}
        features = TechnicalFeatures(config)
        
        result = features.calculate(sample_data)
        
        assert result.shape[0] == len(sample_data)
        # RSI(1) + MACD(3) + BB(4) + ATR(1) + Stoch(2) = 11
        assert result.shape[1] == 11
        assert not np.any(np.isnan(result))
    
    def test_get_feature_names(self, sample_data):
        """Test getting feature names."""
        config = {'indicators': ['rsi', 'macd']}
        features = TechnicalFeatures(config)
        
        features.calculate(sample_data)
        names = features.get_feature_names()
        
        assert 'rsi' in names
        assert 'macd' in names
        assert 'macd_signal' in names
        assert 'macd_hist' in names
    
    def test_default_config(self):
        """Test default configuration."""
        config = {}
        features = TechnicalFeatures(config)
        
        assert features.indicators == ['rsi', 'macd', 'bbands', 'atr', 'stoch']
        assert features.rsi_period == 14
        assert features.macd_fast == 12
