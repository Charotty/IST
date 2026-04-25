"""
Unit tests for OrderBookFeatures.
"""
import pytest
import numpy as np
import pandas as pd
from its_project.features.orderbook import (
    OrderBookFeatures,
    _ofi_from_levels,
    _spread,
    _volume_imbalance,
    _depth_measures,
    _price_density
)


@pytest.mark.unit
@pytest.mark.features_layer
class TestOrderBookHelpers:
    """Test orderbook helper functions."""
    
    def test_ofi_from_levels(self):
        """Test OFI calculation from orderbook levels."""
        bids = np.array([[100, 10], [99, 20], [98, 15]])
        asks = np.array([[101, 10], [102, 20], [103, 15]])
        result = _ofi_from_levels(bids, asks, depth=3)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert not np.any(np.isnan(result))
    
    def test_ofi_from_levels_insufficient_depth(self):
        """Test OFI with insufficient depth."""
        bids = np.array([[100, 10], [99, 20]])
        asks = np.array([[101, 10], [102, 20], [103, 15]])
        result = _ofi_from_levels(bids, asks, depth=5)
        
        assert len(result) == 2
    
    def test_spread(self):
        """Test spread calculation."""
        bids = np.array([[100, 10], [99, 20]])
        asks = np.array([[101, 10], [102, 20]])
        result = _spread(bids, asks)
        
        assert result == 1.0
    
    def test_spread_empty(self):
        """Test spread with empty arrays."""
        bids = np.array([])
        asks = np.array([])
        result = _spread(bids, asks)
        
        assert np.isnan(result)
    
    def test_volume_imbalance(self):
        """Test volume imbalance calculation."""
        bids = np.array([[100, 10], [99, 20], [98, 15]])
        asks = np.array([[101, 10], [102, 20], [103, 15]])
        result = _volume_imbalance(bids, asks, levels=3)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert -1 <= result <= 1
    
    def test_volume_imbalance_insufficient_levels(self):
        """Test volume imbalance with insufficient levels."""
        bids = np.array([[100, 10]])
        asks = np.array([[101, 10], [102, 20]])
        result = _volume_imbalance(bids, asks, levels=10)
        
        assert not np.isnan(result)
    
    def test_depth_measures(self):
        """Test depth measures calculation."""
        bids = np.array([[100, 10], [99, 20], [98, 15]])
        asks = np.array([[101, 10], [102, 20], [103, 15]])
        result = _depth_measures(bids, asks, levels=3)
        
        assert isinstance(result, dict)
        assert "bid_depth" in result
        assert "ask_depth" in result
        assert "bid_vwap" in result
        assert "ask_vwap" in result
        assert result["bid_depth"] == 45
        assert result["ask_depth"] == 45
    
    def test_depth_measures_zero_volume(self):
        """Test depth measures with zero volume."""
        bids = np.array([[100, 0], [99, 0]])
        asks = np.array([[101, 10], [102, 20]])
        result = _depth_measures(bids, asks, levels=2)
        
        assert result["bid_depth"] == 0
        assert np.isnan(result["bid_vwap"])
    
    def test_price_density(self):
        """Test price density calculation."""
        bids = np.array([[100, 10], [99, 20], [98, 15]])
        asks = np.array([[101, 10], [102, 20], [103, 15]])
        result = _price_density(bids, asks)
        
        assert isinstance(result, dict)
        assert "bid_density" in result
        assert "ask_density" in result
        assert not np.isnan(result["bid_density"])
        assert not np.isnan(result["ask_density"])
    
    def test_price_density_insufficient_levels(self):
        """Test price density with insufficient levels."""
        bids = np.array([[100, 10]])
        asks = np.array([[101, 10]])
        result = _price_density(bids, asks)
        
        assert np.isnan(result["bid_density"])
        assert np.isnan(result["ask_density"])


@pytest.mark.unit
@pytest.mark.features_layer
class TestOrderBookFeatures:
    """Test OrderBookFeatures class."""
    
    @pytest.fixture
    def sample_orderbook_data(self):
        """Create sample orderbook data."""
        data = []
        for i in range(10):
            ob = {
                "bids": [[100 - i, 10 + i], [99 - i, 20 + i], [98 - i, 15 + i]],
                "asks": [[101 + i, 10 + i], [102 + i, 20 + i], [103 + i, 15 + i]]
            }
            data.append(ob)
        return pd.DataFrame({"data": data})
    
    def test_initialization(self):
        """Test OrderBookFeatures initialization."""
        config = {'depth_levels': 5, 'imbalance_levels': 10}
        features = OrderBookFeatures(config)
        
        assert features.depth_levels == 5
        assert features.imbalance_levels == 10
    
    def test_initialization_defaults(self):
        """Test initialization with default config."""
        config = {}
        features = OrderBookFeatures(config)
        
        assert features.depth_levels == 5
        assert features.imbalance_levels == 10
    
    def test_calculate(self, sample_orderbook_data):
        """Test feature calculation."""
        config = {'depth_levels': 3, 'imbalance_levels': 3}
        features = OrderBookFeatures(config)
        
        result = features.calculate(sample_orderbook_data)
        
        assert result.shape[0] == len(sample_orderbook_data)
        # 9 base features + depth_levels OFI features
        expected_features = 9 + config['depth_levels']
        assert result.shape[1] == expected_features
        assert not np.any(np.isnan(result))
    
    def test_calculate_missing_data(self):
        """Test calculation with missing orderbook data."""
        config = {'depth_levels': 3, 'imbalance_levels': 3}
        features = OrderBookFeatures(config)
        
        # Create data with missing orderbook
        data = pd.DataFrame({"data": [None, {"bids": [[100, 10]], "asks": [[101, 10]]}]})
        
        result = features.calculate(data)
        
        assert result.shape[0] == 2
        expected_features = 9 + config['depth_levels']
        assert result.shape[1] == expected_features
        # NaNs should be replaced with 0
        assert not np.any(np.isnan(result))
    
    def test_calculate_invalid_data(self):
        """Test calculation with invalid orderbook format."""
        config = {'depth_levels': 3, 'imbalance_levels': 3}
        features = OrderBookFeatures(config)
        
        # Create data with invalid orderbook
        data = pd.DataFrame({"data": [{"invalid": "data"}]})
        
        result = features.calculate(data)
        
        assert result.shape[0] == 1
        expected_features = 9 + config['depth_levels']
        assert result.shape[1] == expected_features
        assert not np.any(np.isnan(result))
    
    def test_calculate_missing_columns(self):
        """Test calculation with missing required columns."""
        config = {'depth_levels': 3, 'imbalance_levels': 3}
        features = OrderBookFeatures(config)
        
        data = pd.DataFrame({"other_column": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing columns"):
            features.calculate(data)
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        config = {'depth_levels': 3, 'imbalance_levels': 10}
        features = OrderBookFeatures(config)
        
        # Initially empty before calculate
        names = features.get_feature_names()
        assert names == []
        
        # After calculate, should have names
        data = pd.DataFrame({"data": [{"bids": [[100, 10]], "asks": [[101, 10]]}]})
        features.calculate(data)
        
        names = features.get_feature_names()
        
        assert 'spread' in names
        assert 'spread_pct' in names
        assert 'volume_imbalance' in names
        assert 'bid_depth' in names
        assert 'ask_depth' in names
        assert 'bid_vwap' in names
        assert 'ask_vwap' in names
        assert 'bid_density' in names
        assert 'ask_density' in names
        assert 'ofi_0' in names
        assert 'ofi_1' in names
        assert 'ofi_2' in names
        expected_features = 9 + config['depth_levels']
        assert len(names) == expected_features
    
    def test_calculate_custom_depth(self, sample_orderbook_data):
        """Test calculation with custom depth levels."""
        config = {'depth_levels': 2, 'imbalance_levels': 2}
        features = OrderBookFeatures(config)
        
        result = features.calculate(sample_orderbook_data)
        
        assert result.shape[0] == len(sample_orderbook_data)
        # 9 base features + 2 OFI features
        expected_features = 9 + config['depth_levels']
        assert result.shape[1] == expected_features
