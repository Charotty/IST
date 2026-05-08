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
    _price_density,
    _depth_imbalance,
    _microprice
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
    
    def test_depth_imbalance(self):
        """Test depth imbalance calculation."""
        bids = np.array([[100, 10], [99, 20], [98, 15]])
        asks = np.array([[101, 10], [102, 20], [103, 15]])
        result = _depth_imbalance(bids, asks, levels=3)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert -1 <= result <= 1
        
        # Calculate manually: (bid_volume - ask_volume) / (bid_volume + ask_volume)
        bid_volume = 10 + 20 + 15  # 45
        ask_volume = 10 + 20 + 15  # 45
        expected = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        assert abs(result - expected) < 1e-10
    
    def test_depth_imbalance_insufficient_levels(self):
        """Test depth imbalance with insufficient levels."""
        bids = np.array([[100, 10], [99, 20]])
        asks = np.array([[101, 10], [102, 20], [103, 15]])
        result = _depth_imbalance(bids, asks, levels=5)
        
        # Should use available levels
        bid_volume = 10 + 20  # 30
        ask_volume = 10 + 20 + 15  # 45
        expected = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        assert abs(result - expected) < 1e-10
    
    def test_depth_imbalance_zero_volume(self):
        """Test depth imbalance with zero volume."""
        bids = np.array([[100, 0], [99, 0]])
        asks = np.array([[101, 10], [102, 20]])
        result = _depth_imbalance(bids, asks, levels=2)
        
        # All bid volume is zero
        assert result == -1.0
    
    def test_microprice(self):
        """Test microprice calculation."""
        bids = np.array([[100, 10], [99, 20]])
        asks = np.array([[101, 10], [102, 20]])
        result = _microprice(bids, asks)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert 100 <= result <= 101  # Should be between best bid and ask
        
        # Calculate manually: (bid_price * ask_volume + ask_price * bid_volume) / (bid_volume + ask_volume)
        bid_price, bid_volume = 100, 10
        ask_price, ask_volume = 101, 10
        expected = (bid_price * ask_volume + ask_price * bid_volume) / (bid_volume + ask_volume)
        assert abs(result - expected) < 1e-10
    
    def test_microprice_weighted_by_volume(self):
        """Test microprice with volume weighting."""
        bids = np.array([[100, 20], [99, 10]])  # More volume at best bid
        asks = np.array([[101, 5], [102, 25]])   # Less volume at best ask
        result = _microprice(bids, asks)
        
        # Should be closer to bid price due to higher bid volume
        bid_price, bid_volume = 100, 20
        ask_price, ask_volume = 101, 5
        expected = (bid_price * ask_volume + ask_price * bid_volume) / (bid_volume + ask_volume)
        assert abs(result - expected) < 1e-10
        assert result < 100.5  # Should be closer to bid
    
    def test_microprice_empty_orderbook(self):
        """Test microprice with empty orderbook."""
        bids = np.array([])
        asks = np.array([])
        result = _microprice(bids, asks)
        
        assert np.isnan(result)
    
    def test_microprice_one_sided(self):
        """Test microprice with one-sided orderbook."""
        bids = np.array([[100, 10]])
        asks = np.array([])
        result = _microprice(bids, asks)
        
        assert np.isnan(result)


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
        # 11 base features (including depth_imbalance and microprice) + depth_levels OFI features
        expected_features = 11 + config['depth_levels']
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
        expected_features = 11 + config['depth_levels']  # Updated for new features
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
        expected_features = 11 + config['depth_levels']  # Updated for new features
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
        assert 'depth_imbalance' in names
        assert 'microprice' in names
        assert 'bid_depth' in names
        assert 'ask_depth' in names
        assert 'bid_vwap' in names
        assert 'ask_vwap' in names
        assert 'bid_density' in names
        assert 'ask_density' in names
        assert 'ofi_0' in names
        assert 'ofi_1' in names
        assert 'ofi_2' in names
        expected_features = 11 + config['depth_levels']  # Updated for new features
        assert len(names) == expected_features
    
    def test_calculate_custom_depth(self, sample_orderbook_data):
        """Test calculation with custom depth levels."""
        config = {'depth_levels': 2, 'imbalance_levels': 2}
        features = OrderBookFeatures(config)
        
        result = features.calculate(sample_orderbook_data)
        
        assert result.shape[0] == len(sample_orderbook_data)
        # 11 base features (including depth_imbalance and microprice) + 2 OFI features
        expected_features = 11 + config['depth_levels']
        assert result.shape[1] == expected_features
    
    def test_ofi_features_calculation(self, sample_orderbook_data):
        """Test OFI (Order Flow Imbalance) features calculation."""
        config = {'depth_levels': 5, 'imbalance_levels': 3}
        features = OrderBookFeatures(config)
        
        result = features.calculate(sample_orderbook_data)
        
        # Should have 5 OFI features (ofi_0 through ofi_4)
        ofi_features = [col for col in features.get_feature_names() if col.startswith('ofi_')]
        assert len(ofi_features) == 5
        
        # OFI features should be numeric and not all zeros
        for i in range(5):
            ofi_col = f'ofi_{i}'
            assert ofi_col in features.get_feature_names()
            assert not np.all(result[:, features.get_feature_names().index(ofi_col)] == 0)
    
    def test_depth_imbalance_feature_calculation(self, sample_orderbook_data):
        """Test depth imbalance feature calculation."""
        config = {'depth_levels': 3, 'imbalance_levels': 3}
        features = OrderBookFeatures(config)
        
        result = features.calculate(sample_orderbook_data)
        
        # Check depth_imbalance feature exists and is valid
        names = features.get_feature_names()
        assert 'depth_imbalance' in names
        
        depth_imbalance_idx = names.index('depth_imbalance')
        depth_imbalance_values = result[:, depth_imbalance_idx]
        
        # Values should be between -1 and 1
        assert np.all(depth_imbalance_values >= -1)
        assert np.all(depth_imbalance_values <= 1)
        
        # Should not be all NaN
        assert not np.all(np.isnan(depth_imbalance_values))
    
    def test_microprice_feature_calculation(self, sample_orderbook_data):
        """Test microprice feature calculation."""
        config = {'depth_levels': 3, 'imbalance_levels': 3}
        features = OrderBookFeatures(config)
        
        result = features.calculate(sample_orderbook_data)
        
        # Check microprice feature exists and is valid
        names = features.get_feature_names()
        assert 'microprice' in names
        
        microprice_idx = names.index('microprice')
        microprice_values = result[:, microprice_idx]
        
        # Microprice should be between best bid and ask
        # For our sample data, this should be reasonable
        assert not np.all(np.isnan(microprice_values))
        
        # Check some reasonable bounds (assuming prices around 100)
        assert np.all(microprice_values > 50)  # Lower bound
        assert np.all(microprice_values < 200)  # Upper bound
    
    def test_all_lob_features_integration(self, sample_orderbook_data):
        """Test integration of all LOB features including OFI, depth imbalance, and microprice."""
        config = {'depth_levels': 3, 'imbalance_levels': 5}
        features = OrderBookFeatures(config)
        
        result = features.calculate(sample_orderbook_data)
        
        # Check all expected features are present
        names = features.get_feature_names()
        expected_features = [
            'spread', 'spread_pct', 'volume_imbalance', 'depth_imbalance', 'microprice',
            'bid_depth', 'ask_depth', 'bid_vwap', 'ask_vwap', 'bid_density', 'ask_density',
            'ofi_0', 'ofi_1', 'ofi_2'
        ]
        
        for feature in expected_features:
            assert feature in names, f"Missing feature: {feature}"
        
        # Check total feature count
        expected_count = 11 + config['depth_levels']  # 11 base + 3 OFI
        assert len(names) == expected_count
        assert result.shape[1] == expected_count
        
        # Check no NaN values in the result
        assert not np.any(np.isnan(result))
        
        # Check feature value ranges are reasonable
        for i, feature_name in enumerate(names):
            feature_values = result[:, i]
            
            if feature_name == 'spread':
                assert np.all(feature_values >= 0)  # Spread should be non-negative
            elif feature_name == 'spread_pct':
                assert np.all(feature_values >= 0)  # Spread percentage should be non-negative
            elif feature_name in ['volume_imbalance', 'depth_imbalance']:
                assert np.all(feature_values >= -1) and np.all(feature_values <= 1)
            elif feature_name in ['bid_depth', 'ask_depth']:
                assert np.all(feature_values >= 0)  # Depth should be non-negative
            elif feature_name == 'microprice':
                assert not np.all(np.isnan(feature_values))  # Microprice should be calculated
