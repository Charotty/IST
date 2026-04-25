"""
Unit tests for data normalizer.
"""
import pytest
import pandas as pd
import numpy as np
from its_project.preprocessing.normalizer import DataNormalizer


@pytest.mark.unit
@pytest.mark.preprocessing_layer
class TestDataNormalizer:
    """Test data normalizer functionality."""
    
    def test_normalizer_initialization(self):
        """Test normalizer initialization."""
        normalizer = DataNormalizer(method='minmax')
        assert normalizer.method == 'minmax'
    
    def test_minmax_scaling(self, sample_market_data):
        """Test min-max scaling."""
        normalizer = DataNormalizer(method='minmax')
        
        # Scale price columns
        price_cols = ['open', 'high', 'low', 'close']
        scaled = normalizer.fit_transform(sample_market_data[price_cols])
        
        # Verify values in [0, 1]
        assert scaled.min() >= 0
        assert scaled.max() <= 1
    
    def test_zscore_normalization(self, sample_market_data):
        """Test z-score normalization."""
        normalizer = DataNormalizer(method='zscore')
        
        # Scale price columns
        price_cols = ['open', 'high', 'low', 'close']
        scaled = normalizer.fit_transform(sample_market_data[price_cols])
        
        # Verify mean ~ 0, std ~ 1
        assert abs(scaled.mean().mean()) < 0.1
        assert abs(scaled.std().mean() - 1.0) < 0.1
    
    def test_log_transformation(self, sample_market_data):
        """Test log transformation."""
        normalizer = DataNormalizer(method='log')
        
        # Apply log to positive values
        data = sample_market_data['close'].clip(lower=1)
        transformed = normalizer.transform(data)
        
        # Verify transformation applied
        assert transformed is not None
        assert len(transformed) == len(data)
    
    def test_inverse_transform_minmax(self, sample_market_data):
        """Test inverse transform for min-max scaling."""
        normalizer = DataNormalizer(method='minmax')
        
        price_cols = ['open', 'high', 'low', 'close']
        scaled = normalizer.fit_transform(sample_market_data[price_cols])
        original = normalizer.inverse_transform(scaled)
        
        # Verify inverse returns original values
        assert np.allclose(original.values, sample_market_data[price_cols].values, rtol=1e-10)
    
    def test_inverse_transform_zscore(self, sample_market_data):
        """Test inverse transform for z-score normalization."""
        normalizer = DataNormalizer(method='zscore')
        
        price_cols = ['open', 'high', 'low', 'close']
        scaled = normalizer.fit_transform(sample_market_data[price_cols])
        original = normalizer.inverse_transform(scaled)
        
        # Verify inverse returns original values
        assert np.allclose(original.values, sample_market_data[price_cols].values, rtol=1e-10)
    
    def test_parameter_persistence(self, sample_market_data):
        """Test parameter persistence for inverse transform."""
        normalizer = DataNormalizer(method='minmax')
        
        price_cols = ['open', 'high', 'low', 'close']
        normalizer.fit(sample_market_data[price_cols])
        
        # Verify parameters saved
        assert hasattr(normalizer, 'min_')
        assert hasattr(normalizer, 'max_')
    
    def test_multiple_normalization_methods(self, sample_market_data):
        """Test applying multiple normalization methods."""
        # Min-max for prices
        price_normalizer = DataNormalizer(method='minmax')
        price_cols = ['open', 'high', 'low', 'close']
        scaled_prices = price_normalizer.fit_transform(sample_market_data[price_cols])
        
        # Z-score for volume
        volume_normalizer = DataNormalizer(method='zscore')
        scaled_volume = volume_normalizer.fit_transform(sample_market_data[['volume']])
        
        # Verify both transformations applied
        assert scaled_prices.min() >= 0
        assert scaled_prices.max() <= 1
        assert abs(scaled_volume.mean().mean()) < 0.1
