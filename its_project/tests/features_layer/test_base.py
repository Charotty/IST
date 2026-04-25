"""
Unit tests for BaseFeature.
"""
import pytest
import numpy as np
import pandas as pd
from its_project.features.base import BaseFeature


class ConcreteFeature(BaseFeature):
    """Concrete implementation for testing BaseFeature."""
    
    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        self.validate_input(data)
        # Simple feature: return normalized close price
        return self.normalize(data['close'].values, method='zscore')
    
    def get_feature_names(self) -> list:
        return ['normalized_close']


@pytest.mark.unit
@pytest.mark.features_layer
class TestBaseFeature:
    """Test BaseFeature functionality."""
    
    def test_initialization(self):
        """Test BaseFeature initialization."""
        config = {'param1': 'value1', 'param2': 42}
        feature = ConcreteFeature(config)
        
        assert feature.config == config
        assert feature._feature_names == []
    
    def test_validate_input_valid(self):
        """Test validate_input with valid data."""
        feature = ConcreteFeature({})
        
        data = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [2, 3, 4],
            'low': [0.5, 1.5, 2.5],
            'close': [1.5, 2.5, 3.5],
            'volume': [100, 200, 300]
        })
        
        # Should not raise
        feature.validate_input(data)
    
    def test_validate_input_missing_columns(self):
        """Test validate_input with missing columns."""
        feature = ConcreteFeature({})
        
        data = pd.DataFrame({
            'open': [1, 2, 3],
            'close': [1.5, 2.5, 3.5]
        })
        
        with pytest.raises(ValueError, match="Missing columns"):
            feature.validate_input(data)
    
    def test_validate_input_nan_values(self):
        """Test validate_input with NaN values."""
        feature = ConcreteFeature({})
        
        data = pd.DataFrame({
            'open': [1, 2, np.nan],
            'high': [2, 3, 4],
            'low': [0.5, 1.5, 2.5],
            'close': [1.5, 2.5, 3.5],
            'volume': [100, 200, 300]
        })
        
        with pytest.raises(ValueError, match="Input contains NaN values"):
            feature.validate_input(data)
    
    def test_validate_input_invalid_dimensions(self):
        """Test validate_input with invalid dimensions."""
        feature = ConcreteFeature({})
        
        # 1D DataFrame (single column) instead of 2D with required columns
        data = pd.DataFrame({'value': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing columns"):
            feature.validate_input(data)
    
    def test_normalize_zscore(self):
        """Test zscore normalization."""
        feature = ConcreteFeature({})
        
        values = np.array([1, 2, 3, 4, 5])
        normalized = feature.normalize(values, method='zscore')
        
        assert normalized.mean() < 1e-10  # Mean should be ~0
        assert abs(normalized.std() - 1.0) < 0.1  # Std should be ~1
    
    def test_normalize_minmax(self):
        """Test minmax normalization."""
        feature = ConcreteFeature({})
        
        values = np.array([1, 2, 3, 4, 5])
        normalized = feature.normalize(values, method='minmax')
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        assert abs(normalized.min() - 0.0) < 0.1
        assert abs(normalized.max() - 1.0) < 0.1
    
    def test_normalize_unknown_method(self):
        """Test normalize with unknown method."""
        feature = ConcreteFeature({})
        
        values = np.array([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError, match="Unknown method"):
            feature.normalize(values, method='unknown')
    
    def test_calculate(self):
        """Test calculate method."""
        feature = ConcreteFeature({})
        
        data = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [2, 3, 4],
            'low': [0.5, 1.5, 2.5],
            'close': [1.5, 2.5, 3.5],
            'volume': [100, 200, 300]
        })
        
        result = feature.calculate(data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
    
    def test_get_feature_names(self):
        """Test get_feature_names method."""
        feature = ConcreteFeature({})
        
        names = feature.get_feature_names()
        
        assert isinstance(names, list)
        assert names == ['normalized_close']
