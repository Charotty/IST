"""
Unit tests for preprocessing module.
"""
import pytest
import numpy as np
import pandas as pd
from its_project.features.preprocessing import (
    handle_missing,
    normalize_features,
    denormalize_features,
    resample_to_uniform
)


@pytest.mark.unit
@pytest.mark.features_layer
class TestHandleMissing:
    """Test handle_missing function."""
    
    def test_ffill_method(self):
        """Test forward fill method."""
        data = pd.DataFrame({
            'a': [1, np.nan, 3, np.nan, 5],
            'b': [10, 20, np.nan, 40, 50]
        })
        
        result = handle_missing(data, method="ffill", limit=1)
        
        assert result.iloc[1]['a'] == 1  # Filled from previous
        # With limit=1, consecutive NaNs won't be filled
        assert result.iloc[2]['b'] == 20  # Filled from previous
    
    def test_interpolate_method(self):
        """Test interpolate method - requires DatetimeIndex."""
        data = pd.DataFrame({
            'a': [1, np.nan, 3, np.nan, 5]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1s'))
        
        result = handle_missing(data, method="interpolate", limit=2)
        
        assert not pd.isna(result.iloc[1]['a'])
        assert not pd.isna(result.iloc[3]['a'])
    
    def test_drop_method(self):
        """Test drop method."""
        data = pd.DataFrame({
            'a': [1, np.nan, 3, np.nan, 5],
            'b': [10, 20, 30, 40, 50]
        })
        
        result = handle_missing(data, method="drop")
        
        assert len(result) == 3  # Only rows without NaN
        assert list(result['a']) == [1, 3, 5]
    
    def test_unknown_method(self):
        """Test with unknown method."""
        data = pd.DataFrame({'a': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Unknown method"):
            handle_missing(data, method="unknown")
    
    def test_no_missing_values(self):
        """Test with no missing values."""
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        result = handle_missing(data, method="ffill")
        
        assert result.equals(data)


@pytest.mark.unit
@pytest.mark.features_layer
class TestNormalizeFeatures:
    """Test normalize_features function."""
    
    def test_zscore_axis_0(self):
        """Test zscore normalization along axis 0."""
        features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)
        
        norm, stats = normalize_features(features, method="zscore", axis=0)
        
        assert norm.shape == features.shape
        assert stats["method"] == "zscore"
        assert "mean" in stats
        assert "std" in stats
        # Check that mean is approximately 0
        assert np.abs(np.mean(norm, axis=0)).max() < 1e-10
    
    def test_zscore_axis_1(self):
        """Test zscore normalization along axis 1."""
        features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)
        
        norm, stats = normalize_features(features, method="zscore", axis=1)
        
        assert norm.shape == features.shape
        assert stats["method"] == "zscore"
    
    def test_minmax_axis_0(self):
        """Test minmax normalization along axis 0."""
        features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)
        
        norm, stats = normalize_features(features, method="minmax", axis=0)
        
        assert norm.shape == features.shape
        assert stats["method"] == "minmax"
        assert "min" in stats
        assert "max" in stats
        # Check that values are in [0, 1]
        assert norm.min() >= 0
        assert norm.max() <= 1
    
    def test_rolling_zscore_axis_0(self):
        """Test rolling window zscore normalization along axis 0."""
        features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ], dtype=float)
        
        norm, stats = normalize_features(features, method="zscore", axis=0, window=2)
        
        assert norm.shape == features.shape
        assert stats["method"] == "rolling_zscore"
        assert stats["window"] == 2
    
    def test_rolling_minmax_axis_0(self):
        """Test rolling window minmax normalization along axis 0."""
        features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ], dtype=float)
        
        norm, stats = normalize_features(features, method="minmax", axis=0, window=2)
        
        assert norm.shape == features.shape
        assert stats["method"] == "rolling_minmax"
        assert stats["window"] == 2
    
    def test_unknown_method(self):
        """Test with unknown method."""
        features = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        
        with pytest.raises(ValueError, match="Unknown method"):
            normalize_features(features, method="unknown")
    
    def test_window_too_large(self):
        """Test with window larger than data."""
        features = np.array([[1, 2, 3]], dtype=float)
        
        norm, stats = normalize_features(features, method="zscore", window=10)
        
        # Should fall back to standard normalization
        assert stats["method"] == "zscore"
        assert "window" not in stats


@pytest.mark.unit
@pytest.mark.features_layer
class TestDenormalizeFeatures:
    """Test denormalize_features function."""
    
    def test_denormalize_zscore(self):
        """Test denormalizing zscore normalized data."""
        features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)
        
        norm, stats = normalize_features(features, method="zscore", axis=0)
        denorm = denormalize_features(norm, stats, method="zscore")
        
        # Should be close to original
        assert np.allclose(denorm, features, atol=1e-5)
    
    def test_denormalize_minmax(self):
        """Test denormalizing minmax normalized data."""
        features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)
        
        norm, stats = normalize_features(features, method="minmax", axis=0)
        denorm = denormalize_features(norm, stats, method="minmax")
        
        # Should be close to original
        assert np.allclose(denorm, features, atol=1e-5)
    
    def test_denormalize_rolling_zscore(self):
        """Test denormalizing rolling zscore - should raise error."""
        stats = {"method": "rolling_zscore", "window": 5}
        norm = np.array([[1, 2], [3, 4]], dtype=float)
        
        with pytest.raises(NotImplementedError, match="Rolling window normalization cannot be perfectly inverted"):
            denormalize_features(norm, stats, method="zscore")
    
    def test_denormalize_rolling_minmax(self):
        """Test denormalizing rolling minmax - should raise error."""
        stats = {"method": "rolling_minmax", "window": 5}
        norm = np.array([[1, 2], [3, 4]], dtype=float)
        
        with pytest.raises(NotImplementedError, match="Rolling window normalization cannot be perfectly inverted"):
            denormalize_features(norm, stats, method="minmax")
    
    def test_denormalize_unknown_method(self):
        """Test denormalizing with unknown method."""
        stats = {"method": "unknown"}
        norm = np.array([[1, 2], [3, 4]], dtype=float)
        
        with pytest.raises(ValueError, match="Unknown method"):
            denormalize_features(norm, stats, method="unknown")


@pytest.mark.unit
@pytest.mark.features_layer
class TestResampleToUniform:
    """Test resample_to_uniform function."""
    
    def test_default_aggregation(self):
        """Test with default OHLC aggregation."""
        data = pd.DataFrame({
            'open': [1, 2, 3, 4, 5],
            'high': [2, 3, 4, 5, 6],
            'low': [0.5, 1.5, 2.5, 3.5, 4.5],
            'close': [1.5, 2.5, 3.5, 4.5, 5.5],
            'volume': [10, 20, 30, 40, 50]
        }, index=pd.date_range('2024-01-01', periods=5, freq='30s'))
        
        result = resample_to_uniform(data, freq='1min')
        
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns
        assert len(result) <= len(data)
    
    def test_custom_aggregation(self):
        """Test with custom aggregation."""
        data = pd.DataFrame({
            'price': [1, 2, 3, 4, 5],
            'volume': [10, 20, 30, 40, 50]
        }, index=pd.date_range('2024-01-01', periods=5, freq='30s'))
        
        agg = {'price': 'mean', 'volume': 'sum'}
        result = resample_to_uniform(data, freq='1min', agg=agg)
        
        assert 'price' in result.columns
        assert 'volume' in result.columns
        assert len(result) <= len(data)
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame - requires DatetimeIndex and columns for default agg."""
        data = pd.DataFrame({
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        }, index=pd.date_range('2024-01-01', periods=0, freq='1s'))
        
        result = resample_to_uniform(data, freq='1min')
        
        assert len(result) == 0
    
    def test_no_datetime_index(self):
        """Test with DatetimeIndex."""
        data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        }, index=pd.date_range('2024-01-01', periods=5, freq='30s'))
        
        agg = {'value': 'mean'}
        result = resample_to_uniform(data, freq='1min', agg=agg)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(data)
