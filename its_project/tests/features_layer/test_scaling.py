"""
Unit tests for FeatureScaler.
"""
import pytest
import numpy as np
import pandas as pd
from its_project.features.scaling import FeatureScaler


@pytest.mark.unit
@pytest.mark.features_layer
class TestFeatureScaler:
    """Test FeatureScaler functionality."""
    
    def test_initialization(self):
        """Test FeatureScaler initialization."""
        config = {'method': 'zscore', 'axis': 0}
        scaler = FeatureScaler(config)
        
        assert scaler.config == config
        assert scaler.method == 'zscore'
        assert scaler.axis == 0
        assert scaler._stats == {}
    
    def test_default_config(self):
        """Test default configuration values."""
        scaler = FeatureScaler({})
        
        assert scaler.method == 'zscore'
        assert scaler.axis == 0
    
    def test_fit_zscore(self):
        """Test fit with zscore method."""
        scaler = FeatureScaler({'method': 'zscore', 'axis': 0})
        features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)
        
        fitted_scaler = scaler.fit(features)
        
        assert fitted_scaler is scaler
        assert 'mean' in scaler._stats
        assert 'std' in scaler._stats
        assert scaler._stats['mean'].shape == (3,)
        assert scaler._stats['std'].shape == (3,)
    
    def test_fit_minmax(self):
        """Test fit with minmax method."""
        scaler = FeatureScaler({'method': 'minmax', 'axis': 0})
        features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)
        
        scaler.fit(features)
        
        assert 'min' in scaler._stats
        assert 'max' in scaler._stats
        assert scaler._stats['min'].shape == (3,)
        assert scaler._stats['max'].shape == (3,)
    
    def test_fit_unknown_method(self):
        """Test fit with unknown method."""
        scaler = FeatureScaler({'method': 'unknown'})
        features = np.array([[1, 2, 3]], dtype=float)
        
        with pytest.raises(ValueError, match="Unknown method"):
            scaler.fit(features)
    
    def test_transform_zscore(self):
        """Test transform with zscore method."""
        scaler = FeatureScaler({'method': 'zscore', 'axis': 0})
        features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)
        
        scaler.fit(features)
        transformed = scaler.transform(features)
        
        assert transformed.shape == features.shape
        # Check that mean is approximately 0
        assert np.abs(np.mean(transformed, axis=0)).max() < 1e-10
    
    def test_transform_minmax(self):
        """Test transform with minmax method."""
        scaler = FeatureScaler({'method': 'minmax', 'axis': 0})
        features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)
        
        scaler.fit(features)
        transformed = scaler.transform(features)
        
        assert transformed.shape == features.shape
        # Check that values are in [0, 1]
        assert transformed.min() >= 0
        assert transformed.max() <= 1
    
    def test_transform_not_fitted(self):
        """Test transform without fitting first."""
        scaler = FeatureScaler({'method': 'zscore'})
        features = np.array([[1, 2, 3]], dtype=float)
        
        with pytest.raises(RuntimeError, match="Scaler not fitted"):
            scaler.transform(features)
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        scaler = FeatureScaler({'method': 'zscore', 'axis': 0})
        features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)
        
        transformed = scaler.fit_transform(features)
        
        assert transformed.shape == features.shape
        assert 'mean' in scaler._stats
        assert 'std' in scaler._stats
    
    def test_inverse_transform_zscore(self):
        """Test inverse transform with zscore method."""
        scaler = FeatureScaler({'method': 'zscore', 'axis': 0})
        features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)
        
        scaler.fit(features)
        transformed = scaler.transform(features)
        inverse = scaler.inverse_transform(transformed)
        
        # Should be close to original
        assert np.allclose(inverse, features, atol=1e-5)
    
    def test_inverse_transform_minmax(self):
        """Test inverse transform with minmax method."""
        scaler = FeatureScaler({'method': 'minmax', 'axis': 0})
        features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)
        
        scaler.fit(features)
        transformed = scaler.transform(features)
        inverse = scaler.inverse_transform(transformed)
        
        # Should be close to original
        assert np.allclose(inverse, features, atol=1e-5)
    
    def test_inverse_transform_not_fitted(self):
        """Test inverse transform without fitting first."""
        scaler = FeatureScaler({'method': 'zscore'})
        normalized = np.array([[1, 2, 3]], dtype=float)
        
        with pytest.raises(RuntimeError, match="Scaler not fitted"):
            scaler.inverse_transform(normalized)
    
    def test_inverse_transform_unknown_method(self):
        """Test inverse transform with unknown method in scaler."""
        scaler = FeatureScaler({'method': 'unknown'})
        scaler._stats = {'mean': np.array([1, 2, 3]), 'std': np.array([1, 1, 1])}
        normalized = np.array([[1, 2, 3]], dtype=float)
        
        with pytest.raises(ValueError, match="Unknown method"):
            scaler.inverse_transform(normalized)
    
    def test_calculate_not_implemented(self):
        """Test that calculate raises NotImplementedError."""
        scaler = FeatureScaler({})
        data = pd.DataFrame({'a': [1, 2, 3]})
        
        with pytest.raises(NotImplementedError, match="FeatureScaler is a wrapper"):
            scaler.calculate(data)
    
    def test_get_feature_names(self):
        """Test get_feature_names returns empty list."""
        scaler = FeatureScaler({})
        
        names = scaler.get_feature_names()
        
        assert names == []
    
    def test_axis_1_zscore(self):
        """Test zscore scaling along axis 1."""
        scaler = FeatureScaler({'method': 'zscore', 'axis': 1})
        features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)
        
        scaler.fit(features)
        transformed = scaler.transform(features)
        
        assert transformed.shape == features.shape
        assert scaler._stats['mean'].shape == (3,)
    
    def test_axis_1_minmax(self):
        """Test minmax scaling along axis 1."""
        scaler = FeatureScaler({'method': 'minmax', 'axis': 1})
        features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)
        
        scaler.fit(features)
        transformed = scaler.transform(features)
        
        assert transformed.shape == features.shape
        assert scaler._stats['min'].shape == (3,)
