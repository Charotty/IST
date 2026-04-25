"""
Unit tests for FeaturePipeline.
"""
import pytest
import numpy as np
import pandas as pd
from its_project.features.pipeline import FeaturePipeline
from its_project.features.base import BaseFeature


# Mock feature class for testing
class MockFeature(BaseFeature):
    """Mock feature for testing pipeline."""
    
    def __init__(self, config=None):
        super().__init__(config or {})
        self._fitted = False
        self.feature_names = [f"mock_{i}" for i in range(2)]
    
    def calculate(self, data):
        # Return simple 2D array
        return np.array([[1, 2], [3, 4], [5, 6]])
    
    def get_feature_names(self):
        return self.feature_names
    
    def fit(self, data):
        self._fitted = True
        return self


@pytest.mark.unit
@pytest.mark.features_layer
class TestFeaturePipeline:
    """Test FeaturePipeline functionality."""
    
    def test_initialization(self):
        """Test FeaturePipeline initialization."""
        feature1 = MockFeature()
        feature2 = MockFeature()
        
        pipeline = FeaturePipeline([feature1, feature2])
        
        assert len(pipeline.features) == 2
        assert pipeline._cached_feature_names is None
    
    def test_initialization_empty_list(self):
        """Test initialization with empty list raises ValueError."""
        with pytest.raises(ValueError, match="FeaturePipeline requires at least one feature"):
            FeaturePipeline([])
    
    def test_fit(self):
        """Test fitting the pipeline."""
        feature = MockFeature()
        pipeline = FeaturePipeline([feature])
        
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        fitted_pipeline = pipeline.fit(data)
        
        assert fitted_pipeline is pipeline
        assert feature._fitted
    
    def test_transform(self):
        """Test transforming data through pipeline."""
        feature1 = MockFeature()
        feature2 = MockFeature()
        pipeline = FeaturePipeline([feature1, feature2])
        
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        result = pipeline.transform(data)
        
        # Should concatenate features from both mock features
        # Each returns 3x2, so result should be 3x4
        assert result.shape == (3, 4)
        assert pipeline._cached_feature_names is not None
    
    def test_transform_single_feature(self):
        """Test transforming with single feature."""
        feature = MockFeature()
        pipeline = FeaturePipeline([feature])
        
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        result = pipeline.transform(data)
        
        assert result.shape == (3, 2)
    
    def test_transform_1d_feature(self):
        """Test transforming when feature returns 1D array."""
        class OneDFeature(BaseFeature):
            def __init__(self):
                super().__init__({})
            
            def calculate(self, data):
                return np.array([1, 2, 3])
            
            def get_feature_names(self):
                return ["oned"]
        
        pipeline = FeaturePipeline([OneDFeature()])
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        result = pipeline.transform(data)
        
        # Should reshape 1D to 2D
        assert result.shape == (3, 1)
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        feature = MockFeature()
        pipeline = FeaturePipeline([feature])
        
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        result = pipeline.fit_transform(data)
        
        assert result.shape == (3, 2)
        assert feature._fitted
        assert pipeline._cached_feature_names is not None
    
    def test_get_feature_names(self):
        """Test getting feature names after transform."""
        feature1 = MockFeature()
        feature2 = MockFeature()
        pipeline = FeaturePipeline([feature1, feature2])
        
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pipeline.transform(data)
        
        names = pipeline.get_feature_names()
        
        # Should have 4 names (2 from each feature)
        assert len(names) == 4
        assert "mock_0" in names
        assert "mock_1" in names
    
    def test_get_feature_names_before_transform(self):
        """Test getting feature names before transform raises RuntimeError."""
        feature = MockFeature()
        pipeline = FeaturePipeline([feature])
        
        with pytest.raises(RuntimeError, match="Feature names not available"):
            pipeline.get_feature_names()
    
    def test_multiple_features(self):
        """Test pipeline with multiple features."""
        features = [MockFeature() for _ in range(5)]
        pipeline = FeaturePipeline(features)
        
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        result = pipeline.transform(data)
        
        # 5 features, each returns 3x2, result should be 3x10
        assert result.shape == (3, 10)
        assert len(pipeline.get_feature_names()) == 10
