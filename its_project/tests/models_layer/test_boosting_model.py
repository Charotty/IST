"""
Unit tests for BoostingModel.
"""
import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch
from its_project.models.boosting_model import BoostingModel


@pytest.mark.unit
@pytest.mark.models_layer
class TestBoostingModel:
    """Test BoostingModel class."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample config."""
        return {
            'n_estimators': 50,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 1.0,
            'max_features': None,
            'random_state': 42
        }
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def test_initialization(self, sample_config):
        """Test BoostingModel initialization."""
        model = BoostingModel(sample_config)
        
        assert model.n_estimators == 50
        assert model.learning_rate == 0.1
        assert model.max_depth == 3
        assert model.min_samples_split == 2
        assert model.min_samples_leaf == 1
        assert model.subsample == 1.0
        assert model.max_features is None
        assert model.random_state == 42
        assert not model.is_fitted
    
    def test_initialization_defaults(self):
        """Test initialization with default config."""
        config = {}
        model = BoostingModel(config)
        
        assert model.n_estimators == 100
        assert model.learning_rate == 0.1
        assert model.max_depth == 3
        assert model.min_samples_split == 2
        assert model.min_samples_leaf == 1
        assert model.subsample == 1.0
        assert model.max_features is None
        assert model.random_state == 42
    
    def test_fit(self, sample_config, sample_data):
        """Test model fitting."""
        model = BoostingModel(sample_config)
        X, y = sample_data
        
        result = model.fit(X, y)
        
        assert result is model  # Returns self
        assert model.is_fitted
        assert model.model is not None
    
    def test_fit_3d_input(self, sample_config):
        """Test fitting with 3D input (reshapes to 2D)."""
        model = BoostingModel(sample_config)
        np.random.seed(42)
        X = np.random.randn(100, 5, 2)  # 3D input
        y = np.random.randint(0, 2, 100)
        
        with patch.object(model, 'validate_input'):
            model.fit(X, y)
        
        assert model.is_fitted
    
    def test_predict(self, sample_config, sample_data):
        """Test prediction."""
        model = BoostingModel(sample_config)
        X, y = sample_data
        model.fit(X, y)
        
        X_test = np.random.randn(10, 10)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (10,)
        assert predictions.dtype in [np.int32, np.int64]
        assert np.all((predictions == 0) | (predictions == 1))
    
    def test_predict_before_fit(self, sample_config):
        """Test prediction before fitting raises error."""
        model = BoostingModel(sample_config)
        X = np.random.randn(10, 10)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X)
    
    def test_predict_3d_input(self, sample_config, sample_data):
        """Test prediction with 3D input."""
        model = BoostingModel(sample_config)
        X, y = sample_data
        model.fit(X, y)
        
        X_test = np.random.randn(10, 5, 2)  # 3D input
        
        with patch.object(model, 'validate_input'):
            predictions = model.predict(X_test)
        
        assert predictions.shape == (10,)
    
    def test_predict_proba(self, sample_config, sample_data):
        """Test probability prediction."""
        model = BoostingModel(sample_config)
        X, y = sample_data
        model.fit(X, y)
        
        X_test = np.random.randn(10, 10)
        probabilities = model.predict_proba(X_test)
        
        assert probabilities.shape == (10, 2)  # Binary classification
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
    
    def test_predict_proba_before_fit(self, sample_config):
        """Test predict_proba before fitting raises error."""
        model = BoostingModel(sample_config)
        X = np.random.randn(10, 10)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_proba(X)
    
    def test_get_confidence(self, sample_config, sample_data):
        """Test getting prediction confidence."""
        model = BoostingModel(sample_config)
        X, y = sample_data
        model.fit(X, y)
        
        X_test = np.random.randn(10, 10)
        confidence = model.get_confidence(X_test)
        
        assert confidence.shape == (10,)
        assert np.all((confidence >= 0.5) & (confidence <= 1.0))
    
    def test_get_feature_importance(self, sample_config, sample_data):
        """Test getting feature importance."""
        model = BoostingModel(sample_config)
        X, y = sample_data
        model.fit(X, y)
        
        importance = model.get_feature_importance()
        
        assert importance is not None
        assert importance.shape == (10,)
        assert np.all(importance >= 0)
    
    def test_get_feature_importance_before_fit(self, sample_config):
        """Test getting feature importance before fitting returns None."""
        model = BoostingModel(sample_config)
        
        importance = model.get_feature_importance()
        
        assert importance is None
    
    def test_cross_validate(self, sample_config, sample_data):
        """Test cross-validation."""
        model = BoostingModel(sample_config)
        X, y = sample_data
        
        results = model.cross_validate(X, y, cv=3)
        
        assert 'mean_accuracy' in results
        assert 'std_accuracy' in results
        assert 'scores' in results
        assert len(results['scores']) == 3
        assert 0 <= results['mean_accuracy'] <= 1
        assert results['std_accuracy'] >= 0
    
    def test_cross_validate_3d_input(self, sample_config):
        """Test cross-validation with 3D input."""
        model = BoostingModel(sample_config)
        np.random.seed(42)
        X = np.random.randn(100, 5, 2)
        y = np.random.randint(0, 2, 100)
        
        results = model.cross_validate(X, y, cv=3)
        
        assert 'mean_accuracy' in results
    
    def test_save_load(self, sample_config, sample_data):
        """Test saving and loading model."""
        model = BoostingModel(sample_config)
        X, y = sample_data
        model.fit(X, y)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            model.save(temp_path)
            assert os.path.exists(temp_path)
            
            loaded_model = BoostingModel.load(temp_path)
            assert loaded_model.is_fitted
            assert loaded_model.config == sample_config
            
            # Test that loaded model can predict
            X_test = np.random.randn(10, 10)
            predictions = loaded_model.predict(X_test)
            assert predictions.shape == (10,)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_is_fitted_property(self, sample_config, sample_data):
        """Test is_fitted property."""
        model = BoostingModel(sample_config)
        
        assert not model.is_fitted
        
        X, y = sample_data
        model.fit(X, y)
        
        assert model.is_fitted
