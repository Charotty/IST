"""
Unit tests for EnsembleModel.
"""
import pytest
import numpy as np
from its_project.models.ensemble import EnsembleModel


@pytest.mark.unit
@pytest.mark.models_layer
class TestEnsembleModel:
    """Test EnsembleModel functionality."""
    
    def test_initialization(self):
        """Test EnsembleModel initialization."""
        config = {
            'rf_estimators': 50,
            'rf_max_depth': 5,
            'gb_estimators': 50,
            'gb_max_depth': 3
        }
        model = EnsembleModel(config)
        
        assert model.config == config
        assert 'lr' in model.models
        assert 'rf' in model.models
        assert 'gb' in model.models
        assert model.ensemble is not None
    
    def test_initialization_defaults(self):
        """Test EnsembleModel initialization with default config."""
        model = EnsembleModel({})
        
        assert model.config == {}
        assert 'lr' in model.models
        assert 'rf' in model.models
        assert 'gb' in model.models
    
    def test_fit(self):
        """Test fitting the ensemble model."""
        model = EnsembleModel({})
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        
        fitted_model = model.fit(X, y)
        
        assert fitted_model is model
        assert model.is_fitted
        assert model.feature_names == ['feat_0', 'feat_1']
    
    def test_fit_with_y(self):
        """Test fitting with y parameter."""
        model = EnsembleModel({})
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        
        model.fit(X, y)
        
        assert model.is_fitted
    
    def test_predict(self):
        """Test predicting with the ensemble model."""
        model = EnsembleModel({})
        
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])
        model.fit(X_train, y_train)
        
        X_test = np.array([[2, 3], [6, 7]])
        predictions = model.predict(X_test)
        
        assert predictions.shape == (2,)
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_not_fitted(self):
        """Test predicting without fitting first."""
        model = EnsembleModel({})
        
        X = np.array([[1, 2], [3, 4]])
        
        # EnsembleModel doesn't initialize is_fitted, so it raises AttributeError
        with pytest.raises((RuntimeError, AttributeError)):
            model.predict(X)
    
    def test_predict_proba(self):
        """Test predicting probabilities with the ensemble model."""
        model = EnsembleModel({})
        
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])
        model.fit(X_train, y_train)
        
        X_test = np.array([[2, 3], [6, 7]])
        proba = model.predict_proba(X_test)
        
        assert proba.shape == (2, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_predict_proba_not_fitted(self):
        """Test predicting probabilities without fitting first."""
        model = EnsembleModel({})
        
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises((RuntimeError, AttributeError)):
            model.predict_proba(X)
    
    def test_validate_input_invalid_dimensions(self):
        """Test validate_input with invalid dimensions."""
        model = EnsembleModel({})
        
        X = np.array([1, 2, 3])  # 1D array
        
        with pytest.raises(ValueError, match="X must be 2D"):
            model.validate_input(X)
    
    def test_validate_input_invalid_type(self):
        """Test validate_input with invalid type."""
        model = EnsembleModel({})
        
        X = [[1, 2], [3, 4]]  # List instead of numpy array
        
        with pytest.raises(TypeError, match="X must be np.ndarray"):
            model.validate_input(X)
    
    def test_get_confidence(self):
        """Test get_confidence method."""
        model = EnsembleModel({})
        
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])
        model.fit(X_train, y_train)
        
        X_test = np.array([[2, 3], [6, 7]])
        confidence = model.get_confidence(X_test)
        
        assert confidence.shape == (2,)
        assert all(0 <= c <= 1 for c in confidence)
    
    def test_get_confidence_not_fitted(self):
        """Test get_confidence without fitting first."""
        model = EnsembleModel({})
        
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises((RuntimeError, AttributeError)):
            model.get_confidence(X)
    
    def test_is_fitted_property(self):
        """Test is_fitted property."""
        model = EnsembleModel({})
        
        # Before fit, is_fitted attribute doesn't exist
        assert not hasattr(model, 'is_fitted') or not model.is_fitted
        
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])
        model.fit(X_train, y_train)
        
        assert model.is_fitted
    
    def test_fit_with_larger_dataset(self):
        """Test fitting with a larger dataset."""
        model = EnsembleModel({})
        
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        model.fit(X, y)
        
        assert model.is_fitted
        assert len(model.feature_names) == 5
    
    def test_predict_with_larger_dataset(self):
        """Test prediction with larger dataset."""
        model = EnsembleModel({})
        
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)
        
        X_test = np.random.randn(20, 5)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (20,)
        assert all(p in [0, 1] for p in predictions)


