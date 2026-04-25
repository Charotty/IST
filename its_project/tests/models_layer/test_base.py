"""
Unit tests for BaseModel.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
from its_project.models.base import BaseModel


class ConcreteModel(BaseModel):
    """Concrete implementation for testing BaseModel."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        self.validate_input(X, y)
        self._is_fitted = True
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.validate_input(X)
        return np.zeros(len(X), dtype=int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.validate_input(X)
        n_samples = len(X)
        return np.ones((n_samples, 3)) / 3  # Equal probabilities


@pytest.mark.unit
@pytest.mark.models_layer
class TestBaseModel:
    """Test BaseModel functionality."""
    
    def test_initialization(self):
        """Test BaseModel initialization."""
        config = {'param1': 'value1', 'param2': 42}
        model = ConcreteModel(config)
        
        assert model.config == config
        assert model._is_fitted is False
        assert model.feature_names is None
    
    def test_fit_and_predict(self):
        """Test fit and predict workflow."""
        model = ConcreteModel({})
        
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        
        model.fit(X, y)
        
        assert model._is_fitted is True
        assert len(model.feature_names) == 5
        
        predictions = model.predict(X)
        assert predictions.shape == (100,)
        assert predictions.dtype == int
    
    def test_predict_proba(self):
        """Test predict_proba."""
        model = ConcreteModel({})
        
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 3, 50)
        
        model.fit(X, y)
        
        proba = model.predict_proba(X)
        assert proba.shape == (50, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_get_confidence(self):
        """Test get_confidence."""
        
        class VaryingProbaModel(ConcreteModel):
            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                self.validate_input(X)
                n_samples = len(X)
                # Create varying probabilities
                proba = np.random.dirichlet([1, 1, 1], size=n_samples)
                return proba
        
        model = VaryingProbaModel({})
        
        X = np.random.randn(30, 4)
        y = np.random.randint(0, 3, 30)
        
        model.fit(X, y)
        
        confidence = model.get_confidence(X)
        assert confidence.shape == (30,)
        assert np.all(confidence >= 0) and np.all(confidence <= 1)
    
    def test_validate_input(self):
        """Test input validation."""
        model = ConcreteModel({})
        
        # Test invalid X type
        with pytest.raises(TypeError):
            model.validate_input([1, 2, 3])
        
        # Test invalid X dimensions
        with pytest.raises(ValueError):
            model.validate_input(np.random.randn(10))
        
        # Test invalid y type
        with pytest.raises(TypeError):
            model.validate_input(np.random.randn(10, 5), [1, 2, 3])
        
        # Test invalid y dimensions
        with pytest.raises(ValueError):
            model.validate_input(np.random.randn(10, 5), np.random.randn(10, 2))
        
        # Test mismatched lengths
        with pytest.raises(ValueError):
            model.validate_input(np.random.randn(10, 5), np.random.randn(5))
    
    def test_save_and_load(self):
        """Test save and load functionality."""
        model = ConcreteModel({'test_param': 123})
        
        X = np.random.randn(20, 3)
        y = np.random.randint(0, 3, 20)
        
        model.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'model.pkl'
            model.save(save_path)
            
            assert save_path.exists()
            
            loaded_model = ConcreteModel.load(save_path)
            
            assert loaded_model.config == {'test_param': 123}
            assert loaded_model._is_fitted is True
            assert loaded_model.feature_names == model.feature_names
    
    def test_save_unfitted_model(self):
        """Test that saving unfitted model raises error."""
        model = ConcreteModel({})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'model.pkl'
            
            with pytest.raises(RuntimeError, match="Model not fitted"):
                model.save(save_path)
    
    def test_get_metadata(self):
        """Test get_metadata."""
        model = ConcreteModel({'param': 'value'})
        
        metadata = model.get_metadata()
        
        assert 'model_type' in metadata
        assert 'is_fitted' in metadata
        assert 'config' in metadata
        assert metadata['model_type'] == 'ConcreteModel'
        assert metadata['is_fitted'] is False
        assert metadata['config'] == {'param': 'value'}
