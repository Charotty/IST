"""
Unit tests for ModelRegistry.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from its_project.models.registry import ModelRegistry
from its_project.models.base import BaseModel


# Mock model class for testing
class MockModel(BaseModel):
    """Mock model for testing registry."""
    
    def __init__(self, config):
        super().__init__(config)
        self._fitted = False
    
    def fit(self, X, y):
        self._fitted = True
        return self
    
    def predict(self, X):
        return [0] * len(X)
    
    def predict_proba(self, X):
        return [[0.5, 0.5]] * len(X)
    
    def get_confidence(self, X):
        return [0.5] * len(X)


@pytest.mark.unit
@pytest.mark.models_layer
class TestModelRegistry:
    """Test ModelRegistry functionality."""
    
    def test_register_decorator(self):
        """Test registering a model with decorator."""
        @ModelRegistry.register("mock_model")
        class TestModel(BaseModel):
            def __init__(self, config):
                super().__init__(config)
            
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                return [0] * len(X)
            
            def predict_proba(self, X):
                return [[0.5, 0.5]] * len(X)
            
            def get_confidence(self, X):
                return [0.5] * len(X)
        
        assert "mock_model" in ModelRegistry.list_models()
        assert ModelRegistry._models["mock_model"] == TestModel
    
    def test_get_model(self):
        """Test getting a model instance."""
        @ModelRegistry.register("test_model")
        class TestModel(BaseModel):
            def __init__(self, config):
                super().__init__(config)
            
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                return [0] * len(X)
            
            def predict_proba(self, X):
                return [[0.5, 0.5]] * len(X)
            
            def get_confidence(self, X):
                return [0.5] * len(X)
        
        config = {'param': 'value'}
        model = ModelRegistry.get_model("test_model", config)
        
        assert isinstance(model, TestModel)
        assert model.config == config
    
    def test_get_model_not_registered(self):
        """Test getting a model that is not registered."""
        with pytest.raises(ValueError, match="Model 'nonexistent' not registered"):
            ModelRegistry.get_model("nonexistent", {})
    
    def test_list_models(self):
        """Test listing available models."""
        # Clear any existing models
        ModelRegistry._models.clear()
        
        @ModelRegistry.register("model1")
        class Model1(BaseModel):
            def __init__(self, config):
                super().__init__(config)
            def fit(self, X, y): return self
            def predict(self, X): return [0] * len(X)
            def predict_proba(self, X): return [[0.5, 0.5]] * len(X)
            def get_confidence(self, X): return [0.5] * len(X)
        
        @ModelRegistry.register("model2")
        class Model2(BaseModel):
            def __init__(self, config):
                super().__init__(config)
            def fit(self, X, y): return self
            def predict(self, X): return [0] * len(X)
            def predict_proba(self, X): return [[0.5, 0.5]] * len(X)
            def get_confidence(self, X): return [0.5] * len(X)
        
        models = ModelRegistry.list_models()
        
        assert "model1" in models
        assert "model2" in models
        assert len(models) == 2
    
    def test_list_models_empty(self):
        """Test listing models when registry is empty."""
        ModelRegistry._models.clear()
        
        models = ModelRegistry.list_models()
        
        assert models == []
    
    def test_save_model(self):
        """Test saving a model through registry."""
        @ModelRegistry.register("save_test_model")
        class SaveTestModel(BaseModel):
            def __init__(self, config):
                super().__init__(config)
                self._test_data = "test"
            
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                return [0] * len(X)
            
            def predict_proba(self, X):
                return [[0.5, 0.5]] * len(X)
            
            def get_confidence(self, X):
                return [0.5] * len(X)
        
        model = ModelRegistry.get_model("save_test_model", {})
        
        # Mock the save method to avoid actual file I/O
        with patch.object(model, 'save') as mock_save:
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "model.pkl"
                ModelRegistry.save_model(model, path)
                mock_save.assert_called_once_with(path)
    
    def test_load_model(self):
        """Test loading a model through registry."""
        # Mock BaseModel.load to avoid actual file I/O
        mock_model = Mock(spec=BaseModel)
        with patch.object(BaseModel, 'load', return_value=mock_model) as mock_load:
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "model.pkl"
                result = ModelRegistry.load_model(path)
                mock_load.assert_called_once_with(path)
                assert result == mock_model
    
    def test_multiple_registrations(self):
        """Test that multiple models can be registered."""
        ModelRegistry._models.clear()
        
        for i in range(5):
            @ModelRegistry.register(f"model_{i}")
            class DynamicModel(BaseModel):
                def __init__(self, config):
                    super().__init__(config)
                def fit(self, X, y): return self
                def predict(self, X): return [0] * len(X)
                def predict_proba(self, X): return [[0.5, 0.5]] * len(X)
                def get_confidence(self, X): return [0.5] * len(X)
        
        models = ModelRegistry.list_models()
        
        assert len(models) == 5
        for i in range(5):
            assert f"model_{i}" in models
    
    def test_register_overwrites(self):
        """Test that registering the same name overwrites."""
        ModelRegistry._models.clear()
        
        @ModelRegistry.register("duplicate")
        class Model1(BaseModel):
            def __init__(self, config):
                super().__init__(config)
            def fit(self, X, y): return self
            def predict(self, X): return [0] * len(X)
            def predict_proba(self, X): return [[0.5, 0.5]] * len(X)
            def get_confidence(self, X): return [0.5] * len(X)
        
        @ModelRegistry.register("duplicate")
        class Model2(BaseModel):
            def __init__(self, config):
                super().__init__(config)
            def fit(self, X, y): return self
            def predict(self, X): return [1] * len(X)
            def predict_proba(self, X): return [[0.6, 0.4]] * len(X)
            def get_confidence(self, X): return [0.6] * len(X)
        
        model = ModelRegistry.get_model("duplicate", {})
        
        # Should be Model2 (last registered)
        assert isinstance(model, Model2)
    
    def test_decorator_returns_class(self):
        """Test that the decorator returns the class unchanged."""
        @ModelRegistry.register("return_test")
        class TestModel(BaseModel):
            def __init__(self, config):
                super().__init__(config)
            def fit(self, X, y): return self
            def predict(self, X): return [0] * len(X)
            def predict_proba(self, X): return [[0.5, 0.5]] * len(X)
            def get_confidence(self, X): return [0.5] * len(X)
        
        assert TestModel.__name__ == "TestModel"
    
    def test_get_model_with_config(self):
        """Test that config is passed to model constructor."""
        @ModelRegistry.register("config_test")
        class ConfigTestModel(BaseModel):
            def __init__(self, config):
                super().__init__(config)
                self.test_param = config.get('test_param', 'default')
            def fit(self, X, y): return self
            def predict(self, X): return [0] * len(X)
            def predict_proba(self, X): return [[0.5, 0.5]] * len(X)
            def get_confidence(self, X): return [0.5] * len(X)
        
        config = {'test_param': 'custom_value', 'other': 123}
        model = ModelRegistry.get_model("config_test", config)
        
        assert model.test_param == 'custom_value'
        assert model.config == config
    
    def test_registry_is_class_level(self):
        """Test that registry is shared across all instances."""
        ModelRegistry._models.clear()
        
        @ModelRegistry.register("shared_test")
        class SharedModel(BaseModel):
            def __init__(self, config):
                super().__init__(config)
            def fit(self, X, y): return self
            def predict(self, X): return [0] * len(X)
            def predict_proba(self, X): return [[0.5, 0.5]] * len(X)
            def get_confidence(self, X): return [0.5] * len(X)
        
        # The model should be in the registry
        assert "shared_test" in ModelRegistry.list_models()
        
        # Should be able to get it
        model = ModelRegistry.get_model("shared_test", {})
        assert isinstance(model, SharedModel)
    
    def test_decorator_with_none_name(self):
        """Test decorator with empty string name."""
        @ModelRegistry.register("")
        class EmptyNameModel(BaseModel):
            def __init__(self, config):
                super().__init__(config)
            def fit(self, X, y): return self
            def predict(self, X): return [0] * len(X)
            def predict_proba(self, X): return [[0.5, 0.5]] * len(X)
            def get_confidence(self, X): return [0.5] * len(X)
        
        assert "" in ModelRegistry.list_models()
    
    def test_save_model_propagates_exception(self):
        """Test that save_model propagates exceptions from model.save."""
        @ModelRegistry.register("error_save_model")
        class ErrorSaveModel(BaseModel):
            def __init__(self, config):
                super().__init__(config)
            def fit(self, X, y): return self
            def predict(self, X): return [0] * len(X)
            def predict_proba(self, X): return [[0.5, 0.5]] * len(X)
            def get_confidence(self, X): return [0.5] * len(X)
            def save(self, path):
                raise IOError("Save failed")
        
        model = ModelRegistry.get_model("error_save_model", {})
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            with pytest.raises(IOError, match="Save failed"):
                ModelRegistry.save_model(model, path)
    
    def test_load_model_propagates_exception(self):
        """Test that load_model propagates exceptions from BaseModel.load."""
        with patch.object(BaseModel, 'load', side_effect=IOError("Load failed")):
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "model.pkl"
                with pytest.raises(IOError, match="Load failed"):
                    ModelRegistry.load_model(path)
