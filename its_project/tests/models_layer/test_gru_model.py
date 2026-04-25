"""
Unit tests for GRUModel.
"""
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from its_project.models.gru_model import GRUModel


@pytest.mark.unit
@pytest.mark.models_layer
class TestGRUModel:
    """Test GRUModel functionality."""
    
    def test_initialization(self):
        """Test GRUModel initialization."""
        config = {
            'input_size': 50,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'bidirectional': False,
            'num_classes': 3
        }
        
        model = GRUModel(config)
        
        assert model.input_size == 50
        assert model.hidden_size == 128
        assert model.num_layers == 2
        assert model.dropout == 0.2
        assert model.bidirectional == False
        assert model.num_classes == 3
        assert model.model is not None
    
    def test_initialization_defaults(self):
        """Test initialization with default config."""
        config = {}
        
        model = GRUModel(config)
        
        assert model.input_size == 50
        assert model.hidden_size == 128
        assert model.num_layers == 2
        assert model.dropout == 0.2
        assert model.bidirectional == False
        assert model.num_classes == 3
        assert model.epochs == 100
        assert model.batch_size == 32
        assert model.learning_rate == 0.001
        assert model.patience == 10
    
    def test_initialization_bidirectional(self):
        """Test initialization with bidirectional=True."""
        config = {
            'input_size': 50,
            'hidden_size': 128,
            'num_layers': 2,
            'bidirectional': True,
            'num_classes': 3
        }
        
        model = GRUModel(config)
        
        assert model.bidirectional == True
    
    def test_build_model(self):
        """Test model building."""
        config = {'input_size': 10, 'hidden_size': 32, 'num_layers': 1, 'num_classes': 3, 'dropout': 0.1}
        
        model = GRUModel(config)
        
        assert model.model is not None
        assert hasattr(model.model, 'gru')
        assert hasattr(model.model, 'fc')
        assert hasattr(model.model, 'output')
    
    @patch('its_project.models.gru_model.logger')
    @patch.object(GRUModel, 'validate_input', return_value=None)
    def test_fit(self, mock_validate, mock_logger):
        """Test model fitting."""
        config = {
            'input_size': 10,
            'hidden_size': 32,
            'num_layers': 1,
            'num_classes': 3,
            'dropout': 0.1,
            'epochs': 5,
            'batch_size': 16,
            'patience': 10
        }
        
        model = GRUModel(config)
        
        # Create sample 2D data (GRU reshapes to 3D)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        
        model.fit(X, y)
        
        assert model.is_fitted
        assert hasattr(model, 'best_state_dict')
    
    @patch('its_project.models.gru_model.logger')
    @patch.object(GRUModel, 'validate_input', return_value=None)
    def test_fit_3d_input(self, mock_validate, mock_logger):
        """Test fit with 3D input."""
        config = {
            'input_size': 10,
            'hidden_size': 32,
            'num_layers': 1,
            'num_classes': 3,
            'epochs': 2
        }
        
        model = GRUModel(config)
        
        # Create sample 3D data
        X = np.random.randn(50, 10, 10)
        y = np.random.randint(0, 3, 50)
        
        model.fit(X, y)
        
        assert model.is_fitted
    
    @patch('its_project.models.gru_model.logger')
    @patch.object(GRUModel, 'validate_input', return_value=None)
    def test_fit_early_stopping(self, mock_validate, mock_logger):
        """Test early stopping during fit."""
        config = {
            'input_size': 10,
            'hidden_size': 32,
            'num_layers': 1,
            'num_classes': 3,
            'epochs': 100,
            'patience': 2
        }
        
        model = GRUModel(config)
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        
        model.fit(X, y)
        
        assert model.is_fitted
        assert hasattr(model, 'best_state_dict')
    
    def test_predict_before_fit(self):
        """Test predict before model is fitted raises error."""
        config = {'input_size': 10, 'hidden_size': 32, 'num_layers': 1, 'num_classes': 3}
        
        model = GRUModel(config)
        
        X = np.random.randn(10, 10)
        
        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            model.predict(X)
    
    @patch('its_project.models.gru_model.logger')
    @patch.object(GRUModel, 'validate_input', return_value=None)
    def test_predict(self, mock_validate, mock_logger):
        """Test prediction."""
        config = {
            'input_size': 10,
            'hidden_size': 32,
            'num_layers': 1,
            'num_classes': 3,
            'epochs': 2
        }
        
        model = GRUModel(config)
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        model.fit(X, y)
        
        X_test = np.random.randn(10, 10)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (10,)
        assert np.all((predictions >= 0) & (predictions < 3))
    
    @patch('its_project.models.gru_model.logger')
    @patch.object(GRUModel, 'validate_input', return_value=None)
    def test_predict_3d_input(self, mock_validate, mock_logger):
        """Test predict with 3D input."""
        config = {
            'input_size': 10,
            'hidden_size': 32,
            'num_layers': 1,
            'num_classes': 3,
            'epochs': 2
        }
        
        model = GRUModel(config)
        
        X = np.random.randn(50, 10, 10)
        y = np.random.randint(0, 3, 50)
        model.fit(X, y)
        
        X_test = np.random.randn(10, 10, 10)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (10,)
    
    def test_predict_proba_before_fit(self):
        """Test predict_proba before model is fitted raises error."""
        config = {'input_size': 10, 'hidden_size': 32, 'num_layers': 1, 'num_classes': 3}
        
        model = GRUModel(config)
        
        X = np.random.randn(10, 10)
        
        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            model.predict_proba(X)
    
    @patch('its_project.models.gru_model.logger')
    @patch.object(GRUModel, 'validate_input', return_value=None)
    def test_predict_proba(self, mock_validate, mock_logger):
        """Test probability prediction."""
        config = {
            'input_size': 10,
            'hidden_size': 32,
            'num_layers': 1,
            'num_classes': 3,
            'epochs': 2
        }
        
        model = GRUModel(config)
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        model.fit(X, y)
        
        X_test = np.random.randn(10, 10)
        proba = model.predict_proba(X_test)
        
        assert proba.shape == (10, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert np.all((proba >= 0) & (proba <= 1))
    
    @patch('its_project.models.gru_model.logger')
    @patch.object(GRUModel, 'validate_input', return_value=None)
    def test_get_confidence(self, mock_validate, mock_logger):
        """Test confidence scores."""
        config = {
            'input_size': 10,
            'hidden_size': 32,
            'num_layers': 1,
            'num_classes': 3,
            'epochs': 2
        }
        
        model = GRUModel(config)
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        model.fit(X, y)
        
        X_test = np.random.randn(10, 10)
        confidence = model.get_confidence(X_test)
        
        assert confidence.shape == (10,)
        assert np.all((confidence >= 0) & (confidence <= 1))
    
    @patch('its_project.models.gru_model.logger')
    @patch.object(GRUModel, 'validate_input', return_value=None)
    def test_save(self, mock_validate, mock_logger, tmp_path):
        """Test model saving."""
        config = {
            'input_size': 10,
            'hidden_size': 32,
            'num_layers': 1,
            'num_classes': 3,
            'epochs': 1
        }
        
        model = GRUModel(config)
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        model.fit(X, y)
        
        save_path = tmp_path / "gru_model.pt"
        model.save(str(save_path))
        
        assert save_path.exists()
    
    @patch('its_project.models.gru_model.logger')
    @patch.object(GRUModel, 'validate_input', return_value=None)
    def test_load(self, mock_validate, mock_logger, tmp_path):
        """Test model loading."""
        config = {
            'input_size': 10,
            'hidden_size': 32,
            'num_layers': 1,
            'num_classes': 3,
            'epochs': 1
        }
        
        model = GRUModel(config)
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        model.fit(X, y)
        
        save_path = tmp_path / "gru_model.pt"
        model.save(str(save_path))
        
        loaded_model = GRUModel.load(str(save_path))
        
        assert loaded_model.is_fitted
        assert loaded_model.config == config
    
    @patch('its_project.models.gru_model.logger')
    def test_device_selection_cpu(self, mock_logger):
        """Test device selection defaults to CPU when CUDA unavailable."""
        config = {'input_size': 10, 'hidden_size': 32, 'num_layers': 1, 'num_classes': 3}
        
        with patch('torch.cuda.is_available', return_value=False):
            model = GRUModel(config)
            
            assert model.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @patch('its_project.models.gru_model.logger')
    def test_device_selection_cuda(self, mock_logger):
        """Test device selection uses CUDA when available."""
        config = {'input_size': 10, 'hidden_size': 32, 'num_layers': 1, 'num_classes': 3}
        
        model = GRUModel(config)
        
        assert model.device.type == 'cuda'
    
    @patch('its_project.models.gru_model.logger')
    @patch.object(GRUModel, 'validate_input', return_value=None)
    def test_is_fitted_property(self, mock_validate, mock_logger):
        """Test is_fitted property."""
        config = {
            'input_size': 10,
            'hidden_size': 32,
            'num_layers': 1,
            'num_classes': 3,
            'epochs': 1
        }
        
        model = GRUModel(config)
        
        assert not model.is_fitted
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        model.fit(X, y)
        
        assert model.is_fitted
