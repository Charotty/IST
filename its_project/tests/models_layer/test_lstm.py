"""
Unit tests for LSTMModel.
"""
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from its_project.models.lstm import LSTMModel


@pytest.mark.unit
@pytest.mark.models_layer
class TestLSTMModel:
    """Test LSTMModel functionality."""
    
    def test_initialization(self):
        """Test LSTMModel initialization."""
        config = {
            'input_size': 50,
            'hidden_size': 128,
            'num_layers': 2,
            'num_classes': 3,
            'dropout': 0.2
        }
        
        model = LSTMModel(config)
        
        assert model.input_size == 50
        assert model.hidden_size == 128
        assert model.num_layers == 2
        assert model.num_classes == 3
        assert model.dropout == 0.2
        assert model.model is not None
    
    def test_initialization_defaults(self):
        """Test initialization with default config."""
        config = {}
        
        model = LSTMModel(config)
        
        assert model.input_size == 50
        assert model.hidden_size == 128
        assert model.num_layers == 2
        assert model.num_classes == 3
        assert model.dropout == 0.2
    
    def test_build_model(self):
        """Test model building."""
        config = {'input_size': 10, 'hidden_size': 32, 'num_layers': 1, 'num_classes': 3, 'dropout': 0.1}
        
        model = LSTMModel(config)
        
        assert model.model is not None
        assert hasattr(model.model, 'lstm')
        assert hasattr(model.model, 'fc')
    
    @patch('its_project.models.lstm.print')
    @patch.object(LSTMModel, 'validate_input', return_value=None)
    def test_fit(self, mock_validate, mock_print):
        """Test model fitting."""
        config = {
            'input_size': 10,
            'hidden_size': 32,
            'num_layers': 1,
            'num_classes': 3,
            'dropout': 0.1,
            'batch_size': 16,
            'epochs': 5,
            'learning_rate': 0.001
        }
        
        model = LSTMModel(config)
        
        # Create sample 3D data (n_samples, seq_len, n_features)
        X = np.random.randn(50, 10, 10)
        y = np.random.randint(0, 3, 50)
        
        model.fit(X, y)
        
        assert model.is_fitted
        assert len(model.feature_names) == 10
    
    def test_fit_invalid_dimensions(self):
        """Test fit with invalid input dimensions."""
        config = {'input_size': 10, 'hidden_size': 32, 'num_layers': 1, 'num_classes': 3}
        
        model = LSTMModel(config)
        
        # 2D input instead of 3D
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        
        with pytest.raises(ValueError, match="LSTM expects 3D input"):
            model.fit(X, y)
    
    @patch.object(LSTMModel, 'validate_input', return_value=None)
    def test_predict_before_fit(self, mock_validate):
        """Test predict before model is fitted raises error."""
        config = {'input_size': 10, 'hidden_size': 32, 'num_layers': 1, 'num_classes': 3}
        
        model = LSTMModel(config)
        model.is_fitted = False  # Explicitly set to False
        
        X = np.random.randn(10, 10, 10)
        
        with pytest.raises(RuntimeError, match="Model not fitted"):
            model.predict(X)
    
    @patch('its_project.models.lstm.print')
    @patch.object(LSTMModel, 'validate_input', return_value=None)
    def test_predict_invalid_dimensions(self, mock_validate, mock_print):
        """Test predict with invalid input dimensions."""
        config = {'input_size': 10, 'hidden_size': 32, 'num_layers': 1, 'num_classes': 3, 'epochs': 1}
        
        model = LSTMModel(config)
        
        X = np.random.randn(50, 10, 10)
        y = np.random.randint(0, 3, 50)
        model.fit(X, y)
        
        # 2D input instead of 3D
        X_invalid = np.random.randn(10, 10)
        
        with pytest.raises(ValueError, match="LSTM expects 3D input"):
            model.predict(X_invalid)
    
    @patch('its_project.models.lstm.print')
    @patch.object(LSTMModel, 'validate_input', return_value=None)
    def test_predict(self, mock_validate, mock_print):
        """Test prediction."""
        config = {
            'input_size': 10,
            'hidden_size': 32,
            'num_layers': 1,
            'num_classes': 3,
            'epochs': 2
        }
        
        model = LSTMModel(config)
        
        X = np.random.randn(50, 10, 10)
        y = np.random.randint(0, 3, 50)
        model.fit(X, y)
        
        X_test = np.random.randn(10, 10, 10)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (10,)
        assert np.all((predictions >= 0) & (predictions < 3))
    
    @patch('its_project.models.lstm.print')
    @patch.object(LSTMModel, 'validate_input', return_value=None)
    def test_predict_proba(self, mock_validate, mock_print):
        """Test probability prediction."""
        config = {
            'input_size': 10,
            'hidden_size': 32,
            'num_layers': 1,
            'num_classes': 3,
            'epochs': 2
        }
        
        model = LSTMModel(config)
        
        X = np.random.randn(50, 10, 10)
        y = np.random.randint(0, 3, 50)
        model.fit(X, y)
        
        X_test = np.random.randn(10, 10, 10)
        proba = model.predict_proba(X_test)
        
        assert proba.shape == (10, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert np.all((proba >= 0) & (proba <= 1))
    
    @patch.object(LSTMModel, 'validate_input', return_value=None)
    def test_predict_proba_before_fit(self, mock_validate):
        """Test predict_proba before model is fitted raises error."""
        config = {'input_size': 10, 'hidden_size': 32, 'num_layers': 1, 'num_classes': 3}
        
        model = LSTMModel(config)
        model.is_fitted = False  # Explicitly set to False
        
        X = np.random.randn(10, 10, 10)
        
        with pytest.raises(RuntimeError, match="Model not fitted"):
            model.predict_proba(X)
    
    @patch('its_project.models.lstm.print')
    def test_device_selection_cpu(self, mock_print):
        """Test device selection defaults to CPU when CUDA unavailable."""
        config = {'input_size': 10, 'hidden_size': 32, 'num_layers': 1, 'num_classes': 3}
        
        with patch('torch.cuda.is_available', return_value=False):
            model = LSTMModel(config)
            
            assert model.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @patch('its_project.models.lstm.print')
    def test_device_selection_cuda(self, mock_print):
        """Test device selection uses CUDA when available."""
        config = {'input_size': 10, 'hidden_size': 32, 'num_layers': 1, 'num_classes': 3}
        
        model = LSTMModel(config)
        
        assert model.device.type == 'cuda'
    
    @patch('its_project.models.lstm.print')
    @patch.object(LSTMModel, 'validate_input', return_value=None)
    def test_feature_names_stored(self, mock_validate, mock_print):
        """Test that feature names are stored during fit."""
        config = {
            'input_size': 10,
            'hidden_size': 32,
            'num_layers': 1,
            'num_classes': 3,
            'epochs': 1
        }
        
        model = LSTMModel(config)
        
        X = np.random.randn(50, 10, 10)
        y = np.random.randint(0, 3, 50)
        model.fit(X, y)
        
        assert len(model.feature_names) == 10
        assert model.feature_names[0] == 'feat_0'
        assert model.feature_names[9] == 'feat_9'
