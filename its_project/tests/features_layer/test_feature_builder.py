"""
Unit tests for FeatureBuilder.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from its_project.features.feature_builder import FeatureBuilder


@pytest.mark.unit
@pytest.mark.features_layer
class TestFeatureBuilder:
    """Test FeatureBuilder class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock FeatureConfig."""
        config = Mock()
        config.get_active_features.return_value = {
            "technical": ["sma", "rsi"],
            "lob": [],
            "microstructure": []
        }
        config.validate_features.return_value = ["sma", "rsi"]
        config.scaling_method = "zscore"
        config.min_feature_std = 0.01
        config.max_nan_ratio = 0.5
        return config
    
    @pytest.fixture
    def feature_builder(self, mock_config):
        """Create FeatureBuilder instance."""
        return FeatureBuilder(mock_config)
    
    def test_initialization(self, mock_config):
        """Test FeatureBuilder initialization."""
        builder = FeatureBuilder(mock_config)
        
        assert builder.config == mock_config
        assert builder._feature_calculators == {}
        assert builder._feature_names == []
        assert builder._scaling_stats is None
    
    def test_initialize_calculators_technical(self, feature_builder):
        """Test initialization with technical features."""
        feature_builder._initialize_calculators()
        
        assert "technical" in feature_builder._feature_calculators
    
    def test_initialize_calculators_lob(self, mock_config):
        """Test initialization with LOB features."""
        mock_config.get_active_features.return_value = {
            "technical": [],
            "lob": ["ofi_0", "imbalance_0"],
            "microstructure": []
        }
        
        builder = FeatureBuilder(mock_config)
        builder._initialize_calculators()
        
        assert "lob" in builder._feature_calculators
    
    def test_initialize_calculators_microstructure(self, mock_config):
        """Test initialization with microstructure features."""
        mock_config.get_active_features.return_value = {
            "technical": [],
            "lob": [],
            "microstructure": ["spread", "depth"]
        }
        
        builder = FeatureBuilder(mock_config)
        builder._initialize_calculators()
        
        assert "microstructure" in builder._feature_calculators
    
    def test_initialize_calculators_empty(self, mock_config):
        """Test initialization with no active features."""
        mock_config.get_active_features.return_value = {
            "technical": [],
            "lob": [],
            "microstructure": []
        }
        
        builder = FeatureBuilder(mock_config)
        builder._initialize_calculators()
        
        assert builder._feature_calculators == {}
    
    @patch('its_project.features.feature_builder.TechnicalFeatures')
    @patch('its_project.features.feature_builder.normalize_features')
    def test_calculate_features_with_scaling(self, mock_normalize, mock_tech, feature_builder):
        """Test feature calculation with scaling."""
        # Setup mocks
        mock_calculator = Mock()
        mock_calculator.calculate.return_value = np.random.randn(100, 2)
        mock_calculator.get_feature_names.return_value = ["sma", "rsi"]
        feature_builder._feature_calculators = {"technical": mock_calculator}
        
        mock_normalize.return_value = (np.random.randn(100, 2), {"mean": np.zeros(2), "std": np.ones(2)})
        
        data = pd.DataFrame({
            'open': [100] * 100,
            'high': [105] * 100,
            'low': [95] * 100,
            'close': [100] * 100,
            'volume': [1000] * 100
        })
        
        features = feature_builder.calculate_features(data, fit_scaling=True)
        
        assert features.shape == (100, 2)
        assert feature_builder._scaling_stats is not None
        mock_normalize.assert_called_once()
    
    @patch('its_project.features.feature_builder.TechnicalFeatures')
    def test_calculate_features_without_scaling(self, mock_tech, feature_builder):
        """Test feature calculation without scaling."""
        feature_builder.config.scaling_method = None
        
        mock_calculator = Mock()
        mock_calculator.calculate.return_value = np.random.randn(100, 2)
        mock_calculator.get_feature_names.return_value = ["sma", "rsi"]
        feature_builder._feature_calculators = {"technical": mock_calculator}
        
        data = pd.DataFrame({
            'open': [100] * 100,
            'high': [105] * 100,
            'low': [95] * 100,
            'close': [100] * 100,
            'volume': [1000] * 100
        })
        
        features = feature_builder.calculate_features(data, fit_scaling=True)
        
        assert features.shape == (100, 2)
    
    def test_calculate_features_no_calculators(self, feature_builder):
        """Test feature calculation when no calculators initialized."""
        # Directly set calculators to empty to simulate no calculators
        feature_builder._feature_calculators = {}
        
        data = pd.DataFrame({
            'open': [100] * 100,
            'high': [105] * 100,
            'low': [95] * 100,
            'close': [100] * 100,
            'volume': [1000] * 100
        })
        
        # Mock _initialize_calculators to not add any calculators
        with patch.object(feature_builder, '_initialize_calculators'):
            with pytest.raises(ValueError, match="No features could be calculated"):
                feature_builder.calculate_features(data)
    
    @patch('its_project.features.feature_builder.TechnicalFeatures')
    def test_calculate_features_calculator_error(self, mock_tech, feature_builder):
        """Test feature calculation when calculator raises error."""
        mock_calculator = Mock()
        mock_calculator.calculate.side_effect = Exception("Calculator error")
        feature_builder._feature_calculators = {"technical": mock_calculator}
        
        data = pd.DataFrame({
            'open': [100] * 100,
            'high': [105] * 100,
            'low': [95] * 100,
            'close': [100] * 100,
            'volume': [1000] * 100
        })
        
        with pytest.raises(ValueError, match="No features could be calculated"):
            feature_builder.calculate_features(data)
    
    @patch('its_project.features.feature_builder.TechnicalFeatures')
    def test_calculate_features_apply_scaling_no_stats(self, mock_tech, feature_builder):
        """Test applying scaling without fitted stats."""
        mock_calculator = Mock()
        mock_calculator.calculate.return_value = np.random.randn(100, 2)
        mock_calculator.get_feature_names.return_value = ["sma", "rsi"]
        feature_builder._feature_calculators = {"technical": mock_calculator}
        
        data = pd.DataFrame({
            'open': [100] * 100,
            'high': [105] * 100,
            'low': [95] * 100,
            'close': [100] * 100,
            'volume': [1000] * 100
        })
        
        with pytest.raises(ValueError, match="Scaling stats not available"):
            feature_builder.calculate_features(data, fit_scaling=False)
    
    @patch('its_project.features.feature_builder.TechnicalFeatures')
    def test_apply_scaling(self, mock_tech, feature_builder):
        """Test _apply_scaling method."""
        feature_builder._scaling_stats = {"mean": np.array([0.0, 0.0]), "std": np.array([1.0, 1.0])}
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        scaled = feature_builder._apply_scaling(features)
        
        assert np.allclose(scaled, features)
    
    def test_apply_scaling_no_stats(self, feature_builder):
        """Test _apply_scaling with no stats."""
        feature_builder._scaling_stats = None
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        scaled = feature_builder._apply_scaling(features)
        
        assert np.array_equal(scaled, features)
    
    def test_apply_scaling_no_scaling_method(self, feature_builder):
        """Test _apply_scaling with no scaling method."""
        feature_builder.config.scaling_method = None
        feature_builder._scaling_stats = {"mean": np.array([0.0, 0.0]), "std": np.array([1.0, 1.0])}
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        scaled = feature_builder._apply_scaling(features)
        
        assert np.array_equal(scaled, features)
    
    def test_validate_features_constant_features(self, feature_builder):
        """Test validation removes constant features."""
        features = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])  # Constant
        feature_builder._feature_names = ["const1", "const2"]
        
        validated = feature_builder._validate_features(features)
        
        # Should remove constant features
        assert validated.shape[1] == 0
    
    def test_validate_features_nan_features(self, feature_builder):
        """Test validation removes features with too many NaNs."""
        features = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        feature_builder._feature_names = ["good", "bad"]
        
        validated = feature_builder._validate_features(features)
        
        # Should remove NaN-heavy feature
        assert validated.shape[1] == 1
    
    def test_validate_features_fill_nans(self, feature_builder):
        """Test validation fills remaining NaNs."""
        features = np.array([[1.0, 2.0], [np.nan, 3.0], [3.0, 4.0]])
        feature_builder._feature_names = ["feat1", "feat2"]
        
        validated = feature_builder._validate_features(features)
        
        # Should fill NaN with 0
        assert not np.any(np.isnan(validated))
    
    def test_get_feature_names(self, feature_builder):
        """Test getting feature names."""
        feature_builder._feature_names = ["sma", "rsi", "macd"]
        
        names = feature_builder.get_feature_names()
        
        assert names == ["sma", "rsi", "macd"]
    
    def test_get_feature_names_empty(self, feature_builder):
        """Test getting feature names when empty."""
        feature_builder._feature_names = []
        
        names = feature_builder.get_feature_names()
        
        assert names == []
    
    def test_get_scaling_stats(self, feature_builder):
        """Test getting scaling stats."""
        feature_builder._scaling_stats = {"mean": np.array([0.0]), "std": np.array([1.0])}
        
        stats = feature_builder.get_scaling_stats()
        
        assert stats is not None
        assert "mean" in stats
        assert "std" in stats
    
    def test_get_scaling_stats_none(self, feature_builder):
        """Test getting scaling stats when None."""
        feature_builder._scaling_stats = None
        
        stats = feature_builder.get_scaling_stats()
        
        assert stats is None
    
    def test_set_scaling_stats(self, feature_builder):
        """Test setting scaling stats."""
        stats = {"mean": np.array([0.0]), "std": np.array([1.0])}
        
        feature_builder.set_scaling_stats(stats)
        
        assert feature_builder._scaling_stats is not None
        assert np.array_equal(feature_builder._scaling_stats["mean"], stats["mean"])
