"""
Unit tests for FeatureConfig.
"""
import pytest
from its_project.features.feature_config import FeatureConfig


@pytest.mark.unit
@pytest.mark.features_layer
class TestFeatureConfig:
    """Test FeatureConfig class."""
    
    def test_initialization_defaults(self):
        """Test FeatureConfig initialization with defaults."""
        config = FeatureConfig()
        
        assert config.technical_indicators == ["rsi", "macd", "bbands", "atr", "stoch"]
        assert config.lob_features == [
            "spread", "spread_pct", "volume_imbalance", 
            "bid_depth", "ask_depth", "bid_vwap", "ask_vwap",
            "bid_density", "ask_density", "ofi_0", "ofi_1", "ofi_2", "ofi_3", "ofi_4"
        ]
        assert config.microstructure_features == [
            "roll_impact", "vpin", "realized_volatility", "amihud_illiquidity", "kyle_lambda"
        ]
        assert config.exclude_features == [
            "ofi_5", "ofi_6", "ofi_7", "ofi_8", "ofi_9",
            "bid_density", "ask_density"
        ]
        assert config.scaling_method == "zscore"
        assert config.scaling_window is None
        assert config.scaling_rolling is True
        assert config.min_feature_std == 1e-6
        assert config.max_nan_ratio == 0.1
    
    def test_initialization_custom(self):
        """Test FeatureConfig initialization with custom values."""
        config = FeatureConfig(
            technical_indicators=["sma", "ema"],
            lob_features=["spread"],
            microstructure_features=["vpin"],
            exclude_features=["sma"],
            scaling_method="minmax",
            scaling_window=100,
            scaling_rolling=False,
            min_feature_std=0.01,
            max_nan_ratio=0.5
        )
        
        assert config.technical_indicators == ["sma", "ema"]
        assert config.lob_features == ["spread"]
        assert config.microstructure_features == ["vpin"]
        assert config.exclude_features == ["sma"]
        assert config.scaling_method == "minmax"
        assert config.scaling_window == 100
        assert config.scaling_rolling is False
        assert config.min_feature_std == 0.01
        assert config.max_nan_ratio == 0.5
    
    def test_get_active_features(self):
        """Test get_active_filters method."""
        config = FeatureConfig()
        
        active = config.get_active_features()
        
        assert "technical" in active
        assert "lob" in active
        assert "microstructure" in active
        
        # Check that excluded features are not in active
        assert "bid_density" not in active["lob"]
        assert "ask_density" not in active["lob"]
        assert "ofi_5" not in active["lob"]
        
        # Check that non-excluded features are present
        assert "spread" in active["lob"]
        assert "ofi_0" in active["lob"]
    
    def test_get_active_features_custom_exclude(self):
        """Test get_active_features with custom exclusions."""
        config = FeatureConfig(
            technical_indicators=["rsi", "macd", "sma"],
            exclude_features=["rsi"]
        )
        
        active = config.get_active_features()
        
        assert "rsi" not in active["technical"]
        assert "macd" in active["technical"]
        assert "sma" in active["technical"]
    
    def test_get_active_features_empty_categories(self):
        """Test get_active_features when all features are excluded."""
        config = FeatureConfig(
            technical_indicators=["rsi"],
            exclude_features=["rsi"]
        )
        
        active = config.get_active_features()
        
        assert active["technical"] == []
    
    def test_validate_features(self):
        """Test validate_features method."""
        config = FeatureConfig()
        
        feature_names = ["rsi", "macd", "bid_density", "spread"]
        
        validated = config.validate_features(feature_names)
        
        assert "rsi" in validated
        assert "macd" in validated
        assert "spread" in validated
        assert "bid_density" not in validated  # Excluded
    
    def test_validate_features_all_excluded(self):
        """Test validate_features when all are excluded."""
        config = FeatureConfig(
            exclude_features=["rsi", "macd"]
        )
        
        feature_names = ["rsi", "macd"]
        
        validated = config.validate_features(feature_names)
        
        assert validated == []
    
    def test_validate_features_none_excluded(self):
        """Test validate_features when none are excluded."""
        config = FeatureConfig(
            exclude_features=[]
        )
        
        feature_names = ["rsi", "macd"]
        
        validated = config.validate_features(feature_names)
        
        assert validated == ["rsi", "macd"]
    
    def test_validate_features_empty_list(self):
        """Test validate_features with empty list."""
        config = FeatureConfig()
        
        validated = config.validate_features([])
        
        assert validated == []
    
    def test_dataclass_immutability(self):
        """Test that dataclass fields can be modified."""
        config = FeatureConfig()
        
        original_indicators = config.technical_indicators.copy()
        
        # Modify the list
        config.technical_indicators.append("new_indicator")
        
        # Check that it was modified
        assert len(config.technical_indicators) == len(original_indicators) + 1
        assert "new_indicator" in config.technical_indicators
    
    def test_scaling_method_zscore(self):
        """Test zscore scaling method."""
        config = FeatureConfig(scaling_method="zscore")
        
        assert config.scaling_method == "zscore"
    
    def test_scaling_method_minmax(self):
        """Test minmax scaling method."""
        config = FeatureConfig(scaling_method="minmax")
        
        assert config.scaling_method == "minmax"
    
    def test_scaling_window_none(self):
        """Test scaling window None."""
        config = FeatureConfig(scaling_window=None)
        
        assert config.scaling_window is None
    
    def test_scaling_window_int(self):
        """Test scaling window with integer."""
        config = FeatureConfig(scaling_window=50)
        
        assert config.scaling_window == 50
    
    def test_scaling_rolling_true(self):
        """Test scaling rolling True."""
        config = FeatureConfig(scaling_rolling=True)
        
        assert config.scaling_rolling is True
    
    def test_scaling_rolling_false(self):
        """Test scaling rolling False."""
        config = FeatureConfig(scaling_rolling=False)
        
        assert config.scaling_rolling is False
    
    def test_min_feature_std(self):
        """Test min_feature_std parameter."""
        config = FeatureConfig(min_feature_std=0.001)
        
        assert config.min_feature_std == 0.001
    
    def test_max_nan_ratio(self):
        """Test max_nan_ratio parameter."""
        config = FeatureConfig(max_nan_ratio=0.2)
        
        assert config.max_nan_ratio == 0.2
    
    def test_default_lob_features_completeness(self):
        """Test that default LOB features include expected entries."""
        config = FeatureConfig()
        
        expected_lob = [
            "spread", "spread_pct", "volume_imbalance", 
            "bid_depth", "ask_depth", "bid_vwap", "ask_vwap",
            "bid_density", "ask_density", "ofi_0", "ofi_1", "ofi_2", "ofi_3", "ofi_4"
        ]
        
        assert config.lob_features == expected_lob
    
    def test_default_microstructure_features_completeness(self):
        """Test that default microstructure features include expected entries."""
        config = FeatureConfig()
        
        expected_micro = [
            "roll_impact", "vpin", "realized_volatility", "amihud_illiquidity", "kyle_lambda"
        ]
        
        assert config.microstructure_features == expected_micro
    
    def test_default_technical_indicators_completeness(self):
        """Test that default technical indicators include expected entries."""
        config = FeatureConfig()
        
        expected_tech = ["rsi", "macd", "bbands", "atr", "stoch"]
        
        assert config.technical_indicators == expected_tech
