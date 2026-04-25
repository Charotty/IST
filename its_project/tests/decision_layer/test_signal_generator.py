"""
Unit tests for SignalGenerator.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from its_project.decision.signal_generator import SignalGenerator
from its_project.decision.decision import Action, Signal


@pytest.mark.unit
@pytest.mark.decision_layer
class TestSignalGenerator:
    """Test SignalGenerator class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "threshold_strategy": "fixed",
            "fixed_threshold": 0.7,
            "adaptive_window": 100,
            "min_signal_strength": 0.1,
            "signal_smoothing": False,
            "smoothing_window": 5
        }
    
    @pytest.fixture
    def signal_generator(self, config):
        """Create SignalGenerator instance."""
        return SignalGenerator(config)
    
    @pytest.fixture
    def mock_regression_model(self):
        """Create mock regression model."""
        model = Mock()
        model.predict.return_value = np.array([0.01])  # Higher for better confidence
        return model
    
    @pytest.fixture
    def mock_classification_model(self):
        """Create mock classification model."""
        model = Mock()
        model.predict_proba.return_value = np.array([[0.02, 0.03, 0.95]])  # Very high confidence
        model.predict.return_value = np.array([2])
        return model
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features."""
        return np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    
    def test_initialization(self, signal_generator, config):
        """Test SignalGenerator initialization."""
        assert signal_generator.config == config
        assert signal_generator.regression_model is None
        assert signal_generator.classification_model is None
        assert signal_generator.threshold_strategy == "fixed"
        assert signal_generator.fixed_threshold == 0.7
        assert signal_generator.adaptive_window == 100
        assert signal_generator.min_signal_strength == 0.1
        assert signal_generator.signal_smoothing is False
        assert signal_generator.smoothing_window == 5
        assert signal_generator.prediction_history == []
        assert signal_generator.return_history == []
    
    def test_initialization_defaults(self):
        """Test initialization with default values."""
        signal_generator = SignalGenerator({})
        assert signal_generator.threshold_strategy == "fixed"
        assert signal_generator.fixed_threshold == 0.7
        assert signal_generator.adaptive_window == 100
        assert signal_generator.min_signal_strength == 0.1
        assert signal_generator.signal_smoothing is False
        assert signal_generator.smoothing_window == 5
    
    def test_set_models(self, signal_generator, mock_regression_model, mock_classification_model):
        """Test setting regression and classification models."""
        signal_generator.set_models(mock_regression_model, mock_classification_model)
        
        assert signal_generator.regression_model == mock_regression_model
        assert signal_generator.classification_model == mock_classification_model
    
    def test_generate_signal_success(self, signal_generator, mock_regression_model, mock_classification_model, sample_features):
        """Test successful signal generation."""
        signal_generator.set_models(mock_regression_model, mock_classification_model)
        
        signal = signal_generator.generate_signal(
            features=sample_features,
            current_price=50000.0,
            timestamp_ms=1234567890000,
            symbol="BTCUSDT"
        )
        
        assert signal is not None
        assert signal.action == Action.BUY
        assert signal.confidence >= 0.7
        assert signal.timestamp == 1234567890000
        assert signal.symbol == "BTCUSDT"
        assert "price_change" in signal.metadata
        assert "classification" in signal.metadata
        assert "threshold" in signal.metadata
    
    def test_generate_signal_no_models(self, signal_generator, sample_features):
        """Test signal generation when models not set."""
        signal = signal_generator.generate_signal(
            features=sample_features,
            current_price=50000.0,
            timestamp_ms=1234567890000,
            symbol="BTCUSDT"
        )
        
        assert signal is None
    
    def test_generate_signal_low_confidence(self, signal_generator, mock_regression_model, sample_features):
        """Test signal generation with low confidence."""
        mock_classification_model = Mock()
        mock_classification_model.predict_proba.return_value = np.array([[0.4, 0.3, 0.3]])
        mock_classification_model.predict.return_value = np.array([0])
        
        signal_generator.set_models(mock_regression_model, mock_classification_model)
        
        signal = signal_generator.generate_signal(
            features=sample_features,
            current_price=50000.0,
            timestamp_ms=1234567890000,
            symbol="BTCUSDT"
        )
        
        assert signal is None  # Confidence below threshold
    
    def test_generate_signal_classification_filter_reject(self, signal_generator, mock_regression_model, sample_features):
        """Test signal generation rejected by classification filter."""
        mock_classification_model = Mock()
        mock_classification_model.predict_proba.return_value = np.array([[0.05, 0.05, 0.05]])  # Very low confidence
        mock_classification_model.predict.return_value = np.array([2])
        
        signal_generator.set_models(mock_regression_model, mock_classification_model)
        
        signal = signal_generator.generate_signal(
            features=sample_features,
            current_price=50000.0,
            timestamp_ms=1234567890000,
            symbol="BTCUSDT"
        )
        
        assert signal is None  # Classification filter rejects
    
    def test_generate_signal_exception(self, signal_generator, sample_features):
        """Test signal generation with exception."""
        mock_regression_model = Mock()
        mock_regression_model.predict.side_effect = Exception("Model error")
        
        signal_generator.set_models(mock_regression_model, Mock())
        
        signal = signal_generator.generate_signal(
            features=sample_features,
            current_price=50000.0,
            timestamp_ms=1234567890000,
            symbol="BTCUSDT"
        )
        
        assert signal is None
    
    def test_predict_price_change(self, signal_generator, mock_regression_model, sample_features):
        """Test price change prediction."""
        signal_generator.set_models(mock_regression_model, Mock())
        
        price_change = signal_generator._predict_price_change(sample_features)
        
        assert price_change == 0.01
        mock_regression_model.predict.assert_called_once()
    
    def test_predict_price_change_no_predict_method(self, signal_generator, sample_features):
        """Test price change prediction when model has no predict method."""
        signal_generator.regression_model = Mock(spec=[])  # No predict method
        
        with pytest.raises(ValueError, match="predict method"):
            signal_generator._predict_price_change(sample_features)
    
    def test_predict_price_change_scalar_prediction(self, signal_generator, sample_features):
        """Test price change prediction with scalar result."""
        mock_model = Mock()
        mock_model.predict.return_value = 0.003  # Scalar
        
        signal_generator.regression_model = mock_model
        
        price_change = signal_generator._predict_price_change(sample_features)
        
        assert price_change == 0.003
    
    def test_classification_filter_pass(self, signal_generator):
        """Test classification filter passing."""
        class_pred = np.array([2])
        class_proba = np.array([[0.1, 0.2, 0.7]])
        
        result = signal_generator._classification_filter(class_pred, class_proba)
        
        assert result is True
    
    def test_classification_filter_low_confidence(self, signal_generator):
        """Test classification filter with low confidence."""
        class_pred = np.array([2])
        class_proba = np.array([[0.05, 0.05, 0.05]])  # Very low confidence
        
        result = signal_generator._classification_filter(class_pred, class_proba)
        
        assert result is False
    
    def test_classification_filter_hold_signal(self, signal_generator):
        """Test classification filter with hold signal."""
        class_pred = np.array([1])
        class_proba = np.array([[0.1, 0.8, 0.1]])
        
        result = signal_generator._classification_filter(class_pred, class_proba)
        
        assert result is False
    
    def test_classification_filter_strong_hold_signal(self, signal_generator):
        """Test classification filter with strong hold signal."""
        class_pred = np.array([1])
        class_proba = np.array([[0.05, 0.95, 0.0]])
        
        result = signal_generator._classification_filter(class_pred, class_proba)
        
        assert result is True  # Very strong hold signal passes
    
    def test_price_change_to_action_buy(self, signal_generator):
        """Test converting price change to BUY action."""
        action = signal_generator._price_change_to_action(0.003)
        
        assert action == Action.BUY
    
    def test_price_change_to_action_sell(self, signal_generator):
        """Test converting price change to SELL action."""
        action = signal_generator._price_change_to_action(-0.003)
        
        assert action == Action.SELL
    
    def test_price_change_to_action_weak_change(self, signal_generator):
        """Test converting weak price change to HOLD action."""
        action = signal_generator._price_change_to_action(0.001)
        
        assert action == Action.HOLD
    
    def test_price_change_to_action_no_change(self, signal_generator):
        """Test converting no price change to HOLD action."""
        action = signal_generator._price_change_to_action(0.0001)
        
        assert action == Action.HOLD
    
    def test_get_threshold_fixed(self, signal_generator):
        """Test getting fixed threshold."""
        threshold = signal_generator._get_threshold()
        
        assert threshold == 0.7
    
    def test_get_threshold_adaptive(self, signal_generator):
        """Test getting adaptive threshold."""
        signal_generator.threshold_strategy = "adaptive"
        
        threshold = signal_generator._get_threshold()
        
        # With insufficient history, returns fixed threshold
        assert threshold == 0.7
    
    def test_get_threshold_volatility_adjusted(self, signal_generator):
        """Test getting volatility-adjusted threshold."""
        signal_generator.threshold_strategy = "volatility_adjusted"
        
        threshold = signal_generator._get_threshold()
        
        # With insufficient history, returns fixed threshold
        assert threshold == 0.7
    
    def test_get_threshold_unknown_strategy(self, signal_generator):
        """Test getting threshold with unknown strategy."""
        signal_generator.threshold_strategy = "unknown"
        
        threshold = signal_generator._get_threshold()
        
        assert threshold == 0.7  # Falls back to fixed
    
    def test_adaptive_threshold_insufficient_history(self, signal_generator):
        """Test adaptive threshold with insufficient history."""
        threshold = signal_generator._adaptive_threshold()
        
        assert threshold == 0.7
    
    def test_adaptive_threshold_high_success_rate(self, signal_generator):
        """Test adaptive threshold with high success rate."""
        signal_generator.prediction_history = [0.001] * 50 + [-0.001] * 50
        signal_generator.return_history = [0.002] * 50 + [-0.002] * 50  # High correlation
        
        threshold = signal_generator._adaptive_threshold()
        
        assert threshold < 0.7  # Lower threshold for good performance
    
    def test_adaptive_threshold_low_success_rate(self, signal_generator):
        """Test adaptive threshold with low success rate."""
        signal_generator.prediction_history = [0.001] * 50 + [-0.001] * 50
        signal_generator.return_history = [-0.002] * 50 + [0.002] * 50  # Low correlation
        
        threshold = signal_generator._adaptive_threshold()
        
        assert threshold > 0.7  # Higher threshold for poor performance
    
    def test_adaptive_threshold_mixed_success_rate(self, signal_generator):
        """Test adaptive threshold with mixed success rate."""
        signal_generator.prediction_history = [0.001] * 100
        signal_generator.return_history = [0.002] * 60 + [-0.002] * 40  # ~60% success
        
        threshold = signal_generator._adaptive_threshold()
        
        assert threshold == 0.7  # No adjustment for moderate performance
    
    def test_volatility_adjusted_threshold_insufficient_history(self, signal_generator):
        """Test volatility-adjusted threshold with insufficient history."""
        threshold = signal_generator._volatility_adjusted_threshold()
        
        assert threshold == 0.7
    
    def test_volatility_adjusted_threshold_high_volatility(self, signal_generator):
        """Test volatility-adjusted threshold with high volatility."""
        signal_generator.return_history = [0.02, -0.02, 0.015, -0.015] * 5  # High volatility
        
        threshold = signal_generator._volatility_adjusted_threshold()
        
        assert threshold < 0.7  # Lower threshold for high volatility
    
    def test_volatility_adjusted_threshold_low_volatility(self, signal_generator):
        """Test volatility-adjusted threshold with low volatility."""
        signal_generator.return_history = [0.001, -0.001, 0.0015, -0.0015] * 5  # Low volatility
        
        threshold = signal_generator._volatility_adjusted_threshold()
        
        assert threshold > 0.7  # Higher threshold for low volatility
    
    def test_calculate_confidence(self, signal_generator):
        """Test confidence calculation."""
        price_change = 0.003
        class_proba = np.array([[0.1, 0.2, 0.7]])
        
        confidence = signal_generator._calculate_confidence(price_change, class_proba)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high
    
    def test_calculate_confidence_low_price_change(self, signal_generator):
        """Test confidence calculation with low price change."""
        price_change = 0.0001
        class_proba = np.array([[0.1, 0.2, 0.7]])
        
        confidence = signal_generator._calculate_confidence(price_change, class_proba)
        
        assert confidence > 0.0
        assert confidence < 0.7  # Lower due to small price change
    
    def test_update_history(self, signal_generator):
        """Test updating prediction history."""
        signal_generator._update_history(0.003, 0.8)
        
        assert len(signal_generator.prediction_history) == 1
        assert signal_generator.prediction_history[0] == 0.003
    
    def test_update_history_truncation(self, signal_generator):
        """Test history truncation when exceeding limit."""
        signal_generator.prediction_history = [0.001] * 1000
        
        signal_generator._update_history(0.003, 0.8)
        
        assert len(signal_generator.prediction_history) == 1000
        assert signal_generator.prediction_history[-1] == 0.003
    
    def test_update_return_history(self, signal_generator):
        """Test updating return history."""
        signal_generator.update_return_history(0.02)
        
        assert len(signal_generator.return_history) == 1
        assert signal_generator.return_history[0] == 0.02
    
    def test_update_return_history_truncation(self, signal_generator):
        """Test return history truncation when exceeding limit."""
        signal_generator.return_history = [0.01] * 1000
        
        signal_generator.update_return_history(0.02)
        
        assert len(signal_generator.return_history) == 1000
        assert signal_generator.return_history[-1] == 0.02
    
    def test_get_performance_stats_insufficient_data(self, signal_generator):
        """Test performance stats with insufficient data."""
        stats = signal_generator.get_performance_stats()
        
        assert "message" in stats
        assert stats["message"] == "Insufficient data"
    
    def test_get_performance_stats_basic(self, signal_generator):
        """Test basic performance stats."""
        signal_generator.prediction_history = [0.001, -0.002, 0.003, -0.001, 0.002] * 2
        
        stats = signal_generator.get_performance_stats()
        
        assert "total_predictions" in stats
        assert "avg_prediction" in stats
        assert "prediction_std" in stats
        assert stats["total_predictions"] == 10
    
    def test_get_performance_stats_with_returns(self, signal_generator):
        """Test performance stats with return history."""
        signal_generator.prediction_history = [0.001, -0.002, 0.003, -0.001, 0.002] * 2
        signal_generator.return_history = [0.002, -0.003, 0.004, -0.002, 0.003] * 2
        
        stats = signal_generator.get_performance_stats()
        
        assert "accuracy" in stats
        assert "total_returns" in stats
        assert "avg_return" in stats
        assert 0.0 <= stats["accuracy"] <= 1.0
    
    def test_get_performance_stats_accuracy_calculation(self, signal_generator):
        """Test accuracy calculation in performance stats."""
        signal_generator.prediction_history = [0.001, -0.002, 0.003, -0.001, 0.002] * 2  # 10 predictions
        signal_generator.return_history = [0.002, -0.003, 0.004, -0.002, 0.003] * 2  # 5/5 correct
        
        stats = signal_generator.get_performance_stats()
        
        assert "accuracy" in stats
        assert stats["accuracy"] == 1.0
    
    def test_generate_signal_sell_action(self, signal_generator, sample_features):
        """Test signal generation resulting in SELL action."""
        mock_regression_model = Mock()
        mock_regression_model.predict.return_value = np.array([-0.01])  # Higher magnitude
        
        mock_classification_model = Mock()
        mock_classification_model.predict_proba.return_value = np.array([[0.95, 0.03, 0.02]])  # Very high confidence
        mock_classification_model.predict.return_value = np.array([0])
        
        signal_generator.set_models(mock_regression_model, mock_classification_model)
        
        signal = signal_generator.generate_signal(
            features=sample_features,
            current_price=50000.0,
            timestamp_ms=1234567890000,
            symbol="BTCUSDT"
        )
        
        assert signal is not None
        assert signal.action == Action.SELL
