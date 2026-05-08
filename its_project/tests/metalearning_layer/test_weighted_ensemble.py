"""
Tests for Weighted Ensemble with Dynamic Model Selection.

Tests the Score = α·Sharpe + β·PnL - γ·DD formula implementation
and dynamic model switching logic.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from its_project.metalearning.weighted_ensemble import (
    WeightedEnsemble, 
    WeightedEnsembleConfig, 
    ModelPerformance
)
from its_project.models.regression_model import RegressionModel


class TestWeightedEnsembleConfig:
    """Test WeightedEnsembleConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WeightedEnsembleConfig()
        
        assert config.alpha == 0.4
        assert config.beta == 0.4
        assert config.gamma == 0.2
        assert config.window_size == 100
        assert config.min_trades_for_switch == 20
        assert config.switch_threshold == 0.1
        assert config.score_smoothing is True
        assert config.smoothing_window == 10
        assert config.ensemble_weights is None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = WeightedEnsembleConfig(
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            window_size=50,
            switch_threshold=0.05
        )
        
        assert config.alpha == 0.5
        assert config.beta == 0.3
        assert config.gamma == 0.2
        assert config.window_size == 50
        assert config.switch_threshold == 0.05


class TestModelPerformance:
    """Test ModelPerformance tracking."""
    
    def test_performance_update(self):
        """Test performance metric updates."""
        config = WeightedEnsembleConfig(alpha=0.4, beta=0.4, gamma=0.2)
        perf = ModelPerformance("TestModel")
        
        # Update with sample metrics
        perf.update_performance(
            sharpe=1.5,
            pnl=0.1,
            drawdown=0.05,
            config=config
        )
        
        # Check score calculation: 0.4*1.5 + 0.4*0.1 - 0.2*0.05 = 0.6 + 0.04 - 0.01 = 0.63
        assert abs(perf.current_score - 0.63) < 0.001
        assert perf.trades_count == 1
        assert perf.best_score == 0.63
    
    def test_window_metrics(self):
        """Test sliding window metrics calculation."""
        perf = ModelPerformance("TestModel")
        config = WeightedEnsembleConfig(window_size=5)
        
        # Add multiple performance updates
        test_data = [
            (1.0, 0.05, 0.02),  # sharpe, pnl, drawdown
            (1.2, 0.08, 0.03),
            (0.8, -0.02, 0.01),
            (1.5, 0.12, 0.04),
            (1.1, 0.06, 0.02),
        ]
        
        for sharpe, pnl, dd in test_data:
            perf.update_performance(sharpe, pnl, dd, config)
        
        # Get window metrics
        metrics = perf.get_window_metrics(5)
        
        assert abs(metrics["sharpe"] - np.mean([s[0] for s in test_data])) < 0.001
        assert abs(metrics["pnl"] - np.mean([s[1] for s in test_data])) < 0.001
        assert abs(metrics["drawdown"] - np.mean([s[2] for s in test_data])) < 0.001
        assert metrics["trades"] == 5
    
    def test_score_smoothing(self):
        """Test score smoothing functionality."""
        config = WeightedEnsembleConfig(
            alpha=0.4, beta=0.4, gamma=0.2,
            score_smoothing=True, smoothing_window=3
        )
        perf = ModelPerformance("TestModel")
        
        # Add multiple updates
        scores = [1.0, 0.5, 1.5, 0.8, 1.2]
        for i, score in enumerate(scores):
            perf.update_performance(score, 0.01, 0.02, config)
            if i >= 2:  # After 3 updates, smoothing should start
                expected_smooth = np.mean(scores[max(0, i-2):i+1])
                assert abs(perf.current_score - expected_smooth) < 0.001


class TestWeightedEnsemble:
    """Test WeightedEnsemble functionality."""
    
    @pytest.fixture
    def sample_models(self):
        """Create sample models for testing."""
        models = [
            RegressionModel({"model_type": "linear"}),
            RegressionModel({"model_type": "linear"}),
            RegressionModel({"model_type": "linear"})
        ]
        return models
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return WeightedEnsembleConfig(
            alpha=0.4, beta=0.4, gamma=0.2,
            window_size=10, switch_threshold=0.1
        )
    
    def test_initialization(self, sample_models, sample_config):
        """Test ensemble initialization."""
        ensemble = WeightedEnsemble(sample_models, sample_config)
        
        assert len(ensemble.models) == 3
        assert ensemble.config == sample_config
        assert len(ensemble.model_performances) == 3
        assert ensemble.current_model_idx == 0
        assert len(ensemble.ensemble_weights) == 3
        assert all(abs(w - 1/3) < 0.001 for w in ensemble.ensemble_weights)
    
    def test_fit(self, sample_models, sample_config):
        """Test model fitting."""
        ensemble = WeightedEnsemble(sample_models, sample_config)
        
        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        
        # Fit ensemble
        ensemble.fit(X, y)
        
        # Check all models are fitted
        for model in ensemble.models:
            assert model._is_fitted is True
        assert ensemble._is_fitted is True
    
    def test_predict_weighted_ensemble(self, sample_models, sample_config):
        """Test weighted ensemble prediction."""
        ensemble = WeightedEnsemble(sample_models, sample_config)
        
        # Create and fit models
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        ensemble.fit(X, y)
        
        # Test prediction
        X_test = np.random.randn(10, 5)
        predictions = ensemble.predict(X_test)
        
        assert predictions.shape == (10,)
        assert predictions.dtype in [np.int32, np.int64]
        
        # Test probabilities
        probas = ensemble.predict_proba(X_test)
        assert probas.shape == (10, 3)  # 3 classes
        assert np.allclose(np.sum(probas, axis=1), 1.0, atol=0.001)
    
    def test_performance_update(self, sample_models, sample_config):
        """Test performance update mechanism."""
        ensemble = WeightedEnsemble(sample_models, sample_config)
        
        # Simulate trading results
        y_true = np.array([0, 1, 2, 1, 0, 2, 1, 0, 2, 1])
        y_pred = np.array([0, 1, 2, 1, 0, 2, 1, 0, 1, 2])  # Some errors
        price_returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.012, 0.025, -0.008])
        
        # Update performance
        ensemble.update_performance(y_true, y_pred, price_returns)
        
        # Check performance was updated
        current_perf = ensemble.model_performances[ensemble.current_model_name]
        assert current_perf.trades_count > 0
        assert current_perf.current_score != 0.0
    
    def test_model_switching_logic(self, sample_models, sample_config):
        """Test dynamic model switching logic."""
        config = WeightedEnsembleConfig(
            alpha=0.4, beta=0.4, gamma=0.2,
            window_size=5, min_trades_for_switch=3, switch_threshold=0.05
        )
        ensemble = WeightedEnsemble(sample_models, config)
        
        # Simulate performance for different models
        model_names = [m.__class__.__name__ for m in sample_models]
        
        # Make current model perform poorly
        for i in range(10):
            y_true = np.array([0, 1, 2])
            y_pred = np.array([1, 0, 1])  # Poor predictions
            price_returns = np.array([-0.01, -0.005, -0.008])
            ensemble.update_performance(y_true, y_pred, price_returns)
        
        old_model = ensemble.current_model_name
        
        # Simulate better performance for second model
        ensemble.current_model_idx = 1
        ensemble.current_model_name = model_names[1]
        
        for i in range(5):
            y_true = np.array([0, 1, 2])
            y_pred = np.array([0, 1, 2])  # Perfect predictions
            price_returns = np.array([0.02, 0.015, 0.025])
            ensemble.update_performance(y_true, y_pred, price_returns)
        
        # Check if switching would occur (need more trades)
        assert ensemble.total_trades >= 15  # 10 + 5 trades
        
        # Force check
        ensemble._check_and_switch_model()
        
        # Should have switched to better model
        assert ensemble.current_model_name == model_names[1]
        assert ensemble.current_model_name != old_model
    
    def test_custom_ensemble_weights(self, sample_models):
        """Test custom ensemble weights."""
        custom_weights = [0.5, 0.3, 0.2]
        config = WeightedEnsembleConfig(ensemble_weights=custom_weights)
        
        ensemble = WeightedEnsemble(sample_models, config)
        
        assert len(ensemble.ensemble_weights) == 3
        assert all(abs(w - cw) < 0.001 for w, cw in zip(ensemble.ensemble_weights, custom_weights))
    
    def test_get_model_rankings(self, sample_models, sample_config):
        """Test model rankings functionality."""
        ensemble = WeightedEnsemble(sample_models, sample_config)
        
        # Simulate different performance for each model
        performances = [(1.5, 0.1, 0.02), (1.0, 0.05, 0.03), (2.0, 0.15, 0.01)]
        
        for i, (sharpe, pnl, dd) in enumerate(performances):
            model_name = sample_models[i].__class__.__name__
            perf = ensemble.model_performances[model_name]
            for _ in range(10):  # Add some history
                perf.update_performance(sharpe, pnl, dd, sample_config)
        
        rankings = ensemble.get_model_rankings()
        
        assert len(rankings) == 3
        assert "model" in rankings.columns
        assert "window_score" in rankings.columns
        assert "is_current" in rankings.columns
        
        # Best model should be ranked first
        best_model = rankings.iloc[0]
        expected_best_idx = np.argmax([p[0] for p in performances])
        expected_best_name = sample_models[expected_best_idx].__class__.__name__
        assert best_model["model"] == expected_best_name
    
    def test_performance_summary(self, sample_models, sample_config):
        """Test performance summary generation."""
        ensemble = WeightedEnsemble(sample_models, sample_config)
        
        summary = ensemble.get_performance_summary()
        
        assert "current_model" in summary
        assert "config" in summary
        assert "models" in summary
        assert summary["config"]["alpha"] == 0.4
        assert summary["config"]["beta"] == 0.4
        assert summary["config"]["gamma"] == 0.2
    
    def test_reset_performance_tracking(self, sample_models, sample_config):
        """Test performance tracking reset."""
        ensemble = WeightedEnsemble(sample_models, sample_config)
        
        # Add some performance data
        for i in range(5):
            y_true = np.array([0, 1, 2])
            y_pred = np.array([0, 1, 2])
            price_returns = np.array([0.01, 0.005, 0.02])
            ensemble.update_performance(y_true, y_pred, price_returns)
        
        # Verify data was added
        assert ensemble.total_trades > 0
        
        # Reset tracking
        ensemble.reset_performance_tracking()
        
        # Verify reset
        assert ensemble.total_trades == 0
        assert ensemble.last_switch_trade == 0
        
        for perf in ensemble.model_performances.values():
            assert perf.trades_count == 0
            assert perf.current_score == 0.0
            assert perf.best_score == 0.0
            assert len(perf.score_history) == 0
    
    def test_metadata(self, sample_models, sample_config):
        """Test metadata generation."""
        ensemble = WeightedEnsemble(sample_models, sample_config)
        
        metadata = ensemble.get_metadata()
        
        assert metadata["ensemble_type"] == "weighted_dynamic"
        assert metadata["num_models"] == 3
        assert "config" in metadata
        assert "ensemble_weights" in metadata
        assert metadata["config"]["alpha"] == 0.4
        assert metadata["config"]["beta"] == 0.4
        assert metadata["config"]["gamma"] == 0.2


class TestScoreFormula:
    """Test the Score = α·Sharpe + β·PnL - γ·DD formula."""
    
    def test_score_calculation_positive_metrics(self):
        """Test score with positive metrics."""
        config = WeightedEnsembleConfig(alpha=0.4, beta=0.4, gamma=0.2)
        perf = ModelPerformance("TestModel")
        
        perf.update_performance(sharpe=2.0, pnl=0.1, drawdown=0.05, config=config)
        
        # Expected: 0.4*2.0 + 0.4*0.1 - 0.2*0.05 = 0.8 + 0.04 - 0.01 = 0.83
        expected = 0.4 * 2.0 + 0.4 * 0.1 - 0.2 * 0.05
        assert abs(perf.current_score - expected) < 0.001
    
    def test_score_calculation_negative_pnl(self):
        """Test score with negative PnL."""
        config = WeightedEnsembleConfig(alpha=0.5, beta=0.3, gamma=0.2)
        perf = ModelPerformance("TestModel")
        
        perf.update_performance(sharpe=1.5, pnl=-0.05, drawdown=0.1, config=config)
        
        # Expected: 0.5*1.5 + 0.3*(-0.05) - 0.2*0.1 = 0.75 - 0.015 - 0.02 = 0.715
        expected = 0.5 * 1.5 + 0.3 * (-0.05) - 0.2 * 0.1
        assert abs(perf.current_score - expected) < 0.001
    
    def test_score_calculation_high_drawdown_penalty(self):
        """Test score with high drawdown (heavy penalty)."""
        config = WeightedEnsembleConfig(alpha=0.3, beta=0.3, gamma=0.4)  # High gamma
        perf = ModelPerformance("TestModel")
        
        perf.update_performance(sharpe=2.0, pnl=0.15, drawdown=0.2, config=config)
        
        # Expected: 0.3*2.0 + 0.3*0.15 - 0.4*0.2 = 0.6 + 0.045 - 0.08 = 0.565
        expected = 0.3 * 2.0 + 0.3 * 0.15 - 0.4 * 0.2
        assert abs(perf.current_score - expected) < 0.001
    
    def test_different_weight_configurations(self):
        """Test score calculation with different weight configurations."""
        test_configs = [
            {"alpha": 1.0, "beta": 0.0, "gamma": 0.0},  # Sharpe only
            {"alpha": 0.0, "beta": 1.0, "gamma": 0.0},  # PnL only
            {"alpha": 0.0, "beta": 0.0, "gamma": 1.0},  # Drawdown penalty only
            {"alpha": 0.33, "beta": 0.33, "gamma": 0.34},  # Balanced
        ]
        
        sharpe, pnl, dd = 1.5, 0.1, 0.05
        
        for config_dict in test_configs:
            config = WeightedEnsembleConfig(**config_dict)
            perf = ModelPerformance("TestModel")
            perf.update_performance(sharpe, pnl, dd, config)
            
            expected = (
                config.alpha * sharpe + 
                config.beta * pnl - 
                config.gamma * dd
            )
            assert abs(perf.current_score - expected) < 0.001


if __name__ == "__main__":
    pytest.main([__file__])
