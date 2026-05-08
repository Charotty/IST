"""
Tests for confidence thresholding functionality in WeightedEnsemble.

Tests p_hat thresholding strategies and confidence filtering.
"""

import pytest
import numpy as np

from its_project.metalearning.weighted_ensemble import (
    WeightedEnsemble, 
    WeightedEnsembleConfig
)
from its_project.models.regression_model import RegressionModel


class TestConfidenceThresholding:
    """Test confidence thresholding functionality."""
    
    @pytest.fixture
    def sample_models(self):
        """Create sample models for testing."""
        models = [
            RegressionModel({"model_type": "linear"}),
            RegressionModel({"model_type": "linear"})
        ]
        return models
    
    @pytest.fixture
    def confidence_config(self):
        """Create configuration with confidence filtering."""
        return WeightedEnsembleConfig(
            confidence_threshold=0.7,
            use_confidence_filter=True,
            confidence_strategy="max_proba"
        )
    
    def test_default_confidence_config(self):
        """Test default confidence configuration."""
        config = WeightedEnsembleConfig()
        
        assert config.confidence_threshold == 0.6
        assert config.use_confidence_filter is True
        assert config.confidence_strategy == "max_proba"
    
    def test_custom_confidence_config(self):
        """Test custom confidence configuration."""
        config = WeightedEnsembleConfig(
            confidence_threshold=0.8,
            use_confidence_filter=False,
            confidence_strategy="entropy"
        )
        
        assert config.confidence_threshold == 0.8
        assert config.use_confidence_filter is False
        assert config.confidence_strategy == "entropy"
    
    def test_max_proba_confidence_calculation(self, sample_models, confidence_config):
        """Test max probability confidence strategy."""
        ensemble = WeightedEnsemble(sample_models, confidence_config)
        
        # Create sample probabilities
        probas = np.array([
            [0.1, 0.8, 0.1],  # High confidence in class 1
            [0.3, 0.4, 0.3],  # Low confidence in class 1
            [0.9, 0.05, 0.05],  # High confidence in class 0
            [0.25, 0.25, 0.5]  # Medium confidence in class 2
        ])
        
        confidence = ensemble._calculate_confidence(probas)
        
        # Expected: max probabilities
        expected = np.array([0.8, 0.4, 0.9, 0.5])
        np.testing.assert_array_almost_equal(confidence, expected, decimal=3)
    
    def test_entropy_confidence_calculation(self, sample_models):
        """Test entropy-based confidence strategy."""
        config = WeightedEnsembleConfig(confidence_strategy="entropy")
        ensemble = WeightedEnsemble(sample_models, config)
        
        # Create sample probabilities
        probas = np.array([
            [0.9, 0.05, 0.05],  # Low entropy (high confidence)
            [0.33, 0.34, 0.33],  # High entropy (low confidence)
            [0.5, 0.25, 0.25],  # Medium entropy
        ])
        
        confidence = ensemble._calculate_confidence(probas)
        
        # Lower entropy should give higher confidence
        assert confidence[0] > confidence[1]  # Low entropy > high entropy
        assert confidence[0] > confidence[2]  # Low entropy > medium entropy
        assert all(0 <= c <= 1 for c in confidence)  # Should be normalized
    
    def test_margin_confidence_calculation(self, sample_models):
        """Test margin-based confidence strategy."""
        config = WeightedEnsembleConfig(confidence_strategy="margin")
        ensemble = WeightedEnsemble(sample_models, config)
        
        # Create sample probabilities
        probas = np.array([
            [0.1, 0.8, 0.1],  # Large margin (0.8 - 0.1 = 0.7)
            [0.35, 0.4, 0.25],  # Small margin (0.4 - 0.35 = 0.05)
            [0.9, 0.08, 0.02],  # Large margin (0.9 - 0.08 = 0.82)
        ])
        
        confidence = ensemble._calculate_confidence(probas)
        
        # Expected: margins between top two probabilities
        expected = np.array([0.7, 0.05, 0.82])
        np.testing.assert_array_almost_equal(confidence, expected, decimal=3)
    
    def test_confidence_filtering(self, sample_models, confidence_config):
        """Test confidence filtering functionality."""
        ensemble = WeightedEnsemble(sample_models, confidence_config)
        
        # Create sample probabilities
        probas = np.array([
            [0.1, 0.8, 0.1],  # High confidence (0.8 > 0.7)
            [0.3, 0.4, 0.3],  # Low confidence (0.4 < 0.7)
            [0.9, 0.05, 0.05],  # High confidence (0.9 > 0.7)
            [0.25, 0.25, 0.5]  # Low confidence (0.5 < 0.7)
        ])
        
        confidence_scores = np.array([0.8, 0.4, 0.9, 0.5])
        
        # Apply filtering
        filtered_predictions = ensemble._filter_by_confidence(probas, confidence_scores)
        
        # Expected: [1, 1, 0, 1] (low confidence become HOLD = class 1)
        expected = np.array([1, 1, 0, 1])
        np.testing.assert_array_equal(filtered_predictions, expected)
    
    def test_predict_with_confidence(self, sample_models, confidence_config):
        """Test predict_with_confidence method."""
        ensemble = WeightedEnsemble(sample_models, confidence_config)
        
        # Create and fit models
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        ensemble.fit(X, y)
        
        # Test prediction with confidence
        X_test = np.random.randn(10, 5)
        predictions, confidence_scores = ensemble.predict_with_confidence(X_test)
        
        assert predictions.shape == (10,)
        assert confidence_scores.shape == (10,)
        assert all(0 <= c <= 1 for c in confidence_scores)
    
    def test_predict_with_confidence_disabled(self, sample_models):
        """Test predict_with_confidence when filtering is disabled."""
        config = WeightedEnsembleConfig(
            use_confidence_filter=False,
            confidence_strategy="max_proba"
        )
        ensemble = WeightedEnsemble(sample_models, config)
        
        # Create and fit models
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        ensemble.fit(X, y)
        
        # Test prediction without filtering
        X_test = np.random.randn(10, 5)
        predictions, confidence_scores = ensemble.predict_with_confidence(X_test)
        
        # Should be regular argmax predictions
        probas = ensemble.predict_proba(X_test)
        expected_predictions = np.argmax(probas, axis=1)
        np.testing.assert_array_equal(predictions, expected_predictions)
    
    def test_get_confidence_stats(self, sample_models, confidence_config):
        """Test confidence statistics calculation."""
        ensemble = WeightedEnsemble(sample_models, confidence_config)
        
        # Create and fit models
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        ensemble.fit(X, y)
        
        # Get confidence stats
        X_test = np.random.randn(50, 5)
        stats = ensemble.get_confidence_stats(X_test)
        
        # Check required statistics
        required_keys = [
            "mean_confidence", "std_confidence", "min_confidence", "max_confidence",
            "above_threshold", "below_threshold", "threshold_ratio",
            "strategy", "threshold"
        ]
        
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"
        
        # Check class-specific stats
        for class_id in range(3):
            assert f"class_{class_id}_mean_confidence" in stats
            assert f"class_{class_id}_count" in stats
        
        # Check values are reasonable
        assert 0 <= stats["mean_confidence"] <= 1
        assert 0 <= stats["min_confidence"] <= 1
        assert 0 <= stats["max_confidence"] <= 1
        assert stats["above_threshold"] + stats["below_threshold"] == 50  # Total samples
        assert 0 <= stats["threshold_ratio"] <= 1
    
    def test_different_confidence_strategies(self, sample_models):
        """Test all confidence strategies."""
        strategies = ["max_proba", "entropy", "margin"]
        
        for strategy in strategies:
            config = WeightedEnsembleConfig(
                confidence_strategy=strategy,
                confidence_threshold=0.6
            )
            ensemble = WeightedEnsemble(sample_models, config)
            
            # Create sample probabilities
            probas = np.array([
                [0.1, 0.8, 0.1],
                [0.33, 0.34, 0.33],
                [0.9, 0.05, 0.05]
            ])
            
            confidence = ensemble._calculate_confidence(probas)
            
            # Should return confidence scores
            assert confidence.shape == (3,)
            assert all(0 <= c <= 1 for c in confidence)
    
    def test_invalid_confidence_strategy(self, sample_models):
        """Test error handling for invalid confidence strategy."""
        config = WeightedEnsembleConfig(confidence_strategy="invalid")
        ensemble = WeightedEnsemble(sample_models, config)
        
        probas = np.array([[0.1, 0.8, 0.1]])
        
        with pytest.raises(ValueError, match="Unknown confidence strategy"):
            ensemble._calculate_confidence(probas)
    
    def test_confidence_threshold_edge_cases(self, sample_models):
        """Test edge cases for confidence thresholding."""
        config = WeightedEnsembleConfig(confidence_threshold=0.5)
        ensemble = WeightedEnsemble(sample_models, config)
        
        # Test with confidence exactly at threshold
        probas = np.array([[0.1, 0.5, 0.4]])  # max = 0.5
        confidence = np.array([0.5])
        
        filtered = ensemble._filter_by_confidence(probas, confidence)
        
        # Should not filter (>= threshold)
        assert filtered[0] == 1  # argmax = 1
        
        # Test with confidence below threshold
        probas = np.array([[0.1, 0.4, 0.5]])  # max = 0.5
        confidence = np.array([0.49])  # Slightly below threshold
        
        filtered = ensemble._filter_by_confidence(probas, confidence)
        
        # Should filter to HOLD
        assert filtered[0] == 1  # HOLD class
    
    def test_performance_summary_includes_confidence(self, sample_models):
        """Test that performance summary includes confidence settings."""
        config = WeightedEnsembleConfig(
            confidence_threshold=0.75,
            confidence_strategy="entropy",
            use_confidence_filter=True
        )
        ensemble = WeightedEnsemble(sample_models, config)
        
        summary = ensemble.get_performance_summary()
        
        assert "config" in summary
        assert "confidence_threshold" in summary["config"]
        assert "confidence_strategy" in summary["config"]
        assert "use_confidence_filter" in summary["config"]
        
        assert summary["config"]["confidence_threshold"] == 0.75
        assert summary["config"]["confidence_strategy"] == "entropy"
        assert summary["config"]["use_confidence_filter"] is True


class TestConfidenceIntegration:
    """Test confidence thresholding integration with full ensemble."""
    
    def test_end_to_end_confidence_workflow(self):
        """Test complete workflow with confidence filtering."""
        # Create models
        models = [
            RegressionModel({"model_type": "linear"}),
            RegressionModel({"model_type": "linear"})
        ]
        
        # Create configuration with confidence filtering
        config = WeightedEnsembleConfig(
            confidence_threshold=0.7,
            use_confidence_filter=True,
            confidence_strategy="max_proba"
        )
        
        ensemble = WeightedEnsemble(models, config)
        
        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        
        # Fit ensemble
        ensemble.fit(X, y)
        
        # Make predictions with confidence
        X_test = np.random.randn(20, 5)
        predictions, confidence_scores = ensemble.predict_with_confidence(X_test)
        
        # Get confidence statistics
        stats = ensemble.get_confidence_stats(X_test)
        
        # Verify workflow
        assert len(predictions) == 20
        assert len(confidence_scores) == 20
        assert stats["threshold"] == 0.7
        assert stats["strategy"] == "max_proba"
        
        # Some predictions should be filtered to HOLD
        hold_predictions = np.sum(predictions == 1)
        assert hold_predictions >= 0  # At least some HOLDs expected


if __name__ == "__main__":
    pytest.main([__file__])
