"""
Tests for unified orchestration layer.

Tests the TrainingOrchestrator and InferenceOrchestrator with explicit pipeline contract:
features → regime → all model predictions → meta weighting → decision → risk
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any
from dataclasses import replace

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration import TrainingOrchestrator, InferenceOrchestrator, OrchestratorConfig
from meta_learning.dynamic_meta import DynamicMetaWeighting


def _meta_matching_config(cfg: OrchestratorConfig) -> DynamicMetaWeighting:
    w = {k: 1.0 / len(cfg.model_keys) for k in cfg.model_keys}
    return DynamicMetaWeighting(trend_weights=w.copy(), range_weights=w.copy())


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, model_key: str, base_pred: float = 0.5):
        self.model_key = model_key
        self.base_pred = base_pred
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return mock predictions."""
        n_samples = len(features)
        # Add some variation based on model key
        variation = hash(self.model_key) % 100 / 1000.0
        return np.full(n_samples, self.base_pred + variation)
    
    def fit(self, features: pd.DataFrame, targets: pd.Series):
        """Mock fit method."""
        pass


class MockRegimeDetector:
    """Mock regime detector for testing."""
    
    def get_regime_info(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Return mock regime info."""
        n_samples = len(features)
        # Alternate between trend and range
        regime_pred = np.array([1 if i % 20 < 10 else 0 for i in range(n_samples)])
        
        return {
            'market_regime': 'trend' if regime_pred[-1] == 1 else 'range',
            'regime_pred': regime_pred,
            'trade_allowed': True
        }


class MockDecisionEngine:
    """Mock decision engine for testing."""
    
    def make_decision(self, direction_signals: np.ndarray, meta_probabilities: np.ndarray, regime_info: Dict[str, Any]) -> np.ndarray:
        """Return direction signals unchanged."""
        return direction_signals


class MockRiskManager:
    """Mock risk manager for testing."""
    
    def calculate_position_sizes(self, signals: np.ndarray, meta_probabilities: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        """Return unsigned position scale (confidence magnitude only; sign comes from signal)."""
        confidence = np.abs(meta_probabilities - 0.5) * 2
        return np.abs(np.asarray(signals, dtype=float)) * confidence
    
    def calculate_position_size(self, signal: int, meta_probability: float, volatility: float, regime: str) -> Dict[str, float]:
        """Return position size for single prediction."""
        confidence = abs(meta_probability - 0.5) * 2
        return {'position_size': signal * confidence}


@pytest.fixture
def sample_config():
    """Create sample orchestrator config."""
    return OrchestratorConfig(
        model_keys=['lgb', 'gru', 'xgb', 'cnn'],
        regime_keys=['trend', 'range', 'breakout'],
        direction_threshold=0.52,
        meta_threshold=0.5,
        signal_threshold=0.6
    )


@pytest.fixture
def sample_models():
    """Create sample models."""
    return {
        'lgb': MockModel('lgb', 0.55),
        'gru': MockModel('gru', 0.60),
        'xgb': MockModel('xgb', 0.50),
        'cnn': MockModel('cnn', 0.45)
    }


@pytest.fixture
def sample_features():
    """Create sample features."""
    np.random.seed(42)
    n_samples = 500
    return pd.DataFrame({
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n_samples),
        'volatility': np.random.uniform(0.01, 0.05, n_samples),
        'rsi': np.random.uniform(20, 80, n_samples),
    }, index=pd.date_range('2024-01-01', periods=n_samples, freq='h'))


@pytest.fixture
def sample_targets(sample_features):
    """Create sample targets."""
    n_samples = len(sample_features)
    return pd.Series(
        np.random.randint(0, 2, n_samples),
        index=sample_features.index
    )


class TestOrchestratorConfig:
    """Test OrchestratorConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = OrchestratorConfig()
        
        assert config.model_keys == ['lgb', 'gru', 'xgb', 'cnn']
        assert config.regime_keys == ['trend', 'range', 'breakout']
        assert config.direction_threshold == 0.52
        assert config.meta_threshold == 0.5
        assert config.signal_threshold == 0.6
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = OrchestratorConfig(
            model_keys=['lgb', 'xgb'],
            regime_keys=['trend', 'range'],
            direction_threshold=0.55
        )
        
        assert config.model_keys == ['lgb', 'xgb']
        assert config.regime_keys == ['trend', 'range']
        assert config.direction_threshold == 0.55
        assert set(config.trend_weights.keys()) == {'lgb', 'xgb'}
        assert abs(sum(config.trend_weights.values()) - 1.0) < 1e-6
    
    def test_weight_validation(self):
        """Test weight validation."""
        # Should raise error if weight keys don't match model keys
        with pytest.raises(ValueError, match="trend_weights keys"):
            OrchestratorConfig(
                model_keys=['lgb', 'gru'],
                trend_weights={'lgb': 0.5, 'xgb': 0.5}  # xgb not in model_keys
            )
    
    def test_get_regime_weights(self, sample_config):
        """Test getting regime weights."""
        trend_weights = sample_config.get_regime_weights('trend')
        range_weights = sample_config.get_regime_weights('range')
        
        assert set(trend_weights.keys()) == set(sample_config.model_keys)
        assert set(range_weights.keys()) == set(sample_config.model_keys)
        
        # Weights should sum to 1
        assert abs(sum(trend_weights.values()) - 1.0) < 1e-6
        assert abs(sum(range_weights.values()) - 1.0) < 1e-6
    
    def test_set_regime_weights(self, sample_config):
        """Test setting regime weights."""
        new_weights = {'lgb': 0.25, 'gru': 0.25, 'xgb': 0.25, 'cnn': 0.25}
        sample_config.set_regime_weights('trend', new_weights)
        
        trend_weights = sample_config.get_regime_weights('trend')
        assert trend_weights == new_weights
    
    def test_set_regime_weights_validation(self, sample_config):
        """Test that setting invalid weights raises error."""
        invalid_weights = {'lgb': 0.5, 'gru': 0.5}  # Missing xgb, cnn
        
        with pytest.raises(ValueError, match="Weight keys"):
            sample_config.set_regime_weights('trend', invalid_weights)


class TestTrainingOrchestrator:
    """Test TrainingOrchestrator."""
    
    def test_initialization(self, sample_config, sample_models):
        """Test orchestrator initialization."""
        orchestrator = TrainingOrchestrator(sample_config)
        
        regime_detector = MockRegimeDetector()
        meta_weighting = _meta_matching_config(sample_config)
        
        orchestrator.initialize(
            models=sample_models,
            regime_detector=regime_detector,
            meta_weighting=meta_weighting
        )
        
        assert orchestrator.is_initialized
        assert set(orchestrator.models.keys()) == set(sample_config.model_keys)
    
    def test_initialization_validation(self, sample_config):
        """Test that initialization validates model keys."""
        orchestrator = TrainingOrchestrator(sample_config)
        
        # Models don't match config
        invalid_models = {'lgb': MockModel('lgb'), 'xgb': MockModel('xgb')}
        
        with pytest.raises(ValueError, match="Model keys"):
            orchestrator.initialize(
                models=invalid_models,
                regime_detector=MockRegimeDetector(),
                meta_weighting=_meta_matching_config(sample_config)
            )
    
    def test_collect_predictions(self, sample_config, sample_models, sample_features):
        """Test collecting predictions from all models."""
        orchestrator = TrainingOrchestrator(sample_config)
        orchestrator.initialize(
            models=sample_models,
            regime_detector=MockRegimeDetector(),
            meta_weighting=_meta_matching_config(sample_config)
        )
        
        predictions = orchestrator.collect_predictions(sample_features)
        
        # Should have predictions for all models
        assert set(predictions.keys()) == set(sample_config.model_keys)
        
        # All predictions should be numpy arrays
        for model_key, pred in predictions.items():
            assert isinstance(pred, np.ndarray)
            assert len(pred) == len(sample_features)
    
    def test_run_pipeline(self, sample_config, sample_models, sample_features):
        """Test running full pipeline."""
        orchestrator = TrainingOrchestrator(sample_config)
        orchestrator.initialize(
            models=sample_models,
            regime_detector=MockRegimeDetector(),
            meta_weighting=_meta_matching_config(sample_config),
            decision_engine=MockDecisionEngine(),
            risk_manager=MockRiskManager()
        )
        
        result = orchestrator.run_pipeline(sample_features)
        
        # Check result structure
        assert hasattr(result, 'model_predictions')
        assert hasattr(result, 'regime_predictions')
        assert hasattr(result, 'meta_probabilities')
        assert hasattr(result, 'direction_signals')
        assert hasattr(result, 'final_signals')
        
        # Check shapes
        assert len(result.model_predictions) == len(sample_config.model_keys)
        assert len(result.regime_predictions) == len(sample_features)
        assert len(result.meta_probabilities) == len(sample_features)
        assert len(result.direction_signals) == len(sample_features)
        assert len(result.final_signals) == len(sample_features)
        assert len(result.position_sizes) == len(sample_features)
    
    def test_train_test_split(self, sample_config, sample_models, sample_features, sample_targets):
        """Test train-test split."""
        orchestrator = TrainingOrchestrator(sample_config)
        orchestrator.initialize(
            models=sample_models,
            regime_detector=MockRegimeDetector(),
            meta_weighting=_meta_matching_config(sample_config)
        )
        
        train_result, test_result = orchestrator.train_test_split(sample_features, sample_targets)
        
        # Both results should have the expected structure
        assert hasattr(train_result, 'meta_probabilities')
        assert hasattr(test_result, 'meta_probabilities')
        
        # Test set should be smaller
        assert len(test_result.meta_probabilities) < len(train_result.meta_probabilities)
    
    def test_get_model_keys(self, sample_config):
        """Test getting model keys."""
        orchestrator = TrainingOrchestrator(sample_config)
        assert orchestrator.get_model_keys() == sample_config.model_keys
    
    def test_get_regime_keys(self, sample_config):
        """Test getting regime keys."""
        orchestrator = TrainingOrchestrator(sample_config)
        assert orchestrator.get_regime_keys() == sample_config.regime_keys


class TestInferenceOrchestrator:
    """Test InferenceOrchestrator."""
    
    def test_initialization(self, sample_config, sample_models):
        """Test orchestrator initialization."""
        orchestrator = InferenceOrchestrator(sample_config)
        
        orchestrator.initialize(
            models=sample_models,
            regime_detector=MockRegimeDetector(),
            meta_weighting=_meta_matching_config(sample_config)
        )
        
        assert orchestrator.is_initialized
        assert set(orchestrator.models.keys()) == set(sample_config.model_keys)
    
    def test_initialization_validation(self, sample_config):
        """Test that initialization validates model keys."""
        orchestrator = InferenceOrchestrator(sample_config)
        
        invalid_models = {'lgb': MockModel('lgb'), 'xgb': MockModel('xgb')}
        
        with pytest.raises(ValueError, match="Model keys"):
            orchestrator.initialize(
                models=invalid_models,
                regime_detector=MockRegimeDetector(),
                meta_weighting=_meta_matching_config(sample_config)
            )
    
    def test_collect_predictions(self, sample_config, sample_models, sample_features):
        """Test collecting predictions from all models."""
        orchestrator = InferenceOrchestrator(sample_config)
        orchestrator.initialize(
            models=sample_models,
            regime_detector=MockRegimeDetector(),
            meta_weighting=_meta_matching_config(sample_config)
        )
        
        # Use last 24 samples as window
        window = sample_features.iloc[-24:]
        predictions = orchestrator.collect_predictions(window)
        
        # Should have predictions for all models
        assert set(predictions.keys()) == set(sample_config.model_keys)
        
        # All predictions should be floats
        for model_key, pred in predictions.items():
            assert isinstance(pred, float)
    
    def test_predict(self, sample_config, sample_models, sample_features):
        """Test single prediction."""
        orchestrator = InferenceOrchestrator(sample_config)
        orchestrator.initialize(
            models=sample_models,
            regime_detector=MockRegimeDetector(),
            meta_weighting=_meta_matching_config(sample_config),
            decision_engine=MockDecisionEngine(),
            risk_manager=MockRiskManager()
        )
        
        # Use last 24 samples as window
        window = sample_features.iloc[-24:]
        result = orchestrator.predict(window)
        
        # Check result structure
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'position_size')
        assert hasattr(result, 'regime')
        assert hasattr(result, 'model_predictions')
        assert hasattr(result, 'meta_probability')
        assert hasattr(result, 'meta_weights')
        assert hasattr(result, 'active_models')
        
        # Check types
        assert isinstance(result.signal, int)
        assert isinstance(result.confidence, float)
        assert isinstance(result.position_size, float)
        assert isinstance(result.regime, str)
        assert isinstance(result.model_predictions, dict)
        assert isinstance(result.meta_probability, float)
        assert isinstance(result.meta_weights, dict)
        assert isinstance(result.active_models, list)
        
        # Check model predictions contain all models
        assert set(result.model_predictions.keys()) == set(sample_config.model_keys)
        assert set(result.meta_weights.keys()) == set(sample_config.model_keys)
    
    def test_batch_predict(self, sample_config, sample_models, sample_features):
        """Test batch prediction."""
        orchestrator = InferenceOrchestrator(sample_config)
        orchestrator.initialize(
            models=sample_models,
            regime_detector=MockRegimeDetector(),
            meta_weighting=_meta_matching_config(sample_config)
        )
        
        results_df = orchestrator.batch_predict(sample_features, window_size=24)
        
        # Check result structure
        assert isinstance(results_df, pd.DataFrame)
        assert 'signal' in results_df.columns
        assert 'confidence' in results_df.columns
        assert 'position_size' in results_df.columns
        assert 'regime' in results_df.columns
        assert 'meta_probability' in results_df.columns
        
        # Check individual model predictions
        for model_key in sample_config.model_keys:
            assert f'{model_key}_pred' in results_df.columns
            assert f'{model_key}_weight' in results_df.columns
        
        # Should have predictions for each window
        assert len(results_df) == len(sample_features) - 24
    
    def test_get_model_keys(self, sample_config):
        """Test getting model keys."""
        orchestrator = InferenceOrchestrator(sample_config)
        assert orchestrator.get_model_keys() == sample_config.model_keys
    
    def test_get_regime_keys(self, sample_config):
        """Test getting regime keys."""
        orchestrator = InferenceOrchestrator(sample_config)
        assert orchestrator.get_regime_keys() == sample_config.regime_keys


class TestRegimePredIntegration:
    """Ensure real RegimeDetector contract (regime_pred length), not only mocks."""

    def test_pipeline_uses_regime_pred_length(self, sample_config, sample_models, sample_features, monkeypatch):
        pytest.importorskip("lightgbm")
        pytest.importorskip("sklearn")
        from models.regime.regime_detector import RegimeDetector

        cfg = replace(sample_config, apply_decision_pipeline=False)
        det = RegimeDetector()
        det.feature_cols = ["adx", "ema_slope", "volatility", "atr"]
        df = sample_features.copy()
        for c in det.feature_cols:
            if c not in df.columns:
                df[c] = np.random.uniform(0.1, 2.0, len(df))
        det.is_fitted = True
        n = len(df)

        def _proba_sub(d):
            m = len(d)
            return np.column_stack([np.full(m, 0.45), np.full(m, 0.55)])

        monkeypatch.setattr(det, "predict_proba", _proba_sub)
        monkeypatch.setattr(det, "predict", lambda d: np.zeros(len(d), dtype=int))

        orchestrator = TrainingOrchestrator(cfg)
        orchestrator.initialize(
            models=sample_models,
            regime_detector=det,
            meta_weighting=DynamicMetaWeighting(
                trend_weights={k: 0.25 for k in cfg.model_keys},
                range_weights={k: 0.25 for k in cfg.model_keys},
            ),
        )

        res = orchestrator.run_pipeline(df)
        assert len(res.regime_predictions) == n


class TestIntegration:
    """Integration tests for unified orchestration."""

    def test_pipeline_contract(self, sample_config, sample_models, sample_features):
        """Test that pipeline follows explicit contract."""
        orchestrator = InferenceOrchestrator(sample_config)
        orchestrator.initialize(
            models=sample_models,
            regime_detector=MockRegimeDetector(),
            meta_weighting=_meta_matching_config(sample_config)
        )
        
        window = sample_features.iloc[-24:]
        result = orchestrator.predict(window)
        
        # Verify contract: features → regime → all predictions → meta → direction
        assert result.regime in sample_config.regime_keys
        assert len(result.model_predictions) == len(sample_config.model_keys)
        assert 0 <= result.meta_probability <= 1
        assert result.signal in [-1, 0, 1]
    
    def test_config_driven_behavior(self, sample_config, sample_models, sample_features):
        """Test that config drives behavior."""
        # Change signal threshold
        sample_config.signal_threshold = 0.9
        
        orchestrator = InferenceOrchestrator(sample_config)
        orchestrator.initialize(
            models=sample_models,
            regime_detector=MockRegimeDetector(),
            meta_weighting=_meta_matching_config(sample_config)
        )
        
        window = sample_features.iloc[-24:]
        result = orchestrator.predict(window)
        
        # With high threshold, should mostly be neutral
        assert result.signal in [-1, 0, 1]
    
    def test_key_consistency(self, sample_config, sample_models, sample_features):
        """Test that keys are consistent across components."""
        # Training orchestrator
        training_orch = TrainingOrchestrator(sample_config)
        training_orch.initialize(
            models=sample_models,
            regime_detector=MockRegimeDetector(),
            meta_weighting=_meta_matching_config(sample_config)
        )
        
        # Inference orchestrator
        inference_orch = InferenceOrchestrator(sample_config)
        inference_orch.initialize(
            models=sample_models,
            regime_detector=MockRegimeDetector(),
            meta_weighting=_meta_matching_config(sample_config)
        )
        
        # Both should have same keys
        assert training_orch.get_model_keys() == inference_orch.get_model_keys()
        assert training_orch.get_regime_keys() == inference_orch.get_regime_keys()
        assert training_orch.get_model_keys() == sample_config.model_keys


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
