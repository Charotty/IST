"""
Integration tests for Meta-Learning Layer.
"""
import pytest
import numpy as np
from its_project.metalearning.metrics import MLMetrics, TradingMetrics
from its_project.metalearning.cv import TimeSeriesSplitter, WalkForwardValidator
from its_project.metalearning.hyperopt import HyperparameterOptimizer
from its_project.metalearning.selector import ModelSelector
from its_project.metalearning.stacking import StackingEnsemble
from its_project.models.ensemble import EnsembleModel


@pytest.mark.integration
@pytest.mark.metalearning_layer
class TestMetaLearningIntegration:
    """Test meta-learning integration scenarios."""
    
    def test_model_selection_hyperparameter_optimization(self, sample_features, sample_labels):
        """Test model selection with hyperparameter optimization."""
        param_space = {
            'n_estimators': {'type': 'int', 'low': 10, 'high': 50}
        }
        
        optimizer = HyperparameterOptimizer(
            model_class=EnsembleModel,
            param_space=param_space,
            metric='accuracy',
            n_trials=3
        )
        
        result = optimizer.optimize(sample_features, sample_labels, sample_features, sample_labels)
        
        assert result is not None
        assert 'best_params' in result
    
    def test_stacking_ensemble_construction(self, sample_features, sample_labels):
        """Test stacking ensemble construction."""
        base_models = [
            EnsembleModel({'method': 'voting', 'estimators': ['lr']}),
            EnsembleModel({'method': 'voting', 'estimators': ['rf']})
        ]
        meta_model = EnsembleModel({'method': 'voting', 'estimators': ['gb']})
        
        stacking = StackingEnsemble(base_models, meta_model)
        stacking.fit(sample_features, sample_labels)
        predictions = stacking.predict(sample_features)
        
        assert predictions is not None
        assert len(predictions) == len(sample_features)
    
    def test_walk_forward_with_model_selection(self, sample_features, sample_labels):
        """Test walk-forward validation with model selection."""
        validator = WalkForwardValidator(train_size=50, test_size=20, step_size=10)
        
        def model_factory():
            return EnsembleModel({'method': 'voting', 'estimators': ['lr']})
        
        results = validator.validate(sample_features, sample_labels, model_factory)
        
        assert results is not None
        assert len(results) > 0
    
    def test_cross_validation_integration(self, sample_features, sample_labels):
        """Test cross-validation integration."""
        splitter = TimeSeriesSplitter(n_splits=3, gap=10)
        
        splits = list(splitter.split(sample_features))
        
        assert len(splits) == 3
        
        # Train model on each split
        for train_idx, test_idx in splits:
            X_train, X_test = sample_features[train_idx], sample_features[test_idx]
            y_train, y_test = sample_labels[train_idx], sample_labels[test_idx]
            
            model = EnsembleModel({'method': 'voting', 'estimators': ['lr']})
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            assert predictions is not None
            assert len(predictions) == len(X_test)
    
    def test_metrics_calculation_on_predictions(self, sample_features, sample_labels):
        """Test metrics calculation on model predictions."""
        model = EnsembleModel({'method': 'voting', 'estimators': ['lr']})
        model.fit(sample_features, sample_labels)
        predictions = model.predict(sample_features)
        proba = model.predict_proba(sample_features)
        
        # Calculate ML metrics
        ml_metrics = MLMetrics()
        accuracy = ml_metrics.accuracy(sample_labels, predictions)
        f1 = ml_metrics.f1_score(sample_labels, predictions, average='macro')
        
        assert accuracy is not None
        assert f1 is not None
    
    def test_automated_model_improvement(self, sample_features, sample_labels):
        """Test automated model improvement pipeline."""
        # Initial model
        model1 = EnsembleModel({'method': 'voting', 'estimators': ['lr']})
        model1.fit(sample_features, sample_labels)
        
        # Improved model with hyperparameter optimization
        param_space = {
            'n_estimators': {'type': 'int', 'low': 20, 'high': 100}
        }
        
        optimizer = HyperparameterOptimizer(
            model_class=EnsembleModel,
            param_space=param_space,
            metric='accuracy',
            n_trials=3
        )
        
        result = optimizer.optimize(sample_features, sample_labels, sample_features, sample_labels)
        
        assert result is not None
        assert result['best_score'] is not None
    
    def test_performance_tracking(self, sample_features, sample_labels):
        """Test performance tracking across iterations."""
        models = [
            EnsembleModel({'method': 'voting', 'estimators': ['lr']}),
            EnsembleModel({'method': 'voting', 'estimators': ['rf']}),
            EnsembleModel({'method': 'voting', 'estimators': ['gb']})
        ]
        
        selector = ModelSelector(models=models, metrics=['accuracy'], weights=[1.0])
        results = selector.evaluate_all(sample_features, sample_labels, sample_features, sample_labels)
        
        assert results is not None
        assert len(results) == 3
