"""
Unit tests for hyperparameter optimizer.
"""
import pytest
import numpy as np
from its_project.metalearning.hyperopt import HyperparameterOptimizer
from its_project.models.ensemble import EnsembleModel


@pytest.mark.unit
@pytest.mark.metalearning_layer
class TestHyperparameterOptimizer:
    """Test hyperparameter optimizer functionality."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        param_space = {
            'n_estimators': {'type': 'int', 'low': 10, 'high': 100},
            'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.1}
        }
        
        optimizer = HyperparameterOptimizer(
            model_class=EnsembleModel,
            param_space=param_space,
            metric='accuracy',
            n_trials=5
        )
        
        assert optimizer is not None
        assert optimizer.n_trials == 5
    
    def test_parameter_space_definition(self):
        """Test parameter space definition."""
        param_space = {
            'n_estimators': {'type': 'int', 'low': 10, 'high': 100},
            'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.1},
            'criterion': {'type': 'categorical', 'choices': ['gini', 'entropy']}
        }
        
        optimizer = HyperparameterOptimizer(
            model_class=EnsembleModel,
            param_space=param_space,
            metric='accuracy',
            n_trials=5
        )
        
        assert len(optimizer.param_space) == 3
    
    def test_optimization_trials(self, sample_features, sample_labels):
        """Test optimization trials."""
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
    
    def test_best_parameter_selection(self, sample_features, sample_labels):
        """Test best parameter selection."""
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
        
        assert result['best_params'] is not None
        assert 'n_estimators' in result['best_params']
    
    def test_study_persistence(self, sample_features, sample_labels, tmp_path):
        """Test study persistence."""
        param_space = {
            'n_estimators': {'type': 'int', 'low': 10, 'high': 50}
        }
        
        optimizer = HyperparameterOptimizer(
            model_class=EnsembleModel,
            param_space=param_space,
            metric='accuracy',
            n_trials=3,
            study_path=tmp_path / 'study.pkl'
        )
        
        result = optimizer.optimize(sample_features, sample_labels, sample_features, sample_labels)
        
        # Study should be saved
        assert (tmp_path / 'study.pkl').exists()
    
    def test_early_stopping(self, sample_features, sample_labels):
        """Test early stopping."""
        param_space = {
            'n_estimators': {'type': 'int', 'low': 10, 'high': 100}
        }
        
        optimizer = HyperparameterOptimizer(
            model_class=EnsembleModel,
            param_space=param_space,
            metric='accuracy',
            n_trials=10,
            early_stopping_rounds=3
        )
        
        result = optimizer.optimize(sample_features, sample_labels, sample_features, sample_labels)
        
        assert result is not None
