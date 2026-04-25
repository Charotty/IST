"""
Unit tests for model selector.
"""
import pytest
import numpy as np
from its_project.metalearning.selector import ModelSelector
from its_project.models.lstm import LSTMModel
from its_project.models.ensemble import EnsembleModel


@pytest.mark.unit
@pytest.mark.metalearning_layer
class TestModelSelector:
    """Test model selector functionality."""
    
    def test_selector_initialization(self):
        """Test selector initialization."""
        models = [LSTMModel({'input_size': 20, 'hidden_size': 32}), 
                  EnsembleModel({'method': 'voting', 'estimators': ['lr']})]
        selector = ModelSelector(models=models, metrics=['accuracy'], weights=[1.0])
        
        assert selector is not None
        assert len(selector.models) == 2
    
    def test_multi_model_evaluation(self, sample_features, sample_labels):
        """Test evaluating multiple models."""
        models = [
            EnsembleModel({'method': 'voting', 'estimators': ['lr']}),
            EnsembleModel({'method': 'voting', 'estimators': ['rf']})
        ]
        
        selector = ModelSelector(models=models, metrics=['accuracy'], weights=[1.0])
        results = selector.evaluate_all(sample_features, sample_labels, sample_features, sample_labels)
        
        assert results is not None
        assert len(results) == 2
    
    def test_weighted_metrics(self, sample_features, sample_labels):
        """Test weighted metric calculation."""
        models = [
            EnsembleModel({'method': 'voting', 'estimators': ['lr']}),
            EnsembleModel({'method': 'voting', 'estimators': ['rf']})
        ]
        
        selector = ModelSelector(
            models=models,
            metrics=['accuracy', 'f1'],
            weights=[0.6, 0.4]
        )
        results = selector.evaluate_all(sample_features, sample_labels, sample_features, sample_labels)
        
        assert results is not None
    
    def test_best_model_selection(self, sample_features, sample_labels):
        """Test selecting best model."""
        models = [
            EnsembleModel({'method': 'voting', 'estimators': ['lr']}),
            EnsembleModel({'method': 'voting', 'estimators': ['rf']})
        ]
        
        selector = ModelSelector(models=models, metrics=['accuracy'], weights=[1.0])
        selector.evaluate_all(sample_features, sample_labels, sample_features, sample_labels)
        
        best_model, best_scores = selector.select_best()
        
        assert best_model is not None
        assert best_scores is not None
    
    def test_ranking_generation(self, sample_features, sample_labels):
        """Test generating model ranking."""
        models = [
            EnsembleModel({'method': 'voting', 'estimators': ['lr']}),
            EnsembleModel({'method': 'voting', 'estimators': ['rf']}),
            EnsembleModel({'method': 'voting', 'estimators': ['gb']})
        ]
        
        selector = ModelSelector(models=models, metrics=['accuracy'], weights=[1.0])
        selector.evaluate_all(sample_features, sample_labels, sample_features, sample_labels)
        
        ranking = selector.get_ranking()
        
        assert ranking is not None
        assert len(ranking) == 3
