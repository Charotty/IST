"""
Unit tests for stacking ensemble.
"""
import pytest
import numpy as np
from its_project.metalearning.stacking import StackingEnsemble
from its_project.models.ensemble import EnsembleModel


@pytest.mark.unit
@pytest.mark.metalearning_layer
class TestStackingEnsemble:
    """Test stacking ensemble functionality."""
    
    def test_stacking_initialization(self):
        """Test stacking ensemble initialization."""
        base_models = [
            EnsembleModel({'method': 'voting', 'estimators': ['lr']}),
            EnsembleModel({'method': 'voting', 'estimators': ['rf']})
        ]
        meta_model = EnsembleModel({'method': 'voting', 'estimators': ['gb']})
        
        stacking = StackingEnsemble(base_models, meta_model)
        
        assert stacking is not None
        assert len(stacking.base_models) == 2
    
    def test_base_model_training(self, sample_features, sample_labels):
        """Test base model training."""
        base_models = [
            EnsembleModel({'method': 'voting', 'estimators': ['lr']}),
            EnsembleModel({'method': 'voting', 'estimators': ['rf']})
        ]
        meta_model = EnsembleModel({'method': 'voting', 'estimators': ['gb']})
        
        stacking = StackingEnsemble(base_models, meta_model)
        stacking.fit(sample_features, sample_labels)
        
        # All base models should be fitted
        assert all(m.fitted for m in stacking.base_models)
    
    def test_meta_model_training(self, sample_features, sample_labels):
        """Test meta model training."""
        base_models = [
            EnsembleModel({'method': 'voting', 'estimators': ['lr']}),
            EnsembleModel({'method': 'voting', 'estimators': ['rf']})
        ]
        meta_model = EnsembleModel({'method': 'voting', 'estimators': ['gb']})
        
        stacking = StackingEnsemble(base_models, meta_model)
        stacking.fit(sample_features, sample_labels)
        
        # Meta model should be fitted
        assert stacking.meta_model.fitted
    
    def test_prediction_generation(self, sample_features, sample_labels):
        """Test prediction generation."""
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
    
    def test_feature_generation_from_base_models(self, sample_features, sample_labels):
        """Test feature generation from base models."""
        base_models = [
            EnsembleModel({'method': 'voting', 'estimators': ['lr']}),
            EnsembleModel({'method': 'voting', 'estimators': ['rf']})
        ]
        meta_model = EnsembleModel({'method': 'voting', 'estimators': ['gb']})
        
        stacking = StackingEnsemble(base_models, meta_model)
        stacking.fit(sample_features, sample_labels)
        
        # Generate features from base models
        features = stacking._generate_base_features(sample_features)
        
        assert features is not None
        assert features.shape[0] == len(sample_features)
        assert features.shape[1] == len(base_models) * 3  # 3 classes per model
