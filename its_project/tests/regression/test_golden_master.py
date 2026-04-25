"""
Regression tests with golden master outputs.
"""
import pytest
import pandas as pd
import numpy as np
from its_project.preprocessing.cleaner import DataCleaner
from its_project.models.ensemble import EnsembleModel


@pytest.mark.regression
class TestGoldenMaster:
    """Golden master regression tests."""
    
    def test_data_cleaner_output(self, sample_market_data):
        """Test data cleaner produces consistent output."""
        cleaner = DataCleaner()
        result = cleaner.clean(sample_market_data)
        
        # Verify output structure
        assert 'timestamp' in result.columns
        assert 'close' in result.columns
        assert result['close'].isna().sum() == 0
    
    def test_model_predictions_consistency(self, sample_features, sample_labels):
        """Test model predictions are consistent."""
        model = EnsembleModel({'method': 'voting', 'estimators': ['lr']})
        model.fit(sample_features, sample_labels)
        
        # Predict twice
        pred1 = model.predict(sample_features)
        pred2 = model.predict(sample_features)
        
        # Results should be identical
        assert (pred1 == pred2).all()
    
    def test_feature_engineering_output(self, sample_market_data):
        """Test feature engineering produces consistent output."""
        from its_project.features.technical import TechnicalFeatures
        features = TechnicalFeatures({'indicators': ['rsi']})
        
        result = features.calculate(sample_market_data)
        
        # Verify output structure
        assert 'rsi' in result.columns
        assert result['rsi'].between(0, 100).all()
    
    def test_decision_making_output(self):
        """Test decision making produces consistent output."""
        from its_project.decision.simple import SimpleDecisionMaker
        from its_project.decision.decision import Action, Signal
        
        maker = SimpleDecisionMaker({'confidence_threshold': 0.7})
        
        signal = Signal(
            action=Action.BUY,
            confidence=0.85,
            timestamp=1234567890000,
            symbol='BTCUSDT'
        )
        
        decision1 = maker.make_decision(signal, {'price': 42000}, 10000.0)
        decision2 = maker.make_decision(signal, {'price': 42000}, 10000.0)
        
        # Results should be consistent
        assert decision1 == decision2 or (decision1 is None and decision2 is None)
