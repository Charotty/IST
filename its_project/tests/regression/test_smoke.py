"""
Regression smoke tests - quick sanity checks.
"""
import pytest
import pandas as pd
import numpy as np
from its_project.preprocessing.cleaner import DataCleaner
from its_project.models.ensemble import EnsembleModel


@pytest.mark.regression
@pytest.mark.smoke
class TestSmokeTests:
    """Quick smoke tests to verify basic functionality."""
    
    def test_data_cleaner_smoke(self, sample_market_data):
        """Test data cleaner basic functionality."""
        cleaner = DataCleaner()
        result = cleaner.clean(sample_market_data)
        assert result is not None
        assert len(result) > 0
    
    def test_model_smoke(self, sample_features, sample_labels):
        """Test model basic functionality."""
        model = EnsembleModel({'method': 'voting', 'estimators': ['lr']})
        model.fit(sample_features, sample_labels)
        predictions = model.predict(sample_features)
        assert predictions is not None
        assert len(predictions) > 0
    
    def test_config_smoke(self):
        """Test config loading."""
        from its_project.config import load_config
        config = load_config()
        assert config is not None
    
    def test_imports_smoke(self):
        """Test that key modules can be imported."""
        from its_project.preprocessing.cleaner import DataCleaner
        from its_project.models.ensemble import EnsembleModel
        from its_project.decision.simple import SimpleDecisionMaker
        from its_project.execution.paper import PaperTradingExecutor
        assert True
