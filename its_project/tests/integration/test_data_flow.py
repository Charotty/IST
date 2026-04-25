"""
Integration tests for data flow across layers.
"""
import pytest
import pandas as pd
import numpy as np
from its_project.preprocessing.cleaner import DataCleaner
from its_project.preprocessing.normalizer import DataNormalizer
from its_project.features.technical import TechnicalFeatures
from its_project.models.ensemble import EnsembleModel


@pytest.mark.integration
@pytest.mark.e2e
class TestDataFlow:
    """Test data flow across layers."""
    
    def test_data_layer_to_preprocessing(self, sample_market_data):
        """Test data → preprocessing flow."""
        cleaner = DataCleaner()
        cleaned = cleaner.clean(sample_market_data)
        
        assert cleaned is not None
        assert len(cleaned) == len(sample_market_data)
        assert cleaned['close'].isna().sum() == 0
    
    def test_preprocessing_to_features(self, sample_market_data):
        """Test preprocessing → features flow."""
        cleaner = DataCleaner()
        cleaned = cleaner.clean(sample_market_data)
        
        features = TechnicalFeatures({'indicators': ['rsi', 'macd']})
        feature_data = features.calculate(cleaned)
        
        assert 'rsi' in feature_data.columns
        assert 'macd' in feature_data.columns
    
    def test_features_to_model(self, sample_market_data):
        """Test features → model flow."""
        cleaner = DataCleaner()
        cleaned = cleaner.clean(sample_market_data)
        
        features = TechnicalFeatures({'indicators': ['rsi', 'macd']})
        feature_data = features.calculate(cleaned)
        
        model = EnsembleModel({'method': 'voting', 'estimators': ['lr']})
        X = feature_data[['rsi', 'macd']].values
        y = np.array([0, 1, 2] * len(feature_data))[:len(feature_data)]
        model.fit(X, y)
        
        assert model.fitted
    
    def test_data_contract_compliance(self, sample_market_data):
        """Test data contract compliance across layers."""
        # Verify data has required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        assert all(col in sample_market_data.columns for col in required_cols)
        
        cleaner = DataCleaner()
        cleaned = cleaner.clean(sample_market_data)
        
        # Verify cleaning preserves data contract
        assert all(col in cleaned.columns for col in required_cols)
    
    def test_batch_processing(self, sample_market_data):
        """Test batch processing of data."""
        cleaner = DataCleaner()
        
        # Process in batches
        batch_size = 100
        for i in range(0, len(sample_market_data), batch_size):
            batch = sample_market_data.iloc[i:i+batch_size]
            cleaned = cleaner.clean(batch)
            assert cleaned is not None
    
    def test_data_quality_validation(self, sample_market_data):
        """Test data quality validation across pipeline."""
        cleaner = DataCleaner()
        cleaned = cleaner.clean(sample_market_data)
        
        # Validate data quality
        assert cleaned['close'].isna().sum() == 0
        assert cleaned['volume'].isna().sum() == 0
        assert (cleaned['high'] >= cleaned['low']).all()
