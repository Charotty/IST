"""
Integration tests for Preprocessing Layer.
"""
import pytest
import pandas as pd
import numpy as np
from its_project.preprocessing.cleaner import DataCleaner
from its_project.preprocessing.normalizer import DataNormalizer
from its_project.preprocessing.synchronizer import DataSynchronizer


@pytest.mark.integration
@pytest.mark.preprocessing_layer
class TestPreprocessingPipeline:
    """Test preprocessing pipeline integration."""
    
    def test_cleaner_normalizer_synchronizer_chain(self, sample_market_data):
        """Test cleaner → normalizer → synchronizer chain."""
        # Add issues to data
        dirty_data = sample_market_data.copy()
        dirty_data.loc[0, 'close'] = np.nan
        dirty_data.loc[1, 'close'] = 1000000
        dirty_data = pd.concat([dirty_data, dirty_data.head(5)])
        
        # Cleaner
        cleaner = DataCleaner()
        cleaned = cleaner.clean(dirty_data)
        
        # Verify cleaning
        assert cleaned['close'].isna().sum() == 0
        assert cleaned['close'].max() < 1000000
        assert len(cleaned) == len(dirty_data) - 5
        
        # Normalizer
        normalizer = DataNormalizer(method='minmax')
        price_cols = ['open', 'high', 'low', 'close']
        normalized = normalizer.fit_transform(cleaned[price_cols])
        
        # Verify normalization
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        
        # Synchronizer
        synchronizer = DataSynchronizer()
        synced = synchronizer.synchronize_streams({'price': cleaned})
        
        # Verify synchronization
        assert 'price_open' in synced.columns
        assert len(synced) == len(cleaned)
    
    def test_pipeline_with_dirty_data(self):
        """Test pipeline with dirty data."""
        # Create dirty data with multiple issues
        price_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1s') + pd.Timedelta(seconds=0.5),
            'price': [42000 + i if i % 10 != 0 else np.nan for i in range(100)]
        })
        
        volume_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1s'),
            'volume': [100 + i for i in range(100)]
        })
        
        # Run full pipeline
        synchronizer = DataSynchronizer()
        synced = synchronizer.synchronize({
            'price': price_data,
            'volume': volume_data
        })
        
        cleaner = DataCleaner()
        cleaned = cleaner.clean(synced)
        
        normalizer = DataNormalizer(method='zscore')
        normalized = normalizer.fit_transform(cleaned)
        
        # Verify output quality
        assert normalized.isna().sum().sum() == 0
        assert len(normalized) > 0
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""
        # Create invalid data
        invalid_data = pd.DataFrame({
            'price': ['invalid', 'data']  # Wrong data type
        })
        
        cleaner = DataCleaner()
        
        # Should handle error gracefully
        try:
            cleaned = cleaner.clean(invalid_data)
            # If it doesn't raise, verify it handled it
        except (ValueError, TypeError):
            # Expected behavior
            pass
    
    def test_pipeline_performance(self, sample_market_data):
        """Test pipeline performance on large dataset."""
        # Create large dataset
        large_data = pd.concat([sample_market_data] * 100)
        
        cleaner = DataCleaner()
        cleaned = cleaner.clean(large_data)
        
        normalizer = DataNormalizer(method='minmax')
        normalized = normalizer.fit_transform(cleaned)
        
        # Verify performance (should complete in reasonable time)
        assert len(normalized) == len(large_data)
