"""
Unit tests for data cleaner.
"""
import pytest
import pandas as pd
import numpy as np
from its_project.preprocessing.cleaner import DataCleaner


@pytest.mark.unit
@pytest.mark.preprocessing_layer
class TestDataCleaner:
    """Test data cleaner functionality."""
    
    def test_cleaner_initialization(self):
        """Test cleaner initialization."""
        cleaner = DataCleaner()
        assert cleaner is not None
    
    def test_remove_duplicates(self, sample_market_data):
        """Test duplicate removal."""
        # Add duplicates
        data_with_dupes = pd.concat([sample_market_data, sample_market_data.head(5)])
        
        cleaner = DataCleaner()
        cleaned = cleaner.remove_duplicates(data_with_dupes)
        
        # Verify duplicates removed
        assert len(cleaned) == len(sample_market_data)
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        data = pd.DataFrame({
            'price': [42000, np.nan, 42010, 42015, np.nan],
            'volume': [100, 200, np.nan, 400, 500]
        })
        
        cleaner = DataCleaner()
        cleaned = cleaner.handle_missing_values(data, method='forward_fill')
        
        # Verify no missing values
        assert cleaned['price'].isna().sum() == 0
        assert cleaned['volume'].isna().sum() == 0
    
    def test_remove_outliers(self, sample_market_data):
        """Test outlier removal."""
        # Add outliers
        data_with_outliers = sample_market_data.copy()
        data_with_outliers.loc[0, 'close'] = 1000000  # Extreme outlier
        data_with_outliers.loc[1, 'close'] = 100       # Extreme outlier
        
        cleaner = DataCleaner()
        cleaned = cleaner.remove_outliers(data_with_outliers, columns=['close'], std_threshold=3)
        
        # Verify outliers removed
        assert cleaned['close'].max() < 1000000
        assert cleaned['close'].min() > 100
    
    def test_data_type_validation(self, sample_market_data):
        """Test data type validation."""
        cleaner = DataCleaner()
        
        # Should pass with correct types
        assert cleaner.validate_types(sample_market_data)
    
    def test_schema_validation(self, sample_market_data):
        """Test schema validation."""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        cleaner = DataCleaner()
        assert cleaner.validate_schema(sample_market_data, required_columns)
        
        # Should fail with missing column
        incomplete_data = sample_market_data.drop(columns=['volume'])
        assert not cleaner.validate_schema(incomplete_data, required_columns)
    
    def test_cleaning_pipeline(self):
        """Test full cleaning pipeline."""
        # Create dirty data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1s'),
            'open': [42000 + i * 0.1 for i in range(100)],
            'high': [42050 + i * 0.1 for i in range(100)],
            'low': [41950 + i * 0.1 for i in range(100)],
            'close': [42000 + i * 0.1 for i in range(100)],
            'volume': [100 + i for i in range(100)]
        })
        
        # Add issues
        data.loc[0, 'close'] = np.nan
        data.loc[1, 'close'] = 1000000
        data = pd.concat([data, data.head(5)])  # Add duplicates
        
        cleaner = DataCleaner()
        cleaned = cleaner.clean(data)
        
        # Verify all issues resolved
        assert cleaned['close'].isna().sum() == 0
        assert cleaned['close'].max() < 1000000
        assert len(cleaned) == len(data) - 5  # Duplicates removed
