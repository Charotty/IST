"""
Unit tests for data synchronizer.
"""
import pytest
import pandas as pd
import numpy as np
from its_project.preprocessing.synchronizer import DataSynchronizer


@pytest.mark.unit
@pytest.mark.preprocessing_layer
class TestDataSynchronizer:
    """Test data synchronizer functionality."""
    
    def test_synchronizer_initialization(self):
        """Test synchronizer initialization."""
        synchronizer = DataSynchronizer()
        assert synchronizer is not None
    
    def test_time_alignment(self):
        """Test time alignment of multiple data streams."""
        # Create data with different timestamps
        price_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1s'),
            'price': [42000 + i for i in range(100)]
        })
        
        volume_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1s') + pd.Timedelta(seconds=0.5),
            'volume': [100 + i for i in range(100)]
        })
        
        synchronizer = DataSynchronizer()
        aligned = synchronizer.align_timestamps(price_data, volume_data)
        
        # Verify alignment
        assert len(aligned) == len(price_data)
    
    def test_resampling(self):
        """Test resampling to common frequency."""
        # Create high-frequency data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='100ms'),
            'price': [42000 + i * 0.01 for i in range(1000)]
        })
        
        synchronizer = DataSynchronizer()
        resampled = synchronizer.resample(data, freq='1s')
        
        # Verify resampling
        assert len(resampled) < len(data)
        assert len(resampled) == 100  # 1000ms / 1s = 100
    
    def test_gap_filling_forward(self):
        """Test forward gap filling."""
        # Create data with gaps
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1s'),
            'price': [42000 + i if i % 10 != 0 else np.nan for i in range(100)]
        })
        
        synchronizer = DataSynchronizer()
        filled = synchronizer.fill_gaps(data, method='forward')
        
        # Verify no gaps
        assert filled['price'].isna().sum() == 0
    
    def test_gap_filling_backward(self):
        """Test backward gap filling."""
        # Create data with gaps at end
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1s'),
            'price': [42000 + i if i < 90 else np.nan for i in range(100)]
        })
        
        synchronizer = DataSynchronizer()
        filled = synchronizer.fill_gaps(data, method='backward')
        
        # Verify no gaps
        assert filled['price'].isna().sum() == 0
    
    def test_multi_stream_synchronization(self):
        """Test synchronization of multiple data streams."""
        price_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1s'),
            'price': [42000 + i for i in range(100)]
        })
        
        volume_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1s'),
            'volume': [100 + i for i in range(100)]
        })
        
        sentiment_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1s'),
            'sentiment': [0.5 + i * 0.01 for i in range(100)]
        })
        
        synchronizer = DataSynchronizer()
        synced = synchronizer.synchronize_streams({
            'price': price_data,
            'volume': volume_data,
            'sentiment': sentiment_data
        })
        
        # Verify all streams synchronized
        assert 'price_price' in synced.columns
        assert 'volume_volume' in synced.columns
        assert 'sentiment_sentiment' in synced.columns
        assert len(synced) == 100
    
    def test_timestamp_validation(self):
        """Test timestamp validation."""
        # Create data with non-monotonic timestamps
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1s').tolist()[::-1],
            'price': [42000 + i for i in range(100)]
        })
        
        synchronizer = DataSynchronizer()
        is_valid = synchronizer.validate_timestamps(data)
        
        # Should detect non-monotonic timestamps
        assert not is_valid
    
    def test_synchronization_pipeline(self):
        """Test full synchronization pipeline."""
        # Create data with issues
        price_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1s') + pd.Timedelta(seconds=0.5),
            'price': [42000 + i if i % 10 != 0 else np.nan for i in range(100)]
        })
        
        volume_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1s'),
            'volume': [100 + i for i in range(100)]
        })
        
        synchronizer = DataSynchronizer()
        synced = synchronizer.synchronize({
            'price': price_data,
            'volume': volume_data
        })
        
        # Verify clean, synchronized output
        assert synced['price'].isna().sum() == 0
        assert len(synced) == 100
