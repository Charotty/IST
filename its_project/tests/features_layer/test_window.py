"""
Unit tests for WindowedFeatures and create_fixed_length_windows.
"""
import pytest
import numpy as np
import pandas as pd
from its_project.features.window import WindowedFeatures, create_fixed_length_windows


# Mock feature calculator for testing
class MockFeatureCalculator:
    """Mock feature calculator for testing."""
    
    def calculate(self, window):
        # Return simple features: mean of first column
        return np.array([window.iloc[:, 0].mean()])


@pytest.mark.unit
@pytest.mark.features_layer
class TestWindowedFeatures:
    """Test WindowedFeatures functionality."""
    
    def test_initialization(self):
        """Test WindowedFeatures initialization."""
        wf = WindowedFeatures(window_size=5, stride=2)
        
        assert wf.window_size == 5
        assert wf.stride == 2
    
    def test_initialization_default_stride(self):
        """Test initialization with default stride."""
        wf = WindowedFeatures(window_size=5)
        
        assert wf.window_size == 5
        assert wf.stride == 1
    
    def test_initialization_invalid_window_size(self):
        """Test initialization with invalid window size."""
        with pytest.raises(ValueError, match="window_size must be > 0"):
            WindowedFeatures(window_size=0)
        
        with pytest.raises(ValueError, match="window_size must be > 0"):
            WindowedFeatures(window_size=-1)
    
    def test_initialization_invalid_stride(self):
        """Test initialization with invalid stride."""
        with pytest.raises(ValueError, match="stride must be > 0"):
            WindowedFeatures(window_size=5, stride=0)
        
        with pytest.raises(ValueError, match="stride must be > 0"):
            WindowedFeatures(window_size=5, stride=-1)
    
    def test_create_windows(self):
        """Test creating sliding windows."""
        data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        wf = WindowedFeatures(window_size=3, stride=2)
        windows = list(wf.create_windows(data))
        
        assert len(windows) == 4  # (10-3)/2 + 1 = 4
        assert windows[0].iloc[0]['a'] == 1
        assert windows[0].iloc[-1]['a'] == 3
        assert windows[1].iloc[0]['a'] == 3
        assert windows[1].iloc[-1]['a'] == 5
    
    def test_create_windows_stride_1(self):
        """Test creating windows with stride 1."""
        data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5]
        })
        
        wf = WindowedFeatures(window_size=3, stride=1)
        windows = list(wf.create_windows(data))
        
        assert len(windows) == 3  # 5-3+1 = 3
        assert windows[0].iloc[0]['a'] == 1
        assert windows[1].iloc[0]['a'] == 2
        assert windows[2].iloc[0]['a'] == 3
    
    def test_create_windows_window_equals_length(self):
        """Test when window size equals data length."""
        data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5]
        })
        
        wf = WindowedFeatures(window_size=5, stride=1)
        windows = list(wf.create_windows(data))
        
        assert len(windows) == 1
        assert len(windows[0]) == 5
    
    def test_create_windows_window_larger_than_data(self):
        """Test when window size is larger than data length."""
        data = pd.DataFrame({
            'a': [1, 2, 3]
        })
        
        wf = WindowedFeatures(window_size=5, stride=1)
        windows = list(wf.create_windows(data))
        
        assert len(windows) == 0
    
    def test_create_windows_copy(self):
        """Test that windows are copies, not views."""
        data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5]
        })
        
        wf = WindowedFeatures(window_size=3, stride=1)
        windows = list(wf.create_windows(data))
        
        # Modify the first window
        windows[0].iloc[0, 0] = 999
        
        # Original data should be unchanged
        assert data.iloc[0, 0] == 1
    
    def test_transform(self):
        """Test transform with feature calculator."""
        data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6],
            'b': [10, 20, 30, 40, 50, 60]
        })
        
        wf = WindowedFeatures(window_size=3, stride=2)
        calculator = MockFeatureCalculator()
        
        result = wf.transform(data, calculator)
        
        # Should have 2 windows: [1,2,3] and [3,4,5]
        # Means: 2 and 4
        assert result.shape == (2, 1)
        assert result[0, 0] == 2.0
        assert result[1, 0] == 4.0
    
    def test_transform_empty_result(self):
        """Test transform when no windows can be created."""
        data = pd.DataFrame({
            'a': [1, 2]
        })
        
        wf = WindowedFeatures(window_size=5, stride=1)
        calculator = MockFeatureCalculator()
        
        result = wf.transform(data, calculator)
        
        assert result.shape == (0, 0)


@pytest.mark.unit
@pytest.mark.features_layer
class TestCreateFixedLengthWindows:
    """Test create_fixed_length_windows standalone function."""
    
    def test_create_fixed_length_windows(self):
        """Test standalone function."""
        data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6]
        })
        
        windows = list(create_fixed_length_windows(data, window_size=3, stride=2))
        
        assert len(windows) == 2
        assert windows[0].iloc[0]['a'] == 1
        assert windows[0].iloc[-1]['a'] == 3
        assert windows[1].iloc[0]['a'] == 3
        assert windows[1].iloc[-1]['a'] == 5
    
    def test_create_fixed_length_windows_default_stride(self):
        """Test standalone function with default stride."""
        data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5]
        })
        
        windows = list(create_fixed_length_windows(data, window_size=3))
        
        assert len(windows) == 3  # stride defaults to 1
