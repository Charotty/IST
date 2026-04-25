"""
Unit tests for time series cross-validation.
"""
import pytest
import numpy as np
from its_project.metalearning.cv import TimeSeriesSplitter, WalkForwardValidator


@pytest.mark.unit
@pytest.mark.metalearning_layer
class TestTimeSeriesSplitter:
    """Test time series splitter functionality."""
    
    def test_splitter_initialization(self):
        """Test splitter initialization."""
        splitter = TimeSeriesSplitter(n_splits=5, gap=0)
        assert splitter is not None
        assert splitter.n_splits == 5
    
    def test_split_generation(self, sample_features):
        """Test split generation."""
        splitter = TimeSeriesSplitter(n_splits=3, gap=10)
        
        splits = list(splitter.split(sample_features))
        
        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
    
    def test_chronological_ordering(self, sample_features):
        """Test chronological ordering of splits."""
        splitter = TimeSeriesSplitter(n_splits=3, gap=10)
        
        splits = list(splitter.split(sample_features))
        
        for train_idx, test_idx in splits:
            # All test indices should be after all train indices
            assert train_idx.max() < test_idx.min()
    
    def test_gap_parameter(self, sample_features):
        """Test gap parameter."""
        splitter = TimeSeriesSplitter(n_splits=3, gap=20)
        
        splits = list(splitter.split(sample_features))
        
        for train_idx, test_idx in splits:
            # Gap should be present between train and test
            assert test_idx.min() - train_idx.max() >= 20
    
    def test_no_leakage(self, sample_features):
        """Test no future data leakage."""
        splitter = TimeSeriesSplitter(n_splits=3, gap=10)
        
        splits = list(splitter.split(sample_features))
        
        all_train_indices = set()
        all_test_indices = set()
        
        for train_idx, test_idx in splits:
            all_train_indices.update(train_idx)
            all_test_indices.update(test_idx)
        
        # No overlap between train and test across all splits
        assert len(all_train_indices & all_test_indices) == 0


@pytest.mark.unit
@pytest.mark.metalearning_layer
class TestWalkForwardValidator:
    """Test walk-forward validator functionality."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = WalkForwardValidator(train_size=252, test_size=63, step_size=21)
        assert validator is not None
        assert validator.train_size == 252
    
    def test_rolling_window_validation(self, sample_features, sample_labels):
        """Test rolling window validation."""
        validator = WalkForwardValidator(train_size=50, test_size=20, step_size=10)
        
        # Mock model factory
        def model_factory():
            from its_project.models.ensemble import EnsembleModel
            return EnsembleModel({'method': 'voting', 'estimators': ['lr']})
        
        results = validator.validate(sample_features, sample_labels, model_factory)
        
        assert results is not None
        assert len(results) > 0
    
    def test_multi_asset_support(self):
        """Test multi-asset support."""
        validator = WalkForwardValidator(train_size=50, test_size=20, step_size=10)
        
        # Create multi-asset data
        data1 = np.random.randn(100, 20)
        data2 = np.random.randn(100, 20)
        
        labels1 = np.random.randint(0, 3, 100)
        labels2 = np.random.randint(0, 3, 100)
        
        def model_factory():
            from its_project.models.ensemble import EnsembleModel
            return EnsembleModel({'method': 'voting', 'estimators': ['lr']})
        
        # Validate each asset
        results1 = validator.validate(data1, labels1, model_factory)
        results2 = validator.validate(data2, labels2, model_factory)
        
        assert results1 is not None
        assert results2 is not None
    
    def test_parameter_stability(self, sample_features, sample_labels):
        """Test parameter stability across windows."""
        validator = WalkForwardValidator(train_size=50, test_size=20, step_size=10)
        
        def model_factory():
            from its_project.models.ensemble import EnsembleModel
            return EnsembleModel({'method': 'voting', 'estimators': ['lr']})
        
        results = validator.validate(sample_features, sample_labels, model_factory)
        
        # Should have results for multiple windows
        assert len(results) > 1
    
    def test_result_aggregation(self, sample_features, sample_labels):
        """Test result aggregation across windows."""
        validator = WalkForwardValidator(train_size=50, test_size=20, step_size=10)
        
        def model_factory():
            from its_project.models.ensemble import EnsembleModel
            return EnsembleModel({'method': 'voting', 'estimators': ['lr']})
        
        results = validator.validate(sample_features, sample_labels, model_factory)
        aggregated = validator.aggregate_results(results)
        
        assert aggregated is not None
        assert 'mean_score' in aggregated or 'scores' in aggregated
