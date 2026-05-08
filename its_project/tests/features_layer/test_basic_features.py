"""
Unit tests for Basic Features (returns, log returns, lag features, volatility).
"""
import pytest
import numpy as np
import pandas as pd
from its_project.features.basic import BasicFeatures


@pytest.mark.unit
@pytest.mark.features_layer
class TestBasicFeatures:
    """Test BasicFeatures functionality."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        np.random.seed(42)
        n = 100
        # Create realistic price series
        prices = [50000.0]
        for i in range(1, n):
            change = np.random.normal(0, 0.01)  # 1% daily volatility
            prices.append(prices[-1] * (1 + change))
        
        return pd.DataFrame({
            'price': prices,
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1min')
        })
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        prices = [50000.0]
        data = []
        
        for i in range(n):
            if i > 0:
                change = np.random.normal(0, 0.01)
                prices.append(prices[-1] * (1 + change))
            
            current_price = prices[-1]
            high = current_price * (1 + abs(np.random.normal(0, 0.005)))
            low = current_price * (1 - abs(np.random.normal(0, 0.005)))
            volume = np.random.normal(1000, 200)
            
            data.append({
                'open': current_price,
                'high': high,
                'low': low,
                'close': current_price,
                'volume': max(volume, 100)
            })
        
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """Test BasicFeatures initialization."""
        config = {
            'features': ['returns', 'log_returns', 'lags', 'volatility'],
            'return_periods': [1, 5, 15],
            'lag_periods': [1, 2, 3, 5, 10],
            'volatility_windows': [10, 20, 30],
            'atr_periods': [14, 21],
            'price_column': 'close'
        }
        features = BasicFeatures(config)
        
        assert features.features == ['returns', 'log_returns', 'lags', 'volatility']
        assert features.return_periods == [1, 5, 15]
        assert features.lag_periods == [1, 2, 3, 5, 10]
        assert features.volatility_windows == [10, 20, 30]
        assert features.atr_periods == [14, 21]
        assert features.price_column == 'close'
    
    def test_default_config(self):
        """Test default configuration."""
        config = {}
        features = BasicFeatures(config)
        
        assert features.features == ['returns', 'log_returns', 'lags']
        assert features.return_periods == [1, 5, 15]
        assert features.lag_periods == [1, 2, 3, 5, 10]
        assert features.price_column == 'close'
    
    def test_calculate_returns(self, sample_price_data):
        """Test returns calculation."""
        config = {
            'features': ['returns'],
            'return_periods': [1, 5],
            'price_column': 'price'
        }
        features = BasicFeatures(config)
        
        result = features.calculate(sample_price_data)
        
        # Should have 2 return features (1-period and 5-period)
        assert result.shape[0] == len(sample_price_data)
        assert result.shape[1] == 2
        
        # Check that returns are calculated correctly
        # First 5 rows should be NaN for 5-period returns
        assert np.isnan(result.iloc[0, 1])  # 5-period return at row 0
        assert np.isnan(result.iloc[4, 1])  # 5-period return at row 4
        assert not np.isnan(result.iloc[5, 1])  # 5-period return at row 5
        
        # 1-period returns should only be NaN at first row
        assert np.isnan(result.iloc[0, 0])  # 1-period return at row 0
        assert not np.isnan(result.iloc[1, 0])  # 1-period return at row 1
    
    def test_calculate_log_returns(self, sample_price_data):
        """Test log returns calculation."""
        config = {
            'features': ['log_returns'],
            'return_periods': [1, 3],
            'price_column': 'price'
        }
        features = BasicFeatures(config)
        
        result = features.calculate(sample_price_data)
        
        # Should have 2 log return features
        assert result.shape[0] == len(sample_price_data)
        assert result.shape[1] == 2
        
        # Verify log returns calculation
        # log_return = log(price_t / price_{t-n})
        for i in range(3, len(sample_price_data)):
            expected_log_ret_3 = np.log(
                sample_price_data['price'].iloc[i] / 
                sample_price_data['price'].iloc[i-3]
            )
            assert abs(result.iloc[i, 1] - expected_log_ret_3) < 1e-10
    
    def test_calculate_lag_features(self, sample_price_data):
        """Test lag features calculation."""
        config = {
            'features': ['lags'],
            'lag_periods': [1, 2, 5],
            'price_column': 'price'
        }
        features = BasicFeatures(config)
        
        result = features.calculate(sample_price_data)
        
        # Should have 3 lag features
        assert result.shape[0] == len(sample_price_data)
        assert result.shape[1] == 3
        
        # Check lag values
        for i in range(5, len(sample_price_data)):
            assert result.iloc[i, 0] == sample_price_data['price'].iloc[i-1]  # lag 1
            assert result.iloc[i, 1] == sample_price_data['price'].iloc[i-2]  # lag 2
            assert result.iloc[i, 2] == sample_price_data['price'].iloc[i-5]  # lag 5
        
        # Check NaN values for early rows
        assert np.isnan(result.iloc[0, 0])  # lag 1 at row 0
        assert np.isnan(result.iloc[0, 1])  # lag 2 at row 0
        assert np.isnan(result.iloc[0, 2])  # lag 5 at row 0
        assert np.isnan(result.iloc[4, 2])  # lag 5 at row 4
        assert not np.isnan(result.iloc[5, 2])  # lag 5 at row 5
    
    def test_calculate_all_features(self, sample_price_data):
        """Test calculating all basic features."""
        config = {
            'features': ['returns', 'log_returns', 'lags', 'volatility'],
            'return_periods': [1, 5],
            'lag_periods': [1, 3],
            'volatility_windows': [10],
            'atr_periods': [14],  # Will be ignored for price-only data
            'price_column': 'price'
        }
        features = BasicFeatures(config)
        
        result = features.calculate(sample_price_data)
        
        # Should have: returns(2) + log_returns(2) + lags(2) + volatility(1) = 7 features
        assert result.shape[0] == len(sample_price_data)
        assert result.shape[1] == 7
        
        # Check that no unexpected NaN values exist after all windows have enough history.
        assert not np.isnan(result.iloc[15, 0])  # 1-period return
        assert not np.isnan(result.iloc[15, 1])  # 5-period return
        assert not np.isnan(result.iloc[15, 2])  # 1-period log return
        assert not np.isnan(result.iloc[15, 3])  # 5-period log return
        assert not np.isnan(result.iloc[15, 4])  # lag 1
        assert not np.isnan(result.iloc[15, 5])  # lag 3
        assert not np.isnan(result.iloc[15, 6])  # rolling std 10
    
    def test_get_feature_names(self, sample_price_data):
        """Test getting feature names."""
        config = {
            'features': ['returns', 'log_returns', 'lags', 'volatility'],
            'return_periods': [1, 5],
            'lag_periods': [1, 3],
            'volatility_windows': [10],
            'atr_periods': [14],
            'price_column': 'price'
        }
        features = BasicFeatures(config)
        
        features.calculate(sample_price_data)
        names = features.get_feature_names()
        
        expected_names = [
            'return_1', 'return_5',
            'log_return_1', 'log_return_5',
            'lag_1', 'lag_3',
            'rolling_std_10'  # ATR should be ignored for price-only data
        ]
        
        assert len(names) == 7
        for name in expected_names:
            assert name in names
    
    def test_ohlcv_data(self, sample_ohlcv_data):
        """Test features with OHLCV data."""
        config = {
            'features': ['returns', 'log_returns', 'lags', 'volatility'],
            'return_periods': [1],
            'lag_periods': [1, 2],
            'volatility_windows': [10],
            'atr_periods': [14],
            'price_column': 'close'
        }
        features = BasicFeatures(config)
        
        result = features.calculate(sample_ohlcv_data)
        
        # Should have: returns(1) + log_returns(1) + lags(2) + volatility(2) = 6 features
        assert result.shape[0] == len(sample_ohlcv_data)
        assert result.shape[1] == 6
        
        # Verify calculations using close prices
        for i in range(2, len(sample_ohlcv_data)):
            # Check return calculation
            expected_return = (
                sample_ohlcv_data['close'].iloc[i] - 
                sample_ohlcv_data['close'].iloc[i-1]
            ) / sample_ohlcv_data['close'].iloc[i-1]
            assert abs(result.iloc[i, 0] - expected_return) < 1e-10
            
            # Check lag values
            assert result.iloc[i, 2] == sample_ohlcv_data['close'].iloc[i-1]
            assert result.iloc[i, 3] == sample_ohlcv_data['close'].iloc[i-2]
        
        # Verify volatility features are calculated for OHLCV data
        # Rolling std should be calculated
        assert not np.isnan(result.iloc[15, 4])  # rolling_std_10
        # ATR should be calculated for OHLCV data
        assert not np.isnan(result.iloc[20, 5])  # atr_14
    
    def test_empty_data(self):
        """Test handling of empty data."""
        config = {'features': ['returns']}
        features = BasicFeatures(config)
        
        empty_data = pd.DataFrame({'price': []})
        result = features.calculate(empty_data)
        
        assert result.shape[0] == 0
        assert result.shape[1] == 1
    
    def test_single_row_data(self):
        """Test handling of single row data."""
        config = {'features': ['returns', 'lags']}
        features = BasicFeatures(config)
        
        single_row = pd.DataFrame({'price': [50000.0]})
        result = features.calculate(single_row)
        
        assert result.shape[0] == 1
        assert result.shape[1] == 2
        # Both should be NaN for single row
        assert np.isnan(result.iloc[0, 0])  # return
        assert np.isnan(result.iloc[0, 1])  # lag
    
    def test_constant_prices(self):
        """Test handling of constant prices."""
        config = {'features': ['returns', 'log_returns']}
        features = BasicFeatures(config)
        
        constant_data = pd.DataFrame({
            'price': [50000.0] * 10
        })
        
        result = features.calculate(constant_data)
        
        # Returns should be zero for constant prices
        assert result.shape[0] == 10
        assert result.shape[1] == 2
        
        for i in range(1, 10):
            assert abs(result.iloc[i, 0]) < 1e-10  # return should be ~0
            assert abs(result.iloc[i, 1]) < 1e-10  # log return should be ~0
    
    def test_negative_prices(self):
        """Test handling of negative prices (should not occur in reality but test robustness)."""
        config = {'features': ['returns']}
        features = BasicFeatures(config)
        
        # Mix of positive and negative prices
        price_data = pd.DataFrame({
            'price': [100.0, -50.0, 25.0, -12.5, 6.25]
        })
        
        result = features.calculate(price_data)
        
        assert result.shape[0] == 5
        assert result.shape[1] == 1
        
        # Check return calculations
        expected_return_1 = (-50.0 - 100.0) / 100.0  # -1.5
        assert abs(result.iloc[1, 0] - expected_return_1) < 1e-10
    
    def test_zero_prices(self):
        """Test handling of zero prices."""
        config = {'features': ['returns', 'log_returns']}
        features = BasicFeatures(config)
        
        # Data with zero price
        price_data = pd.DataFrame({
            'price': [100.0, 0.0, 50.0, 25.0]
        })
        
        result = features.calculate(price_data)
        
        assert result.shape[0] == 4
        assert result.shape[1] == 2
        
        # Return with zero denominator should be handled
        assert np.isnan(result.iloc[1, 0])  # return with zero denominator
        assert np.isnan(result.iloc[1, 1])  # log return with zero
    
    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        config = {
            'features': ['returns', 'log_returns', 'lags'],
            'return_periods': [1, 5, 15],
            'lag_periods': [1, 2, 3, 5, 10]
        }
        features = BasicFeatures(config)
        
        # Create larger dataset
        np.random.seed(42)
        n = 10000
        prices = [50000.0]
        for i in range(1, n):
            change = np.random.normal(0, 0.001)
            prices.append(prices[-1] * (1 + change))
        
        large_data = pd.DataFrame({'price': prices})
        
        # Should complete without errors
        result = features.calculate(large_data)
        
        # Should have: returns(3) + log_returns(3) + lags(5) = 11 features
        assert result.shape[0] == n
        assert result.shape[1] == 11
    
    def test_feature_consistency(self, sample_price_data):
        """Test that features are consistent across multiple calls."""
        config = {
            'features': ['returns', 'log_returns'],
            'return_periods': [1, 5],
            'price_column': 'price'
        }
        features = BasicFeatures(config)
        
        result1 = features.calculate(sample_price_data)
        result2 = features.calculate(sample_price_data)
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_invalid_feature_config(self):
        """Test handling of invalid feature configuration."""
        # Test with empty features list
        config = {'features': []}
        features = BasicFeatures(config)
        
        result = features.calculate(pd.DataFrame({'price': [1, 2, 3]}))
        assert result.shape[1] == 0  # No features calculated
    
    def test_missing_price_column(self):
        """Test handling of missing price column."""
        config = {'features': ['returns'], 'price_column': 'missing_col'}
        features = BasicFeatures(config)
        
        data = pd.DataFrame({'price': [1, 2, 3]})
        
        # Should raise an error for missing column
        with pytest.raises(KeyError):
            features.calculate(data)
    
    def test_feature_names_consistency(self, sample_price_data):
        """Test that feature names match result columns."""
        config = {
            'features': ['returns', 'log_returns', 'lags'],
            'return_periods': [1, 3],
            'lag_periods': [1, 2]
        }
        features = BasicFeatures(config)
        
        result = features.calculate(sample_price_data)
        names = features.get_feature_names()
        
        assert len(names) == result.shape[1]
        
        # Check that names are in the right order
        expected_order = ['return_1', 'return_3', 'log_return_1', 'log_return_3', 'lag_1', 'lag_2']
        assert names == expected_order
    
    def test_calculate_rolling_std(self, sample_price_data):
        """Test rolling standard deviation calculation."""
        config = {
            'features': ['volatility'],
            'volatility_windows': [10, 20],
            'price_column': 'price'
        }
        features = BasicFeatures(config)
        
        result = features.calculate(sample_price_data)
        
        # Should have 2 rolling std features (10-window and 20-window)
        assert result.shape[0] == len(sample_price_data)
        assert result.shape[1] == 2
        
        # Check that early rows are NaN for rolling windows
        assert np.isnan(result.iloc[0, 0])  # 10-window std at row 0
        assert np.isnan(result.iloc[9, 0])  # 10-window std at row 9
        assert not np.isnan(result.iloc[10, 0])  # 10-window std at row 10
        
        assert np.isnan(result.iloc[0, 1])  # 20-window std at row 0
        assert np.isnan(result.iloc[19, 1])  # 20-window std at row 19
        assert not np.isnan(result.iloc[20, 1])  # 20-window std at row 20
        
        # Verify rolling std calculation manually for a specific row
        window = 10
        manual_std = sample_price_data['price'].iloc[10-window:10].std()
        assert abs(result.iloc[10, 0] - manual_std) < 1e-10
    
    def test_calculate_atr(self, sample_ohlcv_data):
        """Test Average True Range (ATR) calculation."""
        config = {
            'features': ['volatility'],
            'atr_periods': [14],
            'price_column': 'close'
        }
        features = BasicFeatures(config)
        
        result = features.calculate(sample_ohlcv_data)
        
        # Should have 1 ATR feature (14-period)
        assert result.shape[0] == len(sample_ohlcv_data)
        assert result.shape[1] == 1
        
        # Check that early rows are NaN for ATR
        assert np.isnan(result.iloc[0, 0])  # ATR at row 0
        assert np.isnan(result.iloc[13, 0])  # ATR at row 13
        assert not np.isnan(result.iloc[14, 0])  # ATR at row 14
        
        # Verify ATR calculation manually for a specific row
        # ATR = average of true ranges over period
        # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        period = 14
        true_ranges = []
        for i in range(1, period + 1):
            high = sample_ohlcv_data['high'].iloc[i]
            low = sample_ohlcv_data['low'].iloc[i]
            prev_close = sample_ohlcv_data['close'].iloc[i-1]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        manual_atr = np.mean(true_ranges)
        assert abs(result.iloc[14, 0] - manual_atr) < 1e-10
    
    def test_calculate_all_volatility_features(self, sample_ohlcv_data):
        """Test calculating all volatility features."""
        config = {
            'features': ['volatility'],
            'volatility_windows': [10, 20],
            'atr_periods': [14, 21],
            'price_column': 'close'
        }
        features = BasicFeatures(config)
        
        result = features.calculate(sample_ohlcv_data)
        
        # Should have: rolling_std(2) + atr(2) = 4 features
        assert result.shape[0] == len(sample_ohlcv_data)
        assert result.shape[1] == 4
        assert not np.any(np.isnan(result.iloc[25:]))  # All non-NaN after sufficient data
    
    def test_volatility_with_price_data(self, sample_price_data):
        """Test volatility features with simple price data (no OHLCV)."""
        config = {
            'features': ['volatility'],
            'volatility_windows': [10],
            'atr_periods': [14],  # Should be ignored for price-only data
            'price_column': 'price'
        }
        features = BasicFeatures(config)
        
        result = features.calculate(sample_price_data)
        
        # Should only have rolling std (1 feature) since ATR requires OHLCV
        assert result.shape[0] == len(sample_price_data)
        assert result.shape[1] == 1
        
        # Verify rolling std is calculated
        assert not np.isnan(result.iloc[10, 0])
    
    def test_volatility_feature_names(self, sample_ohlcv_data):
        """Test volatility feature naming."""
        config = {
            'features': ['volatility'],
            'volatility_windows': [10, 20],
            'atr_periods': [14],
            'price_column': 'close'
        }
        features = BasicFeatures(config)
        
        features.calculate(sample_ohlcv_data)
        names = features.get_feature_names()
        
        expected_names = ['rolling_std_10', 'rolling_std_20', 'atr_14']
        assert len(names) == 3
        for name in expected_names:
            assert name in names
    
    def test_volatility_edge_cases(self):
        """Test volatility features with edge cases."""
        config = {
            'features': ['volatility'],
            'volatility_windows': [5],
            'atr_periods': [5],
            'price_column': 'price'
        }
        features = BasicFeatures(config)
        
        # Test with constant prices (zero volatility)
        constant_data = pd.DataFrame({
            'price': [50000.0] * 10,
            'high': [50000.0] * 10,
            'low': [50000.0] * 10,
            'close': [50000.0] * 10
        })
        
        result = features.calculate(constant_data)
        
        # Rolling std should be zero for constant prices
        assert result.shape[1] == 2  # rolling_std + atr
        assert abs(result.iloc[5, 0]) < 1e-10  # rolling_std should be ~0
        assert abs(result.iloc[5, 1]) < 1e-10  # ATR should be ~0
    
    def test_volatility_with_all_features(self, sample_ohlcv_data):
        """Test volatility features combined with other basic features."""
        config = {
            'features': ['returns', 'log_returns', 'lags', 'volatility'],
            'return_periods': [1],
            'lag_periods': [1],
            'volatility_windows': [10],
            'atr_periods': [14],
            'price_column': 'close'
        }
        features = BasicFeatures(config)
        
        result = features.calculate(sample_ohlcv_data)
        
        # Should have: returns(1) + log_returns(1) + lags(1) + volatility(2) = 5 features
        assert result.shape[0] == len(sample_ohlcv_data)
        assert result.shape[1] == 5
        
        # Verify feature names
        names = features.get_feature_names()
        expected_names = ['return_1', 'log_return_1', 'lag_1', 'rolling_std_10', 'atr_14']
        assert names == expected_names
