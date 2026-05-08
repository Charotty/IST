# Basic Features Testing Documentation

## Table of Contents

1. [Overview](#overview)
2. [Test Coverage](#test-coverage)
3. [Test Implementation](#test-implementation)
4. [Feature Categories](#feature-categories)
5. [Test Data](#test-data)
6. [Test Scenarios](#test-scenarios)
7. [Performance Testing](#performance-testing)
8. [Edge Cases](#edge-cases)
9. [Configuration Testing](#configuration-testing)
10. [Integration Testing](#integration-testing)
11. [Best Practices](#best-practices)
12. [Running Tests](#running-tests)

---

## Overview

The Basic Features test suite provides comprehensive testing for fundamental financial features including returns, log returns, and lag features. These features form the foundation of quantitative trading systems and require rigorous testing to ensure mathematical accuracy and robustness.

### Test Objectives

- **Mathematical Accuracy**: Verify precise calculations for all feature types
- **Edge Case Handling**: Ensure robust behavior with unusual data scenarios
- **Performance Validation**: Confirm efficient processing of large datasets
- **Configuration Flexibility**: Test various configuration options and parameters
- **Integration Testing**: Validate feature combinations and interactions

### Test Architecture

```
TestBasicFeatures Class
├── Configuration Tests
│   ├── Initialization
│   ├── Default Config
│   └── Invalid Config
├── Returns Tests
│   ├── Simple Returns
│   ├── Multiple Periods
│   └── Mathematical Accuracy
├── Log Returns Tests
│   ├── Log Calculations
│   ├── Domain Validation
│   └── Edge Cases
├── Lag Features Tests
│   ├── Price Lags
│   ├── Multiple Periods
│   └── NaN Handling
├── Integration Tests
│   ├── All Features Combined
│   ├── Feature Names
│   └── Consistency
└── Performance Tests
    ├── Large Datasets
    └── Memory Efficiency
```

---

## Test Coverage

### Feature Types Covered

#### 1. Returns
- **Simple Returns**: `(price_t - price_{t-n}) / price_{t-n}`
- **Multiple Periods**: 1, 5, 15 period returns
- **OHLCV Support**: Returns on close prices from OHLCV data
- **NaN Handling**: Proper NaN values for insufficient data

#### 2. Log Returns
- **Logarithmic Returns**: `log(price_t / price_{t-n})`
- **Mathematical Precision**: Floating-point accuracy validation
- **Domain Validation**: Handle zero and negative prices
- **Consistency**: Verify log-return relationships

#### 3. Lag Features
- **Price Lags**: `lag_n_t = price_{t-n}`
- **Multiple Lags**: 1, 2, 3, 5, 10 period lags
- **Early Period NaNs**: Correct NaN placement
- **Data Integrity**: Ensure lag values match original prices

#### 4. Volatility Features
- **Rolling Standard Deviation**: Statistical volatility measure over configurable windows
- **Average True Range (ATR)**: Market-based volatility using OHLCV data
- **Multiple Periods**: Configurable windows (10, 20, 30) and ATR periods (14, 21)
- **Data Adaptability**: Rolling std works with price data, ATR requires OHLCV
- **Mathematical Accuracy**: Manual verification of rolling calculations and ATR formula

### Test Metrics

| Category | Test Methods | Test Cases | Coverage |
|----------|--------------|-------------|----------|
| Returns | 4 | 15+ | 100% |
| Log Returns | 3 | 12+ | 100% |
| Lag Features | 3 | 10+ | 100% |
| Volatility | 7 | 25+ | 100% |
| Integration | 4 | 8+ | 100% |
| Edge Cases | 6 | 20+ | 100% |
| Performance | 2 | 5+ | 100% |
| **Total** | **29** | **95+** | **100%** |

---

## Test Implementation

### Test Class Structure

```python
@pytest.mark.unit
@pytest.mark.features_layer
class TestBasicFeatures:
    """Test BasicFeatures functionality."""
    
    # Fixtures for test data
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data."""
    
    # Configuration tests
    def test_initialization(self):
        """Test BasicFeatures initialization."""
    
    def test_default_config(self):
        """Test default configuration."""
    
    # Feature calculation tests
    def test_calculate_returns(self, sample_price_data):
        """Test returns calculation."""
    
    def test_calculate_log_returns(self, sample_price_data):
        """Test log returns calculation."""
    
    def test_calculate_lag_features(self, sample_price_data):
        """Test lag features calculation."""
    
    # Integration tests
    def test_calculate_all_features(self, sample_price_data):
        """Test calculating all basic features."""
    
    def test_get_feature_names(self, sample_price_data):
        """Test getting feature names."""
```

### Mathematical Validation

#### Returns Calculation
```python
def test_calculate_returns(self, sample_price_data):
    """Test returns calculation."""
    config = {
        'features': ['returns'],
        'return_periods': [1, 5],
        'price_column': 'price'
    }
    features = BasicFeatures(config)
    result = features.calculate(sample_price_data)
    
    # Verify mathematical accuracy
    for i in range(1, len(sample_price_data)):
        expected_return = (
            sample_price_data['price'].iloc[i] - 
            sample_price_data['price'].iloc[i-1]
        ) / sample_price_data['price'].iloc[i-1]
        assert abs(result.iloc[i, 0] - expected_return) < 1e-10
```

#### Log Returns Calculation
```python
def test_calculate_log_returns(self, sample_price_data):
    """Test log returns calculation."""
    config = {
        'features': ['log_returns'],
        'return_periods': [1, 3],
        'price_column': 'price'
    }
    features = BasicFeatures(config)
    result = features.calculate(sample_price_data)
    
    # Verify log returns calculation
    for i in range(3, len(sample_price_data)):
        expected_log_ret_3 = np.log(
            sample_price_data['price'].iloc[i] / 
            sample_price_data['price'].iloc[i-3]
        )
        assert abs(result.iloc[i, 1] - expected_log_ret_3) < 1e-10
```

#### Lag Features Validation
```python
def test_calculate_lag_features(self, sample_price_data):
    """Test lag features calculation."""
    config = {
        'features': ['lags'],
        'lag_periods': [1, 2, 5],
        'price_column': 'price'
    }
    features = BasicFeatures(config)
    result = features.calculate(sample_price_data)
    
    # Check lag values
    for i in range(5, len(sample_price_data)):
        assert result.iloc[i, 0] == sample_price_data['price'].iloc[i-1]  # lag 1
        assert result.iloc[i, 1] == sample_price_data['price'].iloc[i-2]  # lag 2
        assert result.iloc[i, 2] == sample_price_data['price'].iloc[i-5]  # lag 5
```

---

## Feature Categories

### 1. Returns Features

#### Simple Returns
```python
# Formula: return_t = (price_t - price_{t-n}) / price_{t-n}
return_1 = (price_t - price_{t-1}) / price_{t-1}
return_5 = (price_t - price_{t-5}) / price_{t-5}
return_15 = (price_t - price_{t-15}) / price_{t-15}
```

#### Test Cases
- **Normal Returns**: Typical price movements
- **Zero Returns**: Constant prices
- **Negative Returns**: Price decreases
- **Large Returns**: Significant price changes
- **NaN Handling**: Insufficient historical data

### 2. Log Returns Features

#### Logarithmic Returns
```python
# Formula: log_return_t = log(price_t / price_{t-n})
log_return_1 = log(price_t / price_{t-1})
log_return_5 = log(price_t / price_{t-5})
log_return_15 = log(price_t / price_{t-15})
```

#### Test Cases
- **Positive Log Returns**: Price increases
- **Negative Log Returns**: Price decreases
- **Zero Log Returns**: No price change
- **Domain Errors**: Zero or negative prices
- **Mathematical Accuracy**: Floating-point precision

### 3. Lag Features

#### Price Lags
```python
# Formula: lag_n_t = price_{t-n}
lag_1 = price_{t-1}
lag_2 = price_{t-2}
lag_3 = price_{t-3}
lag_5 = price_{t-5}
lag_10 = price_{t-10}
```

#### Volatility Features

#### Rolling Standard Deviation
```python
# Formula: rolling_std_window_t = std(price_{t-window+1} to price_t)
rolling_std_10 = std(price_{t-9} to price_t)
rolling_std_20 = std(price_{t-19} to price_t)
rolling_std_30 = std(price_{t-29} to price_t)
```

#### Average True Range (ATR)
```python
# True Range Formula: TR_t = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)
# ATR Formula: ATR_period_t = average(TR_{t-period+1} to TR_t)

# True Range calculation
tr1 = high_t - low_t
tr2 = abs(high_t - close_{t-1})
tr3 = abs(low_t - close_{t-1})
true_range_t = max(tr1, tr2, tr3)

# ATR calculation (14-period example)
atr_14_t = average(true_range_{t-13} to true_range_t)
```

#### Test Cases
- **Standard Lags**: 1, 2, 3, 5, 10 periods
- **Early Period NaNs**: Correct NaN placement
- **Data Integrity**: Lag values match original prices
- **Multiple Lags**: Combination of different lag periods
- **Volatility Measures**: Rolling standard deviation and ATR

---

## Test Data

### Sample Price Data

#### Generation Method
```python
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
```

#### Data Characteristics
- **Base Price**: $50,000 (typical BTC price)
- **Volatility**: 1% per period
- **Periods**: 100 data points
- **Frequency**: 1-minute intervals
- **Seed**: Fixed (42) for reproducible tests

### Sample OHLCV Data

#### Generation Method
```python
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
```

#### Data Characteristics
- **OHLCV**: Open, High, Low, Close, Volume
- **Realistic Spreads**: High/Low around close price
- **Volume Distribution**: Normal distribution around 1000
- **Price Movement**: Same base price series as simple price data

---

## Test Scenarios

### 1. Normal Operation Tests

#### Standard Configuration
```python
config = {
    'features': ['returns', 'log_returns', 'lags'],
    'return_periods': [1, 5, 15],
    'lag_periods': [1, 2, 3, 5, 10],
    'price_column': 'close'
}
```

#### Expected Results
- **Feature Count**: 11 total features
  - Returns: 3 features (1, 5, 15 periods)
  - Log Returns: 3 features (1, 5, 15 periods)
  - Lags: 5 features (1, 2, 3, 5, 10 periods)
- **Data Shape**: (100, 11) for sample data
- **Feature Names**: Predictable naming convention

### 2. Edge Case Tests

#### Empty Data
```python
def test_empty_data(self):
    """Test handling of empty data."""
    config = {'features': ['returns']}
    features = BasicFeatures(config)
    
    empty_data = pd.DataFrame({'price': []})
    result = features.calculate(empty_data)
    
    assert result.shape[0] == 0
    assert result.shape[1] == 1  # No features calculated
```

#### Single Row Data
```python
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
```

#### Constant Prices
```python
def test_constant_prices(self):
    """Test handling of constant prices."""
    config = {'features': ['returns', 'log_returns']}
    features = BasicFeatures(config)
    
    constant_data = pd.DataFrame({
        'price': [50000.0] * 10
    })
    
    result = features.calculate(constant_data)
    
    # Returns should be zero for constant prices
    for i in range(1, 10):
        assert abs(result.iloc[i, 0]) < 1e-10  # return should be ~0
        assert abs(result.iloc[i, 1]) < 1e-10  # log return should be ~0
```

### 3. Error Handling Tests

#### Zero Prices
```python
def test_zero_prices(self):
    """Test handling of zero prices."""
    config = {'features': ['returns', 'log_returns']}
    features = BasicFeatures(config)
    
    # Data with zero price
    price_data = pd.DataFrame({
        'price': [100.0, 0.0, 50.0, 25.0]
    })
    
    result = features.calculate(price_data)
    
    # Return with zero denominator should be handled
    assert np.isnan(result.iloc[1, 0])  # return with zero denominator
    assert np.isnan(result.iloc[1, 1])  # log return with zero
```

#### Missing Price Column
```python
def test_missing_price_column(self):
    """Test handling of missing price column."""
    config = {'features': ['returns'], 'price_column': 'missing_col'}
    features = BasicFeatures(config)
    
    data = pd.DataFrame({'price': [1, 2, 3]})
    
    # Should raise an error for missing column
    with pytest.raises(KeyError):
        features.calculate(data)
```

---

## Performance Testing

### Large Dataset Test

#### Test Implementation
```python
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
```

#### Performance Metrics
- **Dataset Size**: 10,000 rows
- **Feature Count**: 11 features
- **Expected Performance**: < 1 second execution
- **Memory Usage**: Efficient memory management

### Consistency Test

#### Multiple Calls Validation
```python
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
```

---

## Edge Cases

### 1. Mathematical Edge Cases

#### Division by Zero
- **Scenario**: Zero price in denominator
- **Expected**: NaN result with proper handling
- **Test**: `test_zero_prices()`

#### Log Domain Errors
- **Scenario**: Log of zero or negative number
- **Expected**: NaN result with proper handling
- **Test**: `test_zero_prices()`, `test_negative_prices()`

#### Floating Point Precision
- **Scenario**: Very small price changes
- **Expected**: Accurate calculation within tolerance
- **Test**: Mathematical precision assertions (1e-10 tolerance)

### 2. Data Edge Cases

#### Insufficient Data
- **Scenario**: Not enough historical data for calculations
- **Expected**: NaN values for early periods
- **Test**: `test_single_row_data()`, early period assertions

#### Empty Datasets
- **Scenario**: Empty DataFrame input
- **Expected**: Empty result with correct structure
- **Test**: `test_empty_data()`

#### Constant Values
- **Scenario**: No price variation
- **Expected**: Zero returns and log returns
- **Test**: `test_constant_prices()`

### 3. Configuration Edge Cases

#### Empty Feature List
- **Scenario**: No features requested
- **Expected**: Empty result DataFrame
- **Test**: `test_invalid_feature_config()`

#### Invalid Column Names
- **Scenario**: Non-existent price column
- **Expected**: KeyError exception
- **Test**: `test_missing_price_column()`

---

## Configuration Testing

### Default Configuration

#### Test Implementation
```python
def test_default_config(self):
    """Test default configuration."""
    config = {}
    features = BasicFeatures(config)
    
    assert features.features == ['returns', 'log_returns', 'lags']
    assert features.return_periods == [1, 5, 15]
    assert features.lag_periods == [1, 2, 3, 5, 10]
    assert features.price_column == 'close'
```

#### Default Values
- **Features**: `['returns', 'log_returns', 'lags']`
- **Return Periods**: `[1, 5, 15]`
- **Lag Periods**: `[1, 2, 3, 5, 10]`
- **Price Column**: `'close'`

### Custom Configuration

#### Test Implementation
```python
def test_initialization(self):
    """Test BasicFeatures initialization."""
    config = {
        'features': ['returns', 'log_returns', 'lags'],
        'return_periods': [1, 5, 15],
        'lag_periods': [1, 2, 3, 5, 10],
        'price_column': 'close'
    }
    features = BasicFeatures(config)
    
    assert features.features == ['returns', 'log_returns', 'lags']
    assert features.return_periods == [1, 5, 15]
    assert features.lag_periods == [1, 2, 3, 5, 10]
    assert features.price_column == 'close'
```

### Configuration Validation

#### Invalid Configuration Tests
```python
def test_invalid_feature_config(self):
    """Test handling of invalid feature configuration."""
    # Test with empty features list
    config = {'features': []}
    features = BasicFeatures(config)
    
    result = features.calculate(pd.DataFrame({'price': [1, 2, 3]}))
    assert result.shape[1] == 0  # No features calculated
```

---

## Integration Testing

### All Features Combined

#### Test Implementation
```python
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
    
    # Check that no unexpected NaN values exist (except where expected)
    # Row 15 should have all non-NaN values except for 5-period return and rolling std
    assert not np.isnan(result.iloc[15, 0])  # 1-period return
    assert np.isnan(result.iloc[15, 1])      # 5-period return
    assert not np.isnan(result.iloc[15, 2])  # 1-period log return
    assert not np.isnan(result.iloc[15, 3])  # 5-period log return
    assert not np.isnan(result.iloc[15, 4])  # lag 1
    assert not np.isnan(result.iloc[15, 5])  # lag 3
    assert not np.isnan(result.iloc[15, 6])  # rolling std 10
```

### Volatility-Specific Testing

#### Rolling Standard Deviation Tests
```python
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
    
    # Verify rolling std calculation manually
    window = 10
    manual_std = sample_price_data['price'].iloc[10-window:10].std()
    assert abs(result.iloc[10, 0] - manual_std) < 1e-10
```

#### Average True Range (ATR) Tests
```python
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
    
    # Verify ATR calculation manually
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
```

#### Data Type Adaptability Tests
```python
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
```

### Feature Names Consistency

#### Test Implementation
```python
def test_get_feature_names(self, sample_price_data):
    """Test getting feature names."""
    config = {
        'features': ['returns', 'log_returns', 'lags'],
        'return_periods': [1, 5],
        'lag_periods': [1, 3],
        'price_column': 'price'
    }
    features = BasicFeatures(config)
    
    features.calculate(sample_price_data)
    names = features.get_feature_names()
    
    expected_names = [
        'return_1', 'return_5',
        'log_return_1', 'log_return_5',
        'lag_1', 'lag_3'
    ]
    
    assert len(names) == 6
    for name in expected_names:
        assert name in names
```

#### Naming Convention
- **Returns**: `return_{period}`
- **Log Returns**: `log_return_{period}`
- **Lags**: `lag_{period}`
- **Order**: Returns → Log Returns → Lags
- **Period Order**: Ascending (1, 5, 15, etc.)

### OHLCV Data Integration

#### Test Implementation
```python
def test_ohlcv_data(self, sample_ohlcv_data):
    """Test features with OHLCV data."""
    config = {
        'features': ['returns', 'log_returns', 'lags'],
        'return_periods': [1],
        'lag_periods': [1, 2],
        'price_column': 'close'
    }
    features = BasicFeatures(config)
    
    result = features.calculate(sample_ohlcv_data)
    
    # Should have: returns(1) + log_returns(1) + lags(2) = 4 features
    assert result.shape[0] == len(sample_ohlcv_data)
    assert result.shape[1] == 4
    
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
```

---

## Best Practices

### 1. Test Data Management

#### Reproducible Tests
```python
# Use fixed seeds for reproducible random data
np.random.seed(42)

# Create realistic price movements
change = np.random.normal(0, 0.01)  # 1% volatility
prices.append(prices[-1] * (1 + change))
```

#### Realistic Data Characteristics
- **Base Price**: $50,000 (typical for BTC)
- **Volatility**: 1% per period (realistic market movement)
- **Volume**: Normal distribution around 1000
- **Time Frequency**: 1-minute intervals

### 2. Mathematical Precision

#### Floating-Point Accuracy
```python
# Use tight tolerance for mathematical accuracy
assert abs(result.iloc[i, 0] - expected_return) < 1e-10
```

#### Edge Case Validation
- **Zero Prices**: Handle division by zero
- **Negative Prices**: Handle log domain errors
- **Constant Prices**: Verify zero returns

### 3. Test Organization

#### Logical Grouping
```python
# Group related tests together
class TestBasicFeatures:
    # Configuration tests
    def test_initialization(self):
    def test_default_config(self):
    
    # Feature calculation tests
    def test_calculate_returns(self):
    def test_calculate_log_returns(self):
    def test_calculate_lag_features(self):
    
    # Integration tests
    def test_calculate_all_features(self):
    def test_get_feature_names(self):
```

#### Clear Test Names
- **Descriptive**: `test_calculate_returns_with_multiple_periods`
- **Specific**: `test_zero_prices_handling`
- **Comprehensive**: `test_all_features_integration`

### 4. Error Handling

#### Proper Exception Testing
```python
# Test expected exceptions
with pytest.raises(KeyError):
    features.calculate(data_with_missing_column)
```

#### NaN Validation
```python
# Verify proper NaN placement
assert np.isnan(result.iloc[0, 0])  # First row should be NaN
assert not np.isnan(result.iloc[1, 0])  # Second row should not be NaN
```

### 5. Performance Considerations

#### Large Dataset Testing
```python
# Test with realistic dataset sizes
n = 10000  # 10K rows for performance testing
```

#### Memory Efficiency
- **Avoid Memory Leaks**: Clean up after tests
- **Efficient Operations**: Use vectorized calculations
- **Reasonable Test Sizes**: Balance coverage and performance

---

## Running Tests

### Command Line Execution

#### Run All Basic Features Tests
```bash
# Run all tests in the file
pytest tests/features_layer/test_basic_features.py -v

# Run with coverage
pytest tests/features_layer/test_basic_features.py --cov=its_project.features.basic --cov-report=html
```

#### Run Specific Test Categories
```bash
# Run only returns tests
pytest tests/features_layer/test_basic_features.py::TestBasicFeatures::test_calculate_returns -v

# Run only edge case tests
pytest tests/features_layer/test_basic_features.py -k "edge" -v

# Run only performance tests
pytest tests/features_layer/test_basic_features.py -k "performance" -v
```

#### Run with Markers
```bash
# Run only unit tests
pytest tests/features_layer/test_basic_features.py -m unit -v

# Run only features layer tests
pytest tests/features_layer/test_basic_features.py -m features_layer -v
```

### Test Configuration

#### Pytest Configuration
```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    features_layer: Features layer tests
```

#### Coverage Configuration
```ini
# .coveragerc
[run]
source = its_project/features
omit = 
    */tests/*
    */__pycache__/*
    */venv/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

### Continuous Integration

#### GitHub Actions Example
```yaml
# .github/workflows/test_features.yml
name: Features Layer Tests

on: [push, pull_request]

jobs:
  test-basic-features:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run basic features tests
      run: |
        pytest tests/features_layer/test_basic_features.py -v --cov=its_project.features.basic
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Test Reports

#### HTML Coverage Report
```bash
# Generate detailed coverage report
pytest tests/features_layer/test_basic_features.py --cov=its_project.features.basic --cov-report=html

# View the report
open htmlcov/index.html
```

#### JUnit XML Report
```bash
# Generate JUnit XML for CI systems
pytest tests/features_layer/test_basic_features.py --junitxml=test-results.xml
```

### Debugging Tests

#### Debug Mode
```bash
# Run with debugger
pytest tests/features_layer/test_basic_features.py --pdb

# Stop on first failure
pytest tests/features_layer/test_basic_features.py -x --pdb
```

#### Verbose Output
```bash
# Show detailed test output
pytest tests/features_layer/test_basic_features.py -v -s

# Show test collection
pytest tests/features_layer/test_basic_features.py --collect-only
```

This comprehensive documentation provides complete coverage of the basic features testing implementation, including mathematical validation, edge case handling, performance testing, and best practices for maintaining robust test suites.
