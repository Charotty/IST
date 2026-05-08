# OrderBook Extended Features Documentation

## Table of Contents

1. [Overview](#overview)
2. [Extended Features](#extended-features)
3. [Order Flow Imbalance (OFI)](#order-flow-imbalance-ofi)
4. [Depth Imbalance](#depth-imbalance)
5. [Microprice](#microprice)
6. [Mathematical Formulations](#mathematical-formulations)
7. [Test Implementation](#test-implementation)
8. [Feature Integration](#feature-integration)
9. [Configuration Options](#configuration-options)
10. [Use Cases](#use-cases)
11. [Performance Considerations](#performance-considerations)
12. [Best Practices](#best-practices)

---

## Overview

The OrderBook Extended Features module provides advanced market microstructure indicators for high-frequency trading and quantitative analysis. These features extend the basic orderbook metrics with sophisticated measures of order flow, depth imbalance, and price discovery mechanisms.

### Extended Features Summary

| Feature | Description | Range | Use Case |
|---------|-------------|-------|----------|
| **OFI** | Order Flow Imbalance at multiple depth levels | [-∞, +∞] | Order flow analysis, price prediction |
| **Depth Imbalance** | Volume-based depth imbalance | [-1, +1] | Liquidity assessment, market pressure |
| **Microprice** | Volume-weighted mid price | [bid, ask] | Price discovery, execution optimization |

---

## Extended Features

### Feature Categories

#### 1. Order Flow Imbalance (OFI)
- **Multi-Level Analysis**: OFI calculated at configurable depth levels
- **Flow Direction**: Positive = buying pressure, Negative = selling pressure
- **Temporal Dynamics**: Captures order flow changes over time
- **Predictive Power**: Strong predictor of short-term price movements

#### 2. Depth Imbalance
- **Volume-Based**: Uses total volume at multiple depth levels
- **Normalized Range**: [-1, 1] for consistent interpretation
- **Liquidity Metric**: Measures market depth and balance
- **Risk Indicator**: Identifies liquidity stress scenarios

#### 3. Microprice
- **Volume-Weighted**: Reflects true market price pressure
- **Price Discovery**: More accurate than simple mid-price
- **Execution Optimization**: Better execution price estimation
- **Market Efficiency**: Measures price efficiency and discovery

---

## Order Flow Imbalance (OFI)

### Concept

Order Flow Imbalance measures the net flow of orders at different depth levels in the orderbook. It captures the imbalance between buy and sell pressure by analyzing changes in order volumes at each price level.

### Mathematical Definition

```python
# OFI at level i: OFI_i = ΔBidVolume_i - ΔAskVolume_i
# Where Δ represents change from previous timestep

def calculate_ofi(bids_current, asks_current, bids_previous, asks_previous, level):
    """
    Calculate Order Flow Imbalance at specified depth level.
    
    Args:
        bids_current: Current bid levels [[price, volume], ...]
        asks_current: Current ask levels [[price, volume], ...]
        bids_previous: Previous bid levels [[price, volume], ...]
        asks_previous: Previous ask levels [[price, volume], ...]
        level: Depth level (0 = best bid/ask)
    
    Returns:
        ofi: Order flow imbalance at specified level
    """
    # Volume changes at bid side (new orders = positive, cancellations = negative)
    bid_volume_change = bids_current[level][1] - bids_previous[level][1]
    
    # Volume changes at ask side (new orders = positive, cancellations = negative)
    ask_volume_change = asks_current[level][1] - asks_previous[level][1]
    
    # OFI = bid flow - ask flow
    ofi = bid_volume_change - ask_volume_change
    
    return ofi
```

### Interpretation

| OFI Value | Interpretation | Market Implication |
|-----------|----------------|-------------------|
| **> 0** | Net buying pressure | Price likely to increase |
| **< 0** | Net selling pressure | Price likely to decrease |
| **≈ 0** | Balanced flow | Price likely stable |

### Multi-Level Analysis

```python
# Calculate OFI at multiple depth levels
ofi_levels = []
for level in range(depth_levels):
    ofi = calculate_ofi(bids_current, asks_current, 
                       bids_previous, asks_previous, level)
    ofi_levels.append(ofi)

# Feature names: ['ofi_0', 'ofi_1', 'ofi_2', ...]
```

---

## Depth Imbalance

### Concept

Depth Imbalance measures the relative volume imbalance between buy and sell sides of the orderbook across multiple depth levels. It provides a normalized measure of market pressure and liquidity distribution.

### Mathematical Definition

```python
def calculate_depth_imbalance(bids, asks, levels):
    """
    Calculate depth imbalance across specified levels.
    
    Args:
        bids: Bid levels [[price, volume], ...]
        asks: Ask levels [[price, volume], ...]
        levels: Number of depth levels to consider
    
    Returns:
        imbalance: Normalized depth imbalance [-1, 1]
    """
    # Sum volumes across specified levels
    bid_volume = sum(bids[i][1] for i in range(min(levels, len(bids))))
    ask_volume = sum(asks[i][1] for i in range(min(levels, len(asks))))
    
    total_volume = bid_volume + ask_volume
    
    if total_volume == 0:
        return 0.0  # No liquidity
    
    # Normalize to [-1, 1] range
    imbalance = (bid_volume - ask_volume) / total_volume
    
    return imbalance
```

### Interpretation

| Imbalance Value | Market State | Trading Implications |
|----------------|--------------|---------------------|
| **> 0.5** | Strong buy-side depth | Potential upward pressure, good for selling |
| **< -0.5** | Strong sell-side depth | Potential downward pressure, good for buying |
| **[-0.2, 0.2]** | Balanced depth | Stable market, neutral strategy |
| **≈ ±1** | One-sided market | High volatility risk, careful execution |

### Level Sensitivity

```python
# Different levels capture different market dynamics
imbalance_1 = calculate_depth_imbalance(bids, asks, 1)   # Best levels only
imbalance_5 = calculate_depth_imbalance(bids, asks, 5)   # Top 5 levels
imbalance_10 = calculate_depth_imbalance(bids, asks, 10) # Top 10 levels

# Higher levels = more stable but less responsive
# Lower levels = more responsive but more noisy
```

---

## Microprice

### Concept

Microprice is a volume-weighted price that reflects the true market price by considering both price and volume at the best bid and ask levels. It provides a more accurate measure of the market's equilibrium price than the simple mid-price.

### Mathematical Definition

```python
def calculate_microprice(bids, asks):
    """
    Calculate volume-weighted microprice.
    
    Args:
        bids: Bid levels [[price, volume], ...]
        asks: Ask levels [[price, volume], ...]
    
    Returns:
        microprice: Volume-weighted equilibrium price
    """
    if len(bids) == 0 or len(asks) == 0:
        return np.nan
    
    # Best bid and ask
    best_bid_price, best_bid_volume = bids[0]
    best_ask_price, best_ask_volume = asks[0]
    
    total_volume = best_bid_volume + best_ask_volume
    
    if total_volume == 0:
        return (best_bid_price + best_ask_price) / 2  # Fall back to mid-price
    
    # Volume-weighted price
    microprice = (best_bid_price * best_ask_volume + best_ask_price * best_bid_volume) / total_volume
    
    return microprice
```

### Interpretation

| Microprice Position | Relative to Mid-Price | Market Implication |
|--------------------|----------------------|-------------------|
| **> Mid-Price** | Above center | Buy-side pressure, upward bias |
| **< Mid-Price** | Below center | Sell-side pressure, downward bias |
| **≈ Mid-Price** | Near center | Balanced market |

### Volume Sensitivity

```python
# Microprice responds to volume imbalances
def microprice_sensitivity_example():
    # Equal volumes - microprice = mid-price
    bids = [[100, 100]]
    asks = [[101, 100]]
    microprice = 100.5  # Exactly mid-price
    
    # More bid volume - microprice closer to bid
    bids = [[100, 200]]  # Double bid volume
    asks = [[101, 100]]
    microprice = 100.33  # Closer to bid price
    
    # More ask volume - microprice closer to ask
    bids = [[100, 100]]
    asks = [[101, 200]]  # Double ask volume
    microprice = 100.67  # Closer to ask price
```

---

## Mathematical Formulations

### Complete Feature Set

```python
def calculate_extended_lob_features(orderbook, config):
    """
    Calculate all extended LOB features.
    
    Args:
        orderbook: Current orderbook with bids and asks
        config: Configuration dictionary
    
    Returns:
        features: Dictionary of all calculated features
    """
    features = {}
    
    # Basic features
    features['spread'] = calculate_spread(orderbook.bids, orderbook.asks)
    features['spread_pct'] = calculate_spread_pct(orderbook.bids, orderbook.asks)
    features['volume_imbalance'] = calculate_volume_imbalance(orderbook.bids, orderbook.asks)
    
    # Extended features
    features['depth_imbalance'] = calculate_depth_imbalance(
        orderbook.bids, orderbook.asks, config['imbalance_levels']
    )
    features['microprice'] = calculate_microprice(orderbook.bids, orderbook.asks)
    
    # OFI features (requires previous orderbook)
    if hasattr(orderbook, 'previous'):
        for level in range(config['depth_levels']):
            ofi = calculate_ofi(
                orderbook.bids, orderbook.asks,
                orderbook.previous.bids, orderbook.previous.asks,
                level
            )
            features[f'ofi_{level}'] = ofi
    
    return features
```

### Feature Vector Construction

```python
def construct_feature_vector(features, config):
    """
    Construct ordered feature vector for ML models.
    
    Args:
        features: Dictionary of calculated features
        config: Configuration with feature ordering
    
    Returns:
        vector: Ordered numpy array of features
    """
    vector = []
    
    # Basic features (9 total)
    basic_features = [
        'spread', 'spread_pct', 'volume_imbalance',
        'bid_depth', 'ask_depth', 'bid_vwap', 'ask_vwap',
        'bid_density', 'ask_density'
    ]
    
    # Extended features (2 total)
    extended_features = ['depth_imbalance', 'microprice']
    
    # OFI features (configurable)
    ofi_features = [f'ofi_{i}' for i in range(config['depth_levels'])]
    
    # Combine in order
    all_features = basic_features + extended_features + ofi_features
    
    for feature_name in all_features:
        vector.append(features.get(feature_name, 0.0))
    
    return np.array(vector)
```

---

## Test Implementation

### Test Structure

```python
@pytest.mark.unit
@pytest.mark.features_layer
class TestOrderBookExtendedFeatures:
    """Test extended orderbook features."""
    
    # Helper function tests
    def test_depth_imbalance(self):
        """Test depth imbalance calculation."""
        
    def test_microprice(self):
        """Test microprice calculation."""
        
    def test_ofi_calculation(self):
        """Test OFI calculation."""
    
    # Integration tests
    def test_extended_features_integration(self):
        """Test all extended features working together."""
        
    def test_feature_vector_construction(self):
        """Test feature vector ordering and construction."""
    
    # Edge case tests
    def test_empty_orderbook_handling(self):
        """Test handling of empty orderbook scenarios."""
        
    def test_insufficient_depth_handling(self):
        """Test handling of insufficient depth scenarios."""
```

### Mathematical Validation

```python
def test_depth_imbalance_mathematical_accuracy(self):
    """Test mathematical accuracy of depth imbalance calculation."""
    # Test case: Equal volumes
    bids = np.array([[100, 10], [99, 20], [98, 15]])
    asks = np.array([[101, 10], [102, 20], [103, 15]])
    
    result = _depth_imbalance(bids, asks, levels=3)
    
    # Manual calculation
    bid_volume = 10 + 20 + 15  # 45
    ask_volume = 10 + 20 + 15  # 45
    expected = (bid_volume - ask_volume) / (bid_volume + ask_volume)  # 0.0
    
    assert abs(result - expected) < 1e-10

def test_microprice_mathematical_accuracy(self):
    """Test mathematical accuracy of microprice calculation."""
    bids = np.array([[100, 10], [99, 20]])
    asks = np.array([[101, 10], [102, 20]])
    
    result = _microprice(bids, asks)
    
    # Manual calculation
    bid_price, bid_volume = 100, 10
    ask_price, ask_volume = 101, 10
    expected = (bid_price * ask_volume + ask_price * bid_volume) / (bid_volume + ask_volume)
    
    assert abs(result - expected) < 1e-10
```

---

## Feature Integration

### Feature Count Updates

```python
# Original feature count: 9 base + depth_levels OFI
# Extended feature count: 11 base + depth_levels OFI

def get_feature_count(config):
    """Calculate total feature count."""
    base_features = 11  # Added depth_imbalance and microprice
    ofi_features = config['depth_levels']
    return base_features + ofi_features

# Example configurations
config_3_levels = {'depth_levels': 3, 'imbalance_levels': 5}
# Total features: 11 + 3 = 14

config_5_levels = {'depth_levels': 5, 'imbalance_levels': 10}
# Total features: 11 + 5 = 16
```

### Feature Naming Convention

```python
def get_feature_names(config):
    """Get ordered list of feature names."""
    names = []
    
    # Basic features (9)
    names.extend(['spread', 'spread_pct', 'volume_imbalance'])
    names.extend(['bid_depth', 'ask_depth', 'bid_vwap', 'ask_vwap'])
    names.extend(['bid_density', 'ask_density'])
    
    # Extended features (2)
    names.extend(['depth_imbalance', 'microprice'])
    
    # OFI features (configurable)
    for i in range(config['depth_levels']):
        names.append(f'ofi_{i}')
    
    return names
```

### Integration with Existing Features

```python
class OrderBookFeatures:
    """Enhanced OrderBookFeatures with extended metrics."""
    
    def calculate(self, data):
        """Calculate all features including extended metrics."""
        # Calculate basic features
        basic_features = self._calculate_basic_features(data)
        
        # Calculate extended features
        extended_features = self._calculate_extended_features(data)
        
        # Calculate OFI features
        ofi_features = self._calculate_ofi_features(data)
        
        # Combine all features
        all_features = {**basic_features, **extended_features, **ofi_features}
        
        # Convert to feature vector
        feature_vector = self._construct_feature_vector(all_features)
        
        return feature_vector
```

---

## Configuration Options

### Extended Configuration

```python
default_extended_config = {
    # Basic configuration
    'depth_levels': 5,           # Number of OFI levels
    'imbalance_levels': 10,       # Levels for depth imbalance
    
    # Extended feature configuration
    'enable_depth_imbalance': True,
    'enable_microprice': True,
    'enable_ofi': True,
    
    # Advanced options
    'ofi_smoothing_window': 1,    # Smoothing for OFI calculation
    'microprice_fallback_mid': True,  # Fall back to mid-price
    'normalize_depth_imbalance': True,  # Normalize to [-1, 1]
    
    # Performance options
    'vectorize_calculations': True,
    'cache_intermediate': False,
    'parallel_processing': False
}
```

### Custom Configuration Examples

```python
# High-frequency trading configuration
hft_config = {
    'depth_levels': 3,           # Fast calculation
    'imbalance_levels': 5,
    'enable_depth_imbalance': True,
    'enable_microprice': True,
    'enable_ofi': True,
    'ofi_smoothing_window': 1
}

# Medium-frequency trading configuration
medium_freq_config = {
    'depth_levels': 10,          # More depth levels
    'imbalance_levels': 20,
    'enable_depth_imbalance': True,
    'enable_microprice': True,
    'enable_ofi': True,
    'ofi_smoothing_window': 3     # Some smoothing
}

# Research/analysis configuration
research_config = {
    'depth_levels': 20,          # Maximum depth
    'imbalance_levels': 50,
    'enable_depth_imbalance': True,
    'enable_microprice': True,
    'enable_ofi': True,
    'ofi_smoothing_window': 5,
    'cache_intermediate': True    # Cache for analysis
}
```

---

## Use Cases

### 1. High-Frequency Trading

```python
def hft_strategy_signals(orderbook_features):
    """Generate HFT trading signals from extended features."""
    signals = {}
    
    # OFI-based signals
    ofi_0 = orderbook_features['ofi_0']
    if ofi_0 > threshold_buy:
        signals['action'] = 'BUY'
        signals['confidence'] = min(ofi_0 / max_ofi, 1.0)
    elif ofi_0 < -threshold_sell:
        signals['action'] = 'SELL'
        signals['confidence'] = min(abs(ofi_0) / max_ofi, 1.0)
    else:
        signals['action'] = 'HOLD'
        signals['confidence'] = 0.0
    
    # Microprice-based execution
    microprice = orderbook_features['microprice']
    mid_price = (orderbook_features['best_bid'] + orderbook_features['best_ask']) / 2
    
    if microprice > mid_price:
        signals['execution_price'] = microprice  # Aggressive execution
    else:
        signals['execution_price'] = mid_price   # Passive execution
    
    return signals
```

### 2. Market Making

```python
def market_making_pricing(orderbook_features):
    """Calculate optimal bid/ask prices for market making."""
    features = orderbook_features
    
    # Base spread from microprice
    microprice = features['microprice']
    base_spread = features['spread']
    
    # Adjust spread based on depth imbalance
    depth_imbalance = features['depth_imbalance']
    
    if depth_imbalance > 0.3:  # Buy-side pressure
        # Widen ask spread, tighten bid spread
        ask_adjustment = base_spread * 0.2
        bid_adjustment = -base_spread * 0.1
    elif depth_imbalance < -0.3:  # Sell-side pressure
        # Widen bid spread, tighten ask spread
        bid_adjustment = base_spread * 0.2
        ask_adjustment = -base_spread * 0.1
    else:
        bid_adjustment = ask_adjustment = 0
    
    # Calculate optimal quotes
    optimal_bid = microprice - base_spread/2 + bid_adjustment
    optimal_ask = microprice + base_spread/2 + ask_adjustment
    
    return {
        'bid_price': optimal_bid,
        'ask_price': optimal_ask,
        'spread': optimal_ask - optimal_bid
    }
```

### 3. Liquidity Analysis

```python
def liquidity_analysis(orderbook_features):
    """Analyze market liquidity conditions."""
    features = orderbook_features
    
    analysis = {}
    
    # Liquidity score based on depth imbalance
    depth_imbalance = features['depth_imbalance']
    liquidity_score = 1.0 - abs(depth_imbalance)  # Higher = more balanced
    
    # Volatility prediction from OFI
    ofi_volatility = np.std([features[f'ofi_{i}'] for i in range(3)])  # First 3 levels
    volatility_risk = min(ofi_volatility / max_expected_ofi, 1.0)
    
    # Market efficiency from microprice deviation
    microprice = features['microprice']
    mid_price = (features['best_bid'] + features['best_ask']) / 2
    efficiency = 1.0 - abs(microprice - mid_price) / features['spread']
    
    analysis['liquidity_score'] = liquidity_score
    analysis['volatility_risk'] = volatility_risk
    analysis['market_efficiency'] = efficiency
    analysis['overall_quality'] = (liquidity_score + efficiency) / 2
    
    return analysis
```

---

## Performance Considerations

### Computational Complexity

| Feature | Time Complexity | Space Complexity | Optimization |
|---------|----------------|------------------|-------------|
| **OFI** | O(d) | O(d) | Vectorized operations |
| **Depth Imbalance** | O(d) | O(1) | Cumulative sum caching |
| **Microprice** | O(1) | O(1) | Direct calculation |

### Optimization Strategies

```python
class OptimizedOrderBookFeatures:
    """Performance-optimized feature calculation."""
    
    def __init__(self, config):
        self.config = config
        self._volume_cache = {}
        self._ofi_cache = {}
    
    @lru_cache(maxsize=1000)
    def _cached_depth_imbalance(self, bid_volume_hash, ask_volume_hash, levels):
        """Cached depth imbalance calculation."""
        # Implementation with caching
        pass
    
    def vectorized_ofi_calculation(self, orderbooks):
        """Vectorized OFI calculation for multiple orderbooks."""
        # Use numpy vectorization for batch processing
        bids_array = np.array([ob.bids for ob in orderbooks])
        asks_array = np.array([ob.asks for ob in orderbooks])
        
        # Vectorized volume differences
        volume_diffs = bids_array[:, :, 1] - asks_array[:, :, 1]
        
        return volume_diffs
    
    def parallel_feature_calculation(self, orderbook_batch):
        """Parallel processing of orderbook batch."""
        # Use multiprocessing for independent calculations
        with multiprocessing.Pool() as pool:
            results = pool.map(self.calculate_single, orderbook_batch)
        return results
```

### Memory Management

```python
def memory_efficient_calculation(orderbook_stream, config):
    """Memory-efficient calculation for streaming data."""
    features_buffer = []
    
    for orderbook in orderbook_stream:
        # Calculate features for current orderbook
        features = calculate_features(orderbook, config)
        
        # Store only necessary features
        if len(features_buffer) >= config['max_buffer_size']:
            features_buffer.pop(0)  # Remove oldest
        
        features_buffer.append(features)
        
        # Process features (e.g., send to model)
        process_features(features)
    
    return features_buffer
```

---

## Best Practices

### 1. Feature Selection

```python
def select_features_for_strategy(strategy_type, config):
    """Select optimal features for different trading strategies."""
    
    if strategy_type == 'hft':
        # High-frequency: Focus on fast, responsive features
        selected_features = ['ofi_0', 'ofi_1', 'microprice', 'depth_imbalance']
        config['depth_levels'] = 3
        config['imbalance_levels'] = 5
        
    elif strategy_type == 'market_making':
        # Market making: Focus on liquidity and balance
        selected_features = ['depth_imbalance', 'microprice', 'spread', 'volume_imbalance']
        config['depth_levels'] = 5
        config['imbalance_levels'] = 10
        
    elif strategy_type == 'arbitrage':
        # Arbitrage: Focus on price discovery
        selected_features = ['microprice', 'spread', 'spread_pct', 'ofi_0']
        config['depth_levels'] = 2
        config['imbalance_levels'] = 3
        
    return selected_features, config
```

### 2. Data Quality Assurance

```python
def validate_orderbook_data(orderbook):
    """Validate orderbook data before feature calculation."""
    validation_errors = []
    
    # Check for empty orderbook
    if len(orderbook.bids) == 0 or len(orderbook.asks) == 0:
        validation_errors.append("Empty orderbook side")
    
    # Check for price continuity
    if len(orderbook.bids) > 1:
        for i in range(len(orderbook.bids) - 1):
            if orderbook.bids[i][0] <= orderbook.bids[i+1][0]:
                validation_errors.append("Non-monotonic bid prices")
    
    if len(orderbook.asks) > 1:
        for i in range(len(orderbook.asks) - 1):
            if orderbook.asks[i][0] >= orderbook.asks[i+1][0]:
                validation_errors.append("Non-monotonic ask prices")
    
    # Check for spread reasonableness
    if len(orderbook.bids) > 0 and len(orderbook.asks) > 0:
        spread = orderbook.asks[0][0] - orderbook.bids[0][0]
        if spread <= 0:
            validation_errors.append("Negative or zero spread")
    
    return validation_errors
```

### 3. Feature Monitoring

```python
def monitor_feature_quality(features, config):
    """Monitor feature quality and detect anomalies."""
    alerts = []
    
    # Check for NaN values
    for feature_name, value in features.items():
        if np.isnan(value):
            alerts.append(f"NaN value in {feature_name}")
    
    # Check feature ranges
    if 'depth_imbalance' in features:
        imb = features['depth_imbalance']
        if abs(imb) > 1.0:
            alerts.append(f"Depth imbalance out of range: {imb}")
    
    # Check microprice reasonableness
    if 'microprice' in features and 'spread' in features:
        microprice = features['microprice']
        spread = features['spread']
        mid_price = (features['best_bid'] + features['best_ask']) / 2
        
        if abs(microprice - mid_price) > spread:
            alerts.append(f"Microprice too far from mid-price: {microprice}")
    
    return alerts
```

### 4. Model Integration

```python
def prepare_features_for_model(features, feature_names, model_type):
    """Prepare features for specific ML models."""
    
    if model_type == 'neural_network':
        # Normalize features for neural networks
        normalized_features = {}
        for name in feature_names:
            value = features[name]
            if name in feature_normalization_params:
                mean, std = feature_normalization_params[name]
                normalized_features[name] = (value - mean) / std
            else:
                normalized_features[name] = value
        return normalized_features
    
    elif model_type == 'tree_based':
        # Tree-based models work with raw features
        return features
    
    elif model_type == 'linear':
        # Linear models benefit from feature engineering
        engineered_features = features.copy()
        
        # Add interaction terms
        if 'depth_imbalance' in features and 'ofi_0' in features:
            engineered_features['imbalance_ofi_interaction'] = (
                features['depth_imbalance'] * features['ofi_0']
            )
        
        return engineered_features
```

This comprehensive documentation provides complete coverage of the extended OrderBook features including OFI, depth imbalance, and microprice, with mathematical formulations, implementation details, use cases, and best practices for production deployment.
