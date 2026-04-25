"""
Script to generate synthetic test data for testing.
Run this to populate the tests/data directory with sample data.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


TEST_DATA_DIR = Path(__file__).parent


def generate_ohlcv_data(
    start_date: str = '2024-01-01',
    periods: int = 10000,
    symbol: str = 'BTCUSDT'
) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    
    timestamps = pd.date_range(start=start_date, periods=periods, freq='1s')
    
    # Generate price data with random walk
    base_price = 42000.0
    returns = np.random.normal(0, 0.0001, periods)
    prices = base_price * (1 + returns).cumprod()
    
    # Generate OHLCV
    data = pd.DataFrame({
        'timestamp': timestamps,
        'symbol': symbol,
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, periods)),
        'high': prices * (1 + np.random.uniform(0, 0.002, periods)),
        'low': prices * (1 + np.random.uniform(-0.002, 0, periods)),
        'close': prices,
        'volume': np.random.randint(100, 1000, periods),
    })
    
    # Ensure high >= close >= low
    data['high'] = data[['close', 'high']].max(axis=1)
    data['low'] = data[['close', 'low']].min(axis=1)
    
    return data


def generate_orderbook_data(
    n_samples: int = 1000,
    levels: int = 10,
    symbol: str = 'BTCUSDT'
) -> list:
    """Generate synthetic order book data."""
    np.random.seed(42)
    
    orderbooks = []
    base_price = 42000.0
    
    for i in range(n_samples):
        price_offset = np.random.normal(0, 10)
        current_price = base_price + price_offset
        
        bids = [
            {
                'price': current_price - j * 10,
                'quantity': np.random.uniform(0.1, 1.0)
            }
            for j in range(levels)
        ]
        
        asks = [
            {
                'price': current_price + j * 10,
                'quantity': np.random.uniform(0.1, 1.0)
            }
            for j in range(levels)
        ]
        
        orderbooks.append({
            'timestamp': int(pd.Timestamp('2024-01-01').timestamp() * 1000) + i * 1000,
            'symbol': symbol,
            'lastUpdateId': 123456789 + i,
            'bids': bids,
            'asks': asks
        })
    
    return orderbooks


def generate_trade_data(
    n_samples: int = 1000,
    symbol: str = 'BTCUSDT'
) -> pd.DataFrame:
    """Generate synthetic trade data."""
    np.random.seed(42)
    
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1s')
    base_price = 42000.0
    returns = np.random.normal(0, 0.0001, n_samples)
    prices = base_price * (1 + returns).cumprod()
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'symbol': symbol,
        'price': prices,
        'quantity': np.random.uniform(0.001, 0.1, n_samples),
        'side': np.random.choice(['buy', 'sell'], n_samples),
    })
    
    return data


def generate_sentiment_data(
    n_samples: int = 1000
) -> pd.DataFrame:
    """Generate synthetic sentiment data."""
    np.random.seed(42)
    
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'sentiment_score': np.random.uniform(-1, 1, n_samples),
        'sentiment_momentum': np.random.uniform(-0.1, 0.1, n_samples),
        'volume': np.random.randint(10, 100, n_samples),
        'source': np.random.choice(['twitter', 'reddit', 'news'], n_samples),
    })
    
    return data


def generate_onchain_data(
    n_samples: int = 1000,
    symbol: str = 'BTC'
) -> pd.DataFrame:
    """Generate synthetic on-chain data."""
    np.random.seed(42)
    
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1h')
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'symbol': symbol,
        'active_addresses': np.random.randint(500000, 1000000, n_samples),
        'exchange_inflow': np.random.uniform(100, 1000, n_samples),
        'exchange_outflow': np.random.uniform(100, 1000, n_samples),
        'net_flow': np.random.uniform(-500, 500, n_samples),
        'large_transactions': np.random.randint(10, 100, n_samples),
    })
    
    return data


def save_test_data():
    """Generate and save all test data files."""
    print("Generating test data...")
    
    # Generate OHLCV data
    print("  - OHLCV data...")
    ohlcv = generate_ohlcv_data()
    ohlcv.to_parquet(TEST_DATA_DIR / 'ohlcv_test.parquet', index=False)
    
    # Generate order book data
    print("  - Order book data...")
    orderbooks = generate_orderbook_data()
    orderbook_df = pd.DataFrame(orderbooks)
    orderbook_df.to_parquet(TEST_DATA_DIR / 'orderbook_test.parquet', index=False)
    
    # Generate trade data
    print("  - Trade data...")
    trades = generate_trade_data()
    trades.to_parquet(TEST_DATA_DIR / 'trades_test.parquet', index=False)
    
    # Generate sentiment data
    print("  - Sentiment data...")
    sentiment = generate_sentiment_data()
    sentiment.to_parquet(TEST_DATA_DIR / 'sentiment_test.parquet', index=False)
    
    # Generate on-chain data
    print("  - On-chain data...")
    onchain = generate_onchain_data()
    onchain.to_parquet(TEST_DATA_DIR / 'onchain_test.parquet', index=False)
    
    print(f"Test data saved to: {TEST_DATA_DIR}")
    print("Done!")


if __name__ == '__main__':
    save_test_data()
