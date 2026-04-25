"""
Shared pytest fixtures and configuration for ITS project testing.
"""
import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
import pytest
import pytest_asyncio
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock, MagicMock


# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_OUTPUT_DIR = Path(__file__).parent / "output"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_dirs() -> None:
    """Create test data and output directories."""
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """Generate sample market data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1s'),
        'open': np.random.randn(n_samples).cumsum() + 42000,
        'high': np.random.randn(n_samples).cumsum() + 42050,
        'low': np.random.randn(n_samples).cumsum() + 41950,
        'close': np.random.randn(n_samples).cumsum() + 42000,
        'volume': np.random.randint(100, 1000, n_samples),
    })
    
    # Ensure high >= close >= low
    data['high'] = data[['close', 'high']].max(axis=1)
    data['low'] = data[['close', 'low']].min(axis=1)
    
    return data


@pytest.fixture
def sample_orderbook_data() -> dict:
    """Generate sample order book data for testing."""
    np.random.seed(42)
    levels = 10
    
    bids = [
        {
            'price': 42000 - i * 10,
            'quantity': np.random.uniform(0.1, 1.0)
        }
        for i in range(levels)
    ]
    
    asks = [
        {
            'price': 42000 + i * 10,
            'quantity': np.random.uniform(0.1, 1.0)
        }
        for i in range(levels)
    ]
    
    return {
        'lastUpdateId': 123456789,
        'bids': bids,
        'asks': asks,
        'timestamp': int(pd.Timestamp('2024-01-01').timestamp() * 1000)
    }


@pytest.fixture
def sample_features() -> np.ndarray:
    """Generate sample feature array for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    return np.random.randn(n_samples, n_features)


@pytest.fixture
def sample_sequence_features() -> np.ndarray:
    """Generate sample sequence features for deep learning models."""
    np.random.seed(42)
    n_samples = 100
    seq_len = 60
    n_features = 20
    return np.random.randn(n_samples, seq_len, n_features)


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Generate sample labels for testing (0=sell, 1=hold, 2=buy)."""
    np.random.seed(42)
    n_samples = 100
    return np.random.randint(0, 3, n_samples)


@pytest.fixture
def mock_binance_ws_client():
    """Mock Binance WebSocket client for testing."""
    mock = AsyncMock()
    mock.connect = AsyncMock()
    mock.disconnect = AsyncMock()
    mock.subscribe = AsyncMock()
    mock.is_connected = MagicMock(return_value=True)
    return mock


@pytest.fixture
def mock_binance_rest_client():
    """Mock Binance REST client for testing."""
    mock = AsyncMock()
    mock.get_orderbook_snapshot = AsyncMock()
    mock.get_klines = AsyncMock()
    return mock


@pytest.fixture
def mock_glassnode_client():
    """Mock Glassnode client for testing."""
    mock = AsyncMock()
    mock.get_metric = AsyncMock()
    return mock


@pytest.fixture
def mock_twitter_client():
    """Mock Twitter/X client for testing."""
    mock = AsyncMock()
    mock.search_recent = AsyncMock()
    return mock


@pytest.fixture
def temp_config_file() -> Generator[Path, None, None]:
    """Create a temporary configuration file for testing."""
    config_content = """
{
  "glassnode_api_key": "test_key",
  "twitter_bearer_token": "test_token",
  "binance_api_key": "test_key",
  "binance_api_secret": "test_secret",
  "timescale_dsn": "postgresql://test:test@localhost:5432/test",
  "parquet_base_path": "/tmp/test_parquet"
}
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(config_content)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_parquet_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for Parquet storage testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
async def mock_timescale_connection():
    """Mock TimescaleDB connection for testing."""
    mock = AsyncMock()
    mock.execute = AsyncMock()
    mock.fetch = AsyncMock()
    mock.fetchrow = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_model():
    """Mock ML model for testing."""
    mock = MagicMock()
    mock.fit = MagicMock()
    mock.predict = MagicMock(return_value=np.array([1, 1, 0, 2, 1]))
    mock.predict_proba = MagicMock(return_value=np.array([
        [0.1, 0.7, 0.2],
        [0.2, 0.6, 0.2],
        [0.7, 0.2, 0.1],
        [0.1, 0.2, 0.7],
        [0.2, 0.7, 0.1]
    ]))
    mock.get_confidence = MagicMock(return_value=np.array([0.7, 0.6, 0.7, 0.7, 0.7]))
    mock.save = MagicMock()
    return mock


@pytest.fixture
def sample_decision_config() -> dict:
    """Sample decision engine configuration."""
    return {
        "decision": {
            "confidence_threshold": 0.7,
        },
        "risk": {
            "max_position_size": 0.1,
            "max_portfolio_risk": 0.05,
            "max_drawdown": 0.15,
        },
        "sizing": {
            "method": "risk_based",
            "stop_loss_pct": 0.02,
        },
        "portfolio": {
            "max_positions": 5,
        }
    }


@pytest.fixture
def sample_execution_config() -> dict:
    """Sample execution configuration."""
    return {
        "initial_balance": 10000.0,
        "latency_ms": 50,
        "commission_rate": 0.001,
        "slippage_rate": 0.0005,
    }


@pytest.fixture
def sample_backtest_config() -> dict:
    """Sample backtest configuration."""
    return {
        "initial_capital": 10000,
        "commission_rate": 0.001,
        "slippage_rate": 0.0005,
        "min_window": 100,
    }


# Async fixtures for async testing
@pytest_asyncio.fixture
async def async_queue():
    """Create an async queue for testing."""
    return asyncio.Queue()


@pytest_asyncio.fixture
async def sample_async_data_stream():
    """Generate sample async data stream for testing."""
    async def data_generator():
        for i in range(10):
            yield {
                'timestamp': int(pd.Timestamp('2024-01-01').timestamp() * 1000) + i * 1000,
                'symbol': 'BTCUSDT',
                'price': 42000 + i * 10,
                'volume': np.random.randint(100, 1000)
            }
            await asyncio.sleep(0.01)
    
    return data_generator()
