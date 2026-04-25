"""
Unit tests for StorageReadAPI.
"""
import pytest
from datetime import datetime
from its_project.storage.api import StorageReadAPI
from its_project.storage.base import BaseStorage
from its_project.common.types import MarketData, MarketDataType


# Mock storage classes for testing
class MockWarmStorage(BaseStorage):
    """Mock warm storage for testing."""
    
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.data = []
    
    async def read(self, symbol, start, end, data_type=None):
        if self.should_fail:
            raise Exception("Warm storage failed")
        return self.data
    
    async def get_latest(self, symbol, limit=1):
        if self.should_fail:
            raise Exception("Warm storage failed")
        return self.data[:limit]
    
    async def write(self, data):
        pass
    
    async def write_batch(self, data_list):
        pass
    
    async def close(self):
        pass


class MockColdStorage(BaseStorage):
    """Mock cold storage for testing."""
    
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.data = []
    
    async def read(self, symbol, start, end, data_type=None):
        if self.should_fail:
            raise Exception("Cold storage failed")
        return self.data
    
    async def get_latest(self, symbol, limit=1):
        if self.should_fail:
            raise Exception("Cold storage failed")
        return self.data[:limit]
    
    async def write(self, data):
        pass
    
    async def write_batch(self, data_list):
        pass
    
    async def close(self):
        pass


@pytest.mark.unit
@pytest.mark.storage_layer
class TestStorageReadAPI:
    """Test StorageReadAPI functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test StorageReadAPI initialization."""
        warm = MockWarmStorage()
        cold = MockColdStorage()
        
        api = StorageReadAPI(warm, cold)
        
        assert api._warm is warm
        assert api._cold is cold
    
    @pytest.mark.asyncio
    async def test_read_from_warm(self):
        """Test reading from warm storage when it succeeds."""
        warm = MockWarmStorage()
        cold = MockColdStorage()
        
        market_data = MarketData(
            timestamp_ms=1704067200000,
            symbol="BTC/USDT",
            type=MarketDataType.KLINE,
            exchange="binance",
            data={"open": 42000.0, "high": 42500.0, "low": 41500.0, "close": 42300.0, "volume": 100.0}
        )
        warm.data = [market_data]
        
        api = StorageReadAPI(warm, cold)
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        
        result = await api.read("BTC/USDT", start, end)
        
        assert result == [market_data]
    
    @pytest.mark.asyncio
    async def test_read_warm_fails_fallback_to_cold(self):
        """Test fallback to cold storage when warm fails."""
        warm = MockWarmStorage(should_fail=True)
        cold = MockColdStorage()
        
        market_data = MarketData(
            timestamp_ms=1704067200000,
            symbol="BTC/USDT",
            type=MarketDataType.KLINE,
            exchange="binance",
            data={"open": 42000.0, "high": 42500.0, "low": 41500.0, "close": 42300.0, "volume": 100.0}
        )
        cold.data = [market_data]
        
        api = StorageReadAPI(warm, cold)
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        
        result = await api.read("BTC/USDT", start, end)
        
        assert result == [market_data]
    
    @pytest.mark.asyncio
    async def test_read_both_fail(self):
        """Test when both warm and cold storage fail."""
        warm = MockWarmStorage(should_fail=True)
        cold = MockColdStorage(should_fail=True)
        
        api = StorageReadAPI(warm, cold)
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        
        result = await api.read("BTC/USDT", start, end)
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_read_prefer_cold(self):
        """Test reading directly from cold storage when prefer_warm=False."""
        warm = MockWarmStorage()
        cold = MockColdStorage()
        
        market_data = MarketData(
            timestamp_ms=1704067200000,
            symbol="BTC/USDT",
            type=MarketDataType.KLINE,
            exchange="binance",
            data={"open": 42000.0, "high": 42500.0, "low": 41500.0, "close": 42300.0, "volume": 100.0}
        )
        cold.data = [market_data]
        
        api = StorageReadAPI(warm, cold)
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        
        result = await api.read("BTC/USDT", start, end, prefer_warm=False)
        
        assert result == [market_data]
    
    @pytest.mark.asyncio
    async def test_read_with_data_type(self):
        """Test reading with data_type parameter."""
        warm = MockWarmStorage()
        cold = MockColdStorage()
        
        market_data = MarketData(
            timestamp_ms=1704067200000,
            symbol="BTC/USDT",
            type=MarketDataType.KLINE,
            exchange="binance",
            data={"open": 42000.0, "high": 42500.0, "low": 41500.0, "close": 42300.0, "volume": 100.0}
        )
        warm.data = [market_data]
        
        api = StorageReadAPI(warm, cold)
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        
        result = await api.read("BTC/USDT", start, end, data_type="ohlcv")
        
        assert result == [market_data]
    
    @pytest.mark.asyncio
    async def test_get_latest_from_warm(self):
        """Test getting latest from warm storage."""
        warm = MockWarmStorage()
        cold = MockColdStorage()
        
        market_data = MarketData(
            timestamp_ms=1704067200000,
            symbol="BTC/USDT",
            type=MarketDataType.KLINE,
            exchange="binance",
            data={"open": 42000.0, "high": 42500.0, "low": 41500.0, "close": 42300.0, "volume": 100.0}
        )
        warm.data = [market_data]
        
        api = StorageReadAPI(warm, cold)
        
        result = await api.get_latest("BTC/USDT")
        
        assert result == [market_data]
    
    @pytest.mark.asyncio
    async def test_get_latest_warm_fails_fallback_to_cold(self):
        """Test fallback to cold when warm get_latest fails."""
        warm = MockWarmStorage(should_fail=True)
        cold = MockColdStorage()
        
        market_data = MarketData(
            timestamp_ms=1704067200000,
            symbol="BTC/USDT",
            type=MarketDataType.KLINE,
            exchange="binance",
            data={"open": 42000.0, "high": 42500.0, "low": 41500.0, "close": 42300.0, "volume": 100.0}
        )
        cold.data = [market_data]
        
        api = StorageReadAPI(warm, cold)
        
        result = await api.get_latest("BTC/USDT")
        
        assert result == [market_data]
    
    @pytest.mark.asyncio
    async def test_get_latest_both_fail(self):
        """Test when both warm and cold get_latest fail."""
        warm = MockWarmStorage(should_fail=True)
        cold = MockColdStorage(should_fail=True)
        
        api = StorageReadAPI(warm, cold)
        
        result = await api.get_latest("BTC/USDT")
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_latest_with_limit(self):
        """Test get_latest with custom limit."""
        warm = MockWarmStorage()
        cold = MockColdStorage()
        
        market_data1 = MarketData(
            timestamp_ms=1704067200000,
            symbol="BTC/USDT",
            type=MarketDataType.KLINE,
            exchange="binance",
            data={"open": 42000.0, "high": 42500.0, "low": 41500.0, "close": 42300.0, "volume": 100.0}
        )
        market_data2 = MarketData(
            timestamp_ms=1704067300000,
            symbol="BTC/USDT",
            type=MarketDataType.KLINE,
            exchange="binance",
            data={"open": 42300.0, "high": 42800.0, "low": 41800.0, "close": 42600.0, "volume": 110.0}
        )
        warm.data = [market_data1, market_data2]
        
        api = StorageReadAPI(warm, cold)
        
        result = await api.get_latest("BTC/USDT", limit=2)
        
        assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_get_latest_prefer_cold(self):
        """Test get_latest directly from cold when prefer_warm=False."""
        warm = MockWarmStorage()
        cold = MockColdStorage()
        
        market_data = MarketData(
            timestamp_ms=1704067200000,
            symbol="BTC/USDT",
            type=MarketDataType.KLINE,
            exchange="binance",
            data={"open": 42000.0, "high": 42500.0, "low": 41500.0, "close": 42300.0, "volume": 100.0}
        )
        cold.data = [market_data]
        
        api = StorageReadAPI(warm, cold)
        
        result = await api.get_latest("BTC/USDT", prefer_warm=False)
        
        assert result == [market_data]
