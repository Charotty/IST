"""
Unit tests for TimescaleStorage.
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from its_project.storage.timescale import TimescaleStorage
from its_project.common.types import MarketData, MarketDataType


@pytest.mark.unit
@pytest.mark.storage_layer
class TestTimescaleStorage:
    """Test TimescaleStorage functionality."""
    
    def test_initialization(self):
        """Test TimescaleStorage initialization."""
        storage = TimescaleStorage(
            dsn="postgresql://user:pass@localhost/db",
            min_size=5,
            max_size=20
        )
        
        assert storage._dsn == "postgresql://user:pass@localhost/db"
        assert storage._min_size == 5
        assert storage._max_size == 20
        assert storage._pool is None
    
    def test_initialization_defaults(self):
        """Test initialization with default pool sizes."""
        storage = TimescaleStorage(dsn="postgresql://user:pass@localhost/db")
        
        assert storage._min_size == 5
        assert storage._max_size == 20
    
    @pytest.mark.asyncio
    async def test_connect(self):
        """Test connecting to TimescaleDB."""
        storage = TimescaleStorage(dsn="postgresql://user:pass@localhost/db")
        
        with patch('its_project.storage.timescale.asyncpg.create_pool', new_callable=AsyncMock) as mock_create_pool:
            mock_pool = MagicMock()
            mock_pool.close = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            await storage.connect()
            
            assert storage._pool is mock_pool
            mock_create_pool.assert_awaited_once_with(
                "postgresql://user:pass@localhost/db",
                min_size=5,
                max_size=20
            )
    
    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing the connection pool."""
        storage = TimescaleStorage(dsn="postgresql://user:pass@localhost/db")
        
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        storage._pool = mock_pool
        
        await storage.close()
        
        mock_pool.close.assert_awaited_once()
        assert storage._pool is None
    
    @pytest.mark.asyncio
    async def test_close_no_pool(self):
        """Test closing when pool is None."""
        storage = TimescaleStorage(dsn="postgresql://user:pass@localhost/db")
        
        # Should not raise error
        await storage.close()
        assert storage._pool is None
    
    @pytest.mark.asyncio
    async def test_write(self):
        """Test writing a single MarketData."""
        storage = TimescaleStorage(dsn="postgresql://user:pass@localhost/db")
        
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.executemany = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        storage._pool = mock_pool
        
        market_data = MarketData(
            timestamp_ms=1704067200000,
            symbol="BTC/USDT",
            type=MarketDataType.KLINE,
            exchange="binance",
            data={"open": 42000.0, "close": 42300.0}
        )
        
        result = await storage.write(market_data)
        
        assert result == 1
        mock_conn.executemany.assert_awaited_once()
    
    @pytest.mark.asyncio
    async def test_write_no_pool(self):
        """Test writing without pool raises RuntimeError."""
        storage = TimescaleStorage(dsn="postgresql://user:pass@localhost/db")
        
        market_data = MarketData(
            timestamp_ms=1704067200000,
            symbol="BTC/USDT",
            type=MarketDataType.KLINE,
            exchange="binance",
            data={"open": 42000.0}
        )
        
        with pytest.raises(RuntimeError, match="TimescaleDB pool not initialized"):
            await storage.write(market_data)
    
    @pytest.mark.asyncio
    async def test_write_batch(self):
        """Test writing multiple MarketData."""
        storage = TimescaleStorage(dsn="postgresql://user:pass@localhost/db")
        
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.executemany = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        storage._pool = mock_pool
        
        data = [
            MarketData(
                timestamp_ms=1704067200000 + i,
                symbol="BTC/USDT",
                type=MarketDataType.KLINE,
                exchange="binance",
                data={"open": 42000.0 + i}
            )
            for i in range(3)
        ]
        
        result = await storage.write_batch(data)
        
        assert result == 3
        mock_conn.executemany.assert_awaited_once()
    
    @pytest.mark.asyncio
    async def test_write_batch_empty(self):
        """Test writing empty batch returns 0."""
        storage = TimescaleStorage(dsn="postgresql://user:pass@localhost/db")
        
        mock_pool = MagicMock()
        storage._pool = mock_pool
        
        result = await storage.write_batch([])
        
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_write_batch_no_pool(self):
        """Test writing batch without pool raises RuntimeError."""
        storage = TimescaleStorage(dsn="postgresql://user:pass@localhost/db")
        
        with pytest.raises(RuntimeError, match="TimescaleDB pool not initialized"):
            await storage.write_batch([])
    
    @pytest.mark.asyncio
    async def test_read(self):
        """Test reading MarketData."""
        storage = TimescaleStorage(dsn="postgresql://user:pass@localhost/db")
        
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {
                "timestamp_ms": 1704067200000,
                "symbol": "BTC/USDT",
                "type": "kline",
                "exchange": "binance",
                "data": '{"open": 42000.0, "close": 42300.0}'
            }
        ])
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        storage._pool = mock_pool
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        
        result = await storage.read("BTC/USDT", start, end)
        
        assert len(result) == 1
        assert result[0].symbol == "BTC/USDT"
        assert result[0].timestamp_ms == 1704067200000
    
    @pytest.mark.asyncio
    async def test_read_no_pool(self):
        """Test reading without pool raises RuntimeError."""
        storage = TimescaleStorage(dsn="postgresql://user:pass@localhost/db")
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        
        with pytest.raises(RuntimeError, match="TimescaleDB pool not initialized"):
            await storage.read("BTC/USDT", start, end)
    
    @pytest.mark.asyncio
    async def test_read_with_data_type(self):
        """Test reading with data_type filter."""
        storage = TimescaleStorage(dsn="postgresql://user:pass@localhost/db")
        
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        storage._pool = mock_pool
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        
        await storage.read("BTC/USDT", start, end, data_type="kline")
        
        # Verify query includes data_type parameter
        call_args = mock_conn.fetch.call_args
        assert "kline" in call_args[0]
    
    @pytest.mark.asyncio
    async def test_get_latest(self):
        """Test getting latest MarketData."""
        storage = TimescaleStorage(dsn="postgresql://user:pass@localhost/db")
        
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {
                "timestamp_ms": 1704067200000,
                "symbol": "BTC/USDT",
                "type": "kline",
                "exchange": "binance",
                "data": '{"open": 42000.0}'
            }
        ])
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        storage._pool = mock_pool
        
        result = await storage.get_latest("BTC/USDT", limit=1)
        
        assert len(result) == 1
        assert result[0].symbol == "BTC/USDT"
    
    @pytest.mark.asyncio
    async def test_get_latest_no_pool(self):
        """Test get_latest without pool raises RuntimeError."""
        storage = TimescaleStorage(dsn="postgresql://user:pass@localhost/db")
        
        with pytest.raises(RuntimeError, match="TimescaleDB pool not initialized"):
            await storage.get_latest("BTC/USDT")
    
    @pytest.mark.asyncio
    async def test_get_latest_with_limit(self):
        """Test get_latest with custom limit."""
        storage = TimescaleStorage(dsn="postgresql://user:pass@localhost/db")
        
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        storage._pool = mock_pool
        
        await storage.get_latest("BTC/USDT", limit=5)
        
        # Verify limit parameter is passed
        call_args = mock_conn.fetch.call_args
        assert call_args[0][2] == 5
