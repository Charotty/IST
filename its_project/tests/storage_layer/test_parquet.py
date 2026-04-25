"""
Unit tests for Parquet storage.
"""
import pytest
from datetime import datetime, date
from pathlib import Path
import tempfile
from its_project.storage.parquet import ParquetStorage
from its_project.common.types import MarketData, MarketDataType


@pytest.mark.unit
@pytest.mark.storage_layer
class TestParquetStorage:
    """Test ParquetStorage functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def parquet_storage(self, temp_dir):
        """Create a ParquetStorage instance."""
        storage = ParquetStorage(base_path=temp_dir)
        yield storage
    
    def test_initialization(self, temp_dir):
        """Test storage initialization."""
        storage = ParquetStorage(base_path=temp_dir)
        assert storage._base_path == temp_dir
        assert temp_dir.exists()
    
    @pytest.mark.asyncio
    async def test_write_single_record(self, parquet_storage):
        """Test writing a single record."""
        data = MarketData(
            timestamp_ms=1704067200000,
            symbol='BTCUSDT',
            type=MarketDataType.TRADE,
            exchange='binance',
            data={'price': '42000.0', 'qty': '0.1'}
        )
        
        result = await parquet_storage.write(data)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_write_batch_records(self, parquet_storage):
        """Test writing batch records."""
        data_list = [
            MarketData(
                timestamp_ms=1704067200000,
                symbol='BTCUSDT',
                type=MarketDataType.TRADE,
                exchange='binance',
                data={'price': '42000.0', 'qty': '0.1'}
            ),
            MarketData(
                timestamp_ms=1704067260000,
                symbol='BTCUSDT',
                type=MarketDataType.TRADE,
                exchange='binance',
                data={'price': '42010.0', 'qty': '0.2'}
            )
        ]
        
        result = await parquet_storage.write_batch(data_list)
        assert result == 2
    
    @pytest.mark.asyncio
    async def test_read_records(self, parquet_storage):
        """Test reading records."""
        data_list = [
            MarketData(
                timestamp_ms=1704067200000,
                symbol='BTCUSDT',
                type=MarketDataType.TRADE,
                exchange='binance',
                data={'price': '42000.0'}
            ),
            MarketData(
                timestamp_ms=1704153600000,
                symbol='BTCUSDT',
                type=MarketDataType.TRADE,
                exchange='binance',
                data={'price': '42010.0'}
            )
        ]
        await parquet_storage.write_batch(data_list)
        
        result = await parquet_storage.read(
            symbol='BTCUSDT',
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 3)
        )
        
        assert len(result) == 2
        assert result[0].symbol == 'BTCUSDT'
    
    @pytest.mark.asyncio
    async def test_get_latest(self, parquet_storage):
        """Test getting latest records."""
        import time
        now = int(time.time() * 1000)
        
        data_list = [
            MarketData(
                timestamp_ms=now - 10000,
                symbol='BTCUSDT',
                type=MarketDataType.TRADE,
                exchange='binance',
                data={'price': '42000.0'}
            ),
            MarketData(
                timestamp_ms=now,
                symbol='BTCUSDT',
                type=MarketDataType.TRADE,
                exchange='binance',
                data={'price': '42010.0'}
            )
        ]
        await parquet_storage.write_batch(data_list)
        
        result = await parquet_storage.get_latest(symbol='BTCUSDT', limit=1)
        assert len(result) == 1
        assert result[0].timestamp_ms == now
    
    @pytest.mark.asyncio
    async def test_close(self, parquet_storage):
        """Test closing storage."""
        await parquet_storage.close()
    
    @pytest.mark.asyncio
    async def test_write_batch_empty(self, parquet_storage):
        """Test writing empty batch."""
        result = await parquet_storage.write_batch([])
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_read_with_data_type_filter(self, parquet_storage):
        """Test reading records with data type filter."""
        data_list = [
            MarketData(
                timestamp_ms=1704067200000,
                symbol='BTCUSDT',
                type=MarketDataType.TRADE,
                exchange='binance',
                data={'price': '42000.0'}
            ),
            MarketData(
                timestamp_ms=1704067260000,
                symbol='BTCUSDT',
                type=MarketDataType.ORDERBOOK,
                exchange='binance',
                data={'price': '42010.0'}
            )
        ]
        await parquet_storage.write_batch(data_list)
        
        result = await parquet_storage.read(
            symbol='BTCUSDT',
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 3),
            data_type='trade'
        )
        
        assert len(result) == 1
        assert result[0].type == MarketDataType.TRADE
    
    @pytest.mark.asyncio
    async def test_read_with_corrupted_file(self, parquet_storage, temp_dir):
        """Test reading with corrupted parquet file."""
        # Create a corrupted parquet file
        corrupted_file = temp_dir / "2024-01-01.parquet"
        corrupted_file.write_text("corrupted data")
        
        result = await parquet_storage.read(
            symbol='BTCUSDT',
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2)
        )
        
        # Should handle exception gracefully and return empty list
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_latest_with_corrupted_file(self, parquet_storage, temp_dir):
        """Test get_latest with corrupted parquet file."""
        # Create a corrupted parquet file
        today = date.today()
        corrupted_file = temp_dir / f"{today.isoformat()}.parquet"
        corrupted_file.write_text("corrupted data")
        
        result = await parquet_storage.get_latest(symbol='BTCUSDT', limit=1)
        
        # Should handle exception gracefully and return empty list
        assert result == []
    
    @pytest.mark.asyncio
    async def test_write_batch_multiple_dates(self, parquet_storage):
        """Test writing batch records spanning multiple dates."""
        data_list = [
            MarketData(
                timestamp_ms=1704067200000,  # 2024-01-01
                symbol='BTCUSDT',
                type=MarketDataType.TRADE,
                exchange='binance',
                data={'price': '42000.0'}
            ),
            MarketData(
                timestamp_ms=1704153600000,  # 2024-01-02
                symbol='BTCUSDT',
                type=MarketDataType.TRADE,
                exchange='binance',
                data={'price': '42010.0'}
            )
        ]
        
        result = await parquet_storage.write_batch(data_list)
        assert result == 2
    
    @pytest.mark.asyncio
    async def test_read_no_matching_symbol(self, parquet_storage):
        """Test reading when no records match symbol."""
        data_list = [
            MarketData(
                timestamp_ms=1704067200000,
                symbol='ETHUSDT',
                type=MarketDataType.TRADE,
                exchange='binance',
                data={'price': '42000.0'}
            )
        ]
        await parquet_storage.write_batch(data_list)
        
        result = await parquet_storage.read(
            symbol='BTCUSDT',
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 3)
        )
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_latest_no_matching_symbol(self, parquet_storage):
        """Test get_latest when no records match symbol."""
        data_list = [
            MarketData(
                timestamp_ms=1704067200000,
                symbol='ETHUSDT',
                type=MarketDataType.TRADE,
                exchange='binance',
                data={'price': '42000.0'}
            )
        ]
        await parquet_storage.write_batch(data_list)
        
        result = await parquet_storage.get_latest(symbol='BTCUSDT', limit=1)
        
        assert result == []
