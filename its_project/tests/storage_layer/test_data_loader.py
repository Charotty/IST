"""
Unit tests for DataLoader.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from its_project.storage.data_loader import DataLoader
from its_project.common.types import MarketData, MarketDataType


@pytest.mark.unit
@pytest.mark.storage_layer
@pytest.mark.asyncio
class TestDataLoader:
    """Test DataLoader class."""
    
    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "parquet_store"
    
    @pytest.fixture
    def data_loader(self, temp_dir):
        """Create DataLoader instance."""
        return DataLoader(
            parquet_base_path=str(temp_dir),
            timescale_dsn="postgresql://test",
            parquet_priority=True
        )
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample MarketData."""
        return MarketData(
            timestamp_ms=1234567890000,
            symbol="BTCUSDT",
            type=MarketDataType.ORDERBOOK,
            exchange="binance",
            data={
                "_kind": "orderBook",
                "_reconstructed": False,
                "lastUpdateId": 12345,
                "bids": [["100", "1.0"]],
                "asks": [["101", "1.0"]],
                "_ts_recv_ms": 1234567890000,
                "p": "",
                "q": "",
                "t": 0
            }
        )
    
    def test_initialization(self, temp_dir):
        """Test DataLoader initialization."""
        loader = DataLoader(
            parquet_base_path=str(temp_dir),
            timescale_dsn="postgresql://test",
            parquet_priority=True
        )
        
        assert loader.parquet_priority is True
        assert loader._connected is False
        assert loader.parquet_store is not None
        assert loader.timescale_client is not None
    
    def test_initialization_default_priority(self, temp_dir):
        """Test initialization with default parquet priority."""
        loader = DataLoader(
            parquet_base_path=str(temp_dir),
            timescale_dsn="postgresql://test"
        )
        
        assert loader.parquet_priority is True
    
    async def test_connect(self, data_loader):
        """Test connecting to storage backends."""
        with patch.object(data_loader.timescale_client, 'connect', new_callable=AsyncMock):
            with patch.object(data_loader.timescale_client, 'initialize_schema', new_callable=AsyncMock):
                await data_loader.connect()
                
                assert data_loader._connected is True
    
    async def test_close(self, data_loader):
        """Test closing storage connections."""
        data_loader._connected = True
        
        with patch.object(data_loader.timescale_client, 'close', new_callable=AsyncMock):
            await data_loader.close()
            
            assert data_loader._connected is False
    
    async def test_read_data_not_connected(self, data_loader):
        """Test reading data when not connected."""
        with pytest.raises(RuntimeError, match="DataLoader not connected"):
            await data_loader.read_data(
                symbol="BTCUSDT",
                start_time=datetime(2024, 1, 1),
                end_time=datetime(2024, 1, 2)
            )
    
    async def test_read_data_parquet_success(self, data_loader, sample_market_data):
        """Test reading data from Parquet successfully."""
        data_loader._connected = True
        
        with patch.object(data_loader.parquet_store, 'read_raw_lob', return_value=[sample_market_data]):
            result = await data_loader.read_data(
                symbol="BTCUSDT",
                start_time=datetime(2024, 1, 1),
                end_time=datetime(2024, 1, 2),
                data_type="orderbook",
                use_raw=True
            )
            
            assert len(result) == 1
            assert result[0].symbol == "BTCUSDT"
    
    async def test_read_data_parquet_fallback(self, data_loader, sample_market_data):
        """Test reading data with Parquet fallback to TimescaleDB."""
        data_loader._connected = True
        
        with patch.object(data_loader.parquet_store, 'read_raw_lob', side_effect=Exception("Parquet error")):
            with patch.object(data_loader.timescale_client, 'read_aggregated', new_callable=AsyncMock, return_value=[sample_market_data]):
                result = await data_loader.read_data(
                    symbol="BTCUSDT",
                    start_time=datetime(2024, 1, 1),
                    end_time=datetime(2024, 1, 2),
                    data_type="orderbook"
                )
                
                assert len(result) == 1
    
    async def test_read_data_both_fail(self, data_loader):
        """Test reading data when both backends fail."""
        data_loader._connected = True
        
        with patch.object(data_loader.parquet_store, 'read_raw_lob', side_effect=Exception("Parquet error")):
            with patch.object(data_loader.timescale_client, 'read_aggregated', new_callable=AsyncMock, side_effect=Exception("Timescale error")):
                result = await data_loader.read_data(
                    symbol="BTCUSDT",
                    start_time=datetime(2024, 1, 1),
                    end_time=datetime(2024, 1, 2)
                )
                
                assert result == []
    
    async def test_read_data_with_limit(self, data_loader, sample_market_data):
        """Test reading data with limit."""
        data_loader._connected = True
        
        with patch.object(data_loader.parquet_store, 'read_raw_lob', return_value=[sample_market_data] * 100):
            result = await data_loader.read_data(
                symbol="BTCUSDT",
                start_time=datetime(2024, 1, 1),
                end_time=datetime(2024, 1, 2),
                use_raw=True,
                limit=10
            )
            
            assert len(result) == 10
    
    async def test_write_data_not_connected(self, data_loader, sample_market_data):
        """Test writing data when not connected."""
        with pytest.raises(RuntimeError, match="DataLoader not connected"):
            await data_loader.write_data(sample_market_data)
    
    async def test_write_data_single(self, data_loader, sample_market_data):
        """Test writing single data item."""
        data_loader._connected = True
        
        with patch.object(data_loader.parquet_store, 'write_batch_raw_lob'):
            result = await data_loader.write_data(sample_market_data, use_raw=True)
            
            assert result == 1
    
    async def test_write_data_sequence(self, data_loader, sample_market_data):
        """Test writing sequence of data items."""
        data_loader._connected = True
        
        data_list = [sample_market_data, sample_market_data]
        
        with patch.object(data_loader.parquet_store, 'write_batch_raw_lob'):
            result = await data_loader.write_data(data_list, use_raw=True)
            
            assert result == 2
    
    async def test_write_data_empty(self, data_loader):
        """Test writing empty data."""
        data_loader._connected = True
        
        result = await data_loader.write_data([])
        
        assert result == 0
    
    async def test_write_data_to_timescale(self, data_loader, sample_market_data):
        """Test writing data to TimescaleDB."""
        data_loader._connected = True
        
        with patch.object(data_loader.timescale_client, 'write_batch_aggregated', new_callable=AsyncMock, return_value=1):
            result = await data_loader.write_data(sample_market_data, use_raw=False)
            
            assert result == 1
    
    async def test_write_data_parquet_error(self, data_loader, sample_market_data):
        """Test writing data when Parquet fails."""
        data_loader._connected = True
        
        with patch.object(data_loader.parquet_store, 'write_batch_raw_lob', side_effect=Exception("Write error")):
            result = await data_loader.write_data(sample_market_data, use_raw=True)
            
            assert result == 0
    
    async def test_get_latest_data_not_connected(self, data_loader):
        """Test getting latest data when not connected."""
        with pytest.raises(RuntimeError, match="DataLoader not connected"):
            await data_loader.get_latest_data("BTCUSDT")
    
    async def test_get_latest_data_parquet_success(self, data_loader, sample_market_data):
        """Test getting latest data from Parquet."""
        data_loader._connected = True
        
        with patch.object(data_loader.parquet_store, 'read_latest_raw_lob', return_value=[sample_market_data]):
            result = await data_loader.get_latest_data("BTCUSDT", use_raw=True)
            
            assert len(result) == 1
    
    async def test_get_latest_data_fallback(self, data_loader, sample_market_data):
        """Test getting latest data with fallback to TimescaleDB."""
        data_loader._connected = True
        
        with patch.object(data_loader.parquet_store, 'read_latest_raw_lob', side_effect=Exception("Parquet error")):
            with patch.object(data_loader.timescale_client, 'get_latest_aggregated', new_callable=AsyncMock, return_value=[sample_market_data]):
                result = await data_loader.get_latest_data("BTCUSDT")
                
                assert len(result) == 1
    
    async def test_get_latest_data_both_fail(self, data_loader):
        """Test getting latest data when both backends fail."""
        data_loader._connected = True
        
        with patch.object(data_loader.parquet_store, 'read_latest_raw_lob', side_effect=Exception("Parquet error")):
            with patch.object(data_loader.timescale_client, 'get_latest_aggregated', new_callable=AsyncMock, side_effect=Exception("Timescale error")):
                result = await data_loader.get_latest_data("BTCUSDT")
                
                assert result == []
    
    async def test_get_ohlcv_data_not_connected(self, data_loader):
        """Test getting OHLCV data when not connected."""
        with pytest.raises(RuntimeError, match="DataLoader not connected"):
            await data_loader.get_ohlcv_data("BTCUSDT", datetime(2024, 1, 1), datetime(2024, 1, 2))
    
    async def test_get_ohlcv_data(self, data_loader):
        """Test getting OHLCV data."""
        data_loader._connected = True
        
        with patch.object(data_loader.timescale_client, 'aggregate_ohlcv', new_callable=AsyncMock, return_value=[]):
            result = await data_loader.get_ohlcv_data(
                "BTCUSDT",
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                "1m"
            )
            
            assert isinstance(result, list)
    
    async def test_create_dataset_version_not_connected(self, data_loader):
        """Test creating dataset version when not connected."""
        with pytest.raises(RuntimeError, match="DataLoader not connected"):
            await data_loader.create_dataset_version(
                "BTCUSDT",
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                "v1"
            )
    
    async def test_create_dataset_version(self, data_loader):
        """Test creating a dataset version."""
        data_loader._connected = True
        
        with patch.object(data_loader, 'read_data', new_callable=AsyncMock, return_value=[]):
            mock_conn = AsyncMock()
            mock_conn.execute = AsyncMock()
            mock_conn.fetchval = AsyncMock(return_value=1)
            
            mock_pool = MagicMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            data_loader.timescale_client.pool = mock_pool
            
            result = await data_loader.create_dataset_version(
                "BTCUSDT",
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                "v1",
                "Test version"
            )
            
            assert "1:v1" in result
    
    async def test_list_dataset_versions_not_connected(self, data_loader):
        """Test listing dataset versions when not connected."""
        with pytest.raises(RuntimeError, match="DataLoader not connected"):
            await data_loader.list_dataset_versions()
    
    async def test_list_dataset_versions(self, data_loader):
        """Test listing dataset versions."""
        data_loader._connected = True
        
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        data_loader.timescale_client.pool = mock_pool
        
        result = await data_loader.list_dataset_versions()
        
        assert isinstance(result, list)
    
    async def test_list_dataset_versions_with_symbol(self, data_loader):
        """Test listing dataset versions with symbol filter."""
        data_loader._connected = True
        
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        data_loader.timescale_client.pool = mock_pool
        
        result = await data_loader.list_dataset_versions(symbol="BTCUSDT")
        
        assert isinstance(result, list)
    
    async def test_load_dataset_version_not_connected(self, data_loader):
        """Test loading dataset version when not connected."""
        with pytest.raises(RuntimeError, match="DataLoader not connected"):
            await data_loader.load_dataset_version("v1")
    
    async def test_load_dataset_version(self, data_loader, sample_market_data):
        """Test loading a dataset version."""
        data_loader._connected = True
        
        mock_row = {
            "symbol": "BTCUSDT",
            "start_time": datetime(2024, 1, 1),
            "end_time": datetime(2024, 1, 2)
        }
        
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)
        
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        data_loader.timescale_client.pool = mock_pool
        
        with patch.object(data_loader, 'read_data', new_callable=AsyncMock, return_value=[sample_market_data]):
            result = await data_loader.load_dataset_version("v1")
            
            assert len(result) == 1
    
    async def test_load_dataset_version_not_found(self, data_loader):
        """Test loading non-existent dataset version."""
        data_loader._connected = True
        
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        data_loader.timescale_client.pool = mock_pool
        
        with pytest.raises(ValueError, match="Dataset version v1 not found"):
            await data_loader.load_dataset_version("v1")
    
    def test_get_storage_statistics(self, data_loader):
        """Test getting storage statistics."""
        with patch.object(data_loader.parquet_store, 'get_statistics', return_value={"total_size": 100}):
            stats = data_loader.get_storage_statistics()
            
            assert "parquet" in stats
            assert "timescale" in stats
    
    def test_get_storage_statistics_with_timescale(self, data_loader):
        """Test getting storage statistics with TimescaleDB."""
        data_loader._connected = True
        
        with patch.object(data_loader.parquet_store, 'get_statistics', return_value={"total_size": 100}):
            with patch.object(data_loader.timescale_client, 'get_statistics', new_callable=AsyncMock, return_value={"total_rows": 1000}):
                with patch('asyncio.run', return_value={"total_rows": 1000}):
                    stats = data_loader.get_storage_statistics()
                    
                    assert stats["parquet"]["total_size"] == 100
                    assert stats["timescale"]["total_rows"] == 1000
    
    async def test_optimize_storage(self, data_loader):
        """Test optimizing storage."""
        with patch.object(data_loader.parquet_store, 'optimize_dataset'):
            with patch.object(data_loader.parquet_store, 'cleanup_old_data'):
                await data_loader.optimize_storage()
                
                # Should not raise error
                assert True
    
    async def test_optimize_storage_parquet_error(self, data_loader):
        """Test optimizing storage when Parquet fails."""
        with patch.object(data_loader.parquet_store, 'optimize_dataset', side_effect=Exception("Optimize error")):
            with patch.object(data_loader.parquet_store, 'cleanup_old_data'):
                await data_loader.optimize_storage()
                
                # Should not raise error, just log
                assert True
