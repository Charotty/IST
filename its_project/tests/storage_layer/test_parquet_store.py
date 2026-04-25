"""
Unit tests for ParquetStore.
"""
import pytest
import pandas as pd
import pyarrow as pa
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from its_project.storage.parquet_store import ParquetStore
from its_project.common.types import MarketData, MarketDataType


@pytest.mark.unit
@pytest.mark.storage_layer
class TestParquetStore:
    """Test ParquetStore class."""
    
    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "parquet_store"
    
    @pytest.fixture
    def parquet_store(self, temp_dir):
        """Create ParquetStore instance."""
        return ParquetStore(
            base_path=temp_dir,
            partition_cols=["symbol", "date"],
            compression="snappy",
            row_group_size=100000
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
                "bids": [["100", "1.0"], ["99", "2.0"]],
                "asks": [["101", "1.0"], ["102", "2.0"]],
                "_ts_recv_ms": 1234567890000,
                "p": "",
                "q": "",
                "t": 0
            }
        )
    
    def test_initialization(self, temp_dir):
        """Test ParquetStore initialization."""
        store = ParquetStore(
            base_path=temp_dir,
            partition_cols=["symbol", "date"],
            compression="snappy",
            row_group_size=100000
        )
        
        assert store.base_path == temp_dir
        assert store.partition_cols == ["symbol", "date"]
        assert store.compression == "snappy"
        assert store.row_group_size == 100000
        assert temp_dir.exists()
    
    def test_initialization_default_partition_cols(self, temp_dir):
        """Test initialization with default partition columns."""
        store = ParquetStore(base_path=temp_dir)
        
        assert store.partition_cols == ["symbol", "date"]
    
    def test_initialize_dataset(self, parquet_store):
        """Test dataset initialization."""
        # The dataset should be initialized during __init__
        assert parquet_store.base_path.exists()
    
    @patch('its_project.storage.parquet_store.ds.write_dataset')
    def test_write_raw_lob(self, mock_write, parquet_store, sample_market_data):
        """Test writing raw LOB data."""
        parquet_store.write_raw_lob(sample_market_data)
        
        mock_write.assert_called_once()
    
    @patch('its_project.storage.parquet_store.ds.write_dataset')
    def test_write_batch_raw_lob(self, mock_write, parquet_store, sample_market_data):
        """Test writing batch of raw LOB data."""
        data_list = [sample_market_data, sample_market_data]
        
        parquet_store.write_batch_raw_lob(data_list)
        
        mock_write.assert_called_once()
    
    def test_write_batch_raw_lob_empty(self, parquet_store):
        """Test writing empty batch."""
        with patch('its_project.storage.parquet_store.ds.write_dataset') as mock_write:
            parquet_store.write_batch_raw_lob([])
            
            mock_write.assert_not_called()
    
    @patch('its_project.storage.parquet_store.ds.dataset')
    def test_read_raw_lob(self, mock_dataset, parquet_store):
        """Test reading raw LOB data."""
        # Mock dataset
        mock_ds_instance = Mock()
        mock_table = pa.Table.from_arrays([
            [1234567890000],
            ["BTCUSDT"],
            ["orderbook"],
            ["binance"]
        ], names=["timestamp_ms", "symbol", "data_type", "exchange"])
        
        # Add data column
        data_array = pa.array([{
            "_kind": "orderBook",
            "_reconstructed": False,
            "lastUpdateId": 12345,
            "bids": [],
            "asks": [],
            "_ts_recv_ms": 1234567890000,
            "p": "",
            "q": "",
            "t": 0
        }], type=pa.struct([
            pa.field("_kind", pa.string()),
            pa.field("_reconstructed", pa.bool_()),
            pa.field("lastUpdateId", pa.int64()),
            pa.field("bids", pa.list_(pa.list_(pa.string()))),
            pa.field("asks", pa.list_(pa.list_(pa.string()))),
            pa.field("_ts_recv_ms", pa.int64()),
            pa.field("p", pa.string()),
            pa.field("q", pa.string()),
            pa.field("t", pa.int64())
        ]))
        
        mock_table = mock_table.add_column(4, "data", data_array)
        
        mock_ds_instance.to_table.return_value = mock_table
        mock_dataset.return_value = mock_ds_instance
        
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 1, 2)
        
        results = parquet_store.read_raw_lob("BTCUSDT", start_time, end_time)
        
        assert isinstance(results, list)
    
    @patch('its_project.storage.parquet_store.ds.dataset')
    def test_read_latest_raw_lob(self, mock_dataset, parquet_store):
        """Test reading latest raw LOB data."""
        # Mock dataset
        mock_ds_instance = Mock()
        mock_table = pa.Table.from_arrays([
            [1234567890000],
            ["BTCUSDT"],
            ["orderbook"],
            ["binance"]
        ], names=["timestamp_ms", "symbol", "data_type", "exchange"])
        
        # Add data column
        data_array = pa.array([{
            "_kind": "orderBook",
            "_reconstructed": False,
            "lastUpdateId": 12345,
            "bids": [],
            "asks": [],
            "_ts_recv_ms": 1234567890000,
            "p": "",
            "q": "",
            "t": 0
        }], type=pa.struct([
            pa.field("_kind", pa.string()),
            pa.field("_reconstructed", pa.bool_()),
            pa.field("lastUpdateId", pa.int64()),
            pa.field("bids", pa.list_(pa.list_(pa.string()))),
            pa.field("asks", pa.list_(pa.list_(pa.string()))),
            pa.field("_ts_recv_ms", pa.int64()),
            pa.field("p", pa.string()),
            pa.field("q", pa.string()),
            pa.field("t", pa.int64())
        ]))
        
        mock_table = mock_table.add_column(4, "data", data_array)
        
        mock_ds_instance.to_table.return_value = mock_table
        mock_dataset.return_value = mock_ds_instance
        
        results = parquet_store.read_latest_raw_lob("BTCUSDT", limit=100)
        
        assert isinstance(results, list)
    
    @patch('its_project.storage.parquet_store.ds.dataset')
    def test_read_latest_raw_lob_error(self, mock_dataset, parquet_store):
        """Test reading latest raw LOB data with error."""
        mock_dataset.side_effect = Exception("Test error")
        
        results = parquet_store.read_latest_raw_lob("BTCUSDT", limit=100)
        
        assert results == []
    
    def test_marketdata_to_table(self, parquet_store, sample_market_data):
        """Test converting MarketData to Arrow Table."""
        date_str = "2024-01-01"
        
        table = parquet_store._marketdata_to_table([sample_market_data], [date_str])
        
        assert isinstance(table, pa.Table)
        assert len(table) == 1
        assert table.column("timestamp_ms")[0].as_py() == sample_market_data.timestamp_ms
        assert table.column("symbol")[0].as_py() == sample_market_data.symbol
    
    def test_marketdata_to_table_batch(self, parquet_store, sample_market_data):
        """Test converting batch of MarketData to Arrow Table."""
        data_list = [sample_market_data, sample_market_data]
        date_strs = ["2024-01-01", "2024-01-02"]
        
        table = parquet_store._marketdata_to_table(data_list, date_strs)
        
        assert isinstance(table, pa.Table)
        assert len(table) == 2
    
    def test_marketdata_to_table_missing_fields(self, parquet_store):
        """Test converting MarketData with missing fields."""
        # Create MarketData with minimal data
        minimal_data = MarketData(
            timestamp_ms=1234567890000,
            symbol="BTCUSDT",
            type=MarketDataType.ORDERBOOK,
            exchange="binance",
            data={}  # Empty data dict
        )
        
        date_str = "2024-01-01"
        
        table = parquet_store._marketdata_to_table([minimal_data], [date_str])
        
        assert isinstance(table, pa.Table)
        assert len(table) == 1
    
    def test_table_to_marketdata(self, parquet_store):
        """Test converting Arrow Table to MarketData list."""
        # Create a simple table
        table = pa.Table.from_arrays([
            [1234567890000],
            ["BTCUSDT"],
            ["orderbook"],
            ["binance"]
        ], names=["timestamp_ms", "symbol", "data_type", "exchange"])
        
        # Add data column
        data_array = pa.array([{
            "_kind": "orderBook",
            "_reconstructed": False,
            "lastUpdateId": 12345,
            "bids": [],
            "asks": [],
            "_ts_recv_ms": 1234567890000,
            "p": "",
            "q": "",
            "t": 0
        }], type=pa.struct([
            pa.field("_kind", pa.string()),
            pa.field("_reconstructed", pa.bool_()),
            pa.field("lastUpdateId", pa.int64()),
            pa.field("bids", pa.list_(pa.list_(pa.string()))),
            pa.field("asks", pa.list_(pa.list_(pa.string()))),
            pa.field("_ts_recv_ms", pa.int64()),
            pa.field("p", pa.string()),
            pa.field("q", pa.string()),
            pa.field("t", pa.int64())
        ]))
        
        table = table.add_column(4, "data", data_array)
        
        results = parquet_store._table_to_marketdata(table)
        
        assert len(results) == 1
        assert isinstance(results[0], MarketData)
        assert results[0].symbol == "BTCUSDT"
    
    def test_table_to_marketdata_empty(self, parquet_store):
        """Test converting empty Arrow Table."""
        table = pa.Table.from_arrays([
            [],
            [],
            [],
            []
        ], names=["timestamp_ms", "symbol", "data_type", "exchange"])
        
        # Add empty data column
        data_array = pa.array([], type=pa.struct([
            pa.field("_kind", pa.string()),
            pa.field("_reconstructed", pa.bool_()),
            pa.field("lastUpdateId", pa.int64()),
            pa.field("bids", pa.list_(pa.list_(pa.string()))),
            pa.field("asks", pa.list_(pa.list_(pa.string()))),
            pa.field("_ts_recv_ms", pa.int64()),
            pa.field("p", pa.string()),
            pa.field("q", pa.string()),
            pa.field("t", pa.int64())
        ]))
        
        table = table.add_column(4, "data", data_array)
        
        results = parquet_store._table_to_marketdata(table)
        
        assert results == []
    
    @patch('its_project.storage.parquet_store.ds.dataset')
    def test_get_statistics(self, mock_dataset, parquet_store):
        """Test getting storage statistics."""
        # Mock dataset
        mock_ds_instance = Mock()
        mock_ds_instance.partitions.dictionaries = [{"symbol": ["BTCUSDT", "ETHUSDT"]}]
        mock_ds_instance.head.return_value = pa.Table.from_arrays([
            list(range(100))
        ], names=["col"])
        
        mock_dataset.return_value = mock_ds_instance
        
        # Create a dummy parquet file
        (parquet_store.base_path / "test.parquet").write_text("dummy")
        
        stats = parquet_store.get_statistics()
        
        assert "total_size_bytes" in stats
        assert "total_size_mb" in stats
        assert "file_count" in stats
        assert "unique_symbols" in stats
        assert "base_path" in stats
    
    @patch('its_project.storage.parquet_store.ds.dataset')
    def test_get_statistics_error(self, mock_dataset, parquet_store):
        """Test getting statistics with error."""
        mock_dataset.side_effect = Exception("Test error")
        
        stats = parquet_store.get_statistics()
        
        assert "error" in stats
    
    @patch('shutil.rmtree')
    def test_cleanup_old_data(self, mock_rmtree, parquet_store):
        """Test cleaning up old data."""
        # Create a mock directory structure
        symbol_dir = parquet_store.base_path / "symbol=BTCUSDT"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        parquet_store.cleanup_old_data(days_to_keep=30)
        
        # Should not raise error
        assert True
    
    @patch('its_project.storage.parquet_store.ds.dataset')
    @patch('its_project.storage.parquet_store.ds.write_dataset')
    def test_optimize_dataset(self, mock_write, mock_dataset, parquet_store):
        """Test optimizing dataset."""
        # Mock dataset
        mock_ds_instance = Mock()
        mock_ds_instance.partitions.partitioning = ["symbol=BTCUSDT"]
        mock_ds_instance.to_table.return_value = pa.Table.from_arrays([
            [1, 2, 3]
        ], names=["col"])
        
        mock_dataset.return_value = mock_ds_instance
        
        # Create a mock partition directory
        partition_path = parquet_store.base_path / "symbol=BTCUSDT"
        partition_path.mkdir(parents=True, exist_ok=True)
        
        # Create multiple small files
        for i in range(15):
            (partition_path / f"file_{i}.parquet").write_text("dummy")
        
        parquet_store.optimize_dataset()
        
        # Should not raise error
        assert True
    
    @patch('its_project.storage.parquet_store.ds.dataset')
    def test_optimize_dataset_error(self, mock_dataset, parquet_store):
        """Test optimizing dataset with error."""
        mock_dataset.side_effect = Exception("Test error")
        
        # Should not raise error, just log
        parquet_store.optimize_dataset()
        assert True
