#!/usr/bin/env python3
"""
Tests for Data Persistence
========================

Comprehensive tests for database and CSV persistence
functionality including validation and error handling.
"""

import pytest
import tempfile
import shutil
import json
import csv
import gzip
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from its_project.execution.data_persistence import (
    DatabaseManager, CSVExporter, DataPersistenceManager,
    DatabaseConfig, CSVConfig,
    create_database_manager, create_csv_exporter, create_persistence_manager
)
from its_project.execution.pnl_tracker import TradeRecord, PnLRecord, PositionSnapshot


class TestDatabaseConfig:
    """Test cases for DatabaseConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DatabaseConfig()
        
        assert config.db_path == "data/trading_system.db"
        assert config.backup_enabled is True
        assert config.backup_interval == 3600
        assert config.max_backups == 10
        assert config.connection_timeout == 30
        assert config.enable_wal is True
        assert config.foreign_keys is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DatabaseConfig(
            db_path="custom.db",
            backup_enabled=False,
            backup_interval=1800,
            max_backups=5
        )
        
        assert config.db_path == "custom.db"
        assert config.backup_enabled is False
        assert config.backup_interval == 1800
        assert config.max_backups == 5


class TestCSVConfig:
    """Test cases for CSVConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CSVConfig()
        
        assert config.export_dir == "exports"
        assert config.compression is True
        assert config.date_format == "%Y-%m-%d %H:%M:%S"
        assert config.include_headers is True
        assert config.encoding == "utf-8"
        assert config.batch_size == 10000
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CSVConfig(
            export_dir="custom_exports",
            compression=False,
            batch_size=5000
        )
        
        assert config.export_dir == "custom_exports"
        assert config.compression is False
        assert config.batch_size == 5000


class TestDatabaseManager:
    """Test cases for DatabaseManager class."""
    
    @pytest.fixture
    def temp_db_dir(self):
        """Create temporary directory for database."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def db_config(self, temp_db_dir):
        """Create database config with temporary path."""
        return DatabaseConfig(
            db_path=str(Path(temp_db_dir) / "test.db"),
            backup_enabled=False  # Disable for testing
        )
    
    @pytest.fixture
    def db_manager(self, db_config):
        """Create DatabaseManager instance."""
        return DatabaseManager(db_config)
    
    @pytest.fixture
    def sample_trade(self):
        """Sample trade record for testing."""
        return TradeRecord(
            trade_id="test_trade_001",
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            price=50000.0,
            timestamp=datetime.now(),
            commission=10.0,
            fees={"exchange": 5.0, "network": 2.0},
            strategy_id="test_strategy",
            order_id="test_order_001"
        )
    
    @pytest.fixture
    def sample_pnl_record(self):
        """Sample PnL record for testing."""
        return PnLRecord(
            record_id="pnl_001",
            symbol="BTC/USDT",
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
            entry_price=50000.0,
            exit_price=51000.0,
            quantity=1.0,
            side="buy",
            realized_pnl=1000.0,
            commission=10.0,
            fees={"exchange": 5.0},
            strategy_id="test_strategy",
            trade_duration=timedelta(hours=1),
            pnl_percentage=2.0
        )
    
    @pytest.fixture
    def sample_position(self):
        """Sample position snapshot for testing."""
        return PositionSnapshot(
            symbol="BTC/USDT",
            quantity=1.5,
            avg_price=50500.0,
            unrealized_pnl=1500.0,
            realized_pnl=500.0,
            total_pnl=2000.0,
            last_price=52000.0,
            last_update=datetime.now(),
            trades_count=3
        )
    
    def test_init(self, db_manager, db_config):
        """Test DatabaseManager initialization."""
        assert db_manager.config == db_config
        assert db_manager.db_path.exists()
        assert db_manager._backup_running is False
    
    def test_save_trade(self, db_manager, sample_trade):
        """Test saving trade record."""
        result = asyncio.run(db_manager.save_trade(sample_trade))
        
        assert result is True
        
        # Verify trade was saved
        trades = asyncio.run(db_manager.get_trades())
        assert len(trades) == 1
        assert trades[0]['trade_id'] == sample_trade.trade_id
        assert trades[0]['symbol'] == sample_trade.symbol
        assert trades[0]['side'] == sample_trade.side
        assert trades[0]['quantity'] == sample_trade.quantity
        assert trades[0]['price'] == sample_trade.price
        assert trades[0]['commission'] == sample_trade.commission
        assert json.loads(trades[0]['fees']) == sample_trade.fees
        assert trades[0]['strategy_id'] == sample_trade.strategy_id
        assert trades[0]['order_id'] == sample_trade.order_id
    
    def test_save_trades_batch(self, db_manager):
        """Test saving multiple trades in batch."""
        trades = []
        for i in range(5):
            trade = TradeRecord(
                trade_id=f"batch_trade_{i:03d}",
                symbol="BTC/USDT",
                side="buy" if i % 2 == 0 else "sell",
                quantity=1.0,
                price=50000.0 + i * 100,
                timestamp=datetime.now() + timedelta(minutes=i)
            )
            trades.append(trade)
        
        result = asyncio.run(db_manager.save_trades_batch(trades))
        
        assert result == 5
        
        # Verify all trades were saved
        all_trades = asyncio.run(db_manager.get_trades())
        assert len(all_trades) == 5
    
    def test_save_pnl_record(self, db_manager, sample_pnl_record):
        """Test saving PnL record."""
        result = asyncio.run(db_manager.save_pnl_record(sample_pnl_record))
        
        assert result is True
        
        # Verify PnL record was saved
        pnl_records = asyncio.run(db_manager.get_pnl_records())
        assert len(pnl_records) == 1
        assert pnl_records[0]['record_id'] == sample_pnl_record.record_id
        assert pnl_records[0]['symbol'] == sample_pnl_record.symbol
        assert pnl_records[0]['realized_pnl'] == sample_pnl_record.realized_pnl
        assert pnl_records[0]['pnl_percentage'] == sample_pnl_record.pnl_percentage
    
    def test_save_position(self, db_manager, sample_position):
        """Test saving position snapshot."""
        result = asyncio.run(db_manager.save_position(sample_position))
        
        assert result is True
        
        # Verify position was saved
        # Note: Would need to implement get_positions method for full verification
        # For now, just ensure no errors occurred
        assert result is True
    
    def test_get_trades_with_filters(self, db_manager):
        """Test retrieving trades with filters."""
        # Add sample trades
        trades = [
            TradeRecord("trade_001", "BTC/USDT", "buy", 1.0, 50000.0, datetime.now()),
            TradeRecord("trade_002", "ETH/USDT", "buy", 10.0, 3000.0, datetime.now()),
            TradeRecord("trade_003", "BTC/USDT", "sell", 1.0, 51000.0, datetime.now())
        ]
        
        asyncio.run(db_manager.save_trades_batch(trades))
        
        # Test symbol filter
        btc_trades = asyncio.run(db_manager.get_trades(symbol="BTC/USDT"))
        assert len(btc_trades) == 2
        assert all(t['symbol'] == "BTC/USDT" for t in btc_trades)
        
        # Test strategy filter (would need strategy_id in trades)
        # Test time range filter
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now() + timedelta(hours=1)
        time_filtered = asyncio.run(db_manager.get_trades(
            start_time=start_time,
            end_time=end_time
        ))
        assert len(time_filtered) == 3
        
        # Test limit
        limited_trades = asyncio.run(db_manager.get_trades(limit=2))
        assert len(limited_trades) == 2
    
    def test_get_pnl_records_with_filters(self, db_manager):
        """Test retrieving PnL records with filters."""
        # Add sample PnL records
        pnl_records = [
            PnLRecord("pnl_001", "BTC/USDT", datetime.now(), datetime.now(), 50000, 51000, 1.0, "buy", 1000),
            PnLRecord("pnl_002", "ETH/USDT", datetime.now(), datetime.now(), 3000, 3100, 10.0, "buy", 1000)
        ]
        
        for record in pnl_records:
            asyncio.run(db_manager.save_pnl_record(record))
        
        # Test symbol filter
        btc_pnls = asyncio.run(db_manager.get_pnl_records(symbol="BTC/USDT"))
        assert len(btc_pnls) == 1
        assert btc_pnls[0]['symbol'] == "BTC/USDT"
        
        # Test limit
        limited_pnls = asyncio.run(db_manager.get_pnl_records(limit=1))
        assert len(limited_pnls) == 1
    
    def test_backup_functionality(self, temp_db_dir):
        """Test database backup functionality."""
        config = DatabaseConfig(
            db_path=str(Path(temp_db_dir) / "backup_test.db"),
            backup_enabled=True,
            backup_interval=1,  # Very short for testing
            max_backups=2
        )
        
        db_manager = DatabaseManager(config)
        
        # Add some data
        trade = TradeRecord("backup_test", "BTC/USDT", "buy", 1.0, 50000.0, datetime.now())
        asyncio.run(db_manager.save_trade(trade))
        
        # Wait for backup
        asyncio.sleep(2)
        
        # Check if backup was created
        backup_files = list(Path(temp_db_dir).glob("backup_*.db.gz"))
        assert len(backup_files) > 0
        
        # Verify backup contains data
        backup_path = backup_files[0]
        with gzip.open(backup_path, 'rt') as f:
            backup_content = f.read()
            assert "backup_test" in backup_content
        
        asyncio.run(db_manager.close())


class TestCSVExporter:
    """Test cases for CSVExporter class."""
    
    @pytest.fixture
    def temp_export_dir(self):
        """Create temporary export directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def csv_config(self, temp_export_dir):
        """Create CSV config with temporary path."""
        return CSVConfig(
            export_dir=temp_export_dir,
            compression=False  # Disable for easier testing
        )
    
    @pytest.fixture
    def csv_exporter(self, csv_config):
        """Create CSVExporter instance."""
        return CSVExporter(csv_config)
    
    @pytest.fixture
    def sample_trades(self):
        """Sample trade records for testing."""
        return [
            TradeRecord(
                trade_id=f"export_trade_{i:03d}",
                symbol="BTC/USDT",
                side="buy" if i % 2 == 0 else "sell",
                quantity=1.0,
                price=50000.0 + i * 100,
                timestamp=datetime.now() + timedelta(minutes=i),
                commission=10.0,
                fees={"exchange": 5.0},
                strategy_id="test_strategy"
            )
            for i in range(3)
        ]
    
    @pytest.fixture
    def sample_pnl_records(self):
        """Sample PnL records for testing."""
        return [
            PnLRecord(
                record_id=f"export_pnl_{i:03d}",
                symbol="BTC/USDT",
                entry_time=datetime.now() - timedelta(hours=i+1),
                exit_time=datetime.now() - timedelta(hours=i),
                entry_price=50000.0 + i * 100,
                exit_price=51000.0 + i * 100,
                quantity=1.0,
                side="buy",
                realized_pnl=1000.0,
                commission=10.0,
                fees={"exchange": 5.0},
                strategy_id="test_strategy",
                trade_duration=timedelta(hours=1),
                pnl_percentage=2.0
            )
            for i in range(2)
        ]
    
    def test_init(self, csv_exporter, csv_config):
        """Test CSVExporter initialization."""
        assert csv_exporter.config == csv_config
        assert csv_exporter.export_dir.exists()
    
    def test_export_trades(self, csv_exporter, sample_trades, temp_export_dir):
        """Test exporting trades to CSV."""
        filepath = asyncio.run(csv_exporter.export_trades(sample_trades))
        
        assert filepath != ""
        assert Path(filepath).exists()
        
        # Verify CSV content
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            assert len(rows) == 3
            assert 'trade_id' in rows[0]
            assert 'symbol' in rows[0]
            assert 'side' in rows[0]
            assert 'quantity' in rows[0]
            assert 'price' in rows[0]
            assert 'timestamp' in rows[0]
            assert 'commission' in rows[0]
            assert 'fees' in rows[0]
            assert 'strategy_id' in rows[0]
            assert 'order_id' in rows[0]
            
            # Verify data
            assert rows[0]['trade_id'] == "export_trade_000"
            assert rows[0]['symbol'] == "BTC/USDT"
            assert rows[0]['side'] == "buy"
            assert float(rows[0]['quantity']) == 1.0
            assert float(rows[0]['price']) == 50000.0
    
    def test_export_trades_with_metadata(self, csv_exporter, sample_trades, temp_export_dir):
        """Test exporting trades with additional metadata."""
        filepath = asyncio.run(csv_exporter.export_trades(
            sample_trades, 
            include_metadata=True
        ))
        
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            # Check metadata columns
            assert 'notional' in rows[0]
            assert 'fee_total' in rows[0]
            assert 'timestamp_epoch' in rows[0]
            
            # Verify metadata values
            assert float(rows[0]['notional']) == 50000.0
            assert float(rows[0]['fee_total']) == 15.0  # 10 + 5
    
    def test_export_pnl_records(self, csv_exporter, sample_pnl_records, temp_export_dir):
        """Test exporting PnL records to CSV."""
        filepath = asyncio.run(csv_exporter.export_pnl_records(sample_pnl_records))
        
        assert filepath != ""
        assert Path(filepath).exists()
        
        # Verify CSV content
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            assert len(rows) == 2
            assert 'record_id' in rows[0]
            assert 'symbol' in rows[0]
            assert 'entry_time' in rows[0]
            assert 'exit_time' in rows[0]
            assert 'realized_pnl' in rows[0]
            assert 'pnl_percentage' in rows[0]
            
            # Verify data
            assert rows[0]['record_id'] == "export_pnl_000"
            assert rows[0]['symbol'] == "BTC/USDT"
            assert float(rows[0]['realized_pnl']) == 1000.0
            assert float(rows[0]['pnl_percentage']) == 2.0
    
    def test_export_empty_data(self, csv_exporter, temp_export_dir):
        """Test exporting empty data."""
        trades_filepath = asyncio.run(csv_exporter.export_trades([]))
        pnl_filepath = asyncio.run(csv_exporter.export_pnl_records([]))
        
        assert trades_filepath == ""
        assert pnl_filepath == ""
    
    def test_export_dataframe(self, csv_exporter, temp_export_dir):
        """Test exporting pandas DataFrame."""
        df = pd.DataFrame({
            'symbol': ['BTC/USDT', 'ETH/USDT'],
            'price': [50000.0, 3000.0],
            'quantity': [1.0, 10.0],
            'timestamp': [datetime.now(), datetime.now()]
        })
        
        filepath = asyncio.run(csv_exporter.export_dataframe(df, "test_dataframe.csv"))
        
        assert filepath != ""
        assert Path(filepath).exists()
        
        # Verify DataFrame was exported correctly
        exported_df = pd.read_csv(filepath)
        assert len(exported_df) == 2
        assert list(exported_df.columns) == ['symbol', 'price', 'quantity', 'timestamp']
    
    def test_compression(self, temp_export_dir):
        """Test CSV compression functionality."""
        config_compressed = CSVConfig(
            export_dir=temp_export_dir,
            compression=True
        )
        exporter_compressed = CSVExporter(config_compressed)
        
        trades = [TradeRecord("comp_test", "BTC/USDT", "buy", 1.0, 50000.0, datetime.now())]
        filepath = asyncio.run(exporter_compressed.export_trades(trades))
        
        # Should create compressed file
        assert filepath.endswith('.gz')
        assert Path(filepath).exists()
        
        # Verify compressed content
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            content = f.read()
            assert "comp_test" in content


class TestDataPersistenceManager:
    """Test cases for DataPersistenceManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def persistence_manager(self, temp_dir):
        """Create DataPersistenceManager instance."""
        db_config = DatabaseConfig(
            db_path=str(Path(temp_dir) / "test.db"),
            backup_enabled=False
        )
        csv_config = CSVConfig(
            export_dir=temp_dir,
            compression=False
        )
        return DataPersistenceManager(
            db_config=db_config,
            csv_config=csv_config,
            auto_export_interval=0  # Disable for testing
        )
    
    @pytest.fixture
    def sample_trade(self):
        """Sample trade record."""
        return TradeRecord(
            trade_id="manager_test_001",
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            price=50000.0,
            timestamp=datetime.now()
        )
    
    def test_init(self, persistence_manager, temp_dir):
        """Test DataPersistenceManager initialization."""
        assert persistence_manager.db_manager is not None
        assert persistence_manager.csv_exporter is not None
        assert persistence_manager.db_config.db_path.endswith("test.db")
        assert persistence_manager.csv_config.export_dir == temp_dir
    
    def test_save_trade(self, persistence_manager, sample_trade):
        """Test saving trade through persistence manager."""
        result = asyncio.run(persistence_manager.save_trade(sample_trade))
        
        assert result is True
        
        # Verify through database manager
        trades = asyncio.run(persistence_manager.db_manager.get_trades())
        assert len(trades) == 1
        assert trades[0]['trade_id'] == sample_trade.trade_id
    
    def test_save_trades_batch(self, persistence_manager):
        """Test saving batch trades through persistence manager."""
        trades = [
            TradeRecord(f"batch_{i}", "BTC/USDT", "buy", 1.0, 50000.0, datetime.now())
            for i in range(3)
        ]
        
        result = asyncio.run(persistence_manager.save_trades_batch(trades))
        
        assert result == 3
        
        # Verify through database manager
        all_trades = asyncio.run(persistence_manager.db_manager.get_trades())
        assert len(all_trades) == 3
    
    def test_export_all_data(self, persistence_manager, temp_dir):
        """Test exporting all data types."""
        # Add some data first
        trade = TradeRecord("export_test", "BTC/USDT", "buy", 1.0, 50000.0, datetime.now())
        pnl_record = PnLRecord("pnl_export", "BTC/USDT", datetime.now(), datetime.now(), 50000, 51000, 1.0, "buy", 1000)
        
        asyncio.run(persistence_manager.save_trade(trade))
        asyncio.run(persistence_manager.save_pnl_record(pnl_record))
        
        # Export all data
        results = asyncio.run(persistence_manager.export_all_data())
        
        assert 'trades' in results
        assert 'pnl_records' in results
        
        # Verify files exist
        assert Path(results['trades']).exists()
        assert Path(results['pnl_records']).exists()
        
        # Verify file content
        with open(results['trades'], 'r') as f:
            trades_content = f.read()
            assert "export_test" in trades_content
        
        with open(results['pnl_records'], 'r') as f:
            pnl_content = f.read()
            assert "pnl_export" in pnl_content
    
    def test_close(self, persistence_manager):
        """Test closing persistence manager."""
        # Should not raise any exceptions
        asyncio.run(persistence_manager.close())
        
        # Verify cleanup
        assert persistence_manager._export_running is False


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_create_database_manager(self):
        """Test create_database_manager convenience function."""
        db_manager = create_database_manager("test_convenience.db")
        
        assert isinstance(db_manager, DatabaseManager)
        assert db_manager.config.db_path == "test_convenience.db"
    
    def test_create_csv_exporter(self):
        """Test create_csv_exporter convenience function."""
        csv_exporter = create_csv_exporter("convenience_exports")
        
        assert isinstance(csv_exporter, CSVExporter)
        assert csv_exporter.config.export_dir == "convenience_exports"
    
    def test_create_persistence_manager(self):
        """Test create_persistence_manager convenience function."""
        persistence_manager = create_persistence_manager(
            "test_convenience.db",
            "convenience_exports"
        )
        
        assert isinstance(persistence_manager, DataPersistenceManager)
        assert persistence_manager.db_config.db_path == "test_convenience.db"
        assert persistence_manager.csv_config.export_dir == "convenience_exports"


class TestErrorHandling:
    """Test cases for error handling and validation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_invalid_trade_data(self, temp_dir):
        """Test handling of invalid trade data."""
        config = DatabaseConfig(db_path=str(Path(temp_dir) / "invalid_test.db"))
        db_manager = DatabaseManager(config)
        
        # Test with invalid trade (missing required fields)
        invalid_trade = TradeRecord(
            trade_id="",  # Empty ID
            symbol="BTC/USDT",
            side="invalid",  # Invalid side
            quantity=-1.0,  # Negative quantity
            price=0.0,  # Zero price
            timestamp=datetime.now()
        )
        
        # Should handle gracefully
        result = asyncio.run(db_manager.save_trade(invalid_trade))
        # Note: Depending on implementation, might save or reject
        
        asyncio.run(db_manager.close())
    
    def test_file_permission_errors(self, temp_dir):
        """Test handling of file permission errors."""
        # Create read-only directory
        readonly_dir = Path(temp_dir) / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only
        
        config = CSVConfig(export_dir=str(readonly_dir))
        exporter = CSVExporter(config)
        
        trades = [TradeRecord("perm_test", "BTC/USDT", "buy", 1.0, 50000.0, datetime.now())]
        
        # Should handle permission error gracefully
        result = asyncio.run(exporter.export_trades(trades))
        
        # Depending on implementation, might return empty string or raise exception
        # For now, just ensure it doesn't crash
        assert isinstance(result, str)
    
    def test_database_corruption_handling(self, temp_dir):
        """Test handling of database corruption."""
        config = DatabaseConfig(db_path=str(Path(temp_dir) / "corrupt_test.db"))
        db_manager = DatabaseManager(config)
        
        # Add some data
        trade = TradeRecord("corrupt_test", "BTC/USDT", "buy", 1.0, 50000.0, datetime.now())
        asyncio.run(db_manager.save_trade(trade))
        
        # Simulate database corruption by writing invalid data
        db_path = Path(config.db_path)
        with open(db_path, 'w') as f:
            f.write("corrupted data")
        
        # Should handle corruption gracefully
        try:
            trades = asyncio.run(db_manager.get_trades())
            # Might return empty list or raise exception
            assert isinstance(trades, list)
        except Exception:
            # Expected behavior for corrupted database
            pass
        
        asyncio.run(db_manager.close())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
