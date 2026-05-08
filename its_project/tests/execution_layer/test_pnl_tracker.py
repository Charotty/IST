#!/usr/bin/env python3
"""
Tests for PnL Tracking System
============================

Comprehensive tests for per-trade and cumulative PnL tracking.
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from its_project.execution.pnl_tracker import (
    PnLTracker, TradeRecord, PositionSnapshot, PnLRecord,
    create_pnl_tracker, calculate_simple_pnl
)
from its_project.execution.cumulative_pnl import (
    CumulativePnLTracker, CumulativePnLSnapshot, PerformanceMetrics
)


class TestPnLTracker:
    """Test cases for PnLTracker class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_pnl.db"
        yield db_path
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def pnl_tracker(self, temp_db):
        """Create PnLTracker instance for testing."""
        return PnLTracker(db_path=str(temp_db), auto_save=False)
    
    def test_init(self, temp_db):
        """Test PnLTracker initialization."""
        tracker = PnLTracker(db_path=str(temp_db))
        
        assert tracker.db_path == Path(temp_db)
        assert tracker.positions == {}
        assert tracker.trades == []
        assert tracker.pnl_records == []
        assert tracker.cumulative_pnl == 0.0
    
    def test_add_trade_buy(self, pnl_tracker):
        """Test adding a buy trade."""
        trade_id = "trade_001"
        symbol = "BTC/USDT"
        side = "buy"
        quantity = 1.0
        price = 50000.0
        
        pnl_tracker.add_trade(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price
        )
        
        # Check trade was added
        assert len(pnl_tracker.trades) == 1
        trade = pnl_tracker.trades[0]
        assert trade.trade_id == trade_id
        assert trade.symbol == symbol
        assert trade.side == side
        assert trade.quantity == quantity
        assert trade.price == price
        
        # Check position was created
        assert symbol in pnl_tracker.positions
        position = pnl_tracker.positions[symbol]
        assert position.quantity == quantity
        assert position.avg_price == price
    
    def test_add_trade_sell(self, pnl_tracker):
        """Test adding a sell trade."""
        # First add a buy to create position
        pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0)
        
        # Then add a sell trade
        pnl_tracker.add_trade("sell_001", "BTC/USDT", "sell", 1.0, 51000.0)
        
        # Check position is closed
        assert "BTC/USDT" not in pnl_tracker.positions
        
        # Check PnL record was created
        assert len(pnl_tracker.pnl_records) == 1
        pnl_record = pnl_tracker.pnl_records[0]
        assert pnl_record.symbol == "BTC/USDT"
        assert pnl_record.realized_pnl == 1000.0  # 51000 - 50000
    
    def test_partial_position_close(self, pnl_tracker):
        """Test partial position closing."""
        # Add buy trade
        pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 2.0, 50000.0)
        
        # Partial sell
        pnl_tracker.add_trade("sell_001", "BTC/USDT", "sell", 1.0, 51000.0)
        
        # Check remaining position
        position = pnl_tracker.positions["BTC/USDT"]
        assert position.quantity == 1.0
        assert position.avg_price == 50000.0
        
        # Check realized PnL
        assert len(pnl_tracker.pnl_records) == 1
        pnl_record = pnl_tracker.pnl_records[0]
        assert pnl_record.realized_pnl == 1000.0
    
    def test_short_position(self, pnl_tracker):
        """Test short position PnL calculation."""
        # Short sell
        pnl_tracker.add_trade("short_001", "BTC/USDT", "sell", 1.0, 50000.0)
        
        # Check position
        position = pnl_tracker.positions["BTC/USDT"]
        assert position.quantity == -1.0
        assert position.avg_price == 50000.0
        
        # Cover short
        pnl_tracker.add_trade("cover_001", "BTC/USDT", "buy", 1.0, 49000.0)
        
        # Check PnL (profit from short: 50000 - 49000 = 1000)
        assert len(pnl_tracker.pnl_records) == 1
        pnl_record = pnl_tracker.pnl_records[0]
        assert pnl_record.realized_pnl == 1000.0
    
    def test_update_market_price(self, pnl_tracker):
        """Test updating market prices for unrealized PnL."""
        # Add position
        pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0)
        
        # Update market price
        pnl_tracker.update_market_price("BTC/USDT", 52000.0)
        
        # Check unrealized PnL
        position = pnl_tracker.positions["BTC/USDT"]
        assert position.unrealized_pnl == 2000.0  # 52000 - 50000
        assert position.total_pnl == 2000.0
    
    def test_cumulative_pnl(self, pnl_tracker):
        """Test cumulative PnL calculation."""
        # Add and close profitable trade
        pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0)
        pnl_tracker.add_trade("sell_001", "BTC/USDT", "sell", 1.0, 51000.0)
        
        # Add another position
        pnl_tracker.add_trade("buy_002", "ETH/USDT", "buy", 10.0, 3000.0)
        pnl_tracker.update_market_price("ETH/USDT", 3100.0)
        
        # Check cumulative PnL
        cumulative = pnl_tracker.get_cumulative_pnl()
        assert cumulative == 2000.0  # 1000 realized + 1000 unrealized
    
    def test_performance_metrics(self, pnl_tracker):
        """Test performance metrics calculation."""
        # Add multiple trades
        pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0)
        pnl_tracker.add_trade("sell_001", "BTC/USDT", "sell", 1.0, 51000.0)  # +1000
        pnl_tracker.add_trade("buy_002", "BTC/USDT", "buy", 1.0, 50000.0)
        pnl_tracker.add_trade("sell_002", "BTC/USDT", "sell", 1.0, 49000.0)  # -1000
        pnl_tracker.add_trade("buy_003", "BTC/USDT", "buy", 1.0, 50000.0)
        pnl_tracker.add_trade("sell_003", "BTC/USDT", "sell", 1.0, 52000.0)  # +2000
        
        metrics = pnl_tracker.get_performance_metrics()
        
        assert metrics['total_trades'] == 3
        assert metrics['winning_trades'] == 2
        assert metrics['losing_trades'] == 1
        assert metrics['win_rate'] == 66.66666666666666
        assert metrics['total_realized_pnl'] == 2000.0
        assert metrics['avg_win'] == 1500.0
        assert metrics['avg_loss'] == -1000.0
    
    def test_strategy_pnl(self, pnl_tracker):
        """Test strategy-specific PnL tracking."""
        strategy_id = "momentum_strategy"
        
        # Add trades with strategy
        pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0, strategy_id=strategy_id)
        pnl_tracker.add_trade("sell_001", "BTC/USDT", "sell", 1.0, 51000.0, strategy_id=strategy_id)
        
        # Check strategy PnL
        strategy_pnl = pnl_tracker.get_strategy_pnl(strategy_id)
        assert strategy_pnl == 1000.0
    
    def test_pnl_dataframe(self, pnl_tracker):
        """Test PnL DataFrame export."""
        # Add trade
        pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0)
        pnl_tracker.add_trade("sell_001", "BTC/USDT", "sell", 1.0, 51000.0)
        
        df = pnl_tracker.get_pnl_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'symbol' in df.columns
        assert 'realized_pnl' in df.columns
        assert df.iloc[0]['symbol'] == "BTC/USDT"
        assert df.iloc[0]['realized_pnl'] == 1000.0
    
    def test_positions_dataframe(self, pnl_tracker):
        """Test positions DataFrame export."""
        # Add position
        pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0)
        pnl_tracker.update_market_price("BTC/USDT", 52000.0)
        
        df = pnl_tracker.get_positions_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'symbol' in df.columns
        assert 'quantity' in df.columns
        assert df.iloc[0]['symbol'] == "BTC/USDT"
        assert df.iloc[0]['quantity'] == 1.0
    
    def test_persistence(self, temp_db):
        """Test data persistence."""
        # Create tracker and add data
        tracker1 = PnLTracker(db_path=str(temp_db))
        tracker1.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0)
        tracker1.save()
        
        # Create new tracker and load data
        tracker2 = PnLTracker(db_path=str(temp_db))
        
        assert len(tracker2.trades) == 1
        assert tracker2.trades[0].trade_id == "buy_001"
    
    def test_reset(self, pnl_tracker):
        """Test resetting PnL tracker."""
        # Add data
        pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0)
        
        # Reset
        pnl_tracker.reset()
        
        # Check data is cleared
        assert len(pnl_tracker.trades) == 0
        assert len(pnl_tracker.positions) == 0
        assert len(pnl_tracker.pnl_records) == 0
        assert pnl_tracker.cumulative_pnl == 0.0


class TestCumulativePnLTracker:
    """Test cases for CumulativePnLTracker class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_pnl.db"
        yield db_path
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def pnl_tracker(self, temp_db):
        """Create PnLTracker instance for testing."""
        return PnLTracker(db_path=str(temp_db), auto_save=False)
    
    @pytest.fixture
    def cumulative_tracker(self, pnl_tracker):
        """Create CumulativePnLTracker instance for testing."""
        return CumulativePnLTracker(pnl_tracker)
    
    def test_init(self, pnl_tracker):
        """Test CumulativePnLTracker initialization."""
        cum_tracker = CumulativePnLTracker(pnl_tracker)
        
        assert cum_tracker.pnl_tracker == pnl_tracker
        assert cum_tracker.equity_curve == []
        assert cum_tracker.daily_returns == []
        assert cum_tracker.high_watermark == 0.0
        assert cum_tracker.max_drawdown == 0.0
    
    def test_update_with_trades(self, cumulative_tracker, pnl_tracker):
        """Test updating cumulative tracker with trades."""
        # Add profitable trade
        pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0)
        pnl_tracker.add_trade("sell_001", "BTC/USDT", "sell", 1.0, 51000.0)
        
        # Update cumulative tracker
        snapshot = cumulative_tracker.update()
        
        assert isinstance(snapshot, CumulativePnLSnapshot)
        assert snapshot.total_pnl == 1000.0
        assert snapshot.realized_pnl == 1000.0
        assert snapshot.unrealized_pnl == 0.0
        assert snapshot.total_trades == 1
        assert snapshot.open_positions == 0
    
    def test_equity_curve_tracking(self, cumulative_tracker, pnl_tracker):
        """Test equity curve tracking."""
        # Add multiple trades over time
        base_time = datetime.now()
        
        pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0)
        cumulative_tracker.update(base_time)
        
        pnl_tracker.add_trade("sell_001", "BTC/USDT", "sell", 1.0, 51000.0)
        cumulative_tracker.update(base_time + timedelta(hours=1))
        
        # Check equity curve
        assert len(cumulative_tracker.equity_curve) == 2
        assert cumulative_tracker.equity_curve[0][1] == 0.0  # After buy, unrealized = 0
        assert cumulative_tracker.equity_curve[1][1] == 1000.0  # After sell, realized = 1000
    
    def test_drawdown_tracking(self, cumulative_tracker, pnl_tracker):
        """Test drawdown tracking."""
        # Add profitable trade
        pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0)
        pnl_tracker.add_trade("sell_001", "BTC/USDT", "sell", 1.0, 55000.0)
        cumulative_tracker.update()
        
        # Add losing trade
        pnl_tracker.add_trade("buy_002", "BTC/USDT", "buy", 1.0, 50000.0)
        pnl_tracker.add_trade("sell_002", "BTC/USDT", "sell", 1.0, 48000.0)
        cumulative_tracker.update()
        
        # Check drawdown
        assert cumulative_tracker.high_watermark == 5000.0  # First trade profit
        assert cumulative_tracker.current_drawdown > 0  # In drawdown
        assert cumulative_tracker.max_drawdown > 0
    
    def test_performance_metrics_calculation(self, cumulative_tracker, pnl_tracker):
        """Test performance metrics calculation."""
        # Add multiple trades
        for i in range(10):
            pnl_tracker.add_trade(f"buy_{i}", "BTC/USDT", "buy", 1.0, 50000.0)
            pnl_tracker.add_trade(f"sell_{i}", "BTC/USDT", "sell", 1.0, 50000.0 + (i * 100))
            cumulative_tracker.update()
        
        metrics = cumulative_tracker._calculate_performance_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_trades == 10
        assert metrics.total_return > 0
        assert metrics.win_rate > 0
    
    def test_equity_curve_dataframe(self, cumulative_tracker, pnl_tracker):
        """Test equity curve DataFrame export."""
        # Add trades
        pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0)
        pnl_tracker.add_trade("sell_001", "BTC/USDT", "sell", 1.0, 51000.0)
        cumulative_tracker.update()
        
        df = cumulative_tracker.get_equity_curve_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert 'pnl' in df.columns
        assert 'returns' in df.columns
        assert len(df) >= 1
    
    def test_drawdown_dataframe(self, cumulative_tracker, pnl_tracker):
        """Test drawdown DataFrame export."""
        # Add trades with drawdown
        pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0)
        pnl_tracker.add_trade("sell_001", "BTC/USDT", "sell", 1.0, 55000.0)
        cumulative_tracker.update()
        
        pnl_tracker.add_trade("buy_002", "BTC/USDT", "buy", 1.0, 50000.0)
        pnl_tracker.add_trade("sell_002", "BTC/USDT", "sell", 1.0, 48000.0)
        cumulative_tracker.update()
        
        df = cumulative_tracker.get_drawdown_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert 'drawdown' in df.columns
        assert len(df) >= 1


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_create_pnl_tracker(self):
        """Test create_pnl_tracker utility function."""
        tracker = create_pnl_tracker()
        
        assert isinstance(tracker, PnLTracker)
        assert tracker.db_path == Path("data/pnl_tracking.db")
    
    def test_calculate_simple_pnl_long(self):
        """Test simple PnL calculation for long position."""
        pnl = calculate_simple_pnl(
            entry_price=50000.0,
            exit_price=51000.0,
            quantity=1.0,
            side="long",
            commission=10.0
        )
        
        assert pnl == 990.0  # (51000 - 50000) * 1 - 10
    
    def test_calculate_simple_pnl_short(self):
        """Test simple PnL calculation for short position."""
        pnl = calculate_simple_pnl(
            entry_price=50000.0,
            exit_price=49000.0,
            quantity=1.0,
            side="short",
            commission=10.0
        )
        
        assert pnl == 990.0  # (50000 - 49000) * 1 - 10
    
    def test_calculate_simple_pnl_loss(self):
        """Test simple PnL calculation for losing trade."""
        pnl = calculate_simple_pnl(
            entry_price=50000.0,
            exit_price=49000.0,
            quantity=1.0,
            side="long",
            commission=10.0
        )
        
        assert pnl == -1010.0  # (49000 - 50000) * 1 - 10


class TestIntegration:
    """Integration tests for PnL tracking system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_integration.db"
        yield db_path
        shutil.rmtree(temp_dir)
    
    def test_full_trading_workflow(self, temp_db):
        """Test complete trading workflow with PnL tracking."""
        # Setup
        pnl_tracker = PnLTracker(db_path=str(temp_db))
        cum_tracker = CumulativePnLTracker(pnl_tracker)
        
        # Simulate trading day
        trading_day = datetime.now()
        
        # Morning trades
        pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0, 
                            timestamp=trading_day.replace(hour=9, minute=30))
        cum_tracker.update()
        
        pnl_tracker.add_trade("buy_002", "ETH/USDT", "buy", 10.0, 3000.0,
                            timestamp=trading_day.replace(hour=10, minute=0))
        cum_tracker.update()
        
        # Mid-day price updates
        pnl_tracker.update_market_price("BTC/USDT", 51000.0)
        pnl_tracker.update_market_price("ETH/USDT", 3100.0)
        cum_tracker.update()
        
        # Afternoon trades
        pnl_tracker.add_trade("sell_001", "BTC/USDT", "sell", 1.0, 51000.0,
                            timestamp=trading_day.replace(hour=14, minute=0))
        cum_tracker.update()
        
        pnl_tracker.add_trade("sell_002", "ETH/USDT", "sell", 10.0, 3100.0,
                            timestamp=trading_day.replace(hour=14, minute=30))
        cum_tracker.update()
        
        # Verify results
        assert pnl_tracker.get_cumulative_pnl() == 2000.0  # 1000 BTC + 1000 ETH
        assert len(pnl_tracker.pnl_records) == 2
        assert len(cumulative_tracker.equity_curve) >= 4
        
        # Check performance metrics
        metrics = cum_tracker._calculate_performance_metrics()
        assert metrics.total_trades == 2
        assert metrics.win_rate == 100.0
        assert metrics.total_return == 2000.0
    
    def test_multi_strategy_tracking(self, temp_db):
        """Test tracking multiple strategies."""
        pnl_tracker = PnLTracker(db_path=str(temp_db))
        
        # Strategy 1 trades
        pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0, 
                            strategy_id="strategy_1")
        pnl_tracker.add_trade("sell_001", "BTC/USDT", "sell", 1.0, 51000.0,
                            strategy_id="strategy_1")
        
        # Strategy 2 trades
        pnl_tracker.add_trade("buy_002", "ETH/USDT", "buy", 10.0, 3000.0,
                            strategy_id="strategy_2")
        pnl_tracker.add_trade("sell_002", "ETH/USDT", "sell", 10.0, 2900.0,
                            strategy_id="strategy_2")
        
        # Check strategy-specific PnL
        assert pnl_tracker.get_strategy_pnl("strategy_1") == 1000.0
        assert pnl_tracker.get_strategy_pnl("strategy_2") == -1000.0
        assert pnl_tracker.get_cumulative_pnl() == 0.0
    
    def test_complex_position_management(self, temp_db):
        """Test complex position management scenarios."""
        pnl_tracker = PnLTracker(db_path=str(temp_db))
        
        # Build position gradually
        pnl_tracker.add_trade("buy_001", "BTC/USDT", "buy", 1.0, 50000.0)
        pnl_tracker.add_trade("buy_002", "BTC/USDT", "buy", 2.0, 51000.0)
        pnl_tracker.add_trade("buy_003", "BTC/USDT", "buy", 1.0, 49000.0)
        
        # Check average price calculation
        position = pnl_tracker.positions["BTC/USDT"]
        expected_avg = (50000*1.0 + 51000*2.0 + 49000*1.0) / 4.0
        assert abs(position.avg_price - expected_avg) < 0.01
        assert position.quantity == 4.0
        
        # Partial close
        pnl_tracker.add_trade("sell_001", "BTC/USDT", "sell", 3.0, 52000.0)
        
        # Check remaining position and realized PnL
        position = pnl_tracker.positions["BTC/USDT"]
        assert position.quantity == 1.0
        assert len(pnl_tracker.pnl_records) == 1
        
        # Complete close
        pnl_tracker.add_trade("sell_002", "BTC/USDT", "sell", 1.0, 53000.0)
        
        # Check position is closed
        assert "BTC/USDT" not in pnl_tracker.positions
        assert len(pnl_tracker.pnl_records) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
