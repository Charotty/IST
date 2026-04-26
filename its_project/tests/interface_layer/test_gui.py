"""Tests for PyQt GUI."""
import pytest
from unittest.mock import Mock, patch
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
import sys


@pytest.fixture
def qapp():
    """Create QApplication instance for testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    # Don't quit - let it persist for other tests


@pytest.fixture
def main_window(qapp):
    """Create main window instance."""
    from gui.main_window import ITSMainWindow
    window = ITSMainWindow()
    yield window
    window.close()


def test_main_window_init(main_window):
    """Test main window initialization."""
    assert main_window.current_symbol == "BTC/USDT"
    assert main_window.current_price == 0.0
    assert main_window.current_signal == "HOLD"
    assert main_window.system_status == "Idle"
    assert main_window.windowTitle() == "ITS - Intelligent Trading System"


def test_main_window_has_controls(main_window):
    """Test that main window has required controls."""
    assert hasattr(main_window, "symbol_combo")
    assert hasattr(main_window, "start_btn")
    assert hasattr(main_window, "stop_btn")
    assert hasattr(main_window, "status_label")
    assert hasattr(main_window, "price_label")
    assert hasattr(main_window, "signal_label")
    assert hasattr(main_window, "confidence_bar")


def test_symbol_combo_has_pairs(main_window):
    """Test that symbol combo has trading pairs."""
    count = main_window.symbol_combo.count()
    assert count > 0
    assert main_window.symbol_combo.itemText(0) == "BTC/USDT"


def test_on_symbol_changed(main_window):
    """Test symbol change handler."""
    main_window.on_symbol_changed("ETH/USDT")
    
    assert main_window.current_symbol == "ETH/USDT"


def test_on_start_trading(main_window):
    """Test start trading button."""
    main_window.on_start_trading()
    
    assert main_window.system_status == "Running"
    assert main_window.start_btn.isEnabled() is False
    assert main_window.stop_btn.isEnabled() is True
    assert main_window.status_label.text() == "Running"


def test_on_stop_trading(main_window):
    """Test stop trading button."""
    # First start
    main_window.on_start_trading()
    
    # Then stop
    main_window.on_stop_trading()
    
    assert main_window.system_status == "Stopped"
    assert main_window.start_btn.isEnabled() is True
    assert main_window.stop_btn.isEnabled() is False
    assert main_window.status_label.text() == "Stopped"


def test_on_price_update(main_window):
    """Test price update handler."""
    main_window.on_price_update("BTC/USDT", 42000.0)
    
    assert main_window.current_price == 42000.0
    assert "$42000.00" in main_window.price_label.text()


def test_on_signal_received_buy(main_window):
    """Test signal received handler for BUY."""
    signal_data = {
        "action": "BUY",
        "confidence": 0.85,
        "price": 42000.0,
        "timestamp": None,
    }
    
    main_window.on_signal_received(signal_data)
    
    assert main_window.current_signal == "BUY"
    assert main_window.current_confidence == 0.85
    assert main_window.signal_label.text() == "BUY"


def test_on_signal_received_sell(main_window):
    """Test signal received handler for SELL."""
    signal_data = {
        "action": "SELL",
        "confidence": 0.75,
        "price": 42000.0,
        "timestamp": None,
    }
    
    main_window.on_signal_received(signal_data)
    
    assert main_window.current_signal == "SELL"
    assert main_window.current_confidence == 0.75


def test_on_status_update_model(main_window):
    """Test status update for model."""
    main_window.on_status_update("model", "GRU-LSTM")
    
    assert main_window.model_name == "GRU-LSTM"
    assert main_window.model_label.text() == "GRU-LSTM"


def test_on_status_update_trading_start(main_window):
    """Test status update for trading start."""
    main_window.on_status_update("trading", "start")
    
    assert main_window.system_status == "Running"


def test_on_status_update_trading_stop(main_window):
    """Test status update for trading stop."""
    main_window.on_status_update("trading", "stop")
    
    assert main_window.system_status == "Stopped"


def test_update_signal_color_buy(main_window):
    """Test signal color for BUY."""
    main_window.current_signal = "BUY"
    main_window.update_signal_color()
    
    # Check that color is green (contains green in stylesheet)
    assert "green" in main_window.signal_label.styleSheet().lower()


def test_update_signal_color_sell(main_window):
    """Test signal color for SELL."""
    main_window.current_signal = "SELL"
    main_window.update_signal_color()
    
    # Check that color is red
    assert "red" in main_window.signal_label.styleSheet().lower()


def test_update_signal_color_hold(main_window):
    """Test signal color for HOLD."""
    main_window.current_signal = "HOLD"
    main_window.update_signal_color()
    
    # Check that color is gray
    assert "gray" in main_window.signal_label.styleSheet().lower()


def test_add_to_history(main_window):
    """Test adding signal to history."""
    from datetime import datetime
    
    main_window.add_to_history(datetime.now(), "BUY", 0.85, 42000.0)
    
    assert main_window.history_table.rowCount() == 1


def test_history_limit(main_window):
    """Test that history is limited to 50 entries."""
    from datetime import datetime
    
    # Add more than 50 entries
    for i in range(60):
        main_window.add_to_history(datetime.now(), "BUY", 0.85, 42000.0)
    
    # Should be limited to 50
    assert main_window.history_table.rowCount() == 50


def test_log_message(main_window):
    """Test logging messages."""
    initial_text = main_window.log_text.toPlainText()
    main_window.log_message("Test message")
    
    new_text = main_window.log_text.toPlainText()
    assert "Test message" in new_text
    assert len(new_text) > len(initial_text)


def test_update_model_info(main_window):
    """Test updating model information."""
    main_window.update_model_info("GRU-LSTM", "2024-01-01", 1000, 5)
    
    assert main_window.model_label.text() == "GRU-LSTM"
    assert main_window.model_update_label.text() == "2024-01-01"
    assert main_window.samples_label.text() == "1000"
    assert main_window.retrain_count_label.text() == "5"


def test_update_performance_metrics(main_window):
    """Test updating performance metrics."""
    main_window.update_performance_metrics(1.5, 2.0, 0.65, 100)
    
    assert main_window.sharpe_label.text() == "1.50"
    assert main_window.sortino_label.text() == "2.00"
    assert main_window.win_rate_label.text() == "65.00%"
    assert main_window.trades_label.text() == "100"


def test_update_position(main_window):
    """Test updating position in table."""
    main_window.update_position("BTC/USDT", "long", 0.1, 42000.0, 100.0)
    
    assert main_window.positions_table.rowCount() == 1
    assert main_window.positions_table.item(0, 0).text() == "BTC/USDT"


def test_update_position_existing(main_window):
    """Test updating existing position."""
    # Add position
    main_window.update_position("BTC/USDT", "long", 0.1, 42000.0, 100.0)
    
    # Update same position
    main_window.update_position("BTC/USDT", "long", 0.2, 43000.0, 200.0)
    
    # Should still be 1 row
    assert main_window.positions_table.rowCount() == 1
    assert main_window.positions_table.item(0, 2).text() == "0.2000"


def test_update_order(main_window):
    """Test updating order in table."""
    main_window.update_order("order123", "BTC/USDT", "buy", 0.1, "open")
    
    assert main_window.orders_table.rowCount() == 1
    assert main_window.orders_table.item(0, 0).text() == "order123"


def test_update_order_existing(main_window):
    """Test updating existing order."""
    # Add order
    main_window.update_order("order123", "BTC/USDT", "buy", 0.1, "open")
    
    # Update same order
    main_window.update_order("order123", "BTC/USDT", "buy", 0.1, "filled")
    
    # Should still be 1 row
    assert main_window.orders_table.rowCount() == 1
    assert main_window.orders_table.item(0, 4).text() == "filled"


def test_signal_receiver_exists(main_window):
    """Test that signal receiver is initialized."""
    assert main_window.signal_receiver is not None
    assert hasattr(main_window.signal_receiver, "signal_received")
    assert hasattr(main_window.signal_receiver, "price_update")
    assert hasattr(main_window.signal_receiver, "status_update")


def test_timer_setup(main_window):
    """Test that update timer is set up."""
    assert main_window.update_timer is not None
    assert main_window.update_timer.isActive() is True


def test_get_spread_from_lob_delta():
    """Test LOB delta recovery spread calculation."""
    from data_layer.lob_delta import LOBDeltaRecovery, LOBSnapshot
    
    recovery = LOBDeltaRecovery()
    
    # Apply snapshot
    snapshot = LOBSnapshot(
        symbol="BTC/USDT",
        timestamp_ms=1234567890,
        bids=[(41990.0, 1.0), (41980.0, 2.0)],
        asks=[(42010.0, 1.0), (42020.0, 2.0)],
        sequence=1,
    )
    recovery.apply_snapshot(snapshot)
    
    spread = recovery.get_spread("BTC/USDT")
    
    assert spread == 20.0  # 42010 - 41990


def test_get_mid_price_from_lob_delta():
    """Test LOB delta recovery mid price calculation."""
    from data_layer.lob_delta import LOBDeltaRecovery, LOBSnapshot
    
    recovery = LOBDeltaRecovery()
    
    # Apply snapshot
    snapshot = LOBSnapshot(
        symbol="BTC/USDT",
        timestamp_ms=1234567890,
        bids=[(41990.0, 1.0)],
        asks=[(42010.0, 1.0)],
        sequence=1,
    )
    recovery.apply_snapshot(snapshot)
    
    mid_price = recovery.get_mid_price("BTC/USDT")
    
    assert mid_price == 42000.0  # (41990 + 42010) / 2
