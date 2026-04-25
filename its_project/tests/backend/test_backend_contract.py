"""
Tests for backend contract (GUI↔backend data schemas).
"""

import pytest
from datetime import datetime

from common.backend_contract import (
    # Events
    PriceUpdateEvent,
    OrderbookUpdateEvent,
    CandleUpdateEvent,
    SignalUpdateEvent,
    PaperTradeUpdateEvent,
    MetricsUpdateEvent,
    LogEvent,
    SystemStatusEvent,
    # Enums
    ChannelType,
    TradingMode,
    ConnectionStatus,
    # Utils
    SymbolMapper,
    validate_symbol_gui_format,
    validate_channel,
    validate_trading_mode,
    parse_event,
    serialize_event,
)


class TestSymbolMapper:
    """Test symbol mapping between GUI and OKX formats."""
    
    def test_gui_to_okx_spot(self):
        """Test GUI to OKX conversion for SPOT."""
        assert SymbolMapper.gui_to_okx("BTC/USDT", "SPOT") == "BTCUSDT"
        assert SymbolMapper.gui_to_okx("ETH/USDT", "SPOT") == "ETHUSDT"
    
    def test_gui_to_okx_swap(self):
        """Test GUI to OKX conversion for SWAP."""
        assert SymbolMapper.gui_to_okx("BTC/USDT", "SWAP") == "BTCUSDT-SWAP"
        assert SymbolMapper.gui_to_okx("ETH/USDT", "SWAP") == "ETHUSDT-SWAP"
    
    def test_okx_to_gui(self):
        """Test OKX to GUI conversion."""
        assert SymbolMapper.okx_to_gui("BTCUSDT") == "BTC/USDT"
        assert SymbolMapper.okx_to_gui("ETHUSDT") == "ETH/USDT"
        assert SymbolMapper.okx_to_gui("BTCUSDT-SWAP") == "BTC/USDT"
    
    def test_roundtrip(self):
        """Test roundtrip conversion."""
        original = "BTC/USDT"
        okx = SymbolMapper.gui_to_okx(original, "SPOT")
        back = SymbolMapper.okx_to_gui(okx)
        assert back == original


class TestValidationHelpers:
    """Test validation helper functions."""
    
    def test_validate_symbol_gui_format_valid(self):
        """Test valid GUI symbol format."""
        assert validate_symbol_gui_format("BTC/USDT") is True
        assert validate_symbol_gui_format("ETH/USDT") is True
        assert validate_symbol_gui_format("SOL/USDT") is True
    
    def test_validate_symbol_gui_format_invalid(self):
        """Test invalid GUI symbol format."""
        assert validate_symbol_gui_format("") is False
        assert validate_symbol_gui_format("BTC") is False
        assert validate_symbol_gui_format("BTCUSDT") is False
        assert validate_symbol_gui_format(None) is False
    
    def test_validate_channel_valid(self):
        """Test valid channel names."""
        assert validate_channel("tickers") is True
        assert validate_channel("orderbook_l2") is True
        assert validate_channel("candles") is True
    
    def test_validate_channel_invalid(self):
        """Test invalid channel names."""
        assert validate_channel("invalid") is False
        assert validate_channel("") is False
        assert validate_channel("ticker") is False  # singular
    
    def test_validate_trading_mode_valid(self):
        """Test valid trading modes."""
        assert validate_trading_mode("paper") is True
        assert validate_trading_mode("live") is True
    
    def test_validate_trading_mode_invalid(self):
        """Test invalid trading modes."""
        assert validate_trading_mode("invalid") is False
        assert validate_trading_mode("") is False
        assert validate_trading_mode("test") is False


class TestEventSchemas:
    """Test event dataclass schemas."""
    
    def test_price_update_event(self):
        """Test PriceUpdateEvent creation."""
        event = PriceUpdateEvent(
            symbol="BTC/USDT",
            last=50000.0,
            bid=49999.0,
            ask=50001.0,
            ts_exchange=1234567890,
            ts_local=1234567891,
            volume_24h=1000.0,
            change_24h=2.5
        )
        assert event.symbol == "BTC/USDT"
        assert event.last == 50000.0
        assert event.bid == 49999.0
        assert event.ask == 50001.0
    
    def test_orderbook_update_event(self):
        """Test OrderbookUpdateEvent creation."""
        event = OrderbookUpdateEvent(
            symbol="BTC/USDT",
            bids=[[50000.0, 1.0], [49999.0, 2.0]],
            asks=[[50001.0, 1.0], [50002.0, 2.0]],
            ts_exchange=1234567890,
            ts_local=1234567891,
            depth=20
        )
        assert event.symbol == "BTC/USDT"
        assert len(event.bids) == 2
        assert len(event.asks) == 2
        assert event.depth == 20
    
    def test_candle_update_event(self):
        """Test CandleUpdateEvent creation."""
        event = CandleUpdateEvent(
            symbol="BTC/USDT",
            timeframe="1m",
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=10.0,
            ts_open=1234567890,
            ts_local=1234567891,
            is_closed=True
        )
        assert event.symbol == "BTC/USDT"
        assert event.timeframe == "1m"
        assert event.is_closed is True
    
    def test_signal_update_event(self):
        """Test SignalUpdateEvent creation."""
        event = SignalUpdateEvent(
            symbol="BTC/USDT",
            action="BUY",
            confidence=0.85,
            predicted_change=2.5,
            model_name="GRU-LSTM",
            ts=1234567890
        )
        assert event.symbol == "BTC/USDT"
        assert event.action == "BUY"
        assert event.confidence == 0.85
    
    def test_paper_trade_update_event(self):
        """Test PaperTradeUpdateEvent creation."""
        event = PaperTradeUpdateEvent(
            id="trade_123",
            symbol="BTC/USDT",
            side="BUY",
            qty=0.1,
            price=50000.0,
            fee=5.0,
            status="OPEN",
            ts=1234567890
        )
        assert event.id == "trade_123"
        assert event.symbol == "BTC/USDT"
        assert event.status == "OPEN"
    
    def test_metrics_update_event(self):
        """Test MetricsUpdateEvent creation."""
        event = MetricsUpdateEvent(
            pnl=10.5,
            drawdown=2.0,
            exposure=5000.0,
            trades_count=5,
            win_rate=60.0,
            ts=1234567890,
            sharpe=1.5,
            profit_factor=2.0
        )
        assert event.pnl == 10.5
        assert event.win_rate == 60.0
        assert event.trades_count == 5
    
    def test_log_event(self):
        """Test LogEvent creation."""
        event = LogEvent(
            level="info",
            message="Test message",
            component="test",
            ts=1234567890
        )
        assert event.level == "info"
        assert event.message == "Test message"
        assert event.component == "test"
    
    def test_system_status_event(self):
        """Test SystemStatusEvent creation."""
        event = SystemStatusEvent(
            connected_okx_ws=ConnectionStatus.CONNECTED,
            connected_okx_rest=ConnectionStatus.CONNECTED,
            running_ingestion=True,
            running_trading=True,
            queues={"price_updates": 10},
            last_heartbeat=1234567890,
            ts=1234567891,
            active_symbols=["BTC/USDT"],
            active_channels=["tickers"]
        )
        assert event.connected_okx_ws == ConnectionStatus.CONNECTED
        assert event.running_ingestion is True
        assert len(event.active_symbols) == 1


class TestEventSerialization:
    """Test event serialization and parsing."""
    
    def test_serialize_price_update(self):
        """Test PriceUpdateEvent serialization."""
        event = PriceUpdateEvent(
            symbol="BTC/USDT",
            last=50000.0,
            bid=49999.0,
            ask=50001.0,
            ts_exchange=1234567890,
            ts_local=1234567891
        )
        data = serialize_event(event)
        assert data["symbol"] == "BTC/USDT"
        assert data["event_type"] == "price_update"
    
    def test_parse_price_update(self):
        """Test PriceUpdateEvent parsing."""
        data = {
            "event_type": "price_update",
            "symbol": "BTC/USDT",
            "last": 50000.0,
            "bid": 49999.0,
            "ask": 50001.0,
            "ts_exchange": 1234567890,
            "ts_local": 1234567891
        }
        event = parse_event("price_update", data)
        assert isinstance(event, PriceUpdateEvent)
        assert event.symbol == "BTC/USDT"
    
    def test_parse_unknown_event_type(self):
        """Test parsing unknown event type."""
        data = {"symbol": "BTC/USDT"}
        event = parse_event("unknown", data)
        assert event is None
    
    def test_serialize_parse_roundtrip(self):
        """Test roundtrip serialization and parsing."""
        original = PriceUpdateEvent(
            symbol="BTC/USDT",
            last=50000.0,
            bid=49999.0,
            ask=50001.0,
            ts_exchange=1234567890,
            ts_local=1234567891
        )
        data = serialize_event(original)
        parsed = parse_event("price_update", data)
        assert parsed.symbol == original.symbol
        assert parsed.last == original.last
