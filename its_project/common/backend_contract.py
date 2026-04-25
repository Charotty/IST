"""
Backend Contract for OKX WS + CCXT Integration
==============================================

This module defines the data contract between GUI and backend service.
It is intentionally independent of backend module imports to keep GUI lightweight.

Schema version: 1.0
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class ChannelType(str, Enum):
    """OKX channel types."""
    TICKERS = "tickers"
    ORDERBOOK_L2 = "orderbook_l2"
    CANDLES = "candles"


class TradingMode(str, Enum):
    """Trading modes."""
    PAPER = "paper"
    LIVE = "live"


class SystemComponent(str, Enum):
    """System components for status reporting."""
    OKX_WS = "okx_ws"
    OKX_REST = "okx_rest"
    INGESTION = "ingestion"
    TRADING = "trading"
    STORAGE = "storage"


class ConnectionStatus(str, Enum):
    """Connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


# ============================================================================
# Event Schemas (WS → GUI)
# ============================================================================

@dataclass
class PriceUpdateEvent:
    """Price update from tickers channel."""
    symbol: str  # GUI format: BTC/USDT
    last: float
    bid: float
    ask: float
    ts_exchange: int  # Exchange timestamp in ms
    ts_local: int  # Local timestamp in ms
    volume_24h: Optional[float] = None
    change_24h: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None


@dataclass
class OrderbookUpdateEvent:
    """Orderbook L2 update."""
    symbol: str  # GUI format: BTC/USDT
    bids: List[List[float]]  # [[price, size], ...]
    asks: List[List[float]]  # [[price, size], ...]
    ts_exchange: int
    ts_local: int
    depth: int = 20  # Number of levels
    checksum: Optional[int] = None


@dataclass
class CandleUpdateEvent:
    """Candle/OHLCV update."""
    symbol: str  # GUI format: BTC/USDT
    timeframe: str  # e.g., "1m", "5m", "1h"
    open: float
    high: float
    low: float
    close: float
    volume: float
    ts_open: int  # Candle open timestamp in ms
    ts_local: int
    is_closed: bool = False  # True if candle is complete


@dataclass
class SignalUpdateEvent:
    """Signal update from decision engine (optional in MVP)."""
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    predicted_change: float  # Percentage
    model_name: str
    ts: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PaperTradeUpdateEvent:
    """Paper trade update."""
    id: str
    symbol: str
    side: str  # "BUY", "SELL"
    qty: float
    price: float
    fee: float
    status: str  # "OPEN", "CLOSED", "CANCELLED"
    ts: int
    exit_price: Optional[float] = None
    pnl: Optional[float] = None


@dataclass
class MetricsUpdateEvent:
    """Trading metrics update."""
    pnl: float  # Percentage
    drawdown: float  # Percentage
    exposure: float  # Current position value
    trades_count: int
    win_rate: float  # Percentage
    ts: int
    sharpe: Optional[float] = None
    profit_factor: Optional[float] = None


@dataclass
class LogEvent:
    """Log event from backend."""
    level: str  # "info", "warning", "error", "debug"
    message: str
    component: str  # Source component
    ts: int
    details: Optional[Dict[str, Any]] = None


@dataclass
class SystemStatusEvent:
    """System status heartbeat."""
    connected_okx_ws: ConnectionStatus
    connected_okx_rest: ConnectionStatus
    running_ingestion: bool
    running_trading: bool
    queues: Dict[str, int]  # Queue sizes
    last_heartbeat: int
    ts: int
    active_symbols: List[str] = None
    active_channels: List[str] = None


# Union type for all events
BackendEvent = Union[
    PriceUpdateEvent,
    OrderbookUpdateEvent,
    CandleUpdateEvent,
    SignalUpdateEvent,
    PaperTradeUpdateEvent,
    MetricsUpdateEvent,
    LogEvent,
    SystemStatusEvent
]


# ============================================================================
# Command Schemas (GUI → HTTP)
# ============================================================================

@dataclass
class ConnectCommand:
    """Connect to backend."""
    pass


@dataclass
class DisconnectCommand:
    """Disconnect from backend."""
    pass


@dataclass
class StartIngestionCommand:
    """Start data ingestion."""
    symbols: List[str]  # GUI format: ["BTC/USDT", "ETH/USDT"]
    channels: List[str]  # ["tickers", "orderbook_l2", "candles"]
    depth: int = 20  # For orderbook
    candle_timeframes: List[str] = None  # e.g., ["1m", "5m"]


@dataclass
class StopIngestionCommand:
    """Stop data ingestion."""
    pass


@dataclass
class StartTradingCommand:
    """Start paper trading."""
    mode: TradingMode = TradingMode.PAPER


@dataclass
class StopTradingCommand:
    """Stop trading."""
    pass


@dataclass
class SetModeCommand:
    """Set trading mode."""
    mode: TradingMode


@dataclass
class UpdateSubscriptionsCommand:
    """Update active subscriptions."""
    symbols: List[str]
    channels: List[str]
    depth: int = 20
    candle_timeframes: List[str] = None


@dataclass
class PaperResetCommand:
    """Reset paper trading portfolio."""
    pass


# ============================================================================
# Response Schemas (HTTP → GUI)
# ============================================================================

@dataclass
class CommandResponse:
    """Generic command response."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    ts: int = None


@dataclass
class HealthResponse:
    """Health check response."""
    healthy: bool
    version: str
    uptime_seconds: int
    components: Dict[str, str]  # Component status
    ts: int


@dataclass
class StatusResponse:
    """Detailed status response."""
    connected_okx_ws: ConnectionStatus
    connected_okx_rest: ConnectionStatus
    running_ingestion: bool
    running_trading: bool
    trading_mode: TradingMode
    active_symbols: List[str]
    active_channels: List[str]
    queues: Dict[str, int]
    last_heartbeat: int
    ts: int


# ============================================================================
# Symbol Mapping Utilities
# ============================================================================

class SymbolMapper:
    """Map between GUI symbol format and OKX symbol format."""
    
    # GUI format: BTC/USDT
    # OKX format: BTC-USDT (SPOT) or BTC-USDT-SWAP (SWAP)
    
    @staticmethod
    def gui_to_okx(symbol: str, market_type: str = "SPOT") -> str:
        """Convert GUI symbol to OKX format."""
        # Replace "/" with "-"
        base_quote = symbol.replace("/", "-")
        if market_type == "SPOT":
            return base_quote
        elif market_type == "SWAP":
            return f"{base_quote}-SWAP"
        else:
            return base_quote
    
    @staticmethod
    def okx_to_gui(symbol: str) -> str:
        """Convert OKX symbol to GUI format."""
        # Remove "-SWAP" suffix and replace "-" with "/"
        if symbol.endswith("-SWAP"):
            symbol = symbol[:-5]
        return symbol.replace("-", "/")  # Fallback


# ============================================================================
# Event Type Discriminator
# ============================================================================

EVENT_TYPE_MAP = {
    "price_update": PriceUpdateEvent,
    "orderbook_update": OrderbookUpdateEvent,
    "candle_update": CandleUpdateEvent,
    "signal_update": SignalUpdateEvent,
    "paper_trade_update": PaperTradeUpdateEvent,
    "metrics_update": MetricsUpdateEvent,
    "log_event": LogEvent,
    "system_status": SystemStatusEvent,
}


def parse_event(event_type: str, data: Dict[str, Any]) -> Optional[BackendEvent]:
    """Parse event from dict to appropriate dataclass."""
    event_class = EVENT_TYPE_MAP.get(event_type)
    if event_class is None:
        logger.warning(f"Unknown event type: {event_type}")
        return None
    
    try:
        return event_class(**data)
    except Exception as e:
        logger.error(f"Failed to parse event {event_type}: {e}")
        return None


def serialize_event(event: BackendEvent) -> Dict[str, Any]:
    """Serialize event to dict for WS transmission."""
    data = asdict(event)
    # Add event type field
    event_type = None
    for key, cls in EVENT_TYPE_MAP.items():
        if isinstance(event, cls):
            event_type = key
            break
    
    if event_type:
        data["event_type"] = event_type
    
    return data


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_symbol_gui_format(symbol: str) -> bool:
    """Validate GUI symbol format (e.g., BTC/USDT)."""
    if not symbol or not isinstance(symbol, str):
        return False
    
    parts = symbol.split("/")
    if len(parts) != 2:
        return False
    
    base, quote = parts
    if not base or not quote:
        return False
    
    if not base.isalnum() or not quote.isalnum():
        return False
    
    return True


def validate_channel(channel: str) -> bool:
    """Validate channel name."""
    return channel in [c.value for c in ChannelType]


def validate_trading_mode(mode: str) -> bool:
    """Validate trading mode."""
    return mode in [m.value for m in TradingMode]
