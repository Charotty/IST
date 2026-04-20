from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from decimal import Decimal

import numpy as np
import pandas as pd

# Backend imports
try:
    from its_project.decision.decision import Action, Signal, Decision
    from its_project.common.types import Order, OrderStatus, OrderType, MarketData, MarketDataType
    from its_project.execution.trader import PaperTrader
    from its_project.metalearning.metrics import TradingMetrics
    from its_project.models.base import BaseModel
except ImportError as e:
    logging.warning(f"Backend modules not available: {e}")
    # Create fallback classes for demo mode
    class Action:
        BUY = "buy"
        SELL = "sell"
        HOLD = "hold"
    
    class Signal:
        def __init__(self, action, confidence, predicted_change, timestamp_ms, symbol):
            self.action = action
            self.confidence = confidence
            self.predicted_change = predicted_change
            self.timestamp_ms = timestamp_ms
            self.symbol = symbol
    
    class Order:
        def __init__(self, id, symbol, order_type, side, amount, price=None):
            self.id = id
            self.symbol = symbol
            self.type = order_type
            self.side = side
            self.amount = amount
            self.price = price
            self.status = "pending"
            self.filled_amount = 0.0
            self.average_price = 0.0
    
    class OrderStatus:
        PENDING = "pending"
        FILLED = "filled"
        PARTIALLY_FILLED = "partially_filled"
        CANCELLED = "cancelled"
        REJECTED = "rejected"
    
    class OrderType:
        MARKET = "market"
        LIMIT = "limit"
    
    class MarketDataType:
        TRADE = "trade"
        ORDERBOOK = "orderbook"
        SNAPSHOT = "snapshot"
    
    class MarketData:
        def __init__(self, timestamp_ms, symbol, type, exchange, data):
            self.timestamp_ms = timestamp_ms
            self.symbol = symbol
            self.type = type
            self.exchange = exchange
            self.data = data

logger = logging.getLogger(__name__)


# ============================================================================
# GUI Data Structures (PyQt Compatible)
# ============================================================================

@dataclass
class GUISignal:
    """GUI-compatible signal format."""
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    predicted_change: float  # Percentage change (-100 to +100)
    timestamp: datetime
    symbol: str
    model_name: str
    strength: float = 1.0  # Signal strength indicator
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GUITrade:
    """GUI-compatible trade format."""
    id: str
    timestamp: str  # "HH:MM:SS"
    pair: str
    type: str  # "BUY", "SELL"
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None  # Percentage
    status: str = "OPEN"  # "OPEN", "CLOSED"
    quantity: float = 1.0
    commission: float = 0.0
    slippage: float = 0.0
    duration: Optional[str] = None  # "00:00:00"


@dataclass
class GUIOrder:
    """GUI-compatible order format."""
    id: str
    symbol: str
    type: str  # "MARKET", "LIMIT"
    side: str  # "BUY", "SELL"
    amount: float
    price: Optional[float] = None
    status: str = "PENDING"
    filled_amount: float = 0.0
    average_price: float = 0.0
    created_at: str = ""
    updated_at: str = ""


@dataclass
class GUIMetrics:
    """GUI-compatible metrics format."""
    total_pnl: float
    win_rate: float
    total_trades: int
    max_drawdown: float
    sharpe_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    avg_trade_duration: Optional[str] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    largest_win: Optional[float] = None
    largest_loss: Optional[float] = None
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    equity_curve: Optional[List[float]] = None


@dataclass
class GUIModel:
    """GUI-compatible model format."""
    name: str
    type: str  # "GRU", "CNN", "Transformer", "Boosting"
    score: float  # 0.0 to 1.0
    is_active: bool
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    trading_metrics: Optional[Dict[str, float]] = None
    last_updated: Optional[datetime] = None
    prediction_count: int = 0
    success_rate: float = 0.0


@dataclass
class GUIPriceData:
    """GUI-compatible price data format."""
    timestamp: datetime
    symbol: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    prediction: Optional[float] = None
    signal: Optional[str] = None


@dataclass
class GUILogEntry:
    """GUI-compatible log entry format."""
    timestamp: str  # "HH:MM:SS"
    level: str  # "info", "warning", "error", "debug"
    message: str
    source: str = "SYSTEM"
    details: Optional[Dict[str, Any]] = None


# ============================================================================
# Data Conversion Utilities
# ============================================================================

class DataConverter:
    """Comprehensive data conversion utilities for GUI-backend compatibility."""
    
    @staticmethod
    def backend_signal_to_gui(backend_signal: Signal, model_name: str = "Unknown") -> GUISignal:
        """Convert backend Signal to GUI format."""
        action_map = {
            Action.BUY: "BUY",
            Action.SELL: "SELL",
            Action.HOLD: "HOLD"
        }
        
        # Calculate signal strength based on confidence and predicted change
        strength = backend_signal.confidence * abs(backend_signal.predicted_change) / 100.0
        strength = min(1.0, max(0.0, strength))
        
        return GUISignal(
            signal=action_map.get(backend_signal.action, "HOLD"),
            confidence=float(backend_signal.confidence),
            predicted_change=float(backend_signal.predicted_change * 100),  # Convert to percentage
            timestamp=datetime.fromtimestamp(backend_signal.timestamp_ms / 1000),
            symbol=backend_signal.symbol,
            model_name=model_name,
            strength=strength,
            metadata={
                "backend_timestamp": backend_signal.timestamp_ms,
                "raw_predicted_change": backend_signal.predicted_change
            }
        )
    
    @staticmethod
    def gui_signal_to_backend(gui_signal: GUISignal) -> Signal:
        """Convert GUI Signal to backend format."""
        reverse_action_map = {
            "BUY": Action.BUY,
            "SELL": Action.SELL,
            "HOLD": Action.HOLD
        }
        
        return Signal(
            action=reverse_action_map.get(gui_signal.signal, Action.HOLD),
            confidence=gui_signal.confidence,
            predicted_change=gui_signal.predicted_change / 100.0,  # Convert from percentage
            timestamp_ms=int(gui_signal.timestamp.timestamp() * 1000),
            symbol=gui_signal.symbol
        )
    
    @staticmethod
    def backend_order_to_gui(order: Order) -> GUIOrder:
        """Convert backend Order to GUI format."""
        status_map = {
            OrderStatus.PENDING: "PENDING",
            OrderStatus.FILLED: "FILLED",
            OrderStatus.PARTIALLY_FILLED: "PARTIALLY_FILLED",
            OrderStatus.CANCELLED: "CANCELLED",
            OrderStatus.REJECTED: "REJECTED"
        }
        
        type_map = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT"
        }
        
        created_at = datetime.fromtimestamp(order.created_at.timestamp() if hasattr(order, 'created_at') and order.created_at else datetime.now()).strftime("%H:%M:%S")
        updated_at = datetime.fromtimestamp(order.updated_at.timestamp() if hasattr(order, 'updated_at') and order.updated_at else datetime.now()).strftime("%H:%M:%S")
        
        return GUIOrder(
            id=order.id,
            symbol=order.symbol,
            type=type_map.get(order.type, "MARKET"),
            side=order.side.upper(),
            amount=float(order.amount),
            price=float(order.price) if order.price else None,
            status=status_map.get(order.status, "PENDING"),
            filled_amount=float(order.filled_amount),
            average_price=float(order.average_price),
            created_at=created_at,
            updated_at=updated_at
        )
    
    @staticmethod
    def gui_order_to_backend(gui_order: GUIOrder) -> Order:
        """Convert GUI Order to backend format."""
        reverse_status_map = {
            "PENDING": OrderStatus.PENDING,
            "FILLED": OrderStatus.FILLED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "CANCELLED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED
        }
        
        reverse_type_map = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT
        }
        
        order = Order(
            id=gui_order.id,
            symbol=gui_order.symbol,
            order_type=reverse_type_map.get(gui_order.type, OrderType.MARKET),
            side=gui_order.side.lower(),
            amount=gui_order.amount,
            price=gui_order.price
        )
        
        order.status = reverse_status_map.get(gui_order.status, OrderStatus.PENDING)
        order.filled_amount = gui_order.filled_amount
        order.average_price = gui_order.average_price
        
        return order
    
    @staticmethod
    def backend_trade_to_gui(order: Order, exit_price: Optional[float] = None) -> GUITrade:
        """Convert backend Order (filled) to GUI Trade format."""
        if order.status not in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
            raise ValueError("Order must be filled or partially filled to convert to trade")
        
        # Calculate PnL
        pnl = None
        if exit_price is not None and order.average_price > 0:
            if order.side.lower() == "buy":
                pnl = ((exit_price - order.average_price) / order.average_price) * 100
            else:
                pnl = ((order.average_price - exit_price) / order.average_price) * 100
        
        # Calculate duration
        duration = None
        if hasattr(order, 'created_at') and hasattr(order, 'updated_at'):
            duration = str(order.updated_at - order.created_at).split('.')[0]
        
        return GUITrade(
            id=order.id,
            timestamp=datetime.fromtimestamp(order.created_at.timestamp() if hasattr(order, 'created_at') and order.created_at else datetime.now()).strftime("%H:%M:%S"),
            pair=order.symbol,
            type=order.side.upper(),
            entry_price=float(order.average_price),
            exit_price=exit_price,
            pnl=pnl,
            status="CLOSED" if exit_price is not None else "OPEN",
            quantity=float(order.filled_amount),
            commission=0.0,  # Would be calculated based on commission rate
            slippage=0.0,   # Would be calculated based on slippage model
            duration=duration
        )
    
    @staticmethod
    def market_data_to_price_data(market_data: List[MarketData]) -> List[GUIPriceData]:
        """Convert MarketData list to GUIPriceData list."""
        price_data = []
        
        for md in market_data:
            if md.type == MarketDataType.TRADE:
                data = md.data
                price_data.append(GUIPriceData(
                    timestamp=datetime.fromtimestamp(md.timestamp_ms / 1000),
                    symbol=md.symbol,
                    open_price=float(data.get("price", 0)),
                    high_price=float(data.get("price", 0)),
                    low_price=float(data.get("price", 0)),
                    close_price=float(data.get("price", 0)),
                    volume=float(data.get("volume", 0))
                ))
            elif md.type == MarketDataType.SNAPSHOT:
                # Handle OHLCV snapshot
                data = md.data
                price_data.append(GUIPriceData(
                    timestamp=datetime.fromtimestamp(md.timestamp_ms / 1000),
                    symbol=md.symbol,
                    open_price=float(data.get("open", 0)),
                    high_price=float(data.get("high", 0)),
                    low_price=float(data.get("low", 0)),
                    close_price=float(data.get("close", 0)),
                    volume=float(data.get("volume", 0))
                ))
        
        return price_data
    
    @staticmethod
    def trading_metrics_to_gui(trading_metrics: TradingMetrics) -> GUIMetrics:
        """Convert backend TradingMetrics to GUI format."""
        return GUIMetrics(
            total_pnl=float(trading_metrics.total_return * 100),  # Convert to percentage
            win_rate=float(trading_metrics.win_rate * 100),     # Convert to percentage
            total_trades=int(trading_metrics.total_trades),
            max_drawdown=float(abs(trading_metrics.max_drawdown) * 100),  # Convert to percentage
            sharpe_ratio=float(trading_metrics.sharpe_ratio) if hasattr(trading_metrics, 'sharpe_ratio') else None,
            profit_factor=float(trading_metrics.profit_factor) if hasattr(trading_metrics, 'profit_factor') else None,
            sortino_ratio=float(trading_metrics.sortino_ratio) if hasattr(trading_metrics, 'sortino_ratio') else None,
            calmar_ratio=float(trading_metrics.calmar_ratio) if hasattr(trading_metrics, 'calmar_ratio') else None
        )
    
    @staticmethod
    def model_to_gui(model: BaseModel, is_active: bool = False, trading_metrics: Optional[Dict[str, float]] = None) -> GUIModel:
        """Convert backend BaseModel to GUI format."""
        # Extract model type from class name
        model_type = "Unknown"
        class_name = model.__class__.__name__.lower()
        if "gru" in class_name:
            model_type = "GRU"
        elif "cnn" in class_name:
            model_type = "CNN"
        elif "transformer" in class_name:
            model_type = "Transformer"
        elif "boosting" in class_name or "gradient" in class_name:
            model_type = "Boosting"
        
        return GUIModel(
            name=model.__class__.__name__,
            type=model_type,
            score=getattr(model, 'score', 0.75),  # Default score if not available
            is_active=is_active,
            accuracy=getattr(model, 'accuracy', None),
            precision=getattr(model, 'precision', None),
            recall=getattr(model, 'recall', None),
            f1_score=getattr(model, 'f1_score', None),
            trading_metrics=trading_metrics,
            last_updated=getattr(model, 'last_updated', datetime.now()),
            prediction_count=getattr(model, 'prediction_count', 0),
            success_rate=getattr(model, 'success_rate', 0.0)
        )
    
    @staticmethod
    def log_to_gui(level: str, message: str, source: str = "SYSTEM", details: Optional[Dict[str, Any]] = None) -> GUILogEntry:
        """Convert log data to GUI format."""
        return GUILogEntry(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            level=level.lower(),
            message=message,
            source=source,
            details=details
        )
    
    @staticmethod
    def calculate_derived_metrics(trades: List[GUITrade]) -> Dict[str, Any]:
        """Calculate derived metrics from trade list."""
        if not trades:
            return {}
        
        closed_trades = [t for t in trades if t.status == "CLOSED" and t.pnl is not None]
        
        if not closed_trades:
            return {
                "avg_win": None,
                "avg_loss": None,
                "largest_win": None,
                "largest_loss": None,
                "consecutive_wins": 0,
                "consecutive_losses": 0,
                "avg_trade_duration": "00:00:00"
            }
        
        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl < 0]
        
        # Calculate averages
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0
        largest_win = max([t.pnl for t in wins]) if wins else 0.0
        largest_loss = min([t.pnl for t in losses]) if losses else 0.0
        
        # Calculate consecutive streaks
        consecutive_wins = 0
        consecutive_losses = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for trade in closed_trades:
            if trade.pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
                consecutive_wins = max(consecutive_wins, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                consecutive_losses = max(consecutive_losses, current_loss_streak)
        
        # Calculate average duration
        durations = []
        for trade in closed_trades:
            if trade.duration:
                # Parse duration string "HH:MM:SS"
                try:
                    h, m, s = map(int, trade.duration.split(':'))
                    duration_seconds = h * 3600 + m * 60 + s
                    durations.append(duration_seconds)
                except:
                    pass
        
        avg_duration_seconds = np.mean(durations) if durations else 0
        avg_duration = f"{int(avg_duration_seconds // 3600):02d}:{int((avg_duration_seconds % 3600) // 60):02d}:{int(avg_duration_seconds % 60):02d}"
        
        return {
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "consecutive_wins": consecutive_wins,
            "consecutive_losses": consecutive_losses,
            "avg_trade_duration": avg_duration
        }
    
    @staticmethod
    def create_equity_curve(trades: List[GUITrade], initial_balance: float = 10000.0) -> List[float]:
        """Create equity curve from trade list."""
        equity = [initial_balance]
        current_balance = initial_balance
        
        closed_trades = sorted([t for t in trades if t.status == "CLOSED" and t.pnl is not None], 
                              key=lambda x: x.timestamp)
        
        for trade in closed_trades:
            # Calculate P&L in absolute terms
            if trade.pnl is not None:
                pnl_amount = current_balance * (trade.pnl / 100.0)
                current_balance += pnl_amount
                equity.append(current_balance)
        
        return equity
    
    @staticmethod
    def validate_gui_signal(signal: GUISignal) -> bool:
        """Validate GUI signal format."""
        try:
            # Check required fields
            if not all([signal.signal, signal.symbol, signal.model_name]):
                return False
            
            # Check signal value
            if signal.signal not in ["BUY", "SELL", "HOLD"]:
                return False
            
            # Check confidence range
            if not (0.0 <= signal.confidence <= 1.0):
                return False
            
            # Check predicted change range (reasonable limits)
            if not (-100.0 <= signal.predicted_change <= 100.0):
                return False
            
            # Check strength range
            if not (0.0 <= signal.strength <= 1.0):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Signal validation error: {e}")
            return False
    
    @staticmethod
    def validate_gui_trade(trade: GUITrade) -> bool:
        """Validate GUI trade format."""
        try:
            # Check required fields
            if not all([trade.id, trade.pair, trade.type, trade.timestamp]):
                return False
            
            # Check trade type
            if trade.type not in ["BUY", "SELL"]:
                return False
            
            # Check status
            if trade.status not in ["OPEN", "CLOSED"]:
                return False
            
            # Check prices
            if trade.entry_price <= 0:
                return False
            
            if trade.exit_price is not None and trade.exit_price <= 0:
                return False
            
            # Check PnL range if present
            if trade.pnl is not None and abs(trade.pnl) > 1000:  # Reasonable limit
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Trade validation error: {e}")
            return False
    
    @staticmethod
    def sanitize_symbol(symbol: str) -> str:
        """Sanitize trading symbol format."""
        # Remove common separators and convert to standard format
        sanitized = symbol.replace("/", "").replace("-", "").replace("_", "")
        
        # Ensure it's uppercase
        sanitized = sanitized.upper()
        
        return sanitized
    
    @staticmethod
    def format_price(price: Union[float, str, Decimal], decimals: int = 2) -> str:
        """Format price for display."""
        try:
            if isinstance(price, str):
                price = float(price)
            elif isinstance(price, Decimal):
                price = float(price)
            
            return f"${price:.{decimals}f}"
        except:
            return "$0.00"
    
    @staticmethod
    def format_percentage(value: Union[float, str, Decimal], decimals: int = 2) -> str:
        """Format percentage for display."""
        try:
            if isinstance(value, str):
                value = float(value)
            elif isinstance(value, Decimal):
                value = float(value)
            
            sign = "+" if value >= 0 else ""
            return f"{sign}{value:.{decimals}f}%"
        except:
            return "0.00%"
    
    @staticmethod
    def format_duration(seconds: Union[int, float]) -> str:
        """Format duration in seconds to HH:MM:SS format."""
        try:
            seconds = int(seconds)
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        except:
            return "00:00:00"


# ============================================================================
# Batch Conversion Utilities
# ============================================================================

class BatchConverter:
    """Batch conversion utilities for large datasets."""
    
    @staticmethod
    def convert_signals_batch(backend_signals: List[Signal], model_name: str = "Unknown") -> List[GUISignal]:
        """Convert multiple backend signals to GUI format."""
        return [DataConverter.backend_signal_to_gui(signal, model_name) for signal in backend_signals]
    
    @staticmethod
    def convert_trades_batch(orders: List[Order], exit_prices: Optional[Dict[str, float]] = None) -> List[GUITrade]:
        """Convert multiple backend orders to GUI trades."""
        trades = []
        for order in orders:
            exit_price = exit_prices.get(order.id) if exit_prices else None
            try:
                trade = DataConverter.backend_trade_to_gui(order, exit_price)
                trades.append(trade)
            except ValueError:
                # Skip orders that aren't filled
                continue
        return trades
    
    @staticmethod
    def convert_orders_batch(backend_orders: List[Order]) -> List[GUIOrder]:
        """Convert multiple backend orders to GUI format."""
        return [DataConverter.backend_order_to_gui(order) for order in backend_orders]
    
    @staticmethod
    def convert_market_data_batch(market_data_list: List[MarketData]) -> List[GUIPriceData]:
        """Convert multiple market data entries to GUI price data."""
        return DataConverter.market_data_to_price_data(market_data_list)
    
    @staticmethod
    def filter_signals_by_confidence(signals: List[GUISignal], min_confidence: float = 0.5) -> List[GUISignal]:
        """Filter signals by minimum confidence."""
        return [s for s in signals if s.confidence >= min_confidence]
    
    @staticmethod
    def filter_trades_by_symbol(trades: List[GUITrade], symbol: str) -> List[GUITrade]:
        """Filter trades by symbol."""
        return [t for t in trades if t.pair == symbol]
    
    @staticmethod
    def aggregate_trades_by_day(trades: List[GUITrade]) -> Dict[str, List[GUITrade]]:
        """Aggregate trades by day."""
        daily_trades = {}
        
        for trade in trades:
            try:
                # Extract date from timestamp
                date_str = trade.timestamp.split(' ')[0] if ' ' in trade.timestamp else trade.timestamp
                if date_str not in daily_trades:
                    daily_trades[date_str] = []
                daily_trades[date_str].append(trade)
            except:
                continue
        
        return daily_trades


# ============================================================================
# Export utilities
# ============================================================================

class DataExporter:
    """Export utilities for GUI data."""
    
    @staticmethod
    def trades_to_dataframe(trades: List[GUITrade]) -> pd.DataFrame:
        """Convert trades to pandas DataFrame."""
        data = []
        for trade in trades:
            data.append({
                'id': trade.id,
                'timestamp': trade.timestamp,
                'pair': trade.pair,
                'type': trade.type,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'status': trade.status,
                'quantity': trade.quantity,
                'commission': trade.commission,
                'slippage': trade.slippage,
                'duration': trade.duration
            })
        return pd.DataFrame(data)
    
    @staticmethod
    def signals_to_dataframe(signals: List[GUISignal]) -> pd.DataFrame:
        """Convert signals to pandas DataFrame."""
        data = []
        for signal in signals:
            data.append({
                'timestamp': signal.timestamp,
                'signal': signal.signal,
                'confidence': signal.confidence,
                'predicted_change': signal.predicted_change,
                'symbol': signal.symbol,
                'model_name': signal.model_name,
                'strength': signal.strength
            })
        return pd.DataFrame(data)
    
    @staticmethod
    def metrics_to_dict(metrics: GUIMetrics) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(metrics)
    
    @staticmethod
    def export_to_csv(data: Union[List[GUITrade], List[GUISignal]], filename: str) -> bool:
        """Export data to CSV file."""
        try:
            if isinstance(data[0], GUITrade):
                df = DataExporter.trades_to_dataframe(data)
            elif isinstance(data[0], GUISignal):
                df = DataExporter.signals_to_dataframe(data)
            else:
                return False
            
            df.to_csv(filename, index=False)
            return True
        except Exception as e:
            logger.error(f"Export to CSV failed: {e}")
            return False
