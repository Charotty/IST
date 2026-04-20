from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal, InvalidOperation

from data_adapters import GUISignal, GUITrade, GUIOrder, GUIMetrics, GUIModel, GUILogEntry

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error for data format validation."""
    
    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"Validation error in field '{field}': {message}")


class FormatValidator:
    """Comprehensive format validation for GUI data structures."""
    
    # Validation constants
    VALID_SIGNALS = {"BUY", "SELL", "HOLD"}
    VALID_LEVELS = {"info", "warning", "error", "debug"}
    VALID_STATUSES = {"OPEN", "CLOSED", "PENDING", "FILLED", "PARTIALLY_FILLED", "CANCELLED", "REJECTED"}
    VALID_TYPES = {"BUY", "SELL"}
    VALID_ORDER_TYPES = {"MARKET", "LIMIT"}
    VALID_MODEL_TYPES = {"GRU", "CNN", "Transformer", "Boosting"}
    
    # Value ranges
    CONFIDENCE_RANGE = (0.0, 1.0)
    STRENGTH_RANGE = (0.0, 1.0)
    SCORE_RANGE = (0.0, 1.0)
    PERCENTAGE_RANGE = (-100.0, 100.0)
    PRICE_RANGE = (0.00001, 1000000.0)  # Reasonable price range
    
    @classmethod
    def validate_signal(cls, signal: GUISignal) -> List[ValidationError]:
        """Validate GUISignal format."""
        errors = []
        
        # Validate signal field
        if not signal.signal:
            errors.append(ValidationError("signal", "Signal cannot be empty"))
        elif signal.signal not in cls.VALID_SIGNALS:
            errors.append(ValidationError("signal", f"Invalid signal value: {signal.signal}", signal.signal))
        
        # Validate confidence
        if not isinstance(signal.confidence, (int, float)):
            errors.append(ValidationError("confidence", "Confidence must be numeric", signal.confidence))
        elif not (cls.CONFIDENCE_RANGE[0] <= signal.confidence <= cls.CONFIDENCE_RANGE[1]):
            errors.append(ValidationError("confidence", 
                                        f"Confidence must be between {cls.CONFIDENCE_RANGE[0]} and {cls.CONFIDENCE_RANGE[1]}",
                                        signal.confidence))
        
        # Validate predicted change
        if not isinstance(signal.predicted_change, (int, float)):
            errors.append(ValidationError("predicted_change", "Predicted change must be numeric", signal.predicted_change))
        elif not (cls.PERCENTAGE_RANGE[0] <= signal.predicted_change <= cls.PERCENTAGE_RANGE[1]):
            errors.append(ValidationError("predicted_change",
                                        f"Predicted change must be between {cls.PERCENTAGE_RANGE[0]} and {cls.PERCENTAGE_RANGE[1]}",
                                        signal.predicted_change))
        
        # Validate timestamp
        if not isinstance(signal.timestamp, datetime):
            errors.append(ValidationError("timestamp", "Timestamp must be datetime object", type(signal.timestamp)))
        elif signal.timestamp > datetime.now():
            errors.append(ValidationError("timestamp", "Timestamp cannot be in the future", signal.timestamp))
        
        # Validate symbol
        if not signal.symbol or not isinstance(signal.symbol, str):
            errors.append(ValidationError("symbol", "Symbol must be non-empty string", signal.symbol))
        elif len(signal.symbol) < 1 or len(signal.symbol) > 20:
            errors.append(ValidationError("symbol", "Symbol length must be between 1 and 20 characters", signal.symbol))
        
        # Validate model name
        if not signal.model_name or not isinstance(signal.model_name, str):
            errors.append(ValidationError("model_name", "Model name must be non-empty string", signal.model_name))
        
        # Validate strength
        if not isinstance(signal.strength, (int, float)):
            errors.append(ValidationError("strength", "Strength must be numeric", signal.strength))
        elif not (cls.STRENGTH_RANGE[0] <= signal.strength <= cls.STRENGTH_RANGE[1]):
            errors.append(ValidationError("strength",
                                        f"Strength must be between {cls.STRENGTH_RANGE[0]} and {cls.STRENGTH_RANGE[1]}",
                                        signal.strength))
        
        return errors
    
    @classmethod
    def validate_trade(cls, trade: GUITrade) -> List[ValidationError]:
        """Validate GUITrade format."""
        errors = []
        
        # Validate ID
        if not trade.id or not isinstance(trade.id, str):
            errors.append(ValidationError("id", "Trade ID must be non-empty string", trade.id))
        
        # Validate timestamp
        if not trade.timestamp or not isinstance(trade.timestamp, str):
            errors.append(ValidationError("timestamp", "Timestamp must be non-empty string", trade.timestamp))
        else:
            # Try to parse timestamp format
            try:
                datetime.strptime(trade.timestamp, "%H:%M:%S")
            except ValueError:
                try:
                    # Try with date part
                    datetime.strptime(trade.timestamp, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    errors.append(ValidationError("timestamp", "Invalid timestamp format, expected HH:MM:SS", trade.timestamp))
        
        # Validate pair
        if not trade.pair or not isinstance(trade.pair, str):
            errors.append(ValidationError("pair", "Pair must be non-empty string", trade.pair))
        
        # Validate type
        if not trade.type or trade.type not in cls.VALID_TYPES:
            errors.append(ValidationError("type", f"Invalid trade type: {trade.type}", trade.type))
        
        # Validate entry price
        if not isinstance(trade.entry_price, (int, float, Decimal)):
            errors.append(ValidationError("entry_price", "Entry price must be numeric", trade.entry_price))
        else:
            entry_price = float(trade.entry_price)
            if not (cls.PRICE_RANGE[0] <= entry_price <= cls.PRICE_RANGE[1]):
                errors.append(ValidationError("entry_price",
                                            f"Entry price must be between {cls.PRICE_RANGE[0]} and {cls.PRICE_RANGE[1]}",
                                            entry_price))
            if entry_price <= 0:
                errors.append(ValidationError("entry_price", "Entry price must be positive", entry_price))
        
        # Validate exit price if present
        if trade.exit_price is not None:
            if not isinstance(trade.exit_price, (int, float, Decimal)):
                errors.append(ValidationError("exit_price", "Exit price must be numeric", trade.exit_price))
            else:
                exit_price = float(trade.exit_price)
                if not (cls.PRICE_RANGE[0] <= exit_price <= cls.PRICE_RANGE[1]):
                    errors.append(ValidationError("exit_price",
                                                f"Exit price must be between {cls.PRICE_RANGE[0]} and {cls.PRICE_RANGE[1]}",
                                                exit_price))
                if exit_price <= 0:
                    errors.append(ValidationError("exit_price", "Exit price must be positive", exit_price))
        
        # Validate PnL if present
        if trade.pnl is not None:
            if not isinstance(trade.pnl, (int, float, Decimal)):
                errors.append(ValidationError("pnl", "PnL must be numeric", trade.pnl))
            else:
                pnl = float(trade.pnl)
                if not (cls.PERCENTAGE_RANGE[0] * 10 <= pnl <= cls.PERCENTAGE_RANGE[1] * 10):  # Allow larger PnL
                    errors.append(ValidationError("pnl",
                                                f"PnL must be between {cls.PERCENTAGE_RANGE[0] * 10} and {cls.PERCENTAGE_RANGE[1] * 10}",
                                                pnl))
        
        # Validate status
        if not trade.status or trade.status not in {"OPEN", "CLOSED"}:
            errors.append(ValidationError("status", f"Invalid trade status: {trade.status}", trade.status))
        
        # Validate quantity
        if not isinstance(trade.quantity, (int, float, Decimal)):
            errors.append(ValidationError("quantity", "Quantity must be numeric", trade.quantity))
        else:
            quantity = float(trade.quantity)
            if quantity <= 0:
                errors.append(ValidationError("quantity", "Quantity must be positive", quantity))
        
        # Validate commission
        if not isinstance(trade.commission, (int, float, Decimal)):
            errors.append(ValidationError("commission", "Commission must be numeric", trade.commission))
        else:
            commission = float(trade.commission)
            if commission < 0:
                errors.append(ValidationError("commission", "Commission cannot be negative", commission))
        
        # Validate slippage
        if not isinstance(trade.slippage, (int, float, Decimal)):
            errors.append(ValidationError("slippage", "Slippage must be numeric", trade.slippage))
        else:
            slippage = float(trade.slippage)
            if slippage < 0:
                errors.append(ValidationError("slippage", "Slippage cannot be negative", slippage))
        
        return errors
    
    @classmethod
    def validate_order(cls, order: GUIOrder) -> List[ValidationError]:
        """Validate GUIOrder format."""
        errors = []
        
        # Validate ID
        if not order.id or not isinstance(order.id, str):
            errors.append(ValidationError("id", "Order ID must be non-empty string", order.id))
        
        # Validate symbol
        if not order.symbol or not isinstance(order.symbol, str):
            errors.append(ValidationError("symbol", "Symbol must be non-empty string", order.symbol))
        
        # Validate type
        if not order.type or order.type not in cls.VALID_ORDER_TYPES:
            errors.append(ValidationError("type", f"Invalid order type: {order.type}", order.type))
        
        # Validate side
        if not order.side or order.side not in cls.VALID_TYPES:
            errors.append(ValidationError("side", f"Invalid order side: {order.side}", order.side))
        
        # Validate amount
        if not isinstance(order.amount, (int, float, Decimal)):
            errors.append(ValidationError("amount", "Amount must be numeric", order.amount))
        else:
            amount = float(order.amount)
            if amount <= 0:
                errors.append(ValidationError("amount", "Amount must be positive", amount))
        
        # Validate price if present (for limit orders)
        if order.type == "LIMIT":
            if order.price is None:
                errors.append(ValidationError("price", "Limit order must have a price"))
            elif not isinstance(order.price, (int, float, Decimal)):
                errors.append(ValidationError("price", "Price must be numeric", order.price))
            else:
                price = float(order.price)
                if not (cls.PRICE_RANGE[0] <= price <= cls.PRICE_RANGE[1]):
                    errors.append(ValidationError("price",
                                                f"Price must be between {cls.PRICE_RANGE[0]} and {cls.PRICE_RANGE[1]}",
                                                price))
                if price <= 0:
                    errors.append(ValidationError("price", "Price must be positive", price))
        
        # Validate status
        if not order.status or order.status not in cls.VALID_STATUSES:
            errors.append(ValidationError("status", f"Invalid order status: {order.status}", order.status))
        
        # Validate filled amount
        if not isinstance(order.filled_amount, (int, float, Decimal)):
            errors.append(ValidationError("filled_amount", "Filled amount must be numeric", order.filled_amount))
        else:
            filled_amount = float(order.filled_amount)
            if filled_amount < 0:
                errors.append(ValidationError("filled_amount", "Filled amount cannot be negative", filled_amount))
            if filled_amount > float(order.amount):
                errors.append(ValidationError("filled_amount", "Filled amount cannot exceed total amount", filled_amount))
        
        # Validate average price
        if not isinstance(order.average_price, (int, float, Decimal)):
            errors.append(ValidationError("average_price", "Average price must be numeric", order.average_price))
        else:
            avg_price = float(order.average_price)
            if avg_price < 0:
                errors.append(ValidationError("average_price", "Average price cannot be negative", avg_price))
        
        return errors
    
    @classmethod
    def validate_metrics(cls, metrics: GUIMetrics) -> List[ValidationError]:
        """Validate GUIMetrics format."""
        errors = []
        
        # Validate total PnL
        if not isinstance(metrics.total_pnl, (int, float, Decimal)):
            errors.append(ValidationError("total_pnl", "Total PnL must be numeric", metrics.total_pnl))
        
        # Validate win rate
        if not isinstance(metrics.win_rate, (int, float, Decimal)):
            errors.append(ValidationError("win_rate", "Win rate must be numeric", metrics.win_rate))
        else:
            win_rate = float(metrics.win_rate)
            if not (0 <= win_rate <= 100):
                errors.append(ValidationError("win_rate", "Win rate must be between 0 and 100", win_rate))
        
        # Validate total trades
        if not isinstance(metrics.total_trades, int):
            errors.append(ValidationError("total_trades", "Total trades must be integer", metrics.total_trades))
        elif metrics.total_trades < 0:
            errors.append(ValidationError("total_trades", "Total trades cannot be negative", metrics.total_trades))
        
        # Validate max drawdown
        if not isinstance(metrics.max_drawdown, (int, float, Decimal)):
            errors.append(ValidationError("max_drawdown", "Max drawdown must be numeric", metrics.max_drawdown))
        else:
            max_dd = float(metrics.max_drawdown)
            if max_dd < 0:
                errors.append(ValidationError("max_drawdown", "Max drawdown cannot be negative", max_dd))
        
        # Validate optional metrics
        optional_metrics = {
            "sharpe_ratio": metrics.sharpe_ratio,
            "profit_factor": metrics.profit_factor,
            "sortino_ratio": metrics.sortino_ratio,
            "calmar_ratio": metrics.calmar_ratio
        }
        
        for name, value in optional_metrics.items():
            if value is not None and not isinstance(value, (int, float, Decimal)):
                errors.append(ValidationError(name, f"{name.replace('_', ' ').title()} must be numeric", value))
        
        # Validate derived metrics
        derived_metrics = {
            "avg_win": metrics.avg_win,
            "avg_loss": metrics.avg_loss,
            "largest_win": metrics.largest_win,
            "largest_loss": metrics.largest_loss
        }
        
        for name, value in derived_metrics.items():
            if value is not None and not isinstance(value, (int, float, Decimal)):
                errors.append(ValidationError(name, f"{name.replace('_', ' ').title()} must be numeric", value))
        
        # Validate streaks
        if not isinstance(metrics.consecutive_wins, int):
            errors.append(ValidationError("consecutive_wins", "Consecutive wins must be integer", metrics.consecutive_wins))
        elif metrics.consecutive_wins < 0:
            errors.append(ValidationError("consecutive_wins", "Consecutive wins cannot be negative", metrics.consecutive_wins))
        
        if not isinstance(metrics.consecutive_losses, int):
            errors.append(ValidationError("consecutive_losses", "Consecutive losses must be integer", metrics.consecutive_losses))
        elif metrics.consecutive_losses < 0:
            errors.append(ValidationError("consecutive_losses", "Consecutive losses cannot be negative", metrics.consecutive_losses))
        
        return errors
    
    @classmethod
    def validate_model(cls, model: GUIModel) -> List[ValidationError]:
        """Validate GUIModel format."""
        errors = []
        
        # Validate name
        if not model.name or not isinstance(model.name, str):
            errors.append(ValidationError("name", "Model name must be non-empty string", model.name))
        
        # Validate type
        if not model.type or model.type not in cls.VALID_MODEL_TYPES:
            errors.append(ValidationError("type", f"Invalid model type: {model.type}", model.type))
        
        # Validate score
        if not isinstance(model.score, (int, float, Decimal)):
            errors.append(ValidationError("score", "Score must be numeric", model.score))
        else:
            score = float(model.score)
            if not (cls.SCORE_RANGE[0] <= score <= cls.SCORE_RANGE[1]):
                errors.append(ValidationError("score",
                                            f"Score must be between {cls.SCORE_RANGE[0]} and {cls.SCORE_RANGE[1]}",
                                            score))
        
        # Validate is_active
        if not isinstance(model.is_active, bool):
            errors.append(ValidationError("is_active", "is_active must be boolean", model.is_active))
        
        # Validate optional metrics
        optional_float_metrics = {
            "accuracy": model.accuracy,
            "precision": model.precision,
            "recall": model.recall,
            "f1_score": model.f1_score,
            "success_rate": model.success_rate
        }
        
        for name, value in optional_float_metrics.items():
            if value is not None:
                if not isinstance(value, (int, float, Decimal)):
                    errors.append(ValidationError(name, f"{name} must be numeric", value))
                else:
                    val = float(value)
                    if not (0 <= val <= 1):
                        errors.append(ValidationError(name, f"{name} must be between 0 and 1", val))
        
        # Validate prediction count
        if not isinstance(model.prediction_count, int):
            errors.append(ValidationError("prediction_count", "Prediction count must be integer", model.prediction_count))
        elif model.prediction_count < 0:
            errors.append(ValidationError("prediction_count", "Prediction count cannot be negative", model.prediction_count))
        
        return errors
    
    @classmethod
    def validate_log_entry(cls, log_entry: GUILogEntry) -> List[ValidationError]:
        """Validate GUILogEntry format."""
        errors = []
        
        # Validate timestamp
        if not log_entry.timestamp or not isinstance(log_entry.timestamp, str):
            errors.append(ValidationError("timestamp", "Timestamp must be non-empty string", log_entry.timestamp))
        else:
            try:
                datetime.strptime(log_entry.timestamp, "%H:%M:%S")
            except ValueError:
                errors.append(ValidationError("timestamp", "Invalid timestamp format, expected HH:MM:SS", log_entry.timestamp))
        
        # Validate level
        if not log_entry.level or log_entry.level not in cls.VALID_LEVELS:
            errors.append(ValidationError("level", f"Invalid log level: {log_entry.level}", log_entry.level))
        
        # Validate message
        if not log_entry.message or not isinstance(log_entry.message, str):
            errors.append(ValidationError("message", "Message must be non-empty string", log_entry.message))
        elif len(log_entry.message) > 1000:
            errors.append(ValidationError("message", "Message too long (max 1000 characters)", len(log_entry.message)))
        
        # Validate source
        if not log_entry.source or not isinstance(log_entry.source, str):
            errors.append(ValidationError("source", "Source must be non-empty string", log_entry.source))
        
        return errors
    
    @classmethod
    def validate_symbol(cls, symbol: str) -> List[ValidationError]:
        """Validate trading symbol format."""
        errors = []
        
        if not symbol or not isinstance(symbol, str):
            errors.append(ValidationError("symbol", "Symbol must be non-empty string", symbol))
            return errors
        
        # Remove common separators for validation
        clean_symbol = symbol.replace("/", "").replace("-", "").replace("_", "")
        
        if len(clean_symbol) < 1 or len(clean_symbol) > 10:
            errors.append(ValidationError("symbol", "Symbol length must be between 1 and 10 characters", symbol))
        
        if not clean_symbol.isalnum():
            errors.append(ValidationError("symbol", "Symbol must contain only alphanumeric characters", symbol))
        
        return errors
    
    @classmethod
    def validate_timestamp(cls, timestamp: Union[str, datetime]) -> List[ValidationError]:
        """Validate timestamp format."""
        errors = []
        
        if isinstance(timestamp, str):
            try:
                # Try different formats
                for fmt in ["%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                    try:
                        parsed = datetime.strptime(timestamp, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    errors.append(ValidationError("timestamp", "Invalid timestamp format", timestamp))
            except Exception as e:
                errors.append(ValidationError("timestamp", f"Timestamp parsing error: {e}", timestamp))
        
        elif isinstance(timestamp, datetime):
            if timestamp > datetime.now():
                errors.append(ValidationError("timestamp", "Timestamp cannot be in the future", timestamp))
        
        else:
            errors.append(ValidationError("timestamp", "Timestamp must be string or datetime", type(timestamp)))
        
        return errors


class DataSanitizer:
    """Data sanitization utilities."""
    
    @staticmethod
    def sanitize_signal(signal: GUISignal) -> GUISignal:
        """Sanitize signal data."""
        # Ensure signal is uppercase
        sanitized_signal = signal.signal.upper() if signal.signal else "HOLD"
        
        # Clip confidence to valid range
        confidence = max(0.0, min(1.0, float(signal.confidence)))
        
        # Clip predicted change to reasonable range
        predicted_change = max(-100.0, min(100.0, float(signal.predicted_change)))
        
        # Clip strength to valid range
        strength = max(0.0, min(1.0, float(signal.strength)))
        
        # Sanitize symbol
        symbol = signal.symbol.replace("/", "").replace("-", "").replace("_", "").upper()
        
        # Sanitize model name
        model_name = signal.model_name.strip()
        
        return GUISignal(
            signal=sanitized_signal,
            confidence=confidence,
            predicted_change=predicted_change,
            timestamp=signal.timestamp,
            symbol=symbol,
            model_name=model_name,
            strength=strength,
            metadata=signal.metadata
        )
    
    @staticmethod
    def sanitize_trade(trade: GUITrade) -> GUITrade:
        """Sanitize trade data."""
        # Ensure type is uppercase
        sanitized_type = trade.type.upper() if trade.type else "BUY"
        
        # Ensure status is valid
        status = trade.status if trade.status in {"OPEN", "CLOSED"} else "OPEN"
        
        # Sanitize prices
        entry_price = max(0.00001, float(trade.entry_price))
        exit_price = max(0.00001, float(trade.exit_price)) if trade.exit_price else None
        
        # Sanitize PnL
        pnl = trade.pnl
        if pnl is not None:
            pnl = max(-1000.0, min(1000.0, float(pnl)))
        
        # Sanitize quantity
        quantity = max(0.00001, float(trade.quantity))
        
        # Sanitize commission and slippage
        commission = max(0.0, float(trade.commission))
        slippage = max(0.0, float(trade.slippage))
        
        # Sanitize symbol
        pair = trade.pair.replace("/", "").replace("-", "").replace("_", "").upper()
        
        return GUITrade(
            id=trade.id,
            timestamp=trade.timestamp,
            pair=pair,
            type=sanitized_type,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            status=status,
            quantity=quantity,
            commission=commission,
            slippage=slippage,
            duration=trade.duration
        )
    
    @staticmethod
    def sanitize_order(order: GUIOrder) -> GUIOrder:
        """Sanitize order data."""
        # Ensure type and side are uppercase
        sanitized_type = order.type.upper() if order.type else "MARKET"
        sanitized_side = order.side.upper() if order.side else "BUY"
        
        # Sanitize amount
        amount = max(0.00001, float(order.amount))
        
        # Sanitize price for limit orders
        price = order.price
        if sanitized_type == "LIMIT" and price is not None:
            price = max(0.00001, float(price))
        
        # Sanitize filled amount
        filled_amount = max(0.0, min(float(order.filled_amount), amount))
        
        # Sanitize average price
        avg_price = max(0.0, float(order.average_price))
        
        # Sanitize symbol
        symbol = order.symbol.replace("/", "").replace("-", "").replace("_", "").upper()
        
        return GUIOrder(
            id=order.id,
            symbol=symbol,
            type=sanitized_type,
            side=sanitized_side,
            amount=amount,
            price=price,
            status=order.status,
            filled_amount=filled_amount,
            average_price=avg_price,
            created_at=order.created_at,
            updated_at=order.updated_at
        )
    
    @staticmethod
    def sanitize_metrics(metrics: GUIMetrics) -> GUIMetrics:
        """Sanitize metrics data."""
        # Sanitize win rate
        win_rate = max(0.0, min(100.0, float(metrics.win_rate)))
        
        # Sanitize total trades
        total_trades = max(0, int(metrics.total_trades))
        
        # Sanitize max drawdown
        max_drawdown = max(0.0, float(metrics.max_drawdown))
        
        # Sanitize optional metrics
        sharpe_ratio = metrics.sharpe_ratio
        if sharpe_ratio is not None:
            sharpe_ratio = float(sharpe_ratio)
        
        profit_factor = metrics.profit_factor
        if profit_factor is not None:
            profit_factor = max(0.0, float(profit_factor))
        
        return GUIMetrics(
            total_pnl=float(metrics.total_pnl),
            win_rate=win_rate,
            total_trades=total_trades,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            sortino_ratio=metrics.sortino_ratio,
            calmar_ratio=metrics.calmar_ratio,
            avg_trade_duration=metrics.avg_trade_duration,
            avg_win=metrics.avg_win,
            avg_loss=metrics.avg_loss,
            largest_win=metrics.largest_win,
            largest_loss=metrics.largest_loss,
            consecutive_wins=max(0, int(metrics.consecutive_wins)),
            consecutive_losses=max(0, int(metrics.consecutive_losses)),
            equity_curve=metrics.equity_curve
        )


class ValidationReport:
    """Validation report for batch validation."""
    
    def __init__(self):
        self.total_items = 0
        self.valid_items = 0
        self.invalid_items = 0
        self.errors: List[ValidationError] = []
        self.warnings: List[str] = []
    
    def add_result(self, is_valid: bool, errors: List[ValidationError]):
        """Add validation result."""
        self.total_items += 1
        if is_valid:
            self.valid_items += 1
        else:
            self.invalid_items += 1
            self.errors.extend(errors)
    
    def add_warning(self, warning: str):
        """Add warning message."""
        self.warnings.append(warning)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            "total_items": self.total_items,
            "valid_items": self.valid_items,
            "invalid_items": self.invalid_items,
            "validity_rate": (self.valid_items / self.total_items * 100) if self.total_items > 0 else 0,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [{"field": e.field, "message": e.message, "value": e.value} for e in self.errors],
            "warnings": self.warnings
        }
    
    def is_valid(self) -> bool:
        """Check if all items are valid."""
        return self.invalid_items == 0


class BatchValidator:
    """Batch validation utilities."""
    
    @staticmethod
    def validate_signals(signals: List[GUISignal]) -> ValidationReport:
        """Validate multiple signals."""
        report = ValidationReport()
        
        for signal in signals:
            errors = FormatValidator.validate_signal(signal)
            is_valid = len(errors) == 0
            report.add_result(is_valid, errors)
        
        return report
    
    @staticmethod
    def validate_trades(trades: List[GUITrade]) -> ValidationReport:
        """Validate multiple trades."""
        report = ValidationReport()
        
        for trade in trades:
            errors = FormatValidator.validate_trade(trade)
            is_valid = len(errors) == 0
            report.add_result(is_valid, errors)
        
        return report
    
    @staticmethod
    def validate_orders(orders: List[GUIOrder]) -> ValidationReport:
        """Validate multiple orders."""
        report = ValidationReport()
        
        for order in orders:
            errors = FormatValidator.validate_order(order)
            is_valid = len(errors) == 0
            report.add_result(is_valid, errors)
        
        return report
    
    @staticmethod
    def validate_metrics_list(metrics_list: List[GUIMetrics]) -> ValidationReport:
        """Validate multiple metrics entries."""
        report = ValidationReport()
        
        for metrics in metrics_list:
            errors = FormatValidator.validate_metrics(metrics)
            is_valid = len(errors) == 0
            report.add_result(is_valid, errors)
        
        return report
