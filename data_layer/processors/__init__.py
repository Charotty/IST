"""
Processors Module

Обработка различных типов данных.
"""

from .ohlcv_processor import OHLCVProcessor
from .orderbook_processor import OrderBookProcessor
from .trades_processor import TradesProcessor

__all__ = ['OHLCVProcessor', 'OrderBookProcessor', 'TradesProcessor']
