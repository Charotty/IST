"""
OKX REST Client via CCXT
=========================

Implements OKX REST API functionality using CCXT for:
- Market data bootstrap
- Order placement (for live trading)
- Account information
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import ccxt

from common.backend_contract import SymbolMapper

logger = logging.getLogger(__name__)


class OKXRESTClient:
    """OKX REST client using CCXT."""
    
    def __init__(self, api_key: Optional[str] = None, secret: Optional[str] = None, passphrase: Optional[str] = None):
        """
        Initialize OKX REST client.
        
        Args:
            api_key: OKX API key (optional for public endpoints)
            secret: OKX API secret (optional for public endpoints)
            passphrase: OKX API passphrase (optional for public endpoints)
        """
        self.exchange = ccxt.okx({
            'apiKey': api_key,
            'secret': secret,
            'password': passphrase,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # spot, futures, swap
            }
        })
        
        self.connected = False
    
    def connect(self) -> bool:
        """Test connection to OKX REST API."""
        try:
            # Load markets to test connection
            self.exchange.load_markets()
            self.connected = True
            logger.info("Connected to OKX REST API")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to OKX REST API: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close connection."""
        self.exchange.close()
        self.connected = False
        logger.info("Disconnected from OKX REST API")
    
    def get_markets(self) -> Dict[str, Any]:
        """Get available markets."""
        try:
            self.exchange.load_markets()
            return self.exchange.markets
        except Exception as e:
            logger.error(f"Failed to get markets: {e}")
            return {}
    
    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current ticker for a symbol.
        
        Args:
            symbol: GUI format symbol (e.g., BTC/USDT)
        
        Returns:
            Ticker data or None
        """
        try:
            okx_symbol = SymbolMapper.gui_to_okx(symbol)
            ticker = self.exchange.fetch_ticker(okx_symbol)
            return ticker
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            return None
    
    def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """
        Get current orderbook for a symbol.
        
        Args:
            symbol: GUI format symbol (e.g., BTC/USDT)
            limit: Number of levels
        
        Returns:
            Orderbook data or None
        """
        try:
            okx_symbol = SymbolMapper.gui_to_okx(symbol)
            orderbook = self.exchange.fetch_order_book(okx_symbol, limit=limit)
            return orderbook
        except Exception as e:
            logger.error(f"Failed to get orderbook for {symbol}: {e}")
            return None
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> List[List[float]]:
        """
        Get OHLCV candles for a symbol.
        
        Args:
            symbol: GUI format symbol (e.g., BTC/USDT)
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
            limit: Number of candles
        
        Returns:
            List of OHLCV candles [[ts, o, h, l, c, v], ...]
        """
        try:
            okx_symbol = SymbolMapper.gui_to_okx(symbol)
            ohlcv = self.exchange.fetch_ohlcv(okx_symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            logger.error(f"Failed to get OHLCV for {symbol}: {e}")
            return []
    
    def get_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[List[float]]:
        """
        Get historical OHLCV candles for a symbol.
        
        Args:
            symbol: GUI format symbol (e.g., BTC/USDT)
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            since: Start timestamp in milliseconds
            limit: Number of candles (max 1000 per request)
        
        Returns:
            List of OHLCV candles [[ts, o, h, l, c, v], ...]
        """
        try:
            okx_symbol = SymbolMapper.gui_to_okx(symbol)
            ohlcv = self.exchange.fetch_ohlcv(okx_symbol, timeframe, since=since, limit=limit)
            return ohlcv
        except Exception as e:
            logger.error(f"Failed to get historical OHLCV for {symbol}: {e}")
            return []
    
    def get_balance(self) -> Optional[Dict[str, Any]]:
        """
        Get account balance (requires API credentials).
        
        Returns:
            Balance data or None
        """
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return None
    
    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create an order (requires API credentials).
        
        Args:
            symbol: GUI format symbol (e.g., BTC/USDT)
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'
            amount: Order amount
            price: Price for limit orders
        
        Returns:
            Order data or None
        """
        try:
            okx_symbol = SymbolMapper.gui_to_okx(symbol)
            
            if order_type == 'market':
                order = self.exchange.create_market_order(okx_symbol, side, amount)
            elif order_type == 'limit':
                if price is None:
                    raise ValueError("Price required for limit orders")
                order = self.exchange.create_limit_order(okx_symbol, side, amount, price)
            else:
                raise ValueError(f"Invalid order type: {order_type}")
            
            return order
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            return None
    
    def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Cancel an order (requires API credentials).
        
        Args:
            order_id: Order ID
            symbol: GUI format symbol (e.g., BTC/USDT)
        
        Returns:
            Cancellation result or None
        """
        try:
            okx_symbol = SymbolMapper.gui_to_okx(symbol)
            result = self.exchange.cancel_order(order_id, okx_symbol)
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return None
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Cancel all open orders (requires API credentials).
        
        Args:
            symbol: GUI format symbol (optional, cancel all if None)
        
        Returns:
            List of cancellation results
        """
        try:
            if symbol:
                okx_symbol = SymbolMapper.gui_to_okx(symbol)
                result = self.exchange.cancel_all_orders(okx_symbol)
            else:
                result = self.exchange.cancel_all_orders()
            return result
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return []
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders (requires API credentials).
        
        Args:
            symbol: GUI format symbol (optional)
        
        Returns:
            List of open orders
        """
        try:
            if symbol:
                okx_symbol = SymbolMapper.gui_to_okx(symbol)
                orders = self.exchange.fetch_open_orders(okx_symbol)
            else:
                orders = self.exchange.fetch_open_orders()
            return orders
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    def get_closed_orders(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get closed orders (requires API credentials).
        
        Args:
            symbol: GUI format symbol (optional)
            limit: Number of orders
        
        Returns:
            List of closed orders
        """
        try:
            if symbol:
                okx_symbol = SymbolMapper.gui_to_okx(symbol)
                orders = self.exchange.fetch_closed_orders(okx_symbol, limit=limit)
            else:
                orders = self.exchange.fetch_closed_orders(limit=limit)
            return orders
        except Exception as e:
            logger.error(f"Failed to get closed orders: {e}")
            return []
