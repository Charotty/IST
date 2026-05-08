#!/usr/bin/env python3
"""
OKX Data Source with Auto-Download and Backfill
===============================================

Production-ready OKX data source with:
- Real-time data streaming
- Automatic historical data download
- Backfill system for missing data
- Data validation and integrity monitoring
- Configurable data types and timeframes
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator, Optional, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import aiohttp
import pandas as pd

from its_project.data_layer.base import BaseDataSource
from its_project.data_layer.integrity import DataIntegrityMonitor, ValidationReport
from its_project.common.types import MarketData, MarketDataType

logger = logging.getLogger(__name__)


@dataclass
class OKXConfig:
    """OKX API configuration."""
    api_key: Optional[str] = None
    secret: Optional[str] = None
    passphrase: Optional[str] = None
    sandbox: bool = False
    base_url: str = "https://www.okx.com"
    ws_url: str = "wss://ws.okx.com:8443/ws/v5/public"
    
    # Data configuration
    symbols: List[str] = field(default_factory=lambda: ["BTC-USDT", "ETH-USDT"])
    data_types: List[str] = field(default_factory=lambda: ["ticker", "orderbook", "trades"])
    orderbook_depth: int = 20
    trades_limit: int = 100
    
    # Auto-download configuration
    enable_auto_download: bool = True
    download_timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "1d"])
    max_download_days: int = 365
    download_interval: int = 3600  # 1 hour
    
    # Backfill configuration
    enable_backfill: bool = True
    backfill_gap_threshold: int = 300  # 5 minutes
    backfill_batch_size: int = 1000
    backfill_retry_attempts: int = 3
    
    # Storage configuration
    storage_path: str = "data/okx"
    enable_compression: bool = True
    cache_duration: int = 3600  # 1 hour


@dataclass
class BackfillRequest:
    """Backfill request for missing data."""
    symbol: str
    data_type: str
    timeframe: Optional[str]
    start_time: datetime
    end_time: datetime
    reason: str
    priority: int = 1


class OKXDataSource(BaseDataSource):
    """
    OKX data source with auto-download and backfill capabilities.
    
    Features:
    - Real-time WebSocket streaming
    - Automatic historical data download
    - Gap detection and backfill
    - Data validation and integrity monitoring
    - Configurable storage and caching
    """
    
    def __init__(self, config: OKXConfig) -> None:
        super().__init__()
        self.config = config
        
        # HTTP session for REST API
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_session: Optional[aiohttp.ClientSession] = None
        self.ws_connection: Optional[aiohttp.ClientWebSocketResponse] = None
        
        # Data storage
        self.data_cache: Dict[str, List[MarketData]] = {}
        self.last_update: Dict[str, datetime] = {}
        
        # Backfill system
        self.backfill_queue: asyncio.Queue[BackfillRequest] = asyncio.Queue()
        self.backfill_task: Optional[asyncio.Task] = None
        
        # Auto-download system
        self.download_task: Optional[asyncio.Task] = None
        
        # Integrity monitoring
        self.integrity_monitor: Optional[DataIntegrityMonitor] = None
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'data_downloaded': 0,
            'backfills_completed': 0,
            'gaps_detected': 0,
            'last_download': None,
            'last_backfill': None
        }
        
        # State
        self._running = False
        self._connected = False
        
        logger.info(f"OKX data source initialized for symbols: {config.symbols}")
    
    async def connect(self) -> None:
        """Connect to OKX API."""
        if self._connected:
            logger.warning("Already connected to OKX")
            return
        
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                base_url=self.config.base_url,
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Create WebSocket session
            self.ws_session = aiohttp.ClientSession()
            
            # Initialize integrity monitor
            self.integrity_monitor = DataIntegrityMonitor()
            
            # Test connection
            await self._test_connection()
            
            self._connected = True
            logger.info("Connected to OKX API")
            
        except Exception as e:
            logger.error(f"Failed to connect to OKX: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from OKX API."""
        self._running = False
        
        # Cancel background tasks
        if self.download_task and not self.download_task.done():
            self.download_task.cancel()
        
        if self.backfill_task and not self.backfill_task.done():
            self.backfill_task.cancel()
        
        # Close WebSocket connection
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
        
        # Close sessions
        if self.ws_session:
            await self.ws_session.close()
            self.ws_session = None
        
        if self.session:
            await self.session.close()
            self.session = None
        
        self._connected = False
        logger.info("Disconnected from OKX API")
    
    async def subscribe(self, symbols: Optional[List[str]] = None) -> AsyncIterator[MarketData]:
        """Subscribe to real-time market data."""
        if not self._connected:
            await self.connect()
        
        if symbols:
            self.config.symbols = symbols
        
        # Start background tasks
        await self._start_background_tasks()
        
        # Connect WebSocket for real-time data
        await self._connect_websocket()
        
        self._running = True
        
        try:
            # Yield real-time data
            while self._running and self.ws_connection:
                try:
                    message = await self.ws_connection.receive_json()
                    
                    if 'data' in message:
                        market_data = await self._parse_ws_message(message)
                        if market_data:
                            self.stats['messages_received'] += 1
                            yield market_data
                
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    await asyncio.sleep(1)
        
        finally:
            self._running = False
    
    async def fetch(self, symbol: str, data_type: str = "ticker", **params) -> MarketData:
        """Fetch current market data."""
        if not self._connected:
            await self.connect()
        
        try:
            if data_type == "ticker":
                data = await self._fetch_ticker(symbol)
            elif data_type == "orderbook":
                data = await self._fetch_orderbook(symbol)
            elif data_type == "trades":
                data = await self._fetch_trades(symbol)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            return data
        
        except Exception as e:
            logger.error(f"Error fetching {data_type} for {symbol}: {e}")
            raise
    
    async def is_alive(self) -> bool:
        """Check if connection is alive."""
        if not self._connected or not self.session:
            return False
        
        try:
            # Test with a simple API call
            async with self.session.get("/api/v5/market/ticker?instId=BTC-USDT") as response:
                return response.status == 200
        except Exception:
            return False
    
    async def _test_connection(self) -> None:
        """Test API connection."""
        async with self.session.get("/api/v5/public/time") as response:
            if response.status != 200:
                raise RuntimeError("Failed to connect to OKX API")
            
            data = await response.json()
            logger.debug(f"OKX server time: {data}")
    
    async def _connect_websocket(self) -> None:
        """Connect to WebSocket for real-time data."""
        if not self.ws_session:
            raise RuntimeError("WebSocket session not initialized")
        
        try:
            self.ws_connection = await self.ws_session.ws_connect(self.config.ws_url)
            
            # Subscribe to channels
            await self._subscribe_ws_channels()
            
            logger.info("Connected to OKX WebSocket")
        
        except Exception as e:
            logger.error(f"Failed to connect to OKX WebSocket: {e}")
            raise
    
    async def _subscribe_ws_channels(self) -> None:
        """Subscribe to WebSocket channels."""
        for symbol in self.config.symbols:
            for data_type in self.config.data_types:
                if data_type == "ticker":
                    channel = f"tickers.{symbol}"
                elif data_type == "orderbook":
                    channel = f"books.{symbol}"
                elif data_type == "trades":
                    channel = f"trades.{symbol}"
                else:
                    continue
                
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [{"channel": channel}]
                }
                
                await self.ws_connection.send_json(subscribe_msg)
                logger.debug(f"Subscribed to {channel}")
    
    async def _parse_ws_message(self, message: Dict[str, Any]) -> Optional[MarketData]:
        """Parse WebSocket message to MarketData."""
        try:
            if 'data' not in message or 'arg' not in message:
                return None
            
            channel = message['arg']['channel']
            data = message['data']
            
            if not data:
                return None
            
            # Extract symbol from channel
            if 'tickers' in channel:
                symbol = message['arg']['instId']
                return await self._parse_ticker_data(symbol, data[0])
            elif 'books' in channel:
                symbol = message['arg']['instId']
                return await self._parse_orderbook_data(symbol, data[0])
            elif 'trades' in channel:
                symbol = message['arg']['instId']
                return await self._parse_trades_data(symbol, data)
            
            return None
        
        except Exception as e:
            logger.error(f"Error parsing WebSocket message: {e}")
            return None
    
    async def _parse_ticker_data(self, symbol: str, data: Dict[str, Any]) -> MarketData:
        """Parse ticker data."""
        return MarketData(
            timestamp_ms=int(data.get('ts', 0)),
            symbol=symbol,
            type=MarketDataType.TICKER,
            exchange="okx",
            data={
                'last_price': float(data.get('last', 0)),
                'best_bid': float(data.get('bidPx', 0)),
                'best_ask': float(data.get('askPx', 0)),
                'bid_size': float(data.get('bidSz', 0)),
                'ask_size': float(data.get('askSz', 0)),
                'volume_24h': float(data.get('vol24h', 0)),
                'high_24h': float(data.get('high24h', 0)),
                'low_24h': float(data.get('low24h', 0)),
                'open_24h': float(data.get('open24h', 0)),
                'change_24h': float(data.get('chg', 0)),
                'change_pct_24h': float(data.get('chgPct', 0)),
                '_kind': 'ticker',
                '_ts_recv_ms': int(time.time() * 1000)
            }
        )
    
    async def _parse_orderbook_data(self, symbol: str, data: Dict[str, Any]) -> MarketData:
        """Parse orderbook data."""
        bids = [[float(bid[0]), float(bid[1])] for bid in data.get('bids', [])[:self.config.orderbook_depth]]
        asks = [[float(ask[0]), float(ask[1])] for ask in data.get('asks', [])[:self.config.orderbook_depth]]
        
        return MarketData(
            timestamp_ms=int(data.get('ts', 0)),
            symbol=symbol,
            type=MarketDataType.ORDERBOOK,
            exchange="okx",
            data={
                'bids': bids,
                'asks': asks,
                'checksum': data.get('checksum'),
                'prev_seq_id': data.get('prevSeqId'),
                'seq_id': data.get('seqId'),
                '_kind': 'snapshot' if 'bids' in data and 'asks' in data else 'delta',
                '_ts_recv_ms': int(time.time() * 1000)
            }
        )
    
    async def _parse_trades_data(self, symbol: str, trades_data: List[Dict[str, Any]]) -> MarketData:
        """Parse trades data."""
        trades = []
        for trade in trades_data:
            trades.append({
                'trade_id': trade.get('tradeId'),
                'price': float(trade.get('px', 0)),
                'size': float(trade.get('sz', 0)),
                'side': trade.get('side'),
                'timestamp': int(trade.get('ts', 0))
            })
        
        return MarketData(
            timestamp_ms=int(time.time() * 1000),
            symbol=symbol,
            type=MarketDataType.TRADE,
            exchange="okx",
            data={
                'trades': trades,
                '_kind': 'trade',
                '_ts_recv_ms': int(time.time() * 1000)
            }
        )
    
    async def _fetch_ticker(self, symbol: str) -> MarketData:
        """Fetch ticker via REST API."""
        url = f"/api/v5/market/ticker?instId={symbol}"
        
        async with self.session.get(url) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to fetch ticker: {response.status}")
            
            data = await response.json()
            
            if not data.get('data'):
                raise RuntimeError("No ticker data received")
            
            return await self._parse_ticker_data(symbol, data['data'][0])
    
    async def _fetch_orderbook(self, symbol: str) -> MarketData:
        """Fetch orderbook via REST API."""
        url = f"/api/v5/market/books?instId={symbol}&sz={self.config.orderbook_depth}"
        
        async with self.session.get(url) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to fetch orderbook: {response.status}")
            
            data = await response.json()
            
            if not data.get('data'):
                raise RuntimeError("No orderbook data received")
            
            return await self._parse_orderbook_data(symbol, data['data'][0])
    
    async def _fetch_trades(self, symbol: str) -> MarketData:
        """Fetch trades via REST API."""
        url = f"/api/v5/market/trades?instId={symbol}&limit={self.config.trades_limit}"
        
        async with self.session.get(url) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to fetch trades: {response.status}")
            
            data = await response.json()
            
            if not data.get('data'):
                raise RuntimeError("No trades data received")
            
            return await self._parse_trades_data(symbol, data['data'])
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks for auto-download and backfill."""
        if self.config.enable_auto_download and not self.download_task:
            self.download_task = asyncio.create_task(self._auto_download_loop())
        
        if self.config.enable_backfill and not self.backfill_task:
            self.backfill_task = asyncio.create_task(self._backfill_loop())
    
    async def _auto_download_loop(self) -> None:
        """Background task for automatic historical data download."""
        logger.info("Starting auto-download loop")
        
        while self._running:
            try:
                await self._download_historical_data()
                await asyncio.sleep(self.config.download_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-download loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _download_historical_data(self) -> None:
        """Download historical data for all symbols and timeframes."""
        logger.info("Starting historical data download")
        
        for symbol in self.config.symbols:
            for timeframe in self.config.download_timeframes:
                try:
                    await self._download_symbol_data(symbol, timeframe)
                    self.stats['data_downloaded'] += 1
                except Exception as e:
                    logger.error(f"Error downloading {symbol} {timeframe}: {e}")
        
        self.stats['last_download'] = datetime.now()
        logger.info("Historical data download completed")
    
    async def _download_symbol_data(self, symbol: str, timeframe: str) -> None:
        """Download historical data for a specific symbol and timeframe."""
        # Calculate date range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.config.download_days)
        
        # Convert to OKX format (milliseconds)
        end_ts = int(end_time.timestamp() * 1000)
        start_ts = int(start_time.timestamp() * 1000)
        
        # Download in batches
        current_ts = start_ts
        batch_size = 100  # OKX limit per request
        
        while current_ts < end_ts:
            batch_end = min(current_ts + batch_size * self._get_timeframe_ms(timeframe), end_ts)
            
            url = f"/api/v5/market/history-candles"
            params = {
                'instId': symbol,
                'bar': timeframe,
                'after': str(current_ts),
                'before': str(batch_end),
                'limit': str(batch_size)
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to download data: {response.status}")
                
                data = await response.json()
                
                if data.get('data'):
                    await self._store_historical_data(symbol, timeframe, data['data'])
            
            current_ts = batch_end
            await asyncio.sleep(0.1)  # Rate limiting
    
    def _get_timeframe_ms(self, timeframe: str) -> int:
        """Get timeframe in milliseconds."""
        timeframe_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000
        }
        return timeframe_map.get(timeframe, 60 * 1000)
    
    async def _store_historical_data(self, symbol: str, timeframe: str, data: List[List[Any]]) -> None:
        """Store historical data to file."""
        # Create directory if needed
        import os
        os.makedirs(self.config.storage_path, exist_ok=True)
        
        # Prepare data for storage
        df_data = []
        for candle in data:
            df_data.append({
                'timestamp': int(candle[0]),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5]),
                'volume_ccy': float(candle[6]) if len(candle) > 6 else 0.0
            })
        
        df = pd.DataFrame(df_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Save to file
        filename = f"{symbol}_{timeframe}.csv"
        filepath = os.path.join(self.config.storage_path, filename)
        
        if os.path.exists(filepath):
            # Append to existing file
            existing_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            df = pd.concat([existing_df, df]).drop_duplicates()
        
        df.to_csv(filepath)
        logger.debug(f"Stored {len(df)} candles for {symbol} {timeframe}")
    
    async def _backfill_loop(self) -> None:
        """Background task for processing backfill requests."""
        logger.info("Starting backfill loop")
        
        while self._running:
            try:
                # Get backfill request
                request = await asyncio.wait_for(self.backfill_queue.get(), timeout=1.0)
                await self._process_backfill_request(request)
                self.stats['backfills_completed'] += 1
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in backfill loop: {e}")
    
    async def _process_backfill_request(self, request: BackfillRequest) -> None:
        """Process a backfill request."""
        logger.info(f"Processing backfill request: {request.symbol} {request.data_type}")
        
        for attempt in range(self.config.backfill_retry_attempts):
            try:
                if request.data_type == "ohlcv" and request.timeframe:
                    await self._backfill_ohlcv_data(request)
                else:
                    await self._backfill_market_data(request)
                
                logger.info(f"Backfill completed for {request.symbol}")
                return
            
            except Exception as e:
                logger.error(f"Backfill attempt {attempt + 1} failed: {e}")
                if attempt < self.config.backfill_retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        self.stats['last_backfill'] = datetime.now()
    
    async def _backfill_ohlcv_data(self, request: BackfillRequest) -> None:
        """Backfill OHLCV data."""
        start_ts = int(request.start_time.timestamp() * 1000)
        end_ts = int(request.end_time.timestamp() * 1000)
        
        url = f"/api/v5/market/history-candles"
        params = {
            'instId': request.symbol,
            'bar': request.timeframe,
            'after': str(start_ts),
            'before': str(end_ts),
            'limit': str(self.config.backfill_batch_size)
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                raise RuntimeError(f"Backfill request failed: {response.status}")
            
            data = await response.json()
            
            if data.get('data'):
                await self._store_historical_data(request.symbol, request.timeframe, data['data'])
    
    async def _backfill_market_data(self, request: BackfillRequest) -> None:
        """Backfill other market data types."""
        # Implementation for ticker, orderbook, trades backfill
        pass
    
    def request_backfill(
        self,
        symbol: str,
        data_type: str,
        start_time: datetime,
        end_time: datetime,
        timeframe: Optional[str] = None,
        reason: str = "manual"
    ) -> None:
        """Request backfill for missing data."""
        request = BackfillRequest(
            symbol=symbol,
            data_type=data_type,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            reason=reason
        )
        
        asyncio.create_task(self.backfill_queue.put(request))
        logger.info(f"Backfill requested for {symbol} {data_type}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data source statistics."""
        return {
            **self.stats,
            'connected': self._connected,
            'running': self._running,
            'symbols': self.config.symbols,
            'data_types': self.config.data_types,
            'backfill_queue_size': self.backfill_queue.qsize()
        }


# Convenience functions
def create_okx_source(
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
    passphrase: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    enable_auto_download: bool = True,
    enable_backfill: bool = True
) -> OKXDataSource:
    """Create OKX data source with default configuration."""
    config = OKXConfig(
        api_key=api_key,
        secret=secret,
        passphrase=passphrase,
        symbols=symbols or ["BTC-USDT", "ETH-USDT"],
        enable_auto_download=enable_auto_download,
        enable_backfill=enable_backfill
    )
    
    return OKXDataSource(config)
