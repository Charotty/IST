#!/usr/bin/env python3
"""
Enhanced CCXT Data Source with Integrity Monitoring
================================================

Production-ready data source with:
- Gap detection and integrity monitoring
- WebSocket reconnection (where available)
- Replay mechanism for missing data
- Enhanced order book depth
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator, Optional, Any, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

import ccxt.async_support as ccxt

from its_project.data_layer.base import BaseDataSource
from its_project.data_layer.integrity import DataIntegrityMonitor, ValidationReport, GapInfo
from its_project.common.types import MarketData, MarketDataType

logger = logging.getLogger(__name__)


@dataclass
class ReplayRequest:
    """Request for data replay."""
    symbol: str
    start_time: int  # ms timestamp
    end_time: int    # ms timestamp
    reason: str      # Why replay is needed (gap, reconnect, etc.)


class EnhancedCCXTDataSource(BaseDataSource):
    """
    Enhanced CCXT data source with production features.
    
    Key features:
    - Integrated data integrity monitoring
    - Automatic gap detection and replay
    - WebSocket support where available
    - Enhanced order book depth
    - Reconnection logic
    """
    
    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        sandbox: bool = False,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.secret = secret
        self.sandbox = sandbox
        self.config = config or {}
        
        # CCXT exchange instance
        self.exchange: Optional[ccxt.Exchange] = None
        
        # Enhanced features
        self.integrity_monitor: Optional[DataIntegrityMonitor] = None
        self.replay_queue: asyncio.Queue[ReplayRequest] = asyncio.Queue()
        self.data_buffer: Dict[str, List[MarketData]] = {}
        self.max_buffer_size = config.get('max_buffer_size', 1000)
        
        # Order book configuration
        self.lob_depth = config.get('lob_depth', 50)  # Enhanced depth
        self.lob_normalization = config.get('lob_normalization', True)
        
        # WebSocket support (where available)
        self.use_websocket = config.get('use_websocket', False)
        self.ws_client: Optional[Any] = None
        self.ws_reconnect_attempts = 0
        self.max_ws_reconnect_attempts = config.get('max_ws_reconnect_attempts', 5)
        self.ws_reconnect_delay = config.get('ws_reconnect_delay', 5.0)
        
        # State
        self._symbols: List[str] = []
        self._running = False
        self._replay_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'gaps_detected': 0,
            'replays_completed': 0,
            'reconnects': 0,
            'last_update': None
        }
    
    async def connect(self) -> None:
        """Connect to exchange with enhanced features."""
        if self._connected:
            logger.warning(f"Already connected to {self.exchange_id}")
            return
        
        try:
            # Initialize CCXT exchange
            exchange_class = getattr(ccxt, self.exchange_id)
            
            ccxt_config = {
                'apiKey': self.api_key,
                'secret': self.secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,  # Handle clock skew
                },
                **self.config
            }
            
            if self.sandbox:
                ccxt_config['sandboxMode'] = True
            
            self.exchange = exchange_class(ccxt_config)
            
            # Load markets
            await self.exchange.load_markets()
            
            # Initialize integrity monitor
            integrity_config = {
                'max_gap_ms': self.config.get('max_gap_ms', 5000),
                'price_spike_threshold': self.config.get('price_spike_threshold', 0.1),
                'volume_spike_threshold': self.config.get('volume_spike_threshold', 10.0),
                'history_size': self.config.get('history_size', 1000)
            }
            self.integrity_monitor = DataIntegrityMonitor(integrity_config)
            
            # Initialize WebSocket if requested and available
            if self.use_websocket and hasattr(self.exchange, 'watchTicker'):
                await self._init_websocket()
            
            # Start replay task
            self._replay_task = asyncio.create_task(self._replay_worker())
            
            self._connected = True
            self._running = True
            
            logger.info(f"Enhanced CCXT connected to {self.exchange_id}")
            logger.info(f"LOB depth: {self.lob_depth}, WebSocket: {self.use_websocket}")
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_id}: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect with cleanup."""
        self._running = False
        self._connected = False
        
        # Stop replay task
        if self._replay_task:
            self._replay_task.cancel()
            try:
                await self._replay_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket
        if self.ws_client:
            try:
                await self.ws_client.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        
        # Close CCXT connection
        if self.exchange:
            await self.exchange.close()
        
        logger.info(f"Enhanced CCXT disconnected from {self.exchange_id}")
    
    async def subscribe(self, symbols: Optional[List[str]] = None) -> AsyncIterator[MarketData]:
        """Subscribe with enhanced monitoring and gap handling."""
        if not self._connected:
            await self.connect()
        
        if symbols:
            self._symbols = symbols
        
        if not self._symbols:
            logger.warning("No symbols to subscribe to")
            return
        
        logger.info(f"Starting enhanced subscription for {len(self._symbols)} symbols")
        
        # Choose subscription method
        if self.use_websocket and self.ws_client:
            async for data in self._websocket_subscribe():
                yield data
        else:
            async for data in self._polling_subscribe():
                yield data
    
    async def _websocket_subscribe(self) -> AsyncIterator[MarketData]:
        """WebSocket subscription with reconnection."""
        while self._running:
            try:
                if not self.ws_client:
                    await self._init_websocket()
                
                # Subscribe to symbols
                for symbol in self._symbols:
                    try:
                        # Watch ticker and order book
                        ticker_task = asyncio.create_task(
                            self.exchange.watch_ticker(symbol)
                        )
                        orderbook_task = asyncio.create_task(
                            self.exchange.watch_order_book(symbol, limit=self.lob_depth)
                        )
                        
                        # Wait for either data
                        done, pending = await asyncio.wait(
                            [ticker_task, orderbook_task],
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        
                        # Cancel pending tasks
                        for task in pending:
                            task.cancel()
                        
                        # Process completed task
                        for task in done:
                            try:
                                result = task.result()
                                market_data = await self._process_websocket_data(symbol, result)
                                if market_data:
                                    yield market_data
                            except Exception as e:
                                logger.error(f"Error processing WebSocket data: {e}")
                    
                    except Exception as e:
                        logger.error(f"Error in WebSocket subscription for {symbol}: {e}")
                        await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await self._handle_websocket_reconnect()
    
    async def _polling_subscribe(self) -> AsyncIterator[MarketData]:
        """Enhanced polling subscription with gap detection."""
        while self._running:
            for symbol in self._symbols:
                try:
                    # Fetch enhanced data
                    data = await self.fetch_enhanced(symbol)
                    if data:
                        yield data
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    await asyncio.sleep(0.1)
            
            # Polling interval
            await asyncio.sleep(self.config.get('polling_interval', 1.0))
    
    async def fetch_enhanced(self, symbol: str, **params) -> Optional[MarketData]:
        """Fetch enhanced market data with deep order book."""
        if not self._connected:
            await self.connect()
        
        try:
            # Fetch ticker data
            ticker = await self.exchange.fetch_ticker(symbol)
            
            # Fetch deep order book
            orderbook = await self.exchange.fetch_order_book(symbol, limit=self.lob_depth)
            
            # Normalize order book if requested
            if self.lob_normalization:
                orderbook = self._normalize_orderbook(orderbook, ticker.get('last'))
            
            # Create enhanced MarketData
            market_data = MarketData(
                timestamp_ms=ticker.get('timestamp', int(time.time() * 1000)),
                symbol=symbol,
                type=MarketDataType.TICKER,
                exchange=self.exchange_id,
                data={
                    'open': ticker.get('open'),
                    'high': ticker.get('high'),
                    'low': ticker.get('low'),
                    'close': ticker.get('last'),
                    'volume': ticker.get('baseVolume'),
                    'bid': ticker.get('bid'),
                    'ask': ticker.get('ask'),
                    'bid_volume': ticker.get('bidVolume'),
                    'ask_volume': ticker.get('askVolume'),
                    'orderbook': {
                        'bids': orderbook.get('bids', [])[:self.lob_depth],
                        'asks': orderbook.get('asks', [])[:self.lob_depth]
                    },
                    # Enhanced features
                    'microprice': self._calculate_microprice(orderbook),
                    'spread': self._calculate_spread(orderbook),
                    'depth_ratio': self._calculate_depth_ratio(orderbook)
                }
            )
            
            # Validate data integrity
            if self.integrity_monitor:
                report = self.integrity_monitor.validate_data_point(market_data)
                
                # Handle gaps
                for gap in report.gaps:
                    self.stats['gaps_detected'] += 1
                    logger.warning(f"Gap detected: {gap.gap_type} ({gap.duration_ms}ms) for {symbol}")
                    
                    # Queue replay request for missing data
                    if gap.gap_type == "missing":
                        replay_req = ReplayRequest(
                            symbol=symbol,
                            start_time=gap.start_time,
                            end_time=gap.end_time,
                            reason=f"gap_{gap.gap_type}"
                        )
                        await self.replay_queue.put(replay_req)
                
                # Log quality issues
                if not report.is_valid:
                    logger.warning(f"Data quality issues for {symbol}: {report.issues}")
            
            # Update buffer
            self._update_buffer(symbol, market_data)
            
            # Update stats
            self.stats['messages_received'] += 1
            self.stats['last_update'] = datetime.now().isoformat()
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching enhanced data for {symbol}: {e}")
            raise
    
    async def _process_websocket_data(self, symbol: str, data: Any) -> Optional[MarketData]:
        """Process WebSocket data into MarketData format."""
        try:
            # This depends on the specific exchange WebSocket format
            # Simplified implementation - would need exchange-specific handling
            
            if isinstance(data, dict) and 'symbol' in data:
                # Ticker data
                if 'bid' in data and 'ask' in data:
                    return MarketData(
                        timestamp_ms=data.get('timestamp', int(time.time() * 1000)),
                        symbol=symbol,
                        type=MarketDataType.TICKER,
                        exchange=self.exchange_id,
                        data=data
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing WebSocket data: {e}")
            return None
    
    async def _init_websocket(self) -> None:
        """Initialize WebSocket connection."""
        try:
            if hasattr(self.exchange, 'watchTicker'):
                logger.info(f"Initializing WebSocket for {self.exchange_id}")
                # WebSocket initialization depends on exchange
                # This is a placeholder - actual implementation varies by exchange
                self.ws_client = self.exchange
                self.ws_reconnect_attempts = 0
            else:
                logger.warning(f"WebSocket not available for {self.exchange_id}")
                self.use_websocket = False
                
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket: {e}")
            self.use_websocket = False
    
    async def _handle_websocket_reconnect(self) -> None:
        """Handle WebSocket reconnection."""
        if self.ws_reconnect_attempts >= self.max_ws_reconnect_attempts:
            logger.error("Max WebSocket reconnection attempts reached, falling back to polling")
            self.use_websocket = False
            self.ws_client = None
            return
        
        self.ws_reconnect_attempts += 1
        self.stats['reconnects'] += 1
        
        logger.info(f"WebSocket reconnection attempt {self.ws_reconnect_attempts}/{self.max_ws_reconnect_attempts}")
        
        # Close existing connection
        if self.ws_client:
            try:
                await self.ws_client.close()
            except:
                pass
            self.ws_client = None
        
        # Wait before reconnecting
        await asyncio.sleep(self.ws_reconnect_delay)
        
        # Try to reconnect
        try:
            await self._init_websocket()
        except Exception as e:
            logger.error(f"WebSocket reconnection failed: {e}")
    
    async def _replay_worker(self) -> None:
        """Background worker for handling replay requests."""
        logger.info("Replay worker started")
        
        while self._running:
            try:
                # Get replay request
                replay_req = await asyncio.wait_for(
                    self.replay_queue.get(), 
                    timeout=1.0
                )
                
                # Process replay
                await self._process_replay_request(replay_req)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in replay worker: {e}")
        
        logger.info("Replay worker stopped")
    
    async def _process_replay_request(self, replay_req: ReplayRequest) -> None:
        """Process a single replay request."""
        try:
            logger.info(f"Processing replay request for {replay_req.symbol}: {replay_req.reason}")
            
            # Fetch historical data to fill gap
            ohlcv = await self.exchange.fetch_ohlcv(
                replay_req.symbol,
                timeframe='1m',  # Use 1m for replay
                since=replay_req.start_time,
                limit=100
            )
            
            # Convert OHLCV to MarketData points
            for candle in ohlcv:
                timestamp, open_price, high, low, close, volume = candle
                
                if replay_req.start_time <= timestamp <= replay_req.end_time:
                    market_data = MarketData(
                        timestamp_ms=timestamp,
                        symbol=replay_req.symbol,
                        type=MarketDataType.OHLCV,
                        exchange=self.exchange_id,
                        data={
                            'open': open_price,
                            'high': high,
                            'low': low,
                            'close': close,
                            'volume': volume,
                            'replay': True,
                            'replay_reason': replay_req.reason
                        }
                    )
                    
                    # Add to buffer (don't yield from replay worker)
                    self._update_buffer(replay_req.symbol, market_data)
            
            self.stats['replays_completed'] += 1
            logger.info(f"Replay completed for {replay_req.symbol}: {len(ohlcv)} candles")
            
        except Exception as e:
            logger.error(f"Error processing replay request: {e}")
    
    def _normalize_orderbook(self, orderbook: Dict, mid_price: Optional[float]) -> Dict:
        """Normalize order book prices and volumes."""
        if not mid_price:
            return orderbook
        
        normalized = {'bids': [], 'asks': []}
        
        # Normalize bids (relative to mid price)
        for price, volume in orderbook.get('bids', []):
            norm_price = (price - mid_price) / mid_price
            norm_volume = volume / mid_price  # Volume in base currency units
            normalized['bids'].append([norm_price, norm_volume])
        
        # Normalize asks
        for price, volume in orderbook.get('asks', []):
            norm_price = (price - mid_price) / mid_price
            norm_volume = volume / mid_price
            normalized['asks'].append([norm_price, norm_volume])
        
        return normalized
    
    def _calculate_microprice(self, orderbook: Dict) -> Optional[float]:
        """Calculate microprice (mid price weighted by volume)."""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return None
        
        best_bid_price, best_bid_volume = bids[0]
        best_ask_price, best_ask_volume = asks[0]
        
        total_volume = best_bid_volume + best_ask_volume
        if total_volume == 0:
            return None
        
        microprice = (best_bid_price * best_ask_volume + best_ask_price * best_bid_volume) / total_volume
        return microprice
    
    def _calculate_spread(self, orderbook: Dict) -> Optional[float]:
        """Calculate bid-ask spread."""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return None
        
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        
        return best_ask - best_bid
    
    def _calculate_depth_ratio(self, orderbook: Dict) -> Optional[float]:
        """Calculate depth ratio (bid volume / ask volume)."""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return None
        
        # Sum top 5 levels
        bid_volume = sum(volume for _, volume in bids[:5])
        ask_volume = sum(volume for _, volume in asks[:5])
        
        if ask_volume == 0:
            return None
        
        return bid_volume / ask_volume
    
    def _update_buffer(self, symbol: str, data: MarketData) -> None:
        """Update circular buffer for symbol."""
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = []
        
        self.data_buffer[symbol].append(data)
        
        # Keep buffer size limited
        if len(self.data_buffer[symbol]) > self.max_buffer_size:
            self.data_buffer[symbol] = self.data_buffer[symbol][-self.max_buffer_size:]
    
    async def get_buffered_data(self, symbol: str, limit: int = 100) -> List[MarketData]:
        """Get buffered data for symbol."""
        if symbol not in self.data_buffer:
            return []
        
        return self.data_buffer[symbol][-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics."""
        base_stats = self.stats.copy()
        
        if self.integrity_monitor:
            base_stats.update(self.integrity_monitor.get_system_stats())
        
        base_stats.update({
            'symbols_count': len(self._symbols),
            'buffer_sizes': {sym: len(data) for sym, data in self.data_buffer.items()},
            'websocket_connected': self.ws_client is not None,
            'reconnect_attempts': self.ws_reconnect_attempts
        })
        
        return base_stats


# Convenience functions
async def create_enhanced_binance_source(
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
    sandbox: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> EnhancedCCXTDataSource:
    """Create enhanced Binance data source."""
    default_config = {
        'lob_depth': 50,
        'lob_normalization': True,
        'use_websocket': True,
        'max_gap_ms': 5000,
        'price_spike_threshold': 0.1
    }
    
    if config:
        default_config.update(config)
    
    source = EnhancedCCXTDataSource(
        exchange_id="binance",
        api_key=api_key,
        secret=secret,
        sandbox=sandbox,
        config=default_config
    )
    
    await source.connect()
    return source


if __name__ == "__main__":
    # Test enhanced data source
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        config = {
            'lob_depth': 20,
            'use_websocket': False,  # Use polling for test
            'max_gap_ms': 5000
        }
        
        source = EnhancedCCXTDataSource("binance", config=config)
        await source.connect()
        
        # Test enhanced fetch
        data = await source.fetch_enhanced("BTC/USDT")
        if data:
            print(f"Enhanced data received for {data.symbol}")
            print(f"  Microprice: {data.data.get('microprice')}")
            print(f"  Spread: {data.data.get('spread')}")
            print(f"  Depth ratio: {data.data.get('depth_ratio')}")
            print(f"  LOB depth: {len(data.data.get('orderbook', {}).get('bids', []))}")
        
        # Test stats
        print(f"\nStats: {source.get_stats()}")
        
        await source.disconnect()
    
    asyncio.run(test())
