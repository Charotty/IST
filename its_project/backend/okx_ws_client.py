"""
OKX WebSocket Client for Market Data
=====================================

Implements OKX WebSocket subscriptions for:
- Tickers
- Orderbook L2
- Candles/OHLCV
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import websockets

from common.backend_contract import (
    PriceUpdateEvent,
    OrderbookUpdateEvent,
    CandleUpdateEvent,
    ChannelType,
    SymbolMapper,
)

logger = logging.getLogger(__name__)


class OKXWSClient:
    """OKX WebSocket client for market data."""
    
    # OKX WS endpoint
    WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
    
    def __init__(self):
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.subscribed_channels: Dict[str, List[str]] = {}  # channel -> symbols
        self.callbacks: Dict[str, List[Callable]] = {
            'price': [],
            'orderbook': [],
            'candle': [],
        }
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Ping/pong mechanism
        self.ping_interval = 20  # seconds (must be < 30)
        self.last_pong_time: Optional[float] = None
        self.ping_task: Optional[asyncio.Task] = None
    
    def on_price(self, callback: Callable[[PriceUpdateEvent], None]):
        """Register callback for price updates."""
        self.callbacks['price'].append(callback)
    
    def on_orderbook(self, callback: Callable[[OrderbookUpdateEvent], None]):
        """Register callback for orderbook updates."""
        self.callbacks['orderbook'].append(callback)
    
    def on_candle(self, callback: Callable[[CandleUpdateEvent], None]):
        """Register callback for candle updates."""
        self.callbacks['candle'].append(callback)
    
    async def connect(self) -> bool:
        """Connect to OKX WebSocket."""
        try:
            logger.info(f"Connecting to OKX WebSocket: {self.WS_URL}")
            self.websocket = await websockets.connect(self.WS_URL)
            self.connected = True
            self.reconnect_attempts = 0
            logger.info("Connected to OKX WebSocket")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to OKX WebSocket: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from OKX WebSocket."""
        self.running = False
        
        # Cancel ping task
        if self.ping_task and not self.ping_task.done():
            self.ping_task.cancel()
            try:
                await self.ping_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.connected = False
        logger.info("Disconnected from OKX WebSocket")
    
    async def subscribe_tickers(self, symbols: List[str]):
        """Subscribe to tickers channel."""
        # Convert GUI symbols to OKX format
        okx_symbols = [SymbolMapper.gui_to_okx(s) for s in symbols]
        
        message = {
            "op": "subscribe",
            "args": [{"channel": "tickers", "instId": inst_id} for inst_id in okx_symbols]
        }
        
        await self._send_message(message)
        self.subscribed_channels['tickers'] = symbols
        logger.info(f"Subscribed to tickers for {len(symbols)} symbols")
    
    async def subscribe_orderbook(self, symbols: List[str], depth: int = 20):
        """Subscribe to orderbook L2 channel."""
        okx_symbols = [SymbolMapper.gui_to_okx(s) for s in symbols]
        
        # OKX uses books5 for top 5 levels, books for full L2
        channel = "books5" if depth <= 5 else "books"
        
        message = {
            "op": "subscribe",
            "args": [{"channel": channel, "instId": inst_id} for inst_id in okx_symbols]
        }
        
        await self._send_message(message)
        self.subscribed_channels['orderbook'] = symbols
        logger.info(f"Subscribed to orderbook ({channel}) for {len(symbols)} symbols")
    
    async def subscribe_candles(self, symbols: List[str], timeframes: List[str]):
        """Subscribe to candles channel."""
        okx_symbols = [SymbolMapper.gui_to_okx(s) for s in symbols]
        
        message = {
            "op": "subscribe",
            "args": [
                {"channel": f"candle{tf}", "instId": inst_id}
                for inst_id in okx_symbols
                for tf in timeframes
            ]
        }
        
        await self._send_message(message)
        self.subscribed_channels['candles'] = symbols
        logger.info(f"Subscribed to candles for {len(symbols)} symbols, timeframes: {timeframes}")
    
    async def unsubscribe_all(self):
        """Unsubscribe from all channels."""
        if not self.subscribed_channels:
            return
        
        args = []
        for channel, symbols in self.subscribed_channels.items():
            okx_symbols = [SymbolMapper.gui_to_okx(s) for s in symbols]
            for inst_id in okx_symbols:
                args.append({"channel": channel, "instId": inst_id})
        
        message = {"op": "unsubscribe", "args": args}
        await self._send_message(message)
        self.subscribed_channels.clear()
        logger.info("Unsubscribed from all channels")
    
    async def _send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket."""
        if self.websocket and self.connected:
            try:
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
    
    async def _message_loop(self):
        """Main message loop."""
        self.running = True
        
        # Start ping task
        self.ping_task = asyncio.create_task(self._ping_loop())
        
        while self.running and self.connected:
            try:
                message = await self.websocket.recv()
                await self._handle_message(message)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.connected = False
                break
            except Exception as e:
                logger.error(f"Error in message loop: {e}")
                break
        
        # Cancel ping task
        if self.ping_task and not self.ping_task.done():
            self.ping_task.cancel()
        
        # Attempt reconnection
        if self.running and not self.connected:
            await self._reconnect()
    
    async def _reconnect(self):
        """Attempt to reconnect."""
        while self.running and self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            logger.info(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
            
            await asyncio.sleep(self.reconnect_delay)
            
            if await self.connect():
                # Resubscribe to previous channels
                if 'tickers' in self.subscribed_channels:
                    await self.subscribe_tickers(self.subscribed_channels['tickers'])
                if 'orderbook' in self.subscribed_channels:
                    await self.subscribe_orderbook(self.subscribed_channels['orderbook'])
                if 'candles' in self.subscribed_channels:
                    await self.subscribe_candles(
                        self.subscribed_channels['candles'],
                        self.subscribed_channels.get('candle_timeframes', ['1m'])
                    )
                
                # Restart message loop
                await self._message_loop()
                return
        
        logger.error("Max reconnection attempts reached")
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            # Handle different message types
            if data.get('event') == 'subscribe':
                logger.debug(f"Subscription confirmed: {data}")
            elif data.get('event') == 'unsubscribe':
                logger.debug(f"Unsubscribe confirmed: {data}")
            elif data.get('event') == 'error':
                logger.error(f"Error from OKX: {data}")
            elif data == 'pong':
                # Handle pong response
                self.last_pong_time = time.time()
                logger.debug("Received pong from OKX")
            elif 'data' in data:
                # Market data message
                await self._handle_market_data(data)
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _ping_loop(self):
        """Ping loop to keep connection alive."""
        while self.running and self.connected:
            try:
                await asyncio.sleep(self.ping_interval)
                
                if not self.connected:
                    break
                
                # Check if we received pong recently
                if self.last_pong_time is not None:
                    time_since_pong = time.time() - self.last_pong_time
                    if time_since_pong > self.ping_interval * 2:
                        logger.warning(f"No pong received for {time_since_pong:.1f}s, reconnecting")
                        self.connected = False
                        break
                
                # Send ping
                await self.websocket.send('ping')
                logger.debug("Sent ping to OKX")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ping loop: {e}")
                self.connected = False
                break
    
    async def _handle_market_data(self, data: Dict[str, Any]):
        """Handle market data message."""
        try:
            arg = data.get('arg', {})
            channel = arg.get('channel', '')
            inst_id = arg.get('instId', '')
            
            # Convert OKX symbol back to GUI format
            gui_symbol = SymbolMapper.okx_to_gui(inst_id)
            
            if channel == 'tickers':
                await self._handle_ticker(data['data'], gui_symbol)
            elif channel in ['books', 'books5']:
                await self._handle_orderbook(data['data'], gui_symbol)
            elif channel.startswith('candle'):
                timeframe = channel.replace('candle', '')
                await self._handle_candle(data['data'], gui_symbol, timeframe)
        except Exception as e:
            logger.error(f"Error handling market data: {e}")
    
    async def _handle_ticker(self, data: List[Dict], symbol: str):
        """Handle ticker data."""
        if not data:
            return
        
        ticker = data[0]
        
        event = PriceUpdateEvent(
            symbol=symbol,
            last=float(ticker.get('last', 0)),
            bid=float(ticker.get('bidPx', 0)),
            ask=float(ticker.get('askPx', 0)),
            ts_exchange=int(ticker.get('ts', 0)),
            ts_local=int(time.time() * 1000),
            volume_24h=float(ticker.get('vol24h', 0)),
            change_24h=float(ticker.get('chg', 0)) if ticker.get('chg') else None,
            high_24h=float(ticker.get('high24h', 0)) if ticker.get('high24h') else None,
            low_24h=float(ticker.get('low24h', 0)) if ticker.get('low24h') else None,
        )
        
        for callback in self.callbacks['price']:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in price callback: {e}")
    
    async def _handle_orderbook(self, data: List[Dict], symbol: str):
        """Handle orderbook data."""
        if not data:
            return
        
        book = data[0]
        
        # OKX format: [price, size, orders, count] - we only need price and size
        bids = [[float(level[0]), float(level[1])] for level in book.get('bids', [])]
        asks = [[float(level[0]), float(level[1])] for level in book.get('asks', [])]
        
        event = OrderbookUpdateEvent(
            symbol=symbol,
            bids=bids,
            asks=asks,
            ts_exchange=int(book.get('ts', 0)),
            ts_local=int(time.time() * 1000),
            depth=len(bids),
            checksum=int(book.get('checksum', 0)) if book.get('checksum') else None,
        )
        
        for callback in self.callbacks['orderbook']:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in orderbook callback: {e}")
    
    async def _handle_candle(self, data: List[Dict], symbol: str, timeframe: str):
        """Handle candle data."""
        if not data:
            return
        
        candle = data[0]
        
        # OKX candle format: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        if isinstance(candle, list):
            ts_open = int(candle[0])
            open_price = float(candle[1])
            high_price = float(candle[2])
            low_price = float(candle[3])
            close_price = float(candle[4])
            volume = float(candle[5])
            is_closed = candle[8] == '1'
        else:
            # Alternative format
            ts_open = int(candle.get('ts', 0))
            open_price = float(candle.get('o', 0))
            high_price = float(candle.get('h', 0))
            low_price = float(candle.get('l', 0))
            close_price = float(candle.get('c', 0))
            volume = float(candle.get('vol', 0))
            is_closed = candle.get('confirm') == '1'
        
        event = CandleUpdateEvent(
            symbol=symbol,
            timeframe=timeframe,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            ts_open=ts_open,
            ts_local=int(time.time() * 1000),
            is_closed=is_closed,
        )
        
        for callback in self.callbacks['candle']:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in candle callback: {e}")
    
    def start(self):
        """Start the WebSocket client in a new event loop."""
        if self.loop is None or self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        # Run connection and message loop
        self.loop.run_until_complete(self._run())
    
    async def _run(self):
        """Run the client."""
        if await self.connect():
            await self._message_loop()
    
    def stop(self):
        """Stop the WebSocket client."""
        if self.loop and not self.loop.is_closed():
            self.loop.run_until_complete(self.disconnect())
            self.loop.close()
